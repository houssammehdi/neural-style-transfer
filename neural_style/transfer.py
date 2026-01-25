"""The optimisation that performs style transfer.

The synthesised image's pixels are the only parameters; VGG-19 stays frozen.
The objective is ``alpha * L_content + beta * L_style + gamma * TV``, where the
style term optionally uses guided Gram matrices so that different styles apply
to different regions (Gatys et al., 2017).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal, get_args

import torch
import torch.nn.functional as F

from .color import ColorMatchMethod, luminance, match_color, match_luminance, preserve_colors
from .image import resize, resize_to_area
from .losses import gram_matrix, guided_gram_matrix, total_variation
from .model import VGG19, FeatureExtractor, Pooling

ColorMode = Literal["style", "luminance", "match"]
OptimizerName = Literal["lbfgs", "adam"]
InitName = Literal["content", "noise"]


@dataclass(frozen=True)
class Preset:
    """Default loss layers and weights for one source of VGG-19 weights."""

    content_layers: tuple[str, ...]
    style_layers: tuple[str, ...]
    content_weight: float
    style_weight: float
    pooling: Pooling


TUTORIAL_PRESET = Preset(
    content_layers=("conv2_2",),
    style_layers=("conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1"),
    content_weight=1.0,
    style_weight=1e6,
    pooling="max",
)
"""The v0.1 defaults (those of the PyTorch neural-transfer tutorial), kept for torchvision weights."""

GATYS_PRESET = Preset(
    content_layers=("relu4_2",),
    style_layers=("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"),
    content_weight=1.0,
    style_weight=1e2,
    pooling="avg",
)
"""Layers and pooling of Gatys et al. (2016); weights calibrated on the Caffe VGG-19 (``docs/method.md``)."""

PRESETS: dict[str, Preset] = {
    "torchvision": TUTORIAL_PRESET,
    "random": TUTORIAL_PRESET,
    "caffe": GATYS_PRESET,
}
"""Default :class:`Preset` for each :class:`~neural_style.model.VGG19` ``source``."""


@dataclass(frozen=True)
class TransferConfig:
    """Hyper-parameters of one style-transfer run.

    ``None`` for the layers, the content/style weights or the pooling means
    "use the :data:`PRESETS` entry of the network's weight source".

    ``style_blend`` weights the style images. Without masks they are normalised
    and their Gram matrices averaged, which interpolates between styles. With
    masks each style owns one region and its weight scales that region's style
    strength (default 1 each).

    ``color`` selects the colour handling of Gatys et al. (2016): ``"style"``
    (plain transfer), ``"luminance"`` (luminance-only transfer, the content's
    colours are restored) or ``"match"`` (style images recoloured to the
    content's colour mean and covariance first, using ``color_match``).
    """

    steps: int = 300
    content_weight: float | None = None
    style_weight: float | None = None
    tv_weight: float = 0.0
    content_layers: tuple[str, ...] | None = None
    style_layers: tuple[str, ...] | None = None
    pooling: Pooling | None = None
    optimizer: OptimizerName = "lbfgs"
    lr: float = 0.02
    init: InitName = "content"
    style_blend: tuple[float, ...] | None = None
    color: ColorMode = "style"
    color_match: ColorMatchMethod = "eigen"
    seed: int = 0

    def __post_init__(self) -> None:
        choices: dict[str, tuple[str | None, ...]] = {
            "pooling": (*get_args(Pooling), None),
            "optimizer": get_args(OptimizerName),
            "init": get_args(InitName),
            "color": get_args(ColorMode),
            "color_match": get_args(ColorMatchMethod),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"unknown {name} {getattr(self, name)!r} (expected one of {allowed})")
        if self.steps < 1:
            raise ValueError("steps must be >= 1")
        for name in ("content_weight", "style_weight", "tv_weight"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.style_blend is not None and (
            any(w < 0 for w in self.style_blend) or sum(self.style_blend) <= 0
        ):
            raise ValueError("style_blend weights must be non-negative with a positive sum")

    def resolved(self, source: str) -> TransferConfig:
        """Return a copy with every ``None`` replaced by the preset for ``source``."""
        preset = PRESETS[source]
        return replace(
            self,
            content_layers=self.content_layers if self.content_layers is not None else preset.content_layers,
            style_layers=self.style_layers if self.style_layers is not None else preset.style_layers,
            content_weight=self.content_weight if self.content_weight is not None else preset.content_weight,
            style_weight=self.style_weight if self.style_weight is not None else preset.style_weight,
            pooling=self.pooling if self.pooling is not None else preset.pooling,
        )


@dataclass(frozen=True)
class LossRecord:
    """Loss terms (already multiplied by their weights) at the start of one optimisation step.

    That is the loss of the image the step starts from, so ``history[k]`` describes
    the result of ``k`` completed steps.
    """

    step: int
    total: float
    content: float
    style: float
    tv: float
    elapsed: float
    """Seconds from the start of the synthesis (target computation included) to this evaluation."""


@dataclass(frozen=True)
class TransferResult:
    """Output of :func:`stylize`."""

    image: torch.Tensor
    """Final RGB image ``(1, 3, H, W)`` in ``[0, 1]``."""
    history: tuple[LossRecord, ...]
    config: TransferConfig
    """The configuration with presets resolved."""
    seconds: float


ProgressFn = Callable[[LossRecord, torch.Tensor], None]
"""Called after every step with the step's losses and a copy of the current RGB image."""


class Objective:
    """The style-transfer loss for one content image, its style targets and optional masks.

    Targets are computed once at construction. Calling the objective on a
    candidate image returns the weighted total and its parts. In luminance mode
    candidates are single-channel luminance images; :meth:`working` converts an
    RGB image into the space the objective expects.

    ``masks`` (one ``(1, 1, h, w)`` guidance channel per style, any size)
    switch on spatial control: style ``r`` is matched with guided Gram
    matrices inside ``masks[r]`` only. ``style_masks`` optionally restrict
    which part of each style image provides the statistics.
    """

    def __init__(
        self,
        vgg: VGG19,
        content: torch.Tensor,
        styles: Sequence[torch.Tensor],
        config: TransferConfig | None = None,
        *,
        masks: Sequence[torch.Tensor] | None = None,
        style_masks: Sequence[torch.Tensor | None] | None = None,
    ) -> None:
        cfg = (config or TransferConfig()).resolved(vgg.source)
        assert cfg.content_layers is not None and cfg.style_layers is not None and cfg.pooling is not None
        if not styles:
            raise ValueError("at least one style image is required")
        if cfg.style_blend is not None and len(cfg.style_blend) != len(styles):
            raise ValueError("style_blend needs exactly one weight per style image")
        if masks is not None and len(masks) != len(styles):
            raise ValueError("masks need exactly one guidance channel per style image")
        if style_masks is not None and (masks is None or len(style_masks) != len(styles)):
            raise ValueError("style_masks need masks, and one entry per style image")
        if not cfg.style_layers and not cfg.content_layers:
            raise ValueError("at least one content or style layer is required")

        self.config = cfg
        self.content = content
        self.channels = 1 if cfg.color == "luminance" else 3
        self.extractor = FeatureExtractor(vgg, cfg.content_layers + cfg.style_layers, cfg.pooling)
        self.extractor.to(content.device)
        height, width = content.shape[-2:]

        with torch.no_grad():
            features = self.extractor(self._network_input(self.working(content)))
            self.content_targets = {name: features[name] for name in cfg.content_layers}
            prepared = [self._prepare_style(s, height * width) for s in styles]
            self.prepared_styles = prepared
            """Style images as used: resized, and recoloured or reduced to luminance if configured."""
            self.style_shapes = [(s.shape[-2], s.shape[-1]) for s in prepared]
            style_features = [self.extractor(self._network_input(s)) for s in prepared]
            blend = cfg.style_blend or (1.0,) * len(styles)

            # One (weight, masks-per-layer, targets-per-layer) entry per region.
            self.regions: list[tuple[float, dict[str, torch.Tensor] | None, dict[str, torch.Tensor]]] = []
            if masks is None:
                total = sum(blend)
                targets = {
                    name: sum(
                        (
                            w / total * gram_matrix(f[name])
                            for w, f in zip(blend, style_features, strict=True)
                        ),
                        torch.zeros(()),
                    )
                    for name in cfg.style_layers
                }
                self.regions.append((1.0, None, targets))
            else:
                guidance = torch.cat([_fit_mask(m, (height, width)) for m in masks], dim=1)
                per_layer = self.extractor.downsample(guidance)
                for r, (weight, feats, style) in enumerate(zip(blend, style_features, prepared, strict=True)):
                    style_mask = style_masks[r] if style_masks is not None else None
                    style_layers = (
                        self.extractor.downsample(_fit_mask(style_mask, (style.shape[-2], style.shape[-1])))
                        if style_mask is not None
                        else None
                    )
                    targets = {
                        name: gram_matrix(feats[name])
                        if style_layers is None
                        else guided_gram_matrix(feats[name], style_layers[name])
                        for name in cfg.style_layers
                    }
                    region = {name: per_layer[name][:, r : r + 1] for name in cfg.style_layers}
                    self.regions.append((weight, region, targets))

    def working(self, image: torch.Tensor) -> torch.Tensor:
        """Convert an RGB image to the space being optimised (luminance or RGB)."""
        return luminance(image) if self.channels == 1 else image

    def compose(self, working: torch.Tensor) -> torch.Tensor:
        """Turn an optimised working image into the final RGB image in ``[0, 1]``."""
        if self.channels == 1:
            return preserve_colors(working, self.content)
        return working.clamp(0, 1)

    def _network_input(self, image: torch.Tensor) -> torch.Tensor:
        return image.expand(-1, 3, -1, -1) if image.shape[1] == 1 else image

    def _prepare_style(self, style: torch.Tensor, content_pixels: int) -> torch.Tensor:
        # Gram matrices are averages over positions, so style images keep their own aspect
        # ratio; they are resized to the content's pixel count rather than to its shape.
        style = resize_to_area(style, content_pixels)
        if self.config.color == "match":
            return match_color(style, self.content, self.config.color_match).clamp(0, 1)
        if self.config.color == "luminance":
            return match_luminance(luminance(style), luminance(self.content)).clamp(0, 1)
        return style

    def __call__(self, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the weighted total loss of ``image`` and its weighted parts."""
        cfg = self.config
        assert cfg.content_weight is not None and cfg.style_weight is not None
        expected = (self.channels, *self.content.shape[-2:])
        if tuple(image.shape[1:]) != expected:
            raise ValueError(
                f"expected an image of shape (1, {', '.join(map(str, expected))}), got {tuple(image.shape)}"
            )
        features = self.extractor(self._network_input(image))
        zero = image.new_zeros(())
        content = sum(
            (F.mse_loss(features[name], target) for name, target in self.content_targets.items()), zero
        )
        style = zero
        for weight, region, targets in self.regions:
            for name, target in targets.items():
                if region is None:
                    style = style + weight * F.mse_loss(gram_matrix(features[name]), target)
                else:
                    mask = region[name]
                    # Weighting by the region's share of the image keeps the per-pixel
                    # balance between content and style independent of region size.
                    share = mask.square().mean()
                    gram = guided_gram_matrix(features[name], mask)
                    style = style + weight * share * F.mse_loss(gram, target)
        tv = total_variation(image) if cfg.tv_weight > 0 else zero
        parts = {
            "content": cfg.content_weight * content,
            "style": cfg.style_weight * style,
            "tv": cfg.tv_weight * tv,
        }
        return parts["content"] + parts["style"] + parts["tv"], parts


def _fit_mask(mask: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    if mask.dim() != 4 or mask.shape[:2] != (1, 1):
        raise ValueError(f"a mask must have shape (1, 1, H, W), got {tuple(mask.shape)}")
    if mask.min() < 0 or mask.max() > 1:
        raise ValueError("mask values must lie in [0, 1]")
    return resize(mask, shape, mode="bilinear").clamp(0, 1)


def _initial_image(objective: Objective, seed: int) -> torch.Tensor:
    content = objective.working(objective.content)
    if objective.config.init == "content":
        return content.clone()
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(content.shape, generator=generator).to(content.device)


def _make_optimizer(config: TransferConfig, image: torch.Tensor) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam([image], lr=config.lr)
    # One function evaluation per step; the curvature pairs persist across steps.
    return torch.optim.LBFGS([image], max_iter=1)


def _optimise(
    objective: Objective,
    image: torch.Tensor,
    steps: int,
    started: float,
    on_progress: ProgressFn | None,
) -> tuple[torch.Tensor, list[LossRecord]]:
    image = image.detach().clone().requires_grad_(True)
    optimizer = _make_optimizer(objective.config, image)
    history: list[LossRecord] = []
    for step in range(1, steps + 1):
        parts: dict[str, float] = {}

        def closure(parts: dict[str, float] = parts) -> float:
            with torch.no_grad():
                image.clamp_(0, 1)
            optimizer.zero_grad()
            total, terms = objective(image)
            torch.autograd.backward(total)
            parts.update({k: float(v.detach()) for k, v in terms.items()}, total=float(total.detach()))
            parts["elapsed"] = time.perf_counter() - started
            return parts["total"]

        optimizer.step(closure)
        record = LossRecord(
            step=step,
            total=parts["total"],
            content=parts["content"],
            style=parts["style"],
            tv=parts["tv"],
            elapsed=parts["elapsed"],
        )
        history.append(record)
        if on_progress is not None:
            with torch.no_grad():
                on_progress(record, objective.compose(image.detach().clamp(0, 1)))
    with torch.no_grad():
        image.clamp_(0, 1)
    return image.detach(), history


def stylize(
    vgg: VGG19,
    content: torch.Tensor,
    styles: Sequence[torch.Tensor],
    config: TransferConfig | None = None,
    *,
    masks: Sequence[torch.Tensor] | None = None,
    style_masks: Sequence[torch.Tensor | None] | None = None,
    on_progress: ProgressFn | None = None,
) -> TransferResult:
    """Optimise an image so its VGG-19 features match ``content`` and ``styles``.

    ``content`` fixes the output resolution; style images may have any size and
    aspect ratio (they are resized to the content's pixel count). ``masks``
    enables spatial control (see :class:`Objective`).

    L-BFGS converges in far fewer steps than Adam here; Adam uses less memory.
    """
    started = time.perf_counter()
    objective = Objective(vgg, content, styles, config, masks=masks, style_masks=style_masks)
    cfg = objective.config
    working, history = _optimise(
        objective, _initial_image(objective, cfg.seed), cfg.steps, started, on_progress
    )
    return TransferResult(
        image=objective.compose(working),
        history=tuple(history),
        config=cfg,
        seconds=time.perf_counter() - started,
    )
