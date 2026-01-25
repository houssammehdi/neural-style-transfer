"""The optimisation that performs style transfer.

The synthesised image's pixels are the only parameters; VGG-19 stays frozen.
The objective is ``alpha * L_content + beta * L_style + gamma * TV``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal, get_args

import torch
import torch.nn.functional as F

from .image import resize_to_area
from .losses import gram_matrix, total_variation
from .model import VGG19, FeatureExtractor, Pooling

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

    ``style_blend`` weights the style images: they are normalised and the
    Gram matrices averaged, which interpolates between styles.
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
    seed: int = 0

    def __post_init__(self) -> None:
        choices: dict[str, tuple[str | None, ...]] = {
            "pooling": (*get_args(Pooling), None),
            "optimizer": get_args(OptimizerName),
            "init": get_args(InitName),
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
    """The style-transfer loss for one content image and its style targets.

    Targets are computed once at construction. Calling the objective on a
    candidate image returns the weighted total and its parts.
    """

    def __init__(
        self,
        vgg: VGG19,
        content: torch.Tensor,
        styles: Sequence[torch.Tensor],
        config: TransferConfig | None = None,
    ) -> None:
        cfg = (config or TransferConfig()).resolved(vgg.source)
        assert cfg.content_layers is not None and cfg.style_layers is not None and cfg.pooling is not None
        if not styles:
            raise ValueError("at least one style image is required")
        if cfg.style_blend is not None and len(cfg.style_blend) != len(styles):
            raise ValueError("style_blend needs exactly one weight per style image")
        if not cfg.style_layers and not cfg.content_layers:
            raise ValueError("at least one content or style layer is required")

        self.config = cfg
        self.content = content
        self.extractor = FeatureExtractor(vgg, cfg.content_layers + cfg.style_layers, cfg.pooling)
        self.extractor.to(content.device)
        height, width = content.shape[-2:]

        with torch.no_grad():
            features = self.extractor(content)
            self.content_targets = {name: features[name] for name in cfg.content_layers}
            # Gram matrices are averages over positions, so style images keep their own aspect
            # ratio; they are resized to the content's pixel count rather than to its shape.
            prepared = [resize_to_area(s, height * width) for s in styles]
            self.prepared_styles = prepared
            """Style images as used, resized to the content's pixel count."""
            self.style_shapes = [(s.shape[-2], s.shape[-1]) for s in prepared]
            style_features = [self.extractor(s) for s in prepared]
            blend = cfg.style_blend or (1.0,) * len(styles)
            total = sum(blend)
            self.style_targets = {
                name: sum(
                    (w / total * gram_matrix(f[name]) for w, f in zip(blend, style_features, strict=True)),
                    torch.zeros(()),
                )
                for name in cfg.style_layers
            }

    def __call__(self, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return the weighted total loss of ``image`` and its weighted parts."""
        cfg = self.config
        assert cfg.content_weight is not None and cfg.style_weight is not None
        if tuple(image.shape[1:]) != tuple(self.content.shape[1:]):
            raise ValueError(
                f"expected an image of shape {tuple(self.content.shape)}, got {tuple(image.shape)}"
            )
        features = self.extractor(image)
        zero = image.new_zeros(())
        content = sum(
            (F.mse_loss(features[name], target) for name, target in self.content_targets.items()), zero
        )
        style = sum(
            (F.mse_loss(gram_matrix(features[name]), target) for name, target in self.style_targets.items()),
            zero,
        )
        tv = total_variation(image) if cfg.tv_weight > 0 else zero
        parts = {
            "content": cfg.content_weight * content,
            "style": cfg.style_weight * style,
            "tv": cfg.tv_weight * tv,
        }
        return parts["content"] + parts["style"] + parts["tv"], parts


def _initial_image(objective: Objective, seed: int) -> torch.Tensor:
    content = objective.content
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
                on_progress(record, image.detach().clamp(0, 1))
    with torch.no_grad():
        image.clamp_(0, 1)
    return image.detach(), history


def stylize(
    vgg: VGG19,
    content: torch.Tensor,
    styles: Sequence[torch.Tensor],
    config: TransferConfig | None = None,
    *,
    on_progress: ProgressFn | None = None,
) -> TransferResult:
    """Optimise an image so its VGG-19 features match ``content`` and ``styles``.

    ``content`` fixes the output resolution; style images may have any size and
    aspect ratio (they are resized to the content's pixel count).

    L-BFGS converges in far fewer steps than Adam here; Adam uses less memory.
    """
    started = time.perf_counter()
    objective = Objective(vgg, content, styles, config)
    cfg = objective.config
    working, history = _optimise(
        objective, _initial_image(objective, cfg.seed), cfg.steps, started, on_progress
    )
    return TransferResult(
        image=working, history=tuple(history), config=cfg, seconds=time.perf_counter() - started
    )
