"""Image I/O and the optimisation loop that performs style transfer."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.transforms import functional as TF

from .losses import total_variation
from .model import DEFAULT_CONTENT_LAYERS, DEFAULT_STYLE_LAYERS, build_style_model


def pick_device(preferred: str = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA, then Apple MPS, then CPU."""
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_image(path: str | Path, size: int | tuple[int, int], device: torch.device) -> torch.Tensor:
    """Load an RGB image as a ``(1, 3, H, W)`` float tensor in ``[0, 1]``.

    An ``int`` size resizes the shorter edge (aspect preserved); a tuple
    forces an exact ``(H, W)``, which is how style images are matched to the
    content image's shape. EXIF orientation (as written by phone cameras) is
    applied, so photos load the way image viewers show them.
    """
    with Image.open(path) as raw:
        img = ImageOps.exif_transpose(raw).convert("RGB")
    tensor = TF.to_tensor(TF.resize(img, size, antialias=True))
    return tensor.unsqueeze(0).to(device)


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    """Write a ``(1, 3, H, W)`` tensor to disk (format from the file extension)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    TF.to_pil_image(tensor.detach().squeeze(0).clamp(0, 1).cpu()).save(path)


# ITU-R BT.601 RGB <-> YIQ; Y carries luminance, I/Q carry colour.
_RGB2YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
_YIQ2RGB = torch.linalg.inv(_RGB2YIQ)


def _convert(img: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,bjhw->bihw", matrix.to(img.device, img.dtype), img)


def preserve_colors(stylised: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
    """Keep the stylised luminance but restore the content image's colours.

    This is the colour-preservation variant from Gatys et al., "Preserving
    Color in Neural Artistic Style Transfer" (2016).
    """
    y = _convert(stylised, _RGB2YIQ)[:, :1]
    iq = _convert(content, _RGB2YIQ)[:, 1:]
    return _convert(torch.cat([y, iq], dim=1), _YIQ2RGB).clamp(0, 1)


@dataclass
class TransferConfig:
    """Hyper-parameters for one style-transfer run."""

    steps: int = 300
    style_weight: float = 1e6
    content_weight: float = 1.0
    tv_weight: float = 0.0
    optimizer: str = "lbfgs"
    lr: float = 0.02
    init: str = "content"
    style_blend: list[float] | None = None
    content_layers: tuple[str, ...] = DEFAULT_CONTENT_LAYERS
    style_layers: tuple[str, ...] = DEFAULT_STYLE_LAYERS
    seed: int = 0
    history: list[dict[str, float]] = field(default_factory=list)


ProgressFn = Callable[[int, dict[str, float], torch.Tensor], None]


def stylize(
    cnn: nn.Sequential,
    content: torch.Tensor,
    styles: list[torch.Tensor],
    config: TransferConfig,
    on_progress: ProgressFn | None = None,
) -> torch.Tensor:
    """Optimise an image so its VGG features match ``content`` and ``styles``.

    The network weights stay frozen; the *pixels* of the input image are the
    only parameters. L-BFGS converges in far fewer steps than Adam for this
    problem, but Adam uses less memory and behaves better on MPS.
    """
    if config.steps < 1:
        raise ValueError("steps must be >= 1")
    torch.manual_seed(config.seed)
    net = build_style_model(
        cnn, content, styles, config.style_blend, config.content_layers, config.style_layers
    )

    if config.init == "content":
        image = content.clone()
    elif config.init == "noise":
        image = torch.rand_like(content)
    else:
        raise ValueError(f"unknown init {config.init!r} (expected 'content' or 'noise')")
    image.requires_grad_(True)

    if config.optimizer == "lbfgs":
        opt: torch.optim.Optimizer = torch.optim.LBFGS([image], max_iter=1)
    elif config.optimizer == "adam":
        opt = torch.optim.Adam([image], lr=config.lr)
    else:
        raise ValueError(f"unknown optimizer {config.optimizer!r} (expected 'lbfgs' or 'adam')")

    for step in range(1, config.steps + 1):
        record: dict[str, float] = {}

        def closure(step: int = step, record: dict[str, float] = record) -> torch.Tensor:
            with torch.no_grad():
                image.clamp_(0, 1)
            opt.zero_grad()
            net.model(image)
            style = config.style_weight * sum(p.loss for p in net.style_losses)
            content_term = config.content_weight * sum(p.loss for p in net.content_losses)
            tv = config.tv_weight * total_variation(image)
            loss = style + content_term + tv
            loss.backward()
            record.update(
                step=step,
                style=float(style.detach()) if torch.is_tensor(style) else float(style),
                content=float(content_term.detach())
                if torch.is_tensor(content_term)
                else float(content_term),
                tv=float(tv.detach()),
                total=float(loss.detach()),
            )
            return loss

        opt.step(closure)
        config.history.append(dict(record))
        if on_progress is not None:
            on_progress(step, record, image.detach())

    with torch.no_grad():
        image.clamp_(0, 1)
    return image.detach()
