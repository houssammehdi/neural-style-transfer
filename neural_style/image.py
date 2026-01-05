"""Image and mask I/O plus aspect-preserving resizing.

Images are ``(1, 3, H, W)`` float tensors in ``[0, 1]``. An integer
``size`` always means the *shorter* edge in
pixels, with the aspect ratio preserved; a ``(height, width)`` tuple forces an
exact shape.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

Size = int | tuple[int, int]


def target_shape(height: int, width: int, size: Size) -> tuple[int, int]:
    """Return the ``(height, width)`` that ``size`` asks for, given the current shape."""
    if isinstance(size, tuple):
        return size
    if size < 1:
        raise ValueError(f"size must be positive, got {size}")
    scale = size / min(height, width)
    return max(1, round(height * scale)), max(1, round(width * scale))


def area_shape(height: int, width: int, pixels: float) -> tuple[int, int]:
    """Return the aspect-preserving ``(height, width)`` with about ``pixels`` pixels."""
    scale = math.sqrt(pixels / (height * width))
    return max(1, round(height * scale)), max(1, round(width * scale))


def _to_tensor(img: Image.Image, device: torch.device | str | None) -> torch.Tensor:
    array = np.asarray(img, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[:, :, None]
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)


def _open(path: str | Path, mode: str, size: Size | None) -> Image.Image:
    with Image.open(path) as raw:
        img = ImageOps.exif_transpose(raw).convert(mode)
    if size is not None:
        height, width = target_shape(img.height, img.width, size)
        if (height, width) != (img.height, img.width):
            img = img.resize((width, height), Image.Resampling.LANCZOS)
    return img


def load_image(
    path: str | Path, size: Size | None = None, device: torch.device | str | None = None
) -> torch.Tensor:
    """Load an RGB image as ``(1, 3, H, W)`` in ``[0, 1]``, honouring EXIF orientation.

    ``size`` is the shorter edge (``int``) or an exact ``(H, W)``; ``None`` keeps
    the file's resolution. Resampling uses Lanczos filtering.
    """
    return _to_tensor(_open(path, "RGB", size), device)


def to_pil_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a ``(1, C, H, W)`` or ``(C, H, W)`` tensor in ``[0, 1]`` to a PIL image."""
    image = tensor.detach()
    if image.dim() == 4:
        image = image[0]
    array = (image.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array[:, :, 0] if array.shape[2] == 1 else array)


def save_image(tensor: torch.Tensor, path: str | Path, quality: int = 95) -> None:
    """Write an image tensor to disk; the format follows the extension (JPEG ``quality`` applies)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(tensor).save(target, quality=quality)


def resize(image: torch.Tensor, size: Size, mode: str = "bicubic") -> torch.Tensor:
    """Resize a ``(B, C, H, W)`` tensor with antialiasing (``size`` as in :func:`load_image`).

    Bicubic results are clamped to ``[0, 1]`` because the filter can overshoot;
    use ``mode="bilinear"`` for masks.
    """
    height, width = target_shape(image.shape[-2], image.shape[-1], size)
    if (height, width) == tuple(image.shape[-2:]):
        return image
    out = F.interpolate(image, size=(height, width), mode=mode, align_corners=False, antialias=True)
    return out.clamp(0, 1) if mode == "bicubic" else out


def resize_to_area(image: torch.Tensor, pixels: float) -> torch.Tensor:
    """Resize ``image`` (aspect preserved) so it has about ``pixels`` pixels."""
    return resize(image, area_shape(image.shape[-2], image.shape[-1], pixels))
