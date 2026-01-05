"""Colour-space helpers: NTSC YIQ and keeping the content photo's colours.

All functions take ``(B, 3, H, W)`` RGB tensors in ``[0, 1]``.
"""

from __future__ import annotations

import torch

# NTSC YIQ (as used by Gatys et al.): Y carries luminance, I and Q carry colour.
RGB_TO_YIQ = torch.tensor(
    [[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], dtype=torch.float64
)
YIQ_TO_RGB = torch.linalg.inv(RGB_TO_YIQ)


def _apply(matrix: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,bjhw->bihw", matrix.to(image.device, image.dtype), image)


def rgb_to_yiq(image: torch.Tensor) -> torch.Tensor:
    """Convert RGB to YIQ."""
    return _apply(RGB_TO_YIQ, image)


def yiq_to_rgb(image: torch.Tensor) -> torch.Tensor:
    """Convert YIQ to RGB (not clamped)."""
    return _apply(YIQ_TO_RGB, image)


def luminance(image: torch.Tensor) -> torch.Tensor:
    """Return the luminance ``Y = 0.299 R + 0.587 G + 0.114 B`` as ``(B, 1, H, W)``."""
    return _apply(RGB_TO_YIQ[:1], image)


def preserve_colors(stylised: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
    """Combine the luminance of ``stylised`` with the chrominance (I, Q) of ``content``.

    Applied after a full-colour stylisation this is only a post-hoc
    approximation of colour preservation: the optimisation has already been
    shaped by the style's colours. Gatys et al. (2016) instead transfer style
    on the luminance channel alone.
    """
    y = luminance(stylised)
    iq = rgb_to_yiq(content)[:, 1:]
    return yiq_to_rgb(torch.cat([y, iq], dim=1)).clamp(0, 1)
