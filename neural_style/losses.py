"""Loss terms used by neural style transfer (Gatys et al., 2016)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Return the normalised Gram matrices of a ``(B, C, H, W)`` feature map as ``(B, C, C)``.

    Each entry ``G[b, i, j]`` is the inner product between the vectorised feature
    maps ``i`` and ``j`` of sample ``b``. It captures which features co-occur,
    independent of *where* they occur, which is why it works as a representation
    of style. The result is divided by the number of elements per sample so that
    layers with large feature maps do not dominate the style loss.
    """
    b, c, h, w = features.shape
    flat = features.reshape(b, c, h * w)
    return flat @ flat.transpose(1, 2) / (c * h * w)


class ContentLoss(nn.Module):
    """Transparent layer that records the MSE to a fixed content activation."""

    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("target", target.detach())
        self.loss = torch.zeros(())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.loss = F.mse_loss(x, self.target)
        return x


class StyleLoss(nn.Module):
    """Transparent layer that records the Gram-matrix MSE to one or more styles.

    When several style images are given, their Gram matrices are blended with
    ``weights`` (normalised to sum to one), which interpolates between styles.
    """

    def __init__(self, targets: list[torch.Tensor], weights: list[float] | None = None) -> None:
        super().__init__()
        if not targets:
            raise ValueError("StyleLoss needs at least one target feature map")
        weights = weights or [1.0] * len(targets)
        if len(weights) != len(targets):
            raise ValueError("one weight per style target is required")
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("style weights must sum to a positive value")
        blended = sum(w / total * gram_matrix(t.detach()) for w, t in zip(weights, targets, strict=True))
        self.register_buffer("target", torch.as_tensor(blended))
        self.loss = torch.zeros(())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.loss = F.mse_loss(gram_matrix(x), self.target)
        return x


def total_variation(img: torch.Tensor) -> torch.Tensor:
    """Anisotropic total-variation penalty; suppresses high-frequency noise."""
    dh = (img[..., 1:, :] - img[..., :-1, :]).abs().mean()
    dw = (img[..., :, 1:] - img[..., :, :-1]).abs().mean()
    return dh + dw
