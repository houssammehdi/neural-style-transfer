"""Loss terms of neural style transfer (Gatys et al., 2016)."""

from __future__ import annotations

import torch


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Return the normalised Gram matrices of a ``(B, C, H, W)`` feature map as ``(B, C, C)``.

    ``G[b, i, j]`` is the inner product of the vectorised feature maps ``i`` and
    ``j`` of sample ``b``, divided by ``C * H * W``. It records which features
    fire *together*, independent of *where*, which is why it works as a
    representation of style. Dividing by the number of positions makes it an
    average, so images of different sizes can be compared; the extra ``1/C``
    follows the common implementations (see ``docs/method.md`` for how this
    relates to the normalisation in the paper).
    """
    b, c, h, w = features.shape
    flat = features.reshape(b, c, h * w)
    return flat @ flat.transpose(1, 2) / (c * h * w)


def total_variation(image: torch.Tensor) -> torch.Tensor:
    """Anisotropic total variation: mean absolute difference between neighbouring pixels.

    Penalising it suppresses high-frequency noise in the synthesised image.
    """
    dh = (image[..., 1:, :] - image[..., :-1, :]).abs().mean()
    dw = (image[..., :, 1:] - image[..., :, :-1]).abs().mean()
    return dh + dw
