"""Loss terms of neural style transfer (Gatys et al., 2016 and 2017)."""

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


def guided_gram_matrix(features: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Gram matrix of the features inside a soft region (Gatys et al., 2017).

    ``mask`` is a guidance channel of shape ``(B, 1, H, W)`` or ``(1, 1, H, W)``
    at the feature map's resolution, with values in ``[0, 1]``. The features
    are weighted by the mask, ``F_r = F * m``, and the Gram matrix is
    normalised by ``C * sum(m ** 2)`` instead of ``C * H * W``: for a binary
    mask this is exactly :func:`gram_matrix` of the masked positions alone, so
    the statistic does not depend on how large the region is. An all-ones mask
    gives :func:`gram_matrix`; an empty mask gives zeros rather than NaNs.
    """
    if mask.dim() != 4 or mask.shape[1] != 1 or mask.shape[-2:] != features.shape[-2:]:
        raise ValueError(f"mask of shape {tuple(mask.shape)} does not match features {tuple(features.shape)}")
    b, c, h, w = features.shape
    flat = (features * mask).reshape(b, c, h * w)
    energy = mask.square().sum(dim=(1, 2, 3)).clamp_min(eps)
    return flat @ flat.transpose(1, 2) / (c * energy).view(-1, 1, 1)


def total_variation(image: torch.Tensor) -> torch.Tensor:
    """Anisotropic total variation: mean absolute difference between neighbouring pixels.

    Penalising it suppresses high-frequency noise in the synthesised image.
    """
    dh = (image[..., 1:, :] - image[..., :-1, :]).abs().mean()
    dw = (image[..., :, 1:] - image[..., :, :-1]).abs().mean()
    return dh + dw
