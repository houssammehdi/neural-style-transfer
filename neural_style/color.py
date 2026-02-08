"""Colour control from Gatys, Bethge, Hertzmann & Shechtman (2016).

*Preserving Color in Neural Artistic Style Transfer* (arXiv:1606.05897) gives
two ways to keep the content photo's colours:

* **luminance-only transfer** -- run style transfer on the luminance channel
  only and put the content image's chrominance back afterwards
  (:func:`luminance`, :func:`match_luminance`, :func:`preserve_colors`);
* **colour histogram matching** -- recolour the style image so its pixel mean
  and covariance equal the content image's before transferring
  (:func:`match_color`).

All functions take ``(B, 3, H, W)`` RGB tensors in ``[0, 1]`` unless stated.
"""

from __future__ import annotations

from typing import Literal

import torch

ColorMatchMethod = Literal["eigen", "cholesky"]

# NTSC YIQ (as used by Gatys et al.): Y carries luminance, I and Q carry colour.
RGB_TO_YIQ = torch.tensor(
    [[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], dtype=torch.float64
)
YIQ_TO_RGB = torch.linalg.inv(RGB_TO_YIQ)


def _apply(matrix: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    # Cast before moving: some devices (Apple MPS) have no float64.
    return torch.einsum("ij,bjhw->bihw", matrix.to(dtype=image.dtype).to(image.device), image)


def _cpu_pixels(image: torch.Tensor) -> torch.Tensor:
    """The pixels of a single RGB image as a (3, N) float64 CPU tensor (MPS has no float64)."""
    if image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError(f"expected a single RGB image (1, 3, H, W), got {tuple(image.shape)}")
    return image.detach().reshape(3, -1).to("cpu", torch.float64)


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

    This is the final step of luminance-only transfer, where ``stylised`` is the
    optimised luminance image (one channel, or three equal channels). Applied to
    a full-colour stylisation it is only a post-hoc approximation, because the
    optimisation has then already been shaped by the style's colours.
    """
    y = stylised[:, :1] if stylised.shape[1] == 1 else luminance(stylised)
    iq = rgb_to_yiq(content)[:, 1:]
    return yiq_to_rgb(torch.cat([y, iq], dim=1)).clamp(0, 1)


def match_luminance(source: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Affinely map luminance ``source`` to the mean and standard deviation of ``target``.

    ``L' = sigma_t / sigma_s * (L - mu_s) + mu_t``, per sample, as recommended by
    Gatys et al. (2016) before luminance-only transfer when the two luminance
    histograms differ strongly.
    """
    dims = (1, 2, 3)
    mu_s, mu_t = source.mean(dims, keepdim=True), target.mean(dims, keepdim=True)
    sd_s = source.std(dims, keepdim=True, unbiased=False).clamp_min(eps)
    sd_t = target.std(dims, keepdim=True, unbiased=False)
    return (source - mu_s) * (sd_t / sd_s) + mu_t


def _pixel_stats(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = _cpu_pixels(image)
    mean = pixels.mean(dim=1)
    centred = pixels - mean[:, None]
    return mean, centred @ centred.T / pixels.shape[1]


def _sqrtm(cov: torch.Tensor, power: float, eps: float) -> torch.Tensor:
    decomposition = torch.linalg.eigh(cov)
    vectors: torch.Tensor = decomposition.eigenvectors
    return vectors @ torch.diag(decomposition.eigenvalues.clamp_min(eps).pow(power)) @ vectors.T


def color_transform(
    source: torch.Tensor,
    target: torch.Tensor,
    method: ColorMatchMethod = "eigen",
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(A, b)`` such that ``A @ x + b`` gives ``source`` pixels ``target``'s mean and covariance.

    With covariances ``S_s`` and ``S_t`` (regularised by ``eps * I``), ``A`` must
    satisfy ``A S_s A^T = S_t``. Two solutions from Gatys et al. (2016):

    * ``"eigen"`` -- the Image Analogies formulation ``A = S_t^(1/2) S_s^(-1/2)``
      with symmetric square roots from an eigendecomposition;
    * ``"cholesky"`` -- ``A = L_t L_s^(-1)`` with ``S = L L^T``.

    ``b = mu_t - A mu_s``. Both are computed in float64 on the CPU and returned there.
    """
    mu_s, cov_s = _pixel_stats(source)
    mu_t, cov_t = _pixel_stats(target)
    eye = torch.eye(3, dtype=torch.float64)
    cov_s, cov_t = cov_s + eps * eye, cov_t + eps * eye
    if method == "eigen":
        a = _sqrtm(cov_t, 0.5, eps) @ _sqrtm(cov_s, -0.5, eps)
    elif method == "cholesky":
        a = torch.linalg.cholesky(cov_t) @ torch.linalg.inv(torch.linalg.cholesky(cov_s))
    else:
        raise ValueError(f"unknown colour-matching method {method!r} (expected 'eigen' or 'cholesky')")
    return a, mu_t - a @ mu_s


def match_color(
    source: torch.Tensor,
    target: torch.Tensor,
    method: ColorMatchMethod = "eigen",
    eps: float = 1e-5,
) -> torch.Tensor:
    """Recolour ``source`` so its pixel mean and covariance match ``target`` (not clamped).

    This is the colour histogram matching of Gatys et al. (2016), applied to
    the style image before style transfer. The result can leave ``[0, 1]``;
    callers clamp it.
    """
    a, b = color_transform(source, target, method, eps)
    out = a @ _cpu_pixels(source) + b[:, None]
    return out.reshape(source.shape).to(dtype=source.dtype).to(source.device)
