"""Colour control maths: YIQ, luminance matching and mean/covariance colour transfer."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from neural_style import match_color, match_luminance, preserve_colors
from neural_style.color import color_transform, luminance, rgb_to_yiq, yiq_to_rgb


def _stats(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = image.reshape(3, -1).double()
    mean = pixels.mean(1)
    centred = pixels - mean[:, None]
    return mean, centred @ centred.T / pixels.shape[1]


def _correlated_image(seed: int, pixels: int = 400) -> torch.Tensor:
    """Pixels with a random mean and a random, well-conditioned colour covariance."""
    g = torch.Generator().manual_seed(seed)
    mixing = (
        torch.randn(3, 3, generator=g, dtype=torch.float64) * 0.15 + torch.eye(3, dtype=torch.float64) * 0.2
    )
    mean = torch.rand(3, 1, generator=g, dtype=torch.float64)
    pixels_ = mixing @ torch.randn(3, pixels, generator=g, dtype=torch.float64) + mean
    return pixels_.reshape(1, 3, 20, pixels // 20)


def test_yiq_round_trip_and_luminance_weights() -> None:
    img = torch.rand(1, 3, 5, 6, dtype=torch.float64)
    torch.testing.assert_close(yiq_to_rgb(rgb_to_yiq(img)), img)
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float64).view(1, 3, 1, 1)
    torch.testing.assert_close(luminance(img), (img * weights).sum(1, keepdim=True))


def test_preserve_colors_keeps_luminance_and_restores_chrominance() -> None:
    g = torch.Generator().manual_seed(0)
    stylised, content = torch.rand(1, 3, 8, 8, generator=g), torch.rand(1, 3, 8, 8, generator=g)
    out = preserve_colors(stylised, content)
    inside = ((out > 0) & (out < 1)).all(dim=1)  # identical wherever no clamping happened
    torch.testing.assert_close(
        luminance(out)[:, 0][inside], luminance(stylised)[:, 0][inside], atol=1e-5, rtol=0
    )
    torch.testing.assert_close(
        rgb_to_yiq(out)[:, 1:].permute(0, 2, 3, 1)[inside],
        rgb_to_yiq(content)[:, 1:].permute(0, 2, 3, 1)[inside],
        atol=1e-5,
        rtol=0,
    )
    # a single-channel luminance image is accepted directly
    torch.testing.assert_close(preserve_colors(luminance(stylised), content), out)


def test_match_luminance_matches_mean_and_std() -> None:
    g = torch.Generator().manual_seed(1)
    source = torch.rand(1, 1, 10, 10, generator=g, dtype=torch.float64) * 0.2
    target = torch.rand(1, 1, 12, 7, generator=g, dtype=torch.float64) * 0.9 + 0.05
    out = match_luminance(source, target)
    torch.testing.assert_close(out.mean(), target.mean())
    torch.testing.assert_close(out.std(unbiased=False), target.std(unbiased=False))


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 100_000), method=st.sampled_from(["eigen", "cholesky"]))
def test_match_color_gives_the_target_mean_and_covariance(seed: int, method: str) -> None:
    source, target = _correlated_image(seed), _correlated_image(seed + 1, pixels=600)
    out = match_color(source, target, method, eps=0.0)  # type: ignore[arg-type]
    mean, cov = _stats(out)
    target_mean, target_cov = _stats(target)
    torch.testing.assert_close(mean, target_mean)
    torch.testing.assert_close(cov, target_cov)


def test_both_decompositions_solve_the_covariance_equation_differently() -> None:
    source, target = _correlated_image(3), _correlated_image(4)
    _, cov_s = _stats(source)
    _, cov_t = _stats(target)
    a_eig, _ = color_transform(source, target, "eigen", eps=0.0)
    a_chol, _ = color_transform(source, target, "cholesky", eps=0.0)
    for a in (a_eig, a_chol):
        torch.testing.assert_close(a @ cov_s @ a.T, cov_t)
    assert not torch.allclose(a_eig, a_chol)  # two genuinely different solutions
    assert torch.allclose(
        torch.linalg.cholesky(cov_t) @ torch.linalg.inv(torch.linalg.cholesky(cov_s)), a_chol
    )


def test_match_color_is_the_identity_for_identical_statistics() -> None:
    img = _correlated_image(5)
    torch.testing.assert_close(match_color(img, img, eps=0.0), img)


def test_match_color_stays_finite_for_a_grey_source() -> None:
    grey = torch.rand(1, 1, 8, 8, generator=torch.Generator().manual_seed(2)).expand(1, 3, 8, 8)
    out = match_color(grey, _correlated_image(6).float())
    assert torch.isfinite(out).all()
    torch.testing.assert_close(_stats(out)[0], _stats(_correlated_image(6))[0], atol=1e-4, rtol=0)


def test_match_color_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="single RGB image"):
        match_color(torch.rand(2, 3, 4, 4), torch.rand(1, 3, 4, 4))
    with pytest.raises(ValueError, match="unknown colour-matching method"):
        match_color(torch.rand(1, 3, 4, 4), torch.rand(1, 3, 4, 4), "pca")  # type: ignore[arg-type]
