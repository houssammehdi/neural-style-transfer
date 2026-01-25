"""Gram matrices, guided Gram matrices and total variation."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from neural_style import gram_matrix, guided_gram_matrix, total_variation

SETTINGS = settings(max_examples=30, deadline=None)


def _features(seed: int, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def test_gram_matrix_matches_definition() -> None:
    x = _features(0, (1, 3, 4, 5))
    flat = x.reshape(3, 20)
    torch.testing.assert_close(gram_matrix(x)[0], flat @ flat.T / (3 * 4 * 5))


def test_gram_matrix_is_computed_per_sample() -> None:
    # Regression: v0.1 reshaped (B, C, H, W) to (B*C, H*W) and mixed samples for B > 1.
    x = _features(1, (2, 3, 4, 5))
    g = gram_matrix(x)
    assert g.shape == (2, 3, 3)
    torch.testing.assert_close(g[1], gram_matrix(x[1:])[0])


@SETTINGS
@given(seed=st.integers(0, 10_000), c=st.integers(1, 6), h=st.integers(1, 7), w=st.integers(1, 7))
def test_gram_matrix_is_symmetric_psd_and_translation_invariant(seed: int, c: int, h: int, w: int) -> None:
    x = _features(seed, (1, c, h, w))
    g = gram_matrix(x)[0]
    torch.testing.assert_close(g, g.T)
    assert torch.linalg.eigvalsh(g).min() >= -1e-10
    # moving features around changes *where* they are, not which ones co-occur
    torch.testing.assert_close(gram_matrix(torch.roll(x, shifts=(1, 2), dims=(-2, -1)))[0], g)


def test_guided_gram_with_full_mask_is_gram_matrix() -> None:
    x = _features(2, (1, 4, 6, 7))
    torch.testing.assert_close(guided_gram_matrix(x, torch.ones(1, 1, 6, 7, dtype=x.dtype)), gram_matrix(x))


@SETTINGS
@given(seed=st.integers(0, 10_000), h=st.integers(2, 8), w=st.integers(2, 8))
def test_guided_gram_of_binary_mask_equals_gram_of_region_pixels(seed: int, h: int, w: int) -> None:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((1, 3, h, w), generator=g, dtype=torch.float64)
    mask = (torch.rand((1, 1, h, w), generator=g) < 0.5).to(x.dtype)
    mask[..., 0, 0] = 1  # never empty
    inside = x[0][:, mask[0, 0].bool()]  # (C, n): only the pixels in the region
    expected = inside @ inside.T / (3 * inside.shape[1])
    torch.testing.assert_close(guided_gram_matrix(x, mask)[0], expected)


def test_guided_gram_does_not_depend_on_region_size() -> None:
    # The same texture filling a small or a large region has the same statistics.
    tile = _features(3, (1, 4, 5, 5))
    doubled = torch.cat([tile, tile], dim=-1)
    left = torch.cat([torch.ones(1, 1, 5, 5), torch.zeros(1, 1, 5, 5)], dim=-1).double()
    torch.testing.assert_close(guided_gram_matrix(doubled, left), gram_matrix(tile))
    torch.testing.assert_close(guided_gram_matrix(doubled, torch.ones_like(left)), gram_matrix(tile))


def test_guided_gram_broadcasts_one_mask_over_a_batch() -> None:
    x = _features(4, (2, 3, 4, 4))
    mask = torch.rand(1, 1, 4, 4, generator=torch.Generator().manual_seed(0), dtype=x.dtype)
    batched = guided_gram_matrix(x, mask)
    torch.testing.assert_close(batched[1], guided_gram_matrix(x[1:], mask)[0])


def test_guided_gram_of_empty_mask_is_zero_not_nan() -> None:
    x = _features(5, (1, 3, 4, 4))
    out = guided_gram_matrix(x, torch.zeros(1, 1, 4, 4, dtype=x.dtype))
    assert torch.equal(out, torch.zeros_like(out))


def test_guided_gram_rejects_mismatched_mask() -> None:
    with pytest.raises(ValueError, match="does not match"):
        guided_gram_matrix(torch.zeros(1, 3, 4, 4), torch.ones(1, 1, 4, 5))
    with pytest.raises(ValueError, match="does not match"):
        guided_gram_matrix(torch.zeros(1, 3, 4, 4), torch.ones(1, 2, 4, 4))


def test_total_variation() -> None:
    assert total_variation(torch.full((1, 3, 5, 5), 0.3)).item() == 0.0
    ramp = torch.arange(4.0).view(1, 1, 1, 4).expand(1, 1, 3, 4)  # steps of 1 along the width only
    assert total_variation(ramp).item() == pytest.approx(1.0)
