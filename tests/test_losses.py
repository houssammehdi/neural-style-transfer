"""Gram matrices and total variation."""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from neural_style import gram_matrix, total_variation

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


def test_total_variation() -> None:
    assert total_variation(torch.full((1, 3, 5, 5), 0.3)).item() == 0.0
    ramp = torch.arange(4.0).view(1, 1, 1, 4).expand(1, 1, 3, 4)  # steps of 1 along the width only
    assert total_variation(ramp).item() == pytest.approx(1.0)
