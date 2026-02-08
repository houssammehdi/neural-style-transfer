"""Shared fixtures: every test runs offline on a randomly initialised VGG-19."""

from __future__ import annotations

import pytest
import torch
from hypothesis import settings

from neural_style import VGG19, load_vgg19

# Property-based tests draw the same examples on every run, so CI results are reproducible.
settings.register_profile("deterministic", derandomize=True, deadline=None)
settings.load_profile("deterministic")


@pytest.fixture(scope="session")
def vgg() -> VGG19:
    """An untrained VGG-19 (fixed seed); its losses behave like the real network's, only cheaper."""
    return load_vgg19("random", seed=0)


@pytest.fixture
def generator() -> torch.Generator:
    """A seeded generator, so tests never depend on the global RNG."""
    return torch.Generator().manual_seed(1234)
