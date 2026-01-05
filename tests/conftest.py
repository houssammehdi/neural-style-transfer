"""Shared fixtures: every test runs offline on a randomly initialised VGG-19."""

from __future__ import annotations

import pytest
import torch

from neural_style import VGG19, load_vgg19


@pytest.fixture(scope="session")
def vgg() -> VGG19:
    """An untrained VGG-19 (fixed seed); its losses behave like the real network's, only cheaper."""
    return load_vgg19("random", seed=0)


@pytest.fixture
def generator() -> torch.Generator:
    """A seeded generator, so tests never depend on the global RNG."""
    return torch.Generator().manual_seed(1234)
