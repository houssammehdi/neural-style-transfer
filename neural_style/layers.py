"""VGG-19 topology and the canonical layer names used throughout the literature."""

from __future__ import annotations

VGG19_BLOCK_DEPTHS = (2, 2, 4, 4, 4)
"""Number of 3x3 convolutions in each of the five blocks."""

VGG19_WIDTHS = (64, 128, 256, 512, 512)
"""Output channels of the convolutions in each block."""


def _layer_names() -> tuple[str, ...]:
    names: list[str] = []
    for block, depth in enumerate(VGG19_BLOCK_DEPTHS, start=1):
        for position in range(1, depth + 1):
            names += [f"conv{block}_{position}", f"relu{block}_{position}"]
        names.append(f"pool{block}")
    return tuple(names)


VGG19_LAYERS = _layer_names()
"""Names of the 37 modules of ``torchvision.models.vgg19().features``, in order.

``conv4_2`` is the second convolution of the fourth block, ``relu4_2`` its
rectified output and ``pool4`` the pooling layer that ends the block, so
``VGG19_LAYERS[i]`` names ``features[i]``.
"""
