"""YIQ conversion and post-hoc colour preservation."""

from __future__ import annotations

import torch

from neural_style import preserve_colors
from neural_style.color import luminance, rgb_to_yiq, yiq_to_rgb


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
