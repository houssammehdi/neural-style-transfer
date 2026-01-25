"""Image I/O and resizing."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from neural_style import load_image, load_mask, resize, save_image
from neural_style.image import area_shape, resize_to_area, target_shape, to_pil_image


def test_png_round_trip(tmp_path: Path) -> None:
    img = torch.rand(1, 3, 20, 30, generator=torch.Generator().manual_seed(0))
    save_image(img, tmp_path / "nested" / "img.png")  # parent directories are created
    back = load_image(tmp_path / "nested" / "img.png")
    torch.testing.assert_close(back, img, atol=1 / 255, rtol=0)


def test_integer_size_is_the_shorter_edge(tmp_path: Path) -> None:
    Image.new("RGB", (300, 200)).save(tmp_path / "wide.png")
    assert load_image(tmp_path / "wide.png", 100).shape == (1, 3, 100, 150)
    assert load_image(tmp_path / "wide.png", (40, 50)).shape == (1, 3, 40, 50)
    assert target_shape(200, 300, 100) == (100, 150)
    with pytest.raises(ValueError):
        target_shape(10, 10, 0)


def test_exif_orientation_is_applied(tmp_path: Path) -> None:
    # Regression: phone photos store rotation in EXIF; v0.1 loaded them sideways.
    img = Image.new("RGB", (40, 20), "black")
    img.paste((255, 0, 0), (0, 0, 40, 5))  # red stripe along the top of the stored pixels
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90 degrees clockwise to display"
    img.save(tmp_path / "rotated.jpg", exif=exif, quality=95)
    loaded = load_image(tmp_path / "rotated.jpg")
    assert loaded.shape == (1, 3, 40, 20)  # displayed upright: portrait
    assert loaded[0, 0, :, -3:].mean() > 0.8  # the stripe is now on the right edge
    assert loaded[0, 0, :, :3].mean() < 0.2


def test_masks_load_as_single_channel(tmp_path: Path) -> None:
    Image.new("L", (8, 4), 255).save(tmp_path / "mask.png")
    mask = load_mask(tmp_path / "mask.png", (2, 4))
    assert mask.shape == (1, 1, 2, 4) and torch.all(mask == 1)


def test_resize_to_area_preserves_the_aspect_ratio() -> None:
    img = torch.rand(1, 3, 30, 120)
    out = resize_to_area(img, 40 * 40)
    assert out.shape[-2:] == (20, 80)
    assert area_shape(30, 120, 30 * 120) == (30, 120)


def test_bicubic_resize_is_clamped_and_same_size_is_a_no_op() -> None:
    checker = (torch.arange(64).view(8, 8) % 2).float().expand(1, 3, 8, 8)
    up = resize(checker, (16, 16))
    assert up.min() >= 0 and up.max() <= 1
    assert resize(checker, (8, 8)) is checker


def test_to_pil_image_handles_greyscale_and_batches() -> None:
    assert to_pil_image(torch.zeros(1, 1, 4, 5)).mode == "L"
    assert to_pil_image(torch.ones(3, 4, 5)).size == (5, 4)


def test_jpeg_quality_is_configurable(tmp_path: Path) -> None:
    img = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(1))
    save_image(img, tmp_path / "hi.jpg", quality=95)
    save_image(img, tmp_path / "lo.jpg", quality=30)
    assert (tmp_path / "lo.jpg").stat().st_size < (tmp_path / "hi.jpg").stat().st_size
