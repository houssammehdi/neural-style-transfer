"""Fast tests that use a randomly initialised VGG-19 (no weight download)."""

import pytest
import torch

from neural_style import (
    StyleLoss,
    TransferConfig,
    build_style_model,
    gram_matrix,
    load_image,
    preserve_colors,
    save_image,
    stylize,
)
from neural_style.cli import build_parser
from neural_style.model import load_vgg19_features


@pytest.fixture(scope="module")
def cnn() -> torch.nn.Sequential:
    torch.manual_seed(0)
    return load_vgg19_features(pretrained=False)


def test_gram_matrix_matches_definition() -> None:
    x = torch.randn(1, 3, 4, 5)
    flat = x.reshape(3, 20)
    expected = flat @ flat.T / (3 * 4 * 5)
    torch.testing.assert_close(gram_matrix(x)[0], expected)


def test_gram_matrix_is_computed_per_sample() -> None:
    # Regression: reshaping (B, C, H, W) to (B*C, H*W) mixed the samples of a batch.
    x = torch.randn(2, 3, 4, 5)
    g = gram_matrix(x)
    assert g.shape == (2, 3, 3)
    torch.testing.assert_close(g[1], gram_matrix(x[1:])[0])


def test_gram_matrix_is_symmetric_and_translation_invariant() -> None:
    x = torch.randn(1, 8, 6, 6)
    g = gram_matrix(x)
    torch.testing.assert_close(g, g.transpose(1, 2))
    # rolling pixels changes *where* features are, not which co-occur
    torch.testing.assert_close(gram_matrix(torch.roll(x, shifts=2, dims=-1)), g)


def test_style_blend_interpolates_gram_targets() -> None:
    a, b = torch.randn(1, 4, 5, 5), torch.randn(1, 4, 5, 5)
    loss = StyleLoss([a, b], [3.0, 1.0])
    torch.testing.assert_close(loss.target, 0.75 * gram_matrix(a) + 0.25 * gram_matrix(b))
    with pytest.raises(ValueError):
        StyleLoss([a, b], [1.0])


def test_model_is_truncated_after_last_probe(cnn: torch.nn.Sequential) -> None:
    img = torch.rand(1, 3, 32, 32)
    net = build_style_model(cnn, img, [img])
    assert len(net.content_losses) == 1
    assert len(net.style_losses) == 5
    assert list(net.model.named_children())[-1][0] == "style_loss_5"
    assert not any(isinstance(m, torch.nn.ReLU) and m.inplace for m in net.model)


def test_unknown_layer_is_rejected(cnn: torch.nn.Sequential) -> None:
    img = torch.rand(1, 3, 32, 32)
    with pytest.raises(ValueError, match="conv_99"):
        build_style_model(cnn, img, [img], style_layers=("conv_1", "conv_99"))


@pytest.mark.parametrize("optimizer", ["lbfgs", "adam"])
def test_stylize_reduces_loss(cnn: torch.nn.Sequential, optimizer: str) -> None:
    torch.manual_seed(1)
    content, style = torch.rand(1, 3, 48, 48), torch.rand(1, 3, 48, 48)
    config = TransferConfig(steps=15, optimizer=optimizer, lr=0.05, init="noise", tv_weight=1e-3)
    out = stylize(cnn, content, [style], config)
    assert out.shape == content.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert config.history[-1]["total"] < config.history[0]["total"]


def test_preserve_colors_keeps_luminance_of_stylised_image() -> None:
    stylised, content = torch.rand(1, 3, 8, 8), torch.rand(1, 3, 8, 8)
    out = preserve_colors(stylised, content)
    luma = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    # identical luminance wherever no clamping happened
    mask = (out > 0) & (out < 1)
    mask = mask.all(dim=1, keepdim=True)
    torch.testing.assert_close(
        ((out * luma).sum(1, keepdim=True))[mask],
        ((stylised * luma).sum(1, keepdim=True))[mask],
        atol=1e-5,
        rtol=0,
    )


def test_image_round_trip(tmp_path) -> None:
    img = torch.rand(1, 3, 20, 30)
    path = tmp_path / "img.png"
    save_image(img, path)
    back = load_image(path, (20, 30), torch.device("cpu"))
    torch.testing.assert_close(back, img, atol=1 / 255, rtol=0)


def test_cli_parser_defaults() -> None:
    args = build_parser().parse_args(["c.jpg", "s1.jpg", "s2.jpg", "--blend", "1", "2"])
    assert len(args.styles) == 2 and args.blend == [1.0, 2.0] and args.optimizer == "lbfgs"
