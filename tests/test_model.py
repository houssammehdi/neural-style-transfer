"""VGG-19 layout, input normalisation and feature extraction."""

from __future__ import annotations

import urllib.error

import pytest
import torch
from torch import nn
from torchvision.models import VGG19_Weights
from torchvision.models.vgg import cfgs, make_layers

from neural_style import VGG19, VGG19_LAYERS, FeatureExtractor, WeightsUnavailableError, load_vgg19
from neural_style.model import Normalization, make_vgg19_features


def test_layer_names_follow_the_vgg_convention() -> None:
    assert len(VGG19_LAYERS) == 37
    assert VGG19_LAYERS[:5] == ("conv1_1", "relu1_1", "conv1_2", "relu1_2", "pool1")
    assert VGG19_LAYERS.index("conv4_2") == 21
    assert VGG19_LAYERS.index("relu5_1") == 29
    assert VGG19_LAYERS[-1] == "pool5"


def test_features_have_torchvision_layout() -> None:
    ours = make_vgg19_features()
    theirs = make_layers(cfgs["E"])  # torchvision's own vgg19 `features` builder
    assert [type(m) for m in ours] == [type(m) for m in theirs]
    assert {k: v.shape for k, v in ours.state_dict().items()} == {
        k: v.shape for k, v in theirs.state_dict().items()
    }
    for name, module in zip(VGG19_LAYERS, ours, strict=True):
        assert isinstance(module, {"conv": nn.Conv2d, "relu": nn.ReLU, "pool": nn.MaxPool2d}[name[:4]])


def test_imagenet_normalization() -> None:
    red = torch.tensor([1.0, 0.0, 0.0]).view(1, 3, 1, 1)
    out = Normalization.imagenet()(red).flatten()
    expected = torch.tensor([(1 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225])
    torch.testing.assert_close(out, expected)


def test_vgg19_is_frozen_and_validates_its_layout(vgg: VGG19) -> None:
    assert not any(p.requires_grad for p in vgg.parameters())
    assert not vgg.training
    broken = make_vgg19_features()
    broken[1] = nn.Identity()
    with pytest.raises(ValueError, match="relu1_1"):
        VGG19(broken, Normalization.imagenet(), "random")
    with pytest.raises(ValueError, match="37"):
        VGG19(nn.Sequential(*list(make_vgg19_features())[:-1]), Normalization.imagenet(), "random")


def test_random_weights_are_seeded_without_touching_the_global_rng() -> None:
    torch.manual_seed(7)
    expected = torch.rand(3)
    torch.manual_seed(7)
    a = load_vgg19("random", seed=3)
    assert torch.equal(torch.rand(3), expected)
    b = load_vgg19("random", seed=3)
    conv_a, conv_b = a.features[0], b.features[0]
    assert isinstance(conv_a, nn.Conv2d) and isinstance(conv_b, nn.Conv2d)
    assert torch.equal(conv_a.weight, conv_b.weight)


def test_unknown_weight_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown weights"):
        load_vgg19("imagenet")  # type: ignore[arg-type]


def test_torchvision_weights_keep_only_the_features(monkeypatch: pytest.MonkeyPatch) -> None:
    reference = make_vgg19_features()
    full = {f"features.{k}": torch.full_like(v, 0.5) for k, v in reference.state_dict().items()}
    full["classifier.0.weight"] = torch.zeros(2, 2)

    def fake(self: object, **kwargs: object) -> dict[str, torch.Tensor]:
        assert kwargs["check_hash"] is True
        return full

    monkeypatch.setattr(type(VGG19_Weights.IMAGENET1K_V1), "get_state_dict", fake)
    vgg = load_vgg19("torchvision")
    assert vgg.source == "torchvision"
    conv = vgg.features[0]
    assert isinstance(conv, nn.Conv2d) and torch.all(conv.weight == 0.5)
    assert isinstance(vgg.normalization, Normalization)


def test_unreachable_torchvision_weights_are_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self: object, **kwargs: object) -> dict[str, torch.Tensor]:
        raise urllib.error.URLError("Tunnel connection failed: 403 Forbidden")

    monkeypatch.setattr(type(VGG19_Weights.IMAGENET1K_V1), "get_state_dict", fail)
    with pytest.raises(WeightsUnavailableError, match="download.pytorch.org"):
        load_vgg19("torchvision")


def test_extractor_truncates_after_the_deepest_layer(vgg: VGG19) -> None:
    extractor = FeatureExtractor(vgg, ["relu2_1", "conv1_1"])
    assert [name for name, _ in extractor.body.named_children()][-1] == "relu2_1"
    assert extractor.layers == ("conv1_1", "relu2_1")  # network order
    out = extractor(torch.rand(1, 3, 16, 16))
    assert set(out) == {"conv1_1", "relu2_1"}
    assert out["relu2_1"].shape == (1, 128, 8, 8)


def test_conv_outputs_are_not_overwritten_by_relu(vgg: VGG19) -> None:
    # In-place ReLUs would clobber the recorded conv1_1 activations.
    out = FeatureExtractor(vgg, ["conv1_1", "relu1_1"])(torch.rand(1, 3, 16, 16))
    assert out["conv1_1"].min() < 0
    torch.testing.assert_close(out["relu1_1"], out["conv1_1"].clamp_min(0))


def test_unknown_layers_are_rejected_with_a_clear_message(vgg: VGG19) -> None:
    with pytest.raises(ValueError, match="conv_99"):
        FeatureExtractor(vgg, ["conv1_1", "conv_99"])
    # Regression: v0.1 failed with "max() arg is an empty sequence" when no layer existed.
    with pytest.raises(ValueError, match="conv_4"):
        FeatureExtractor(vgg, ["conv_4"])
    with pytest.raises(ValueError, match="at least one"):
        FeatureExtractor(vgg, [])


def test_average_pooling_replaces_max_pooling(vgg: VGG19) -> None:
    extractor = FeatureExtractor(vgg, ["relu3_1"], pooling="avg")
    pools = [m for m in extractor.body if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d))]
    assert len(pools) == 2 and all(isinstance(p, nn.AvgPool2d) for p in pools)
    assert any(isinstance(m, nn.MaxPool2d) for m in vgg.features)  # the shared network is untouched
