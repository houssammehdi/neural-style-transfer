"""VGG-19 feature extraction with the input normalisation that matches each weight source."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from typing import Literal, get_args

import torch
from torch import nn

from .layers import VGG19_BLOCK_DEPTHS, VGG19_LAYERS, VGG19_WIDTHS

WeightSource = Literal["torchvision", "random"]
WEIGHT_SOURCES: tuple[WeightSource, ...] = get_args(WeightSource)
Pooling = Literal["max", "avg"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Normalization(nn.Module):
    """Maps an RGB image in ``[0, 1]`` to the input a particular VGG-19 was trained on.

    Computes ``(x - mean) / std`` per channel; torchvision's weights expect the
    ImageNet mean and standard deviation.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

    @classmethod
    def imagenet(cls) -> Normalization:
        """Normalisation for torchvision's ImageNet weights."""
        return cls(IMAGENET_MEAN, IMAGENET_STD)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Normalise a ``(B, 3, H, W)`` RGB batch in ``[0, 1]``."""
        return (image - self.mean) / self.std


def make_vgg19_features(widths: Sequence[int] = VGG19_WIDTHS) -> nn.Sequential:
    """Build the convolutional part of VGG-19 in torchvision's ``features`` layout.

    ``widths`` gives the channels of each of the five blocks; anything other than
    the default is only useful for tests.
    """
    if len(widths) != len(VGG19_BLOCK_DEPTHS):
        raise ValueError(f"expected {len(VGG19_BLOCK_DEPTHS)} block widths, got {len(widths)}")
    layers: list[nn.Module] = []
    in_channels = 3
    for width, depth in zip(widths, VGG19_BLOCK_DEPTHS, strict=True):
        for _ in range(depth):
            layers += [nn.Conv2d(in_channels, width, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            in_channels = width
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


_LAYER_TYPES: dict[str, tuple[type[nn.Module], ...]] = {
    "conv": (nn.Conv2d,),
    "relu": (nn.ReLU,),
    "pool": (nn.MaxPool2d, nn.AvgPool2d),
}


class VGG19(nn.Module):
    """Frozen VGG-19 convolutional layers bundled with the normalisation their weights need.

    Keeping the two together means an input convention can never be paired
    with the wrong weights. ``source`` records where the weights came from and
    selects the default loss settings (see :data:`neural_style.transfer.PRESETS`).
    """

    def __init__(self, features: nn.Sequential, normalization: Normalization, source: str) -> None:
        super().__init__()
        if len(features) != len(VGG19_LAYERS):
            raise ValueError(f"expected {len(VGG19_LAYERS)} feature modules, got {len(features)}")
        for name, module in zip(VGG19_LAYERS, features, strict=True):
            allowed = _LAYER_TYPES[name[:4]]
            if not isinstance(module, allowed):
                raise ValueError(f"{name} should be {allowed[0].__name__}, got {type(module).__name__}")
        self.features = features
        self.normalization = normalization
        self.source = source
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return the ``pool5`` activations of an RGB image in ``[0, 1]``."""
        output: torch.Tensor = self.features(self.normalization(image))
        return output


class WeightsUnavailableError(RuntimeError):
    """The requested pretrained weights could not be downloaded or loaded."""


def _torchvision_state_dict() -> dict[str, torch.Tensor]:
    from torchvision.models import VGG19_Weights

    try:
        full: dict[str, torch.Tensor] = VGG19_Weights.IMAGENET1K_V1.get_state_dict(
            progress=True, check_hash=True
        )
    except (OSError, RuntimeError) as exc:
        raise WeightsUnavailableError(
            f"could not load torchvision's VGG-19 weights ({exc}). They are downloaded from "
            "download.pytorch.org into the torch hub cache, so that host must be reachable "
            "once (or the file placed in the cache by hand)."
        ) from exc
    prefix = "features."
    return {k.removeprefix(prefix): v for k, v in full.items() if k.startswith(prefix)}


def _random_features(seed: int) -> nn.Sequential:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        features = make_vgg19_features()
        for module in features.modules():
            if isinstance(module, nn.Conv2d) and module.bias is not None:
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(module.bias)
    return features


def load_vgg19(weights: WeightSource = "torchvision", *, seed: int = 0) -> VGG19:
    """Load a frozen VGG-19 in eval mode, paired with its input normalisation.

    ``weights`` is one of:

    * ``"torchvision"`` -- torchvision's ImageNet weights (cached by torch hub);
    * ``"random"`` -- an untrained network (He initialisation seeded by
      ``seed``), for tests and offline smoke runs only.

    Raises :class:`WeightsUnavailableError` if the torchvision weights cannot
    be obtained.
    """
    if weights == "torchvision":
        features = make_vgg19_features()
        features.load_state_dict(_torchvision_state_dict())
        return VGG19(features, Normalization.imagenet(), "torchvision")
    if weights == "random":
        return VGG19(_random_features(seed), Normalization.imagenet(), "random")
    raise ValueError(f"unknown weights {weights!r}; expected one of {WEIGHT_SOURCES}")


class FeatureExtractor(nn.Module):
    """Runs VGG-19 up to the deepest requested layer and returns those activations.

    Layers after the deepest requested one cannot influence the loss, so they
    are dropped. ReLUs are made out-of-place, otherwise they would overwrite
    the convolution outputs that were just recorded. With ``pooling="avg"``
    every max-pooling layer is replaced by 2x2 average pooling, which Gatys et
    al. (2016) report gives slightly more appealing results.
    """

    def __init__(self, vgg: VGG19, layers: Iterable[str], pooling: Pooling = "max") -> None:
        super().__init__()
        wanted = set(layers)
        if not wanted:
            raise ValueError("at least one layer is required")
        unknown = sorted(wanted - set(VGG19_LAYERS))
        if unknown:
            raise ValueError(
                f"layers not found in VGG-19: {unknown} (valid names look like conv4_2, relu3_1, pool2)"
            )
        if pooling not in get_args(Pooling):
            raise ValueError(f"unknown pooling {pooling!r} (expected 'max' or 'avg')")
        depth = max(VGG19_LAYERS.index(name) for name in wanted) + 1
        body: OrderedDict[str, nn.Module] = OrderedDict()
        for name, module in zip(VGG19_LAYERS[:depth], vgg.features, strict=False):
            if isinstance(module, nn.ReLU):
                module = nn.ReLU(inplace=False)
            elif isinstance(module, nn.MaxPool2d) and pooling == "avg":
                module = nn.AvgPool2d(kernel_size=2, stride=2)
            body[name] = module
        self.normalization = vgg.normalization
        self.body = nn.Sequential(body)
        self.layers = tuple(name for name in VGG19_LAYERS[:depth] if name in wanted)
        self.source = vgg.source

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return ``{layer: activation}`` for the requested layers of an RGB image in ``[0, 1]``."""
        x: torch.Tensor = self.normalization(image)
        out: dict[str, torch.Tensor] = {}
        for name, module in self.body.named_children():
            x = module(x)
            if name in self.layers:
                out[name] = x
        return out
