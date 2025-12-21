"""Builds a truncated VGG-19 with content/style loss probes inserted."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .losses import ContentLoss, StyleLoss

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_CONTENT_LAYERS = ("conv_4",)
DEFAULT_STYLE_LAYERS = ("conv_1", "conv_2", "conv_3", "conv_4", "conv_5")


class Normalization(nn.Module):
    """Normalises an image with ImageNet statistics so VGG sees familiar inputs."""

    def __init__(
        self, mean: tuple[float, ...] = IMAGENET_MEAN, std: tuple[float, ...] = IMAGENET_STD
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(-1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(-1, 1, 1))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return (img - self.mean) / self.std


def load_vgg19_features(pretrained: bool = True) -> nn.Sequential:
    """Return the convolutional part of VGG-19 in eval mode with frozen weights."""
    from torchvision.models import VGG19_Weights, vgg19

    weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
    features = vgg19(weights=weights).features.eval()
    for p in features.parameters():
        p.requires_grad_(False)
    return features


@dataclass
class StyleModel:
    """A feature extractor whose forward pass populates the loss probes."""

    model: nn.Sequential
    content_losses: list[ContentLoss]
    style_losses: list[StyleLoss]


def build_style_model(
    cnn: nn.Sequential,
    content_img: torch.Tensor,
    style_imgs: list[torch.Tensor],
    style_blend: list[float] | None = None,
    content_layers: tuple[str, ...] = DEFAULT_CONTENT_LAYERS,
    style_layers: tuple[str, ...] = DEFAULT_STYLE_LAYERS,
) -> StyleModel:
    """Copy ``cnn`` layer by layer, inserting loss probes after the chosen convs.

    Layers after the last probe are dropped because they cannot influence the
    loss, which saves both memory and compute. In-place ReLUs are replaced
    with out-of-place ones, otherwise they would overwrite the activations the
    probes just stored.
    """
    device = content_img.device
    model = nn.Sequential(Normalization().to(device))
    content_losses: list[ContentLoss] = []
    style_losses: list[StyleLoss] = []

    conv_idx = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            conv_idx += 1
            name = f"conv_{conv_idx}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{conv_idx}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{conv_idx}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{conv_idx}"
        else:
            raise RuntimeError(f"unrecognised layer: {layer.__class__.__name__}")
        model.add_module(name, layer)

        if name in content_layers:
            probe = ContentLoss(model(content_img))
            model.add_module(f"content_loss_{conv_idx}", probe)
            content_losses.append(probe)

        if name in style_layers:
            targets = [model(s) for s in style_imgs]
            probe = StyleLoss(targets, style_blend)
            model.add_module(f"style_loss_{conv_idx}", probe)
            style_losses.append(probe)

    last = max(i for i, m in enumerate(model) if isinstance(m, (ContentLoss, StyleLoss)))
    unknown = (set(content_layers) | set(style_layers)) - {n for n, _ in model.named_children()}
    if unknown:
        raise ValueError(f"layers not found in network: {sorted(unknown)}")
    return StyleModel(model[: last + 1], content_losses, style_losses)
