"""Neural style transfer with PyTorch and VGG-19, after Gatys et al. (2016, 2017)."""

from importlib.metadata import PackageNotFoundError, version

from .color import match_color, match_luminance, preserve_colors
from .image import load_image, load_mask, resize, save_image
from .layers import VGG19_LAYERS
from .losses import gram_matrix, guided_gram_matrix, total_variation
from .model import VGG19, FeatureExtractor, WeightsUnavailableError, load_vgg19
from .transfer import PRESETS, LossRecord, Objective, TransferConfig, TransferResult, stylize

try:
    __version__ = version("neural-style")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without installing
    __version__ = "0+unknown"

__all__ = [
    "PRESETS",
    "VGG19",
    "VGG19_LAYERS",
    "FeatureExtractor",
    "LossRecord",
    "Objective",
    "TransferConfig",
    "TransferResult",
    "WeightsUnavailableError",
    "__version__",
    "gram_matrix",
    "guided_gram_matrix",
    "load_image",
    "load_mask",
    "load_vgg19",
    "match_color",
    "match_luminance",
    "preserve_colors",
    "resize",
    "save_image",
    "stylize",
    "total_variation",
]
