"""Neural style transfer with PyTorch and VGG-19, after Gatys et al. (2016)."""

from importlib.metadata import PackageNotFoundError, version

from .color import preserve_colors
from .image import load_image, resize, save_image
from .layers import VGG19_LAYERS
from .losses import gram_matrix, total_variation
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
    "load_image",
    "load_vgg19",
    "preserve_colors",
    "resize",
    "save_image",
    "stylize",
    "total_variation",
]
