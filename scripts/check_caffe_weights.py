"""Check the Keras -> PyTorch conversion of the original Caffe VGG-19 on real data.

1. **Convolution check** -- the converted ``conv1_1`` (PyTorch ``conv2d``) is
   compared with a direct NumPy cross-correlation of the raw Keras kernel on a
   Caffe-preprocessed photo.
2. **Classification check** (``--full``) -- the complete VGG-19, classifier
   included, is rebuilt from the 575 MB Keras file and classifies the example
   photos, once with the converted kernels and once with every kernel
   spatially flipped. Only the unflipped network should be right, which shows
   that no flip is needed and that the check can tell the difference.

    python scripts/fetch_examples.py
    python scripts/check_caffe_weights.py            # convolution check (80 MB weights)
    python scripts/check_caffe_weights.py --full     # plus classification (575 MB download)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import VGG19_Weights

from neural_style import load_image, load_vgg19
from neural_style.image import resize
from neural_style.layers import VGG19_LAYERS
from neural_style.model import VGG19, Normalization, make_vgg19_features
from neural_style.weights import CAFFE_VGG19, RemoteFile, fetch, keras_vgg19_state_dict

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "inputs"

FULL_VGG19 = RemoteFile(
    url=(
        "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/"
        "vgg19_weights_tf_dim_ordering_tf_kernels.h5"
    ),
    filename="vgg19_weights_tf_dim_ordering_tf_kernels.h5",
    # MD5 cbe5617147190e668d6c5d5026f83318 is the file_hash keras.applications.vgg19 pins.
    sha256="c872d82e40328910cd7a12763dae043d56eb00efe6a137f31899514d637d4cdb",
    size=574_710_816,
)


def convolution_check(image: torch.Tensor, weights_path: Path) -> float:
    """Return the largest |PyTorch - NumPy| difference for conv1_1 on ``image``."""
    with h5py.File(weights_path, "r") as handle:
        group = handle["block1_conv1"]
        kernel, bias = (np.asarray(group[n.decode()], dtype=np.float64) for n in group.attrs["weight_names"])
    x = Normalization.caffe()(image)[0].double().numpy()  # (3, H, W), BGR, mean-subtracted
    padded = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    height, width = x.shape[1:]
    expected = np.zeros((kernel.shape[3], height, width))
    for dy in range(3):
        for dx in range(3):  # out[o, y, x] = sum_c,dy,dx in[c, y+dy, x+dx] * k[dy, dx, c, o]
            expected += np.einsum("chw,co->ohw", padded[:, dy : dy + height, dx : dx + width], kernel[dy, dx])
    expected += bias[:, None, None]
    state = keras_vgg19_state_dict(weights_path)
    actual = F.conv2d(
        torch.from_numpy(x)[None], state["0.weight"].double(), state["0.bias"].double(), padding=1
    )
    return float(np.abs(actual[0].numpy() - expected).max())


def classifier(path: Path) -> nn.Sequential:
    """Rebuild VGG-19's three fully connected layers from the Keras file."""
    with h5py.File(path, "r") as handle:

        def dense(name: str) -> tuple[np.ndarray, np.ndarray]:
            group = handle[name]
            kernel, bias = (np.asarray(group[n.decode()]) for n in group.attrs["weight_names"])
            return kernel, bias

        (k1, b1), (k2, b2), (k3, b3) = dense("fc1"), dense("fc2"), dense("predictions")
    # Keras flattens pool5 as (h, w, c); PyTorch flattens (c, h, w).
    w1 = k1.reshape(7, 7, 512, 4096).transpose(3, 2, 0, 1).reshape(4096, 25088)
    layers = [nn.Linear(25088, 4096), nn.ReLU(), nn.Linear(4096, 4096), nn.ReLU(), nn.Linear(4096, 1000)]
    for linear, (w, b) in zip(layers[::2], [(w1, b1), (k2.T, b2), (k3.T, b3)], strict=True):
        assert isinstance(linear, nn.Linear)
        linear.weight.data = torch.from_numpy(np.ascontiguousarray(w))
        linear.bias.data = torch.from_numpy(b)
    return nn.Sequential(*layers).eval()


def classify(vgg: VGG19, head: nn.Sequential, image: torch.Tensor) -> list[tuple[str, float]]:
    """Top-3 ImageNet labels for a 224x224 centre crop of ``image``."""
    small = resize(image, 256)
    top, left = (small.shape[-2] - 224) // 2, (small.shape[-1] - 224) // 2
    crop = small[..., top : top + 224, left : left + 224]
    with torch.no_grad():
        probabilities = head(vgg(crop).flatten(1)).softmax(-1)[0]
    categories: list[str] = VGG19_Weights.IMAGENET1K_V1.meta["categories"]
    values, indices = probabilities.topk(3)
    return [(categories[i], float(p)) for p, i in zip(values, indices, strict=True)]


def flipped(vgg: VGG19) -> VGG19:
    """A copy of ``vgg`` whose kernels are rotated by 180 degrees (true convolution)."""
    features = make_vgg19_features()
    features.load_state_dict(vgg.features.state_dict())
    for name, module in zip(VGG19_LAYERS, features, strict=True):
        if name.startswith("conv"):
            assert isinstance(module, nn.Conv2d)
            module.weight.data = module.weight.data.flip(-1, -2)
    return VGG19(features, Normalization.caffe(), "caffe")


def main(argv: list[str] | None = None) -> int:
    """Run the checks and print a Markdown-friendly report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--full", action="store_true", help="also run the classification check (575 MB)")
    parser.add_argument("--cache-dir", type=Path, default=None, help="where the weight files are cached")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)

    photos = {name: load_image(EXAMPLES / name) for name in ("chelsea.png", "coffee.png", "rocket.jpg")}
    notop = fetch(CAFFE_VGG19, args.cache_dir)
    error = convolution_check(photos["chelsea.png"], notop)
    print(f"conv1_1 on chelsea.png: max |PyTorch - NumPy cross-correlation| = {error:.2e}")
    if not args.full:
        return 0

    vgg = load_vgg19("caffe", cache_dir=args.cache_dir)
    head = classifier(fetch(FULL_VGG19, args.cache_dir))
    for label, network in (("converted kernels", vgg), ("flipped kernels", flipped(vgg))):
        print(f"\n{label}:")
        for name, photo in photos.items():
            top = ", ".join(f"{c} {p:.2f}" for c, p in classify(network, head, photo))
            print(f"  {name:12} {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
