"""Keras/Caffe weight conversion and the verified download cache, tested offline."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from neural_style import model as model_module
from neural_style.layers import VGG19_LAYERS
from neural_style.model import Normalization, load_vgg19, make_vgg19_features
from neural_style.weights import (
    CAFFE_VGG19,
    ChecksumError,
    RemoteFile,
    default_cache_dir,
    fetch,
    keras_vgg19_state_dict,
)

h5py = pytest.importorskip("h5py")

TINY_WIDTHS = (4, 5, 6, 7, 7)
KerasWeights = dict[str, tuple[np.ndarray, np.ndarray]]


def write_keras_vgg19(
    path: Path, widths: tuple[int, ...] = TINY_WIDTHS, seed: int = 0, nest: bool = False
) -> KerasWeights:
    """Write an HDF5 file laid out like keras.applications' VGG-19 weights (random values)."""
    rng = np.random.default_rng(seed)
    layers: KerasWeights = {}
    names = ["input_1"]
    in_channels = 3
    for block, (width, depth) in enumerate(zip(widths, (2, 2, 4, 4, 4), strict=True), start=1):
        for position in range(1, depth + 1):
            name = f"block{block}_conv{position}"
            kernel = rng.standard_normal((3, 3, in_channels, width)).astype(np.float32)
            layers[name] = (kernel, rng.standard_normal(width).astype(np.float32))
            names.append(name)
            in_channels = width
        names.append(f"block{block}_pool")
    with h5py.File(path, "w") as handle:
        root = handle.create_group("model_weights") if nest else handle
        root.attrs["layer_names"] = np.array([n.encode() for n in names])
        for name in names:
            group = root.create_group(name)
            if name not in layers:
                group.attrs["weight_names"] = np.array([], dtype="S1")
                continue
            kernel, bias = layers[name]
            weight_names = [f"{name}_W_1:0", f"{name}_b_1:0"]  # the naming used by the real file
            group.attrs["weight_names"] = np.array([n.encode() for n in weight_names])
            group.create_dataset(weight_names[0], data=kernel)
            group.create_dataset(weight_names[1], data=bias)
    return layers


def numpy_conv_same(image: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Direct cross-correlation with 'same' padding, as TensorFlow's conv2d defines it.

    ``image`` is (C, H, W) and ``kernel`` is in Keras layout (kh, kw, C, out).
    """
    channels, height, width = image.shape
    padded = np.pad(image, ((0, 0), (1, 1), (1, 1)))
    out = np.empty((kernel.shape[3], height, width))
    for y in range(height):
        for x in range(width):
            patch = padded[:, y : y + 3, x : x + 3]  # (C, 3, 3)
            out[:, y, x] = np.einsum("chw,hwco->o", patch, kernel) + bias
    return out


def test_converter_produces_the_torchvision_layout(tmp_path: Path) -> None:
    layers = write_keras_vgg19(tmp_path / "tiny.h5")
    state = keras_vgg19_state_dict(tmp_path / "tiny.h5")
    features = make_vgg19_features(TINY_WIDTHS)
    features.load_state_dict(state, strict=True)  # every key present, every shape right
    conv4_2 = VGG19_LAYERS.index("conv4_2")
    kernel, bias = layers["block4_conv2"]
    assert torch.equal(state[f"{conv4_2}.weight"], torch.from_numpy(kernel.transpose(3, 2, 0, 1)))
    assert torch.equal(state[f"{conv4_2}.bias"], torch.from_numpy(bias))


@pytest.mark.parametrize("layer", ["block1_conv1", "block3_conv2"])
def test_converted_convolution_equals_a_direct_numpy_cross_correlation(tmp_path: Path, layer: str) -> None:
    layers = write_keras_vgg19(tmp_path / "tiny.h5")
    state = keras_vgg19_state_dict(tmp_path / "tiny.h5")
    kernel, bias = layers[layer]
    index = VGG19_LAYERS.index(layer.replace("block", "conv").replace("_conv", "_"))
    image = np.random.default_rng(1).standard_normal((kernel.shape[2], 7, 9))

    expected = numpy_conv_same(image, kernel.astype(np.float64), bias.astype(np.float64))
    weight, b = state[f"{index}.weight"].double(), state[f"{index}.bias"].double()
    actual = F.conv2d(torch.from_numpy(image)[None], weight, b, padding=1)[0].numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    # The check is sharp: spatially flipping the kernels (true convolution) would not match.
    flipped = F.conv2d(torch.from_numpy(image)[None], weight.flip(-1, -2), b, padding=1)[0].numpy()
    assert not np.allclose(flipped, expected, atol=1e-3)


def test_converter_accepts_full_model_files(tmp_path: Path) -> None:
    # Files saved with the whole model nest the layers under "model_weights".
    write_keras_vgg19(tmp_path / "nested.h5", nest=True)
    make_vgg19_features(TINY_WIDTHS).load_state_dict(keras_vgg19_state_dict(tmp_path / "nested.h5"))


def test_converter_rejects_missing_or_inconsistent_layers(tmp_path: Path) -> None:
    write_keras_vgg19(tmp_path / "tiny.h5")
    with h5py.File(tmp_path / "tiny.h5", "a") as handle:
        del handle["block5_conv4"]
    with pytest.raises(ValueError, match="block5_conv4"):
        keras_vgg19_state_dict(tmp_path / "tiny.h5")

    write_keras_vgg19(tmp_path / "bad.h5")
    with h5py.File(tmp_path / "bad.h5", "a") as handle:
        group = handle["block2_conv1"]
        del group["block2_conv1_W_1:0"]
        group.create_dataset("block2_conv1_W_1:0", data=np.zeros((3, 3, 99, 5), np.float32))
    with pytest.raises(ValueError, match="block2_conv1"):
        keras_vgg19_state_dict(tmp_path / "bad.h5")


def _remote(source: Path, **overrides: str) -> RemoteFile:
    data = source.read_bytes()
    fields = {
        "url": source.as_uri(),
        "filename": "weights.bin",
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    fields.update(overrides)
    return RemoteFile(size=len(data), **fields)


def test_fetch_downloads_verifies_and_caches(tmp_path: Path) -> None:
    source = tmp_path / "upstream.bin"
    source.write_bytes(b"pretend these are weights" * 1000)
    remote = _remote(source)
    seen: list[int] = []
    path = fetch(remote, tmp_path / "cache", progress=lambda done, total: seen.append(done))
    assert path.read_bytes() == source.read_bytes()
    assert seen and seen[-1] == remote.size
    source.unlink()  # a second call must be served from the cache
    assert fetch(remote, tmp_path / "cache") == path


def test_fetch_rejects_a_bad_download_and_caches_nothing(tmp_path: Path) -> None:
    source = tmp_path / "upstream.bin"
    source.write_bytes(b"tampered")
    with pytest.raises(ChecksumError, match="expected"):
        fetch(_remote(source, sha256="0" * 64), tmp_path / "cache")
    assert list((tmp_path / "cache").iterdir()) == []  # no partial or unverified file left behind


def test_fetch_refuses_a_corrupted_cache_entry(tmp_path: Path) -> None:
    source = tmp_path / "upstream.bin"
    source.write_bytes(b"good weights")
    remote = _remote(source)
    cached = fetch(remote, tmp_path / "cache")
    cached.write_bytes(b"bit rot")
    with pytest.raises(ChecksumError, match="delete it"):
        fetch(remote, tmp_path / "cache")


def test_default_cache_dir_honours_the_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("NEURAL_STYLE_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert default_cache_dir() == tmp_path / "xdg" / "neural-style"
    monkeypatch.setenv("NEURAL_STYLE_CACHE", str(tmp_path / "explicit"))
    assert default_cache_dir() == tmp_path / "explicit"
    monkeypatch.delenv("NEURAL_STYLE_CACHE")
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert default_cache_dir() == Path.home() / ".cache" / "neural-style"


def test_caffe_weights_are_pinned() -> None:
    assert CAFFE_VGG19.url.startswith("https://github.com/fchollet/deep-learning-models/releases/")
    assert CAFFE_VGG19.url.endswith("/" + CAFFE_VGG19.filename)
    assert len(CAFFE_VGG19.sha256) == 64 and int(CAFFE_VGG19.sha256, 16) > 0


def test_load_vgg19_caffe_wires_download_conversion_and_normalisation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_keras_vgg19(tmp_path / "tiny.h5")
    remote = _remote(tmp_path / "tiny.h5", filename="tiny-vgg19.h5")
    monkeypatch.setattr(model_module, "CAFFE_VGG19", remote)
    monkeypatch.setattr(model_module, "make_vgg19_features", lambda: make_vgg19_features(TINY_WIDTHS))
    vgg = load_vgg19("caffe", cache_dir=tmp_path / "cache")
    assert vgg.source == "caffe"
    assert (tmp_path / "cache" / "tiny-vgg19.h5").exists()
    assert isinstance(vgg.normalization, Normalization) and vgg.normalization.bgr
    assert vgg.normalization.scale == 255.0
    conv = vgg.features[0]
    assert isinstance(conv, torch.nn.Conv2d)
    assert torch.equal(conv.weight, keras_vgg19_state_dict(tmp_path / "tiny.h5")["0.weight"])
