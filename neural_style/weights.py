"""Pretrained VGG-19 weights: download cache, SHA-256 verification and Keras conversion.

Two sources of ImageNet VGG-19 weights are supported:

* ``torchvision`` -- the weights shipped with torchvision (downloaded from
  ``download.pytorch.org`` by :mod:`torch.hub`).
* ``caffe`` -- the original weights released by Simonyan & Zisserman (2014) for
  Caffe, which are the ones Gatys et al. used. They are fetched as the Keras
  conversion that François Chollet published as a GitHub release asset and are
  converted here into the layout of ``torchvision.models.vgg19().features``.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .layers import VGG19_LAYERS

ProgressFn = Callable[[int, int | None], None]
"""Download progress callback: ``(bytes_done, bytes_total_or_None)``."""


@dataclass(frozen=True)
class RemoteFile:
    """A file that is downloaded once, verified with SHA-256 and cached."""

    url: str
    filename: str
    sha256: str
    size: int


CAFFE_VGG19 = RemoteFile(
    url=(
        "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/"
        "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"
    ),
    filename="vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5",
    # Computed from the release asset; its MD5 (253f8cb515780f3b799900260a226db6) is the
    # ``file_hash`` that keras.applications.vgg19 pins for the same file.
    sha256="bde7da0dfd621cbf86e11ce04e4946a9a42634c2cfd21318bba090d860699e35",
    size=80_134_624,
)
"""Keras (TensorFlow dim ordering) conversion of the original Caffe VGG-19, without the classifier."""


class ChecksumError(RuntimeError):
    """A downloaded or cached file does not have the expected SHA-256 digest."""


def default_cache_dir() -> Path:
    """Return the weight cache directory.

    ``$NEURAL_STYLE_CACHE`` wins if set, then ``$XDG_CACHE_HOME/neural-style``,
    then ``~/.cache/neural-style``.
    """
    explicit = os.environ.get("NEURAL_STYLE_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "neural-style"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(
    remote: RemoteFile,
    cache_dir: Path | None = None,
    progress: ProgressFn | None = None,
    timeout: float = 60.0,
) -> Path:
    """Return a verified local copy of ``remote``, downloading it on first use.

    The download is streamed to ``<name>.part`` while hashing and only renamed
    into place once the SHA-256 digest matches, so an interrupted or tampered
    download never ends up in the cache. An existing cached file is re-hashed
    on every call; a mismatch raises :class:`ChecksumError` instead of being
    silently replaced.
    """
    directory = cache_dir if cache_dir is not None else default_cache_dir()
    target = directory / remote.filename
    if target.exists():
        actual = sha256_file(target)
        if actual != remote.sha256:
            raise ChecksumError(
                f"{target} has SHA-256 {actual}, expected {remote.sha256}; delete it to re-download"
            )
        return target

    directory.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    done = 0
    try:
        with urllib.request.urlopen(remote.url, timeout=timeout) as response, partial.open("wb") as out:
            length = response.headers.get("Content-Length")
            total = int(length) if length is not None else None
            while chunk := response.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
        if digest.hexdigest() != remote.sha256:
            raise ChecksumError(
                f"download of {remote.url} has SHA-256 {digest.hexdigest()}, expected {remote.sha256}"
            )
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def _decode(names: np.ndarray | list[bytes] | list[str]) -> list[str]:
    return [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in names]


def keras_vgg19_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Convert a Keras VGG-19 HDF5 weight file into a ``vgg19().features`` state dict.

    Keras stores convolution kernels as ``(height, width, in, out)``; PyTorch
    wants ``(out, in, height, width)``. TensorFlow's ``conv2d`` and PyTorch's
    ``conv2d`` both compute a cross-correlation, so the kernels are transposed
    but not spatially flipped (``tests/test_weights.py`` checks this against a
    direct NumPy cross-correlation).

    Only the 16 convolutional layers are read (``block{b}_conv{i}``); any
    classifier weights in the file are ignored. Channel widths are validated for
    consistency but not fixed, which lets the tests use a tiny synthetic file.

    Requires the optional ``h5py`` dependency (``pip install "neural-style[caffe]"``).
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError('converting Keras weights needs h5py: pip install "neural-style[caffe]"') from exc

    state: dict[str, torch.Tensor] = {}
    with h5py.File(path, "r") as handle:
        root = handle.get("model_weights", handle)
        in_channels = 3
        for index, name in enumerate(VGG19_LAYERS):
            if not name.startswith("conv"):
                continue
            block, position = name.removeprefix("conv").split("_")
            keras_name = f"block{block}_conv{position}"
            if keras_name not in root:
                raise ValueError(f"{path}: layer {keras_name!r} not found")
            group = root[keras_name]
            arrays = [np.asarray(group[n]) for n in _decode(group.attrs["weight_names"])]
            kernels = [a for a in arrays if a.ndim == 4]
            biases = [a for a in arrays if a.ndim == 1]
            if len(kernels) != 1 or len(biases) != 1:
                raise ValueError(f"{path}: {keras_name} should hold one kernel and one bias")
            kernel, bias = kernels[0], biases[0]
            height, width, fan_in, fan_out = kernel.shape
            if (height, width) != (3, 3) or fan_in != in_channels or bias.shape != (fan_out,):
                raise ValueError(
                    f"{path}: {keras_name} has kernel {kernel.shape} and bias {bias.shape}, "
                    f"expected (3, 3, {in_channels}, C) and (C,)"
                )
            weight = np.ascontiguousarray(kernel.transpose(3, 2, 0, 1), dtype=np.float32)
            state[f"{index}.weight"] = torch.from_numpy(weight)
            state[f"{index}.bias"] = torch.from_numpy(np.array(bias, dtype=np.float32))
            in_channels = fan_out
    return state
