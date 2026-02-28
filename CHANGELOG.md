# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); while the version is 0.x, minor
releases may break the API.

## [0.2.0] - 2026-09-25

A faithful implementation of the Gatys family of methods, with a real results gallery.

### Added

- **Original Caffe VGG-19 weights** (`--weights caffe`, `load_vgg19("caffe")`): the weights of
  Simonyan & Zisserman that Gatys et al. used, fetched once from a GitHub release asset (80 MB),
  SHA-256 verified, cached in `~/.cache/neural-style` and converted into torchvision's layout,
  with Caffe preprocessing (BGR, 0-255, mean pixel subtracted). The conversion is tested offline
  against a direct NumPy cross-correlation, and `scripts/check_caffe_weights.py` verifies it on
  real photos, including ImageNet classification with the full network.
- **Presets per weight source.** With the Caffe weights the defaults are those of Gatys et al.
  (content `relu4_2`, style `relu1_1`…`relu5_1`, average pooling), with weights calibrated on
  real runs. torchvision weights keep the 0.1.0 defaults.
- **Spatial control** (Gatys et al., 2017): `--masks` applies each style to its own region,
  using guided Gram matrices with masks propagated to every layer through the pooling
  structure; optional style masks.
- **Colour control** (Gatys et al., 2016): `--color luminance` (luminance-only transfer) and
  `--color match` (colour histogram matching of the style image, `--color-match eigen|cholesky`).
- **Scale control** (Gatys et al., 2017): coarse-to-fine synthesis (`--size 256 512 --steps 300
  100`, `stylize_multiscale`) and `--style-scale`.
- `--line-search`: an optional strong-Wolfe line search for L-BFGS.
- `--pooling`, `--content-layers`, `--style-layers`, `--quality`, `--version`; `--weights random`
  for offline smoke runs; the CLI reports the progress of the weight download.
- A results gallery in `docs/gallery/`, generated on CPU by `scripts/gallery.py` from public-domain
  and CC0 images that `scripts/fetch_examples.py` downloads with pinned SHA-256 digests, with
  provenance in `docs/gallery/CREDITS.md`; a style-weight × layer grid; L-BFGS vs Adam
  convergence curves; single-scale vs coarse-to-fine timings.
- `docs/method.md`, a technical write-up of the method and of every deviation from the papers.
- Property-based tests (hypothesis), `mypy --strict` in CI, a `py.typed` marker, and an optional
  monthly CI job that exercises the real weights.

### Changed

- **Breaking:** the pipeline is rebuilt around `VGG19` (weights plus their input normalisation),
  `FeatureExtractor`, `Objective` and `TransferResult`.
- **Breaking:** layers use their canonical VGG names; the old `conv_4` is `conv2_2`.
- `--color luminance` replaces `--preserve-colors` and transfers style on the luminance channel
  instead of swapping the chrominance after an RGB stylisation.
- `stylize` returns a `TransferResult` (image, per-step loss history, resolved config, seconds).
- Style images keep their aspect ratio and are resized by pixel count relative to the content.
- The image is projected onto [0, 1] after each optimisation step rather than clamped inside the
  L-BFGS closure (equivalent without a line search, required with one).
- Dev tools are pinned; numpy is a declared dependency; h5py is optional (extra `caffe`).

### Removed

- **Breaking:** `load_vgg19_features`, `build_style_model`, `ContentLoss`, `StyleLoss` and the
  `--preserve-colors` flag.
- `docs/showcase.jpg`, the 2019 result the README showed: its input images came from
  image-hosting links with unrecorded licences. The 2019 notebooks keep their original outputs.

### Fixed

- Gram matrices of batched inputs mixed the samples of the batch.
- Style images were stretched to the content image's exact shape.
- Photos with an EXIF orientation were stylised sideways; image files were left open.
- `TransferConfig.history` was mutated by every run and accumulated across runs.
- A request in which no layer existed failed with "max() arg is an empty sequence".
- The README's loss formula did not match the code (the code uses mean squared errors).
- Images too small for the deepest layer failed inside PyTorch with "Output size is too small";
  they are now rejected with the minimum size.
- Progress lines were not flushed, so nothing appeared when stdout was a pipe or a file.

## [0.1.0] - 2026-09-25

First packaged release: the 2019 Colab script rewritten as the `neural_style` package (its
`pyproject.toml` declared version 1.0.0; this changelog counts it as 0.1.0).

### Added

- VGG-19 style transfer with content, style (Gram matrix) and total-variation losses.
- Multi-style blending, post-hoc colour preservation, L-BFGS or Adam, content or noise
  initialisation, CUDA / MPS / CPU selection, intermediate frames.
- A command-line interface, an offline test suite with a random VGG-19, and CI.

[0.1.0]: https://github.com/houssammehdi/NeuralStyleTransfer-PyTorch/commit/5104327
