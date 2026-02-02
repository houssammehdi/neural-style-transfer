"""Command-line interface: ``neural-style content.jpg style.jpg -o out.jpg``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import get_args

import torch

from . import __version__
from .color import ColorMatchMethod
from .image import load_image, load_mask, save_image
from .model import WEIGHT_SOURCES, Pooling, WeightsUnavailableError, load_vgg19
from .transfer import (
    ColorMode,
    InitName,
    LossRecord,
    OptimizerName,
    TransferConfig,
    TransferResult,
    stylize,
    stylize_multiscale,
)


def pick_device(preferred: str = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA, then Apple MPS, then CPU."""
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser of the ``neural-style`` command."""
    p = argparse.ArgumentParser(
        prog="neural-style",
        description="Neural style transfer with VGG-19 (Gatys et al., 2016, 2017).",
    )
    p.add_argument("content", type=Path, help="content image")
    p.add_argument("styles", type=Path, nargs="+", help="one or more style images")
    p.add_argument("-o", "--output", type=Path, default=Path("output.jpg"))
    p.add_argument(
        "--weights",
        choices=WEIGHT_SOURCES,
        default="torchvision",
        help="VGG-19 weights: torchvision's ImageNet weights, the original Caffe weights used by "
        "Gatys et al. (80 MB from GitHub, needs h5py), or an untrained network for smoke tests",
    )
    p.add_argument(
        "--size",
        type=int,
        nargs="+",
        help="shorter edge of the output in px (default 512 on GPU, 256 on CPU); several ascending "
        "values run coarse-to-fine synthesis, e.g. --size 256 512",
    )
    p.add_argument(
        "--steps", type=int, nargs="+", default=[300], help="optimisation steps, one value or one per size"
    )
    p.add_argument("--content-weight", type=float, help="content weight alpha (default: preset)")
    p.add_argument("--style-weight", type=float, help="style weight beta (default: preset)")
    p.add_argument("--tv-weight", type=float, default=0.0, help="total-variation weight")
    p.add_argument("--content-layers", nargs="+", metavar="LAYER", help="e.g. relu4_2 (default: preset)")
    p.add_argument(
        "--style-layers", nargs="+", metavar="LAYER", help="e.g. relu1_1 relu2_1 (default: preset)"
    )
    p.add_argument("--pooling", choices=get_args(Pooling), help="max or avg (default: preset)")
    p.add_argument("--blend", type=float, nargs="+", help="weight of each style image")
    p.add_argument(
        "--masks",
        type=Path,
        nargs="+",
        help="spatial control: one greyscale mask per style image; style i is applied where mask i is white",
    )
    p.add_argument(
        "--color",
        choices=get_args(ColorMode),
        default="style",
        help="colour handling: style colours, luminance-only transfer, or style recoloured to the "
        "content's colour statistics (Gatys et al., 2016)",
    )
    p.add_argument("--color-match", choices=get_args(ColorMatchMethod), default="eigen")
    p.add_argument("--style-scale", type=float, default=1.0, help="style image size relative to the content")
    p.add_argument("--optimizer", choices=get_args(OptimizerName), default="lbfgs")
    p.add_argument(
        "--line-search",
        action="store_true",
        help="strong-Wolfe line search for L-BFGS (monotone, ~2x evaluations)",
    )
    p.add_argument("--lr", type=float, default=0.02, help="learning rate (adam only)")
    p.add_argument("--init", choices=get_args(InitName), default="content")
    p.add_argument("--save-every", type=int, default=0, help="also write intermediate frames every N steps")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality of the output")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the command line; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    n_styles = len(args.styles)
    if args.blend is not None and len(args.blend) != n_styles:
        parser.error("--blend needs exactly one weight per style image")
    if args.masks is not None and len(args.masks) != n_styles:
        parser.error("--masks needs exactly one mask per style image")

    device = pick_device(args.device)
    sizes: list[int] = args.size or [512 if device.type != "cpu" else 256]
    if sorted(sizes) != sizes:
        parser.error("--size values must be ascending (coarse to fine)")
    if len(args.steps) not in (1, len(sizes)):
        parser.error("--steps needs one value, or one per --size")
    try:
        config = TransferConfig(
            steps=args.steps[0],
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            tv_weight=args.tv_weight,
            content_layers=tuple(args.content_layers) if args.content_layers else None,
            style_layers=tuple(args.style_layers) if args.style_layers else None,
            pooling=args.pooling,
            optimizer=args.optimizer,
            line_search=args.line_search,
            lr=args.lr,
            init=args.init,
            style_blend=tuple(args.blend) if args.blend else None,
            color=args.color,
            color_match=args.color_match,
            style_scale=args.style_scale,
            seed=args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))

    try:
        vgg = load_vgg19(args.weights).to(device)
    except (WeightsUnavailableError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    content = load_image(args.content, sizes[-1], device)
    styles = [load_image(path, None, device) for path in args.styles]
    masks = [load_mask(path, None, device) for path in args.masks] if args.masks else None
    total_steps = sum(args.steps) if len(args.steps) > 1 else args.steps[0] * len(sizes)
    done = 0

    def report(record: LossRecord, image: torch.Tensor) -> None:
        nonlocal done
        done += 1
        if record.step == 1 or record.step % 50 == 0 or done == total_steps:
            print(
                f"scale {record.scale}  step {record.step:4d}  total {record.total:11.4g}  "
                f"content {record.content:11.4g}  style {record.style:11.4g}  ({record.elapsed:6.1f}s)"
            )
        if args.save_every and done % args.save_every == 0:
            frame = args.output.with_name(f"{args.output.stem}_{done:04d}{args.output.suffix}")
            save_image(image, frame, args.quality)

    print(
        f"device={device} weights={args.weights} sizes={sizes} steps={args.steps} "
        f"styles={n_styles} color={args.color}"
    )
    started = time.perf_counter()
    try:
        result: TransferResult
        if len(sizes) == 1:
            result = stylize(vgg, content, styles, config, masks=masks, on_progress=report)
        else:
            result = stylize_multiscale(
                vgg, content, styles, sizes, args.steps, config, masks=masks, on_progress=report
            )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    save_image(result.image, args.output, args.quality)
    print(f"saved {args.output} ({tuple(result.image.shape[-2:])}) in {time.perf_counter() - started:.1f}s")
    return 0
