"""Command-line interface: ``neural-style content.jpg style.jpg -o out.jpg``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import get_args

import torch

from . import __version__
from .color import preserve_colors
from .image import load_image, save_image
from .model import WEIGHT_SOURCES, Pooling, WeightsUnavailableError, load_vgg19
from .transfer import InitName, LossRecord, OptimizerName, TransferConfig, stylize


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
        prog="neural-style", description="Neural style transfer with VGG-19 (Gatys et al.)."
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
        "--size", type=int, help="shorter edge of the output in px (default 512 on GPU, 256 on CPU)"
    )
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--content-weight", type=float, help="content weight alpha (default: preset)")
    p.add_argument("--style-weight", type=float, help="style weight beta (default: preset)")
    p.add_argument("--tv-weight", type=float, default=0.0, help="total-variation weight")
    p.add_argument("--content-layers", nargs="+", metavar="LAYER", help="e.g. relu4_2 (default: preset)")
    p.add_argument(
        "--style-layers", nargs="+", metavar="LAYER", help="e.g. relu1_1 relu2_1 (default: preset)"
    )
    p.add_argument("--pooling", choices=get_args(Pooling), help="max or avg (default: preset)")
    p.add_argument("--blend", type=float, nargs="+", help="weight of each style image")
    p.add_argument("--preserve-colors", action="store_true", help="keep the content image's colours")
    p.add_argument("--optimizer", choices=get_args(OptimizerName), default="lbfgs")
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
    if args.blend is not None and len(args.blend) != len(args.styles):
        parser.error("--blend needs exactly one weight per style image")
    try:
        config = TransferConfig(
            steps=args.steps,
            content_weight=args.content_weight,
            style_weight=args.style_weight,
            tv_weight=args.tv_weight,
            content_layers=tuple(args.content_layers) if args.content_layers else None,
            style_layers=tuple(args.style_layers) if args.style_layers else None,
            pooling=args.pooling,
            optimizer=args.optimizer,
            lr=args.lr,
            init=args.init,
            style_blend=tuple(args.blend) if args.blend else None,
            seed=args.seed,
        )
    except ValueError as exc:
        parser.error(str(exc))

    device = pick_device(args.device)
    try:
        vgg = load_vgg19(args.weights).to(device)
    except (WeightsUnavailableError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    content = load_image(args.content, args.size or (512 if device.type != "cpu" else 256), device)
    styles = [load_image(path, None, device) for path in args.styles]

    def report(record: LossRecord, image: torch.Tensor) -> None:
        if record.step == 1 or record.step % 50 == 0 or record.step == config.steps:
            print(
                f"step {record.step:4d}/{config.steps}  total {record.total:11.4g}  "
                f"content {record.content:11.4g}  style {record.style:11.4g}  ({record.elapsed:6.1f}s)"
            )
        if args.save_every and record.step % args.save_every == 0:
            frame = args.output.with_name(f"{args.output.stem}_{record.step:04d}{args.output.suffix}")
            save_image(image, frame, args.quality)

    print(f"device={device} weights={args.weights} size={tuple(content.shape[-2:])} styles={len(styles)}")
    started = time.perf_counter()
    try:
        result = stylize(vgg, content, styles, config, on_progress=report)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    image = preserve_colors(result.image, content) if args.preserve_colors else result.image
    save_image(image, args.output, args.quality)
    print(f"saved {args.output} in {time.perf_counter() - started:.1f}s")
    return 0
