"""Command-line interface: ``python -m neural_style content.jpg style.jpg -o out.jpg``."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .model import load_vgg19_features
from .transfer import TransferConfig, load_image, pick_device, preserve_colors, save_image, stylize


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neural_style",
        description="Neural style transfer with VGG-19 (Gatys et al.).",
    )
    p.add_argument("content", type=Path, help="content image")
    p.add_argument("styles", type=Path, nargs="+", help="one or more style images")
    p.add_argument("-o", "--output", type=Path, default=Path("output.jpg"))
    p.add_argument(
        "--size", type=int, default=None, help="shorter edge in px (default: 512 on GPU, 256 on CPU)"
    )
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--style-weight", type=float, default=1e6)
    p.add_argument("--content-weight", type=float, default=1.0)
    p.add_argument("--tv-weight", type=float, default=0.0, help="total-variation smoothing weight")
    p.add_argument("--blend", type=float, nargs="+", help="relative weight of each style image")
    p.add_argument("--optimizer", choices=("lbfgs", "adam"), default="lbfgs")
    p.add_argument("--lr", type=float, default=0.02, help="learning rate (adam only)")
    p.add_argument("--init", choices=("content", "noise"), default="content")
    p.add_argument("--preserve-colors", action="store_true", help="keep the content image's colours")
    p.add_argument("--save-every", type=int, default=0, help="also write intermediate frames every N steps")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.blend is not None and len(args.blend) != len(args.styles):
        raise SystemExit("--blend needs exactly one weight per style image")

    device = pick_device(args.device)
    size = args.size or (512 if device.type != "cpu" else 256)
    content = load_image(args.content, size, device)
    # Gram matrices are averages over positions, so style images need not match the content's
    # shape; resizing them to it would stretch the brush strokes. Match the shorter edge instead.
    styles = [load_image(s, min(content.shape[-2:]), device) for s in args.styles]
    cnn = load_vgg19_features(pretrained=True).to(device)

    config = TransferConfig(
        steps=args.steps,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        tv_weight=args.tv_weight,
        optimizer=args.optimizer,
        lr=args.lr,
        init=args.init,
        style_blend=args.blend,
        seed=args.seed,
    )

    started = time.perf_counter()

    def report(step: int, losses: dict[str, float], image: torch.Tensor) -> None:
        if step == 1 or step % 50 == 0 or step == config.steps:
            print(
                f"step {step:4d}/{config.steps}  style {losses['style']:10.4f}  "
                f"content {losses['content']:8.4f}  total {losses['total']:10.4f}  "
                f"({time.perf_counter() - started:5.1f}s)"
            )
        if args.save_every and step % args.save_every == 0:
            frame = args.output.with_name(f"{args.output.stem}_{step:04d}{args.output.suffix}")
            save_image(image, frame)

    print(f"device={device} size={tuple(content.shape[-2:])} styles={len(styles)}")
    result = stylize(cnn, content, styles, config, on_progress=report)
    if args.preserve_colors:
        result = preserve_colors(result, content)
    save_image(result, args.output)
    print(f"saved {args.output}")
    return 0
