"""Generate the results gallery in ``docs/gallery/`` from real runs with the Caffe VGG-19.

    python scripts/fetch_examples.py
    python scripts/gallery.py styles hero spatial color scale layers   # images
    python scripts/gallery.py speed convergence multiscale --threads 2   # timing-sensitive runs

Each experiment writes web-sized JPEG composites (at most 300 KB each) and a
JSON record of its settings, losses and timings to ``docs/gallery/``. Raw
results are cached as PNG under ``examples/outputs/`` (git-ignored), so a
composite can be rebuilt without recomputing; ``--force`` recomputes.
Timings are wall-clock seconds on whatever machine runs the script; the JSON
records the thread count, CPU model and load average next to them.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from neural_style import (
    VGG19,
    Objective,
    TransferConfig,
    TransferResult,
    load_image,
    load_vgg19,
    stylize,
    stylize_multiscale,
)
from neural_style.image import to_pil_image

ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / "examples" / "inputs"
CACHE = ROOT / "examples" / "outputs"
GALLERY = ROOT / "docs" / "gallery"
MAX_BYTES = 300_000
GAP = 6
BACKGROUND = (252, 252, 251)

CONTENTS = ("chelsea.png", "coffee.png", "rocket.jpg")
STYLES = ("starry_night.jpg", "the_scream.jpg", "shipwreck.jpg", "woman-with-hat-matisse.jpg")


# --------------------------------------------------------------------------- helpers


def machine() -> dict[str, Any]:
    """Describe the machine a timing was taken on."""
    model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    return {
        "cpu": model,
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "loadavg": os.getloadavg(),
    }


def save_web_jpeg(image: Image.Image, path: Path, max_bytes: int = MAX_BYTES) -> int:
    """Save ``image`` as the best-quality JPEG that fits in ``max_bytes``; return the quality."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for quality in range(92, 49, -2):
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=quality, optimize=True, progressive=True)
        if buffer.tell() <= max_bytes:
            path.write_bytes(buffer.getvalue())
            return quality
    raise ValueError(f"{path.name}: cannot fit in {max_bytes} bytes; make the composite smaller")


def fit(img: Image.Image, width: int, height: int) -> Image.Image:
    """Letterbox ``img`` into a ``width`` x ``height`` cell (aspect preserved)."""
    scale = min(width / img.width, height / img.height)
    resized = img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.Resampling.LANCZOS
    )
    cell = Image.new("RGB", (width, height), BACKGROUND)
    cell.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return cell


def grid(rows: Sequence[Sequence[Image.Image | None]], width: int, height: int) -> Image.Image:
    """Arrange images in a grid of equal cells separated by a thin gap (``None`` = empty cell)."""
    n_cols = max(len(r) for r in rows)
    canvas = Image.new(
        "RGB", (n_cols * width + (n_cols - 1) * GAP, len(rows) * height + (len(rows) - 1) * GAP), BACKGROUND
    )
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            if img is not None:
                canvas.paste(fit(img, width, height), (c * (width + GAP), r * (height + GAP)))
    return canvas


def strip(images: Sequence[Image.Image], height: int) -> Image.Image:
    """Put images side by side at a common height."""
    scaled = [
        i.resize((round(i.width * height / i.height), height), Image.Resampling.LANCZOS) for i in images
    ]
    canvas = Image.new("RGB", (sum(i.width for i in scaled) + GAP * (len(scaled) - 1), height), BACKGROUND)
    x = 0
    for img in scaled:
        canvas.paste(img, (x, 0))
        x += img.width + GAP
    return canvas


def regions_preview(content: Image.Image, regions: Sequence[tuple[Image.Image, str]]) -> Image.Image:
    """Tint each (mask, colour) region of ``content``; pixels outside every mask stay unchanged."""
    base = np.asarray(content.convert("RGB"), dtype=np.float32) / 255
    shown = base.copy()
    for mask, colour in regions:
        m = np.asarray(mask.convert("L").resize(content.size), dtype=np.float32)[..., None] / 255
        tint = np.array([int(colour[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float32) / 255
        shown = shown * (1 - m) + m * (0.5 * base + 0.5 * tint)
    return Image.fromarray((shown * 255).round().astype(np.uint8))


def horizon_mask(size: tuple[int, int], horizon: float, ramp: float) -> Image.Image:
    """Sky mask: 1 above ``horizon`` (fraction of the height), a linear ramp of ``ramp`` px, 0 below."""
    width, height = size
    rows = np.arange(height, dtype=np.float32)
    column = np.clip((horizon * height - rows) / ramp + 0.5, 0, 1)
    return Image.fromarray((np.repeat(column[:, None], width, 1) * 255).round().astype(np.uint8))


def porcelain_mask(img: Image.Image) -> Image.Image:
    """Segment the red-and-white cup and saucer of ``coffee.png`` by colour, holes filled."""
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((hue < 0.03) | (hue > 0.95)) & (sat > 0.55) & (val > 0.25)
    white = (sat < 0.25) & (val > 0.75)
    mask = Image.fromarray(((red | white) * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5))
    mask = mask.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(9))  # close the rim
    ImageDraw.floodfill(mask, (0, 0), 128)  # everything reachable from the border is table
    solid = np.where(np.asarray(mask) == 128, 0, 255).astype(np.uint8)
    return Image.fromarray(solid).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.5))


def tensor_mask(img: Image.Image) -> torch.Tensor:
    """A PIL greyscale mask as a ``(1, 1, H, W)`` tensor in ``[0, 1]``."""
    array = np.asarray(img.convert("L"), dtype=np.float32) / 255
    return torch.from_numpy(array)[None, None]


def history_summary(result: TransferResult) -> dict[str, Any]:
    """First and last loss terms of a run plus its duration."""
    first, last = result.history[0], result.history[-1]
    return {
        "seconds": round(result.seconds, 2),
        "steps": len(result.history),
        "first": dataclasses.asdict(first),
        "last": dataclasses.asdict(last),
    }


def config_record(config: TransferConfig) -> dict[str, Any]:
    """The resolved configuration as JSON-friendly values."""
    return {k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(config).items()}


class Runner:
    """Runs (or loads cached) style-transfer results for the experiments."""

    def __init__(self, force: bool) -> None:
        self.vgg: VGG19 = load_vgg19("caffe")
        self.force = force
        self.records: dict[str, Any] = {}

    def image(self, name: str, size: int | None = None) -> torch.Tensor:
        """Load an example input."""
        return load_image(INPUTS / name, size)

    def run(
        self,
        key: str,
        content: str,
        styles: Sequence[str],
        size: int,
        config: TransferConfig,
        *,
        masks: Sequence[torch.Tensor] | None = None,
        sizes: Sequence[int] | None = None,
        steps: Sequence[int] | None = None,
    ) -> Image.Image:
        """Return the result for ``key``, computing it unless a cached copy exists."""
        png, meta = CACHE / f"{key}.png", CACHE / f"{key}.json"
        expected = config.resolved(self.vgg.source)
        if steps is not None:
            expected = dataclasses.replace(expected, steps=sum(steps))  # as stylize_multiscale reports
        if png.exists() and meta.exists() and not self.force:
            cached = json.loads(meta.read_text())
            # A cached result is reused only if it was made with exactly these settings.
            if cached["config"] == config_record(expected) and cached["sizes"] == (
                list(sizes) if sizes else [size]
            ):
                self.records[key] = cached
                return Image.open(png).convert("RGB")
        style_images = [self.image(s) for s in styles]
        print(f"[{time.strftime('%H:%M:%S')}] {key}: {content} x {', '.join(styles)}", flush=True)
        load_start = os.getloadavg()
        if sizes is None:
            result = stylize(self.vgg, self.image(content, size), style_images, config, masks=masks)
        else:
            assert steps is not None
            result = stylize_multiscale(
                self.vgg, self.image(content, size), style_images, sizes, steps, config, masks=masks
            )
        record = {
            "content": content,
            "styles": list(styles),
            "shape": list(result.image.shape[-2:]),
            "sizes": list(sizes) if sizes else [size],
            "config": config_record(result.config),
            **history_summary(result),
            "loadavg_start": load_start,
            "machine": machine(),
        }
        first, last = record["first"]["total"], record["last"]["total"]
        print(f"    {record['seconds']:.0f}s, loss {first:.4g} -> {last:.4g}")
        CACHE.mkdir(parents=True, exist_ok=True)
        to_pil_image(result.image).save(png)
        meta.write_text(json.dumps(record, indent=2))
        self.records[key] = record
        return to_pil_image(result.image)

    def write_record(self, name: str, extra: dict[str, Any] | None = None) -> None:
        """Write the runs used by one experiment to ``docs/gallery/<name>.json``."""
        payload = {"runs": self.records, **(extra or {})}
        (GALLERY / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
        self.records = {}


def picture(name: str) -> Image.Image:
    """Open an example input as a PIL image."""
    return Image.open(INPUTS / name).convert("RGB")


# --------------------------------------------------------------------------- experiments

STEPS = 200


def styles(runner: Runner) -> None:
    """Every content photo in every style: the main gallery grid."""
    config = TransferConfig(steps=STEPS)
    rows: list[list[Image.Image | None]] = [[None, *[picture(s) for s in STYLES]]]
    for content in CONTENTS:
        row: list[Image.Image | None] = [picture(content)]
        for style in STYLES:
            key = f"styles_{Path(content).stem}_{Path(style).stem}"
            row.append(runner.run(key, content, [style], 256, config))
        rows.append(row)
    quality = save_web_jpeg(grid(rows, 300, 200), GALLERY / "styles.jpg")
    runner.write_record("styles", {"jpeg_quality": quality})


def hero(runner: Runner) -> None:
    """The README image: coarse-to-fine at the photo's native 427x640."""
    config = TransferConfig()
    result = runner.run(
        "hero_rocket_starry",
        "rocket.jpg",
        ["starry_night.jpg"],
        427,
        config,
        sizes=[214, 427],
        steps=[300, 100],
    )
    inputs = Image.new("RGB", (320, 427), BACKGROUND)
    inputs.paste(fit(picture("rocket.jpg"), 320, 210), (0, 0))
    inputs.paste(fit(picture("starry_night.jpg"), 320, 210), (0, 217))
    quality = save_web_jpeg(strip([inputs, result], 427), GALLERY / "hero.jpg")
    runner.write_record("hero", {"jpeg_quality": quality})


def spatial(runner: Runner) -> None:
    """Spatial control: two styles split at the horizon, and a region left photographic."""
    config = TransferConfig(steps=STEPS)
    rocket = picture("rocket.jpg")
    sky = horizon_mask(rocket.size, horizon=0.755, ramp=12)
    ground = Image.eval(sky, lambda v: 255 - v)
    split = runner.run(
        "spatial_rocket",
        "rocket.jpg",
        ["starry_night.jpg", "the_scream.jpg"],
        300,
        config,
        masks=[tensor_mask(sky), tensor_mask(ground)],
    )
    preview = regions_preview(rocket, [(sky, SERIES["light"][0]), (ground, SERIES["light"][1])])
    top = strip([rocket, preview, picture("starry_night.jpg"), picture("the_scream.jpg"), split], 200)

    coffee = picture("coffee.png")
    table = Image.eval(porcelain_mask(coffee), lambda v: 255 - v)
    kept = runner.run(
        "spatial_coffee", "coffee.png", ["starry_night.jpg"], 256, config, masks=[tensor_mask(table)]
    )
    preview = regions_preview(coffee, [(table, SERIES["light"][0])])
    bottom = strip([coffee, preview, picture("starry_night.jpg"), kept], 200)

    canvas = Image.new("RGB", (max(top.width, bottom.width), 2 * 200 + GAP), BACKGROUND)
    canvas.paste(top, (0, 0))
    canvas.paste(bottom, (0, 200 + GAP))
    quality = save_web_jpeg(canvas, GALLERY / "spatial.jpg")
    runner.write_record("spatial", {"jpeg_quality": quality, "rocket_horizon": 0.755, "rocket_ramp_px": 12})


def color(runner: Runner) -> None:
    """Colour control: style colours, luminance-only transfer, colour histogram matching."""
    content, style = "coffee.png", "starry_night.jpg"
    # The plain run is the same one as in the styles grid, so it shares the cache key.
    plain = runner.run("styles_coffee_starry_night", content, [style], 256, TransferConfig(steps=STEPS))
    luminance = runner.run(
        "color_luminance", content, [style], 256, TransferConfig(steps=STEPS, color="luminance")
    )
    eigen = runner.run("color_match_eigen", content, [style], 256, TransferConfig(steps=STEPS, color="match"))
    cholesky = runner.run(
        "color_match_cholesky",
        content,
        [style],
        256,
        TransferConfig(steps=STEPS, color="match", color_match="cholesky"),
    )
    recoloured = [
        to_pil_image(
            Objective(
                runner.vgg,
                runner.image(content, 256),
                [runner.image(style)],
                TransferConfig(color="match", color_match=method),
            ).prepared_styles[0]
        )
        for method in ("eigen", "cholesky")
    ]
    rows: list[list[Image.Image | None]] = [
        [picture(content), picture(style), *recoloured],
        [plain, luminance, eigen, cholesky],
    ]
    quality = save_web_jpeg(grid(rows, 330, 220), GALLERY / "color.jpg")
    runner.write_record("color", {"jpeg_quality": quality})


STYLE_SCALES = (0.5, 1.0, 2.0)


def scale(runner: Runner) -> None:
    """--style-scale: the painting's size relative to the photo sets the size of its strokes."""
    results = [
        runner.run(
            f"scale_{factor:g}",
            "chelsea.png",
            ["starry_night.jpg"],
            192,
            TransferConfig(steps=STEPS, style_scale=factor),
        )
        for factor in STYLE_SCALES
    ]
    quality = save_web_jpeg(strip(results, 240), GALLERY / "style-scale.jpg")
    runner.write_record("style-scale", {"jpeg_quality": quality, "style_scales": list(STYLE_SCALES)})


LAYER_SETS = {
    "relu1_1": ("relu1_1",),
    "relu2_1": ("relu1_1", "relu2_1"),
    "relu3_1": ("relu1_1", "relu2_1", "relu3_1"),
    "relu4_1": ("relu1_1", "relu2_1", "relu3_1", "relu4_1"),
    "relu5_1": ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"),
}
STYLE_WEIGHTS = (1e0, 1e1, 1e2, 1e3)


def layers(runner: Runner) -> None:
    """Style-weight x style-layer grid, modelled on the corresponding figure of Gatys et al."""
    rows: list[list[Image.Image | None]] = []
    for label, layer_set in LAYER_SETS.items():
        row: list[Image.Image | None] = []
        for weight in STYLE_WEIGHTS:
            config = TransferConfig(steps=100, style_layers=layer_set, style_weight=weight)
            key = f"layers_{label}_{weight:g}"
            row.append(runner.run(key, "chelsea.png", ["starry_night.jpg"], 160, config))
        rows.append(row)
    quality = save_web_jpeg(grid(rows, 240, 160), GALLERY / "layers.jpg")
    runner.write_record("layers", {"jpeg_quality": quality, "style_weights": list(STYLE_WEIGHTS)})


OPTIMIZERS: dict[str, TransferConfig] = {
    "L-BFGS": TransferConfig(steps=300),
    "L-BFGS + line search": TransferConfig(steps=150, line_search=True),
    "Adam, lr 0.01": TransferConfig(steps=300, optimizer="adam", lr=0.01),
    "Adam, lr 0.02": TransferConfig(steps=300, optimizer="adam", lr=0.02),
    "Adam, lr 0.05": TransferConfig(steps=300, optimizer="adam", lr=0.05),
}
"""About 300 loss evaluations each (a line-search step costs about two)."""


def convergence(runner: Runner) -> None:
    """L-BFGS versus Adam: full loss curves (always recomputed; they are the data)."""
    content = runner.image("chelsea.png", 192)
    style = [runner.image("starry_night.jpg")]
    stylize(runner.vgg, content, style, TransferConfig(steps=3))  # warm-up, discarded
    curves: dict[str, Any] = {}
    images = []
    for label, config in OPTIMIZERS.items():
        print(f"[{time.strftime('%H:%M:%S')}] convergence: {label}", flush=True)
        result = stylize(runner.vgg, content, style, config)
        spent = np.cumsum([r.evaluations for r in result.history]).tolist()
        evaluations = [1, *(n + 1 for n in spent[:-1])]  # evaluations used when each loss was measured
        curves[label] = {
            "config": config_record(result.config),
            "total": [r.total for r in result.history],
            "evaluations": evaluations,
            "elapsed": [round(r.elapsed, 3) for r in result.history],
            "seconds": round(result.seconds, 2),
        }
        images.append(to_pil_image(result.image))
        final = result.history[-1].total
        print(f"    {result.seconds:.0f}s, {spent[-1]} evaluations, final loss {final:.4g}")
    payload = {
        "content": "chelsea.png",
        "style": "starry_night.jpg",
        "size": 192,
        "machine": machine(),
        "curves": curves,
    }
    (GALLERY / "convergence.json").write_text(json.dumps(payload, indent=2) + "\n")
    save_web_jpeg(strip(images, 192), GALLERY / "convergence-results.jpg")
    plot_convergence(GALLERY / "convergence.json")


# Validated categorical slots 1-5 (light and dark steps) of the reference palette, in fixed order.
SERIES = {
    "light": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"),
    "dark": ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"),
}
CHROME = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "secondary": "#c3c2b7",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}


def plot_convergence(data_path: Path) -> None:
    """Loss against evaluations and against time (one y-scale each), in light and dark variants."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(data_path.read_text())
    curves: dict[str, Any] = data["curves"]
    # Direct end labels only where they cannot collide: at least 7% of the log-scale span apart.
    values = [v for curve in curves.values() for v in curve["total"]]
    min_gap = 0.07 * (np.log10(max(values)) - np.log10(min(values)))
    ends = sorted((curve["total"][-1], label) for label, curve in curves.items())
    labelled = {ends[0][1]}
    last = ends[0][0]
    for value, label in ends[1:]:
        if np.log10(value) - np.log10(last) > min_gap:
            labelled.add(label)
            last = value
    for mode in ("light", "dark"):
        chrome = CHROME[mode]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), dpi=110, sharey=True)
        fig.patch.set_facecolor(chrome["surface"])
        panels = (
            (axes[0], "evaluations", "loss evaluations (forward + backward passes)"),
            (axes[1], "elapsed", "wall-clock seconds"),
        )
        for ax, x_key, x_label in panels:
            ax.set_facecolor(chrome["surface"])
            for (label, curve), colour in zip(curves.items(), SERIES[mode], strict=False):
                ax.plot(
                    curve[x_key],
                    curve["total"],
                    color=colour,
                    linewidth=2,
                    solid_capstyle="round",
                    label=label,
                )
                if label in labelled:
                    ax.annotate(
                        label,
                        (curve[x_key][-1], curve["total"][-1]),
                        xytext=(5, 0),
                        textcoords="offset points",
                        va="center",
                        fontsize=8,
                        color=chrome["secondary"],
                    )
            ax.set_yscale("log")
            ax.set_xlabel(x_label, color=chrome["secondary"], fontsize=9)
            ax.grid(True, which="major", color=chrome["grid"], linewidth=0.8)
            ax.tick_params(colors=chrome["secondary"], labelsize=8, which="both")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(chrome["axis"])
            ax.set_xlim(left=0)
            ax.margins(x=0.3)
        axes[0].set_ylabel("total loss, log scale", color=chrome["secondary"], fontsize=9)
        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(labels),
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.5, 0.92),
        )
        for text in legend.get_texts():
            text.set_color(chrome["ink"])
        fig.suptitle(
            f"L-BFGS vs Adam: {data['content']} x {data['style']} at {data['size']} px, Caffe VGG-19",
            color=chrome["ink"],
            fontsize=10,
            y=0.98,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        out = GALLERY / f"convergence-{mode}.png"
        fig.savefig(out, facecolor=chrome["surface"])
        plt.close(fig)
        assert out.stat().st_size <= MAX_BYTES, out


def multiscale(
    runner: Runner,
    repeats: int,
    size: int = 384,
    coarse: int = 192,
    steps: tuple[int, int, int] = (200, 200, 50),
) -> None:
    """Single-scale versus coarse-to-fine: wall-clock time and final full-resolution loss.

    ``steps`` is (single-scale steps, coarse steps, fine steps). A third run gives
    single-scale as many steps as fit in the coarse-to-fine run's time.
    """
    vgg = runner.vgg
    content = runner.image("rocket.jpg", size)
    style = [runner.image("starry_night.jpg")]
    base = TransferConfig()
    objective = Objective(vgg, content, style, base)
    single_steps, coarse_steps, fine_steps = steps

    def measure(label: str, plan: Callable[[], TransferResult]) -> TransferResult:
        print(f"[{time.strftime('%H:%M:%S')}] multiscale: {label}", flush=True)
        results = [plan() for _ in range(repeats)]
        with torch.no_grad():
            total, _ = objective(objective.working(results[-1].image))
        seconds = [round(r.seconds, 2) for r in results]
        report[label] = {
            "seconds": seconds,
            "median_seconds": statistics.median(seconds),
            "final_loss": float(total),
            "loadavg_after": os.getloadavg(),
        }
        images.append(to_pil_image(results[-1].image))
        print(f"    {report[label]}")
        return results[-1]

    # Warm-up (discarded): the first optimisation steps in a process pay one-off allocation costs.
    stylize_multiscale(vgg, content, style, [coarse, size], [3, 3], base)
    report: dict[str, Any] = {}
    images: list[Image.Image] = []
    single_label = f"single-scale {size} px, {single_steps} steps"
    measure(single_label, lambda: stylize(vgg, content, style, dataclasses.replace(base, steps=single_steps)))
    multi_label = f"coarse-to-fine {coarse} -> {size} px, {coarse_steps} + {fine_steps} steps"
    measure(
        multi_label,
        lambda: stylize_multiscale(vgg, content, style, [coarse, size], [coarse_steps, fine_steps], base),
    )
    per_step = report[single_label]["median_seconds"] / single_steps
    budget = max(1, round(report[multi_label]["median_seconds"] / per_step))
    measure(
        f"single-scale {size} px, {budget} steps (same time as coarse-to-fine)",
        lambda: stylize(vgg, content, style, dataclasses.replace(base, steps=budget)),
    )
    payload = {
        "content": "rocket.jpg",
        "style": "starry_night.jpg",
        "repeats": repeats,
        "machine": machine(),
        "results": report,
    }
    (GALLERY / "multiscale.json").write_text(json.dumps(payload, indent=2) + "\n")
    save_web_jpeg(strip(images, 300), GALLERY / "multiscale.jpg")


SPEED_SIZES = (128, 192, 256, 384, 512)


def speed(runner: Runner) -> None:
    """Seconds per L-BFGS step (one forward and backward pass) at several resolutions."""
    style = [runner.image("starry_night.jpg")]
    stylize(runner.vgg, runner.image("rocket.jpg", 64), style, TransferConfig(steps=3))  # warm-up
    rows: dict[str, Any] = {}
    for size in SPEED_SIZES:
        content = runner.image("rocket.jpg", size)
        print(f"[{time.strftime('%H:%M:%S')}] speed: {tuple(content.shape[-2:])}", flush=True)
        load = os.getloadavg()
        history = stylize(runner.vgg, content, style, TransferConfig(steps=12)).history
        per_step = [b.elapsed - a.elapsed for a, b in zip(history[1:], history[2:], strict=False)]
        rows[str(size)] = {
            "shape": list(content.shape[-2:]),
            "median_seconds_per_step": round(statistics.median(per_step), 3),
            "seconds_to_first_step": round(history[0].elapsed, 3),
            "loadavg_start": load,
        }
        print(f"    {rows[str(size)]}")
    payload = {"content": "rocket.jpg", "style": "starry_night.jpg", "machine": machine(), "sizes": rows}
    (GALLERY / "speed.json").write_text(json.dumps(payload, indent=2) + "\n")


EXPERIMENTS: dict[str, Callable[[Runner], None]] = {
    "styles": styles,
    "hero": hero,
    "spatial": spatial,
    "color": color,
    "scale": scale,
    "layers": layers,
    "convergence": convergence,
    "speed": speed,
}


def main(argv: list[str] | None = None) -> int:
    """Run the named experiments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("experiments", nargs="+", choices=[*EXPERIMENTS, "multiscale", "plot", "all"])
    parser.add_argument(
        "--threads", type=int, default=None, help="torch CPU threads (default: torch's choice)"
    )
    parser.add_argument("--repeats", type=int, default=2, help="timing repeats for multiscale")
    parser.add_argument("--force", action="store_true", help="recompute cached results")
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    GALLERY.mkdir(parents=True, exist_ok=True)
    if args.experiments == ["plot"]:
        plot_convergence(GALLERY / "convergence.json")
        return 0
    runner = Runner(force=args.force)
    names = [*EXPERIMENTS, "multiscale"] if "all" in args.experiments else args.experiments
    for name in names:
        if name == "multiscale":
            multiscale(runner, args.repeats)
        elif name != "plot":
            EXPERIMENTS[name](runner)
    return 0


if __name__ == "__main__":
    sys.exit(main())
