"""The command line, end to end on an untrained network."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from neural_style import WeightsUnavailableError, __version__, cli


@pytest.fixture
def pictures(tmp_path: Path) -> dict[str, Path]:
    g = torch.Generator().manual_seed(0)
    paths = {}
    for name, (w, h) in {"content": (36, 24), "style": (30, 40), "style2": (20, 20)}.items():
        pixels = (torch.rand((h, w, 3), generator=g) * 255).to(torch.uint8).numpy()
        paths[name] = tmp_path / f"{name}.png"
        Image.fromarray(pixels).save(paths[name])
    mask = Image.new("L", (36, 24), 0)
    mask.paste(255, (0, 0, 18, 24))
    paths["mask"] = tmp_path / "mask.png"
    mask.save(paths["mask"])
    return paths


def test_parser_defaults() -> None:
    args = cli.build_parser().parse_args(["c.jpg", "s1.jpg", "s2.jpg", "--blend", "1", "2"])
    assert len(args.styles) == 2 and args.blend == [1.0, 2.0]
    assert args.weights == "torchvision" and args.optimizer == "lbfgs" and not args.preserve_colors
    assert args.steps == 300 and args.size is None and args.style_weight is None


def test_end_to_end_run_writes_the_image_and_frames(pictures: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "out" / "result.jpg"
    argv = [str(pictures["content"]), str(pictures["style"]), "-o", str(out), "--preserve-colors"]
    argv += ["--weights", "random", "--size", "16", "--steps", "4", "--save-every", "2", "--device", "cpu"]
    assert cli.main(argv) == 0
    with Image.open(out) as img:
        assert img.size == (24, 16)  # shorter edge 16, aspect of the 36x24 content
    frames = sorted(p.name for p in out.parent.iterdir())
    assert frames == ["result.jpg", "result_0002.jpg", "result_0004.jpg"]


def test_spatial_control_with_masks(pictures: dict[str, Path], tmp_path: Path) -> None:
    out = tmp_path / "masked.png"
    argv = [str(pictures["content"]), str(pictures["style"]), str(pictures["style2"]), "-o", str(out)]
    argv += ["--weights", "random", "--size", "16", "--steps", "2", "--device", "cpu"]
    argv += ["--masks", str(pictures["mask"]), str(pictures["mask"]), "--pooling", "avg"]
    argv += ["--style-layers", "relu1_1", "relu2_1", "--content-layers", "relu2_2"]
    assert cli.main(argv) == 0
    with Image.open(out) as img:
        assert img.size == (24, 16)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--blend", "1", "2"], "one weight per style"),
        (["--masks", "a.png", "b.png"], "one mask per style"),
        (["--style-weight", "-1"], "non-negative"),
    ],
)
def test_invalid_arguments_exit_with_usage_errors(
    pictures: dict[str, Path], capsys: pytest.CaptureFixture[str], extra: list[str], message: str
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main([str(pictures["content"]), str(pictures["style"]), "--weights", "random", *extra])
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_unknown_layer_is_reported(pictures: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    argv = [str(pictures["content"]), str(pictures["style"]), "--weights", "random", "--size", "8"]
    assert cli.main([*argv, "--style-layers", "conv_4"]) == 1
    assert "conv_4" in capsys.readouterr().err


def test_unavailable_weights_are_explained(
    pictures: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def unavailable(weights: str) -> None:
        raise WeightsUnavailableError("download.pytorch.org is unreachable")

    monkeypatch.setattr(cli, "load_vgg19", unavailable)
    assert cli.main([str(pictures["content"]), str(pictures["style"])]) == 1
    assert "unreachable" in capsys.readouterr().err


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert __version__ in capsys.readouterr().out
