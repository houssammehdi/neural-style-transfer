"""The objective and the optimisation loop."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from neural_style import PRESETS, VGG19, LossRecord, Objective, TransferConfig, stylize


def _images(seed: int, *shapes: tuple[int, int]) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.rand((1, 3, h, w), generator=g) for h, w in shapes]


@pytest.mark.parametrize("optimizer", ["lbfgs", "adam"])
def test_stylize_reduces_the_loss(vgg: VGG19, optimizer: str) -> None:
    content, style = _images(1, (48, 48), (48, 48))
    config = TransferConfig(steps=15, optimizer=optimizer, lr=0.05, init="noise", tv_weight=1e-3)  # type: ignore[arg-type]
    result = stylize(vgg, content, [style], config)
    assert result.image.shape == content.shape
    assert result.image.min() >= 0 and result.image.max() <= 1
    assert len(result.history) == 15
    assert result.history[-1].total < result.history[0].total
    assert [r.step for r in result.history] == list(range(1, 16))


def test_history_belongs_to_the_run_not_the_config(vgg: VGG19) -> None:
    # Regression: v0.1 appended to a mutable list on the config, so reusing it mixed runs.
    content, style = _images(2, (24, 24), (24, 24))
    config = TransferConfig(steps=3)
    first, second = stylize(vgg, content, [style], config), stylize(vgg, content, [style], config)
    assert len(first.history) == len(second.history) == 3
    assert not hasattr(config, "history")


def test_presets_follow_the_weight_source() -> None:
    config = TransferConfig(style_weight=5.0)
    caffe, random = config.resolved("caffe"), config.resolved("random")
    assert caffe.content_layers == ("relu4_2",)
    assert caffe.style_layers == ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
    assert random.content_layers == ("conv2_2",)  # v0.1 behaviour for torchvision/random weights
    assert caffe.style_weight == random.style_weight == 5.0  # explicit values win
    assert caffe.content_weight == PRESETS["caffe"].content_weight
    assert (caffe.pooling, random.pooling) == ("avg", "max")
    assert TransferConfig(pooling="max").resolved("caffe").pooling == "max"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 0},
        {"style_weight": -1.0},
        {"lr": 0.0},
        {"style_blend": (1.0, -1.0)},
        {"style_blend": (0.0, 0.0)},
        {"optimizer": "sgd"},
        {"pooling": "min"},
    ],
)
def test_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TransferConfig(**kwargs)  # type: ignore[arg-type]


def test_style_images_keep_their_aspect_ratio(vgg: VGG19) -> None:
    # Style images are resized to the content's pixel count, never stretched to its shape.
    content, style = _images(3, (40, 40), (20, 80))
    objective = Objective(vgg, content, [style], TransferConfig())
    assert objective.style_shapes == [(20, 80)]


def test_objective_validates_its_inputs(vgg: VGG19) -> None:
    content, style = _images(10, (16, 16), (16, 16))
    with pytest.raises(ValueError, match=r"shape \(1, 3, 16, 16\)"):
        Objective(vgg, content, [style])(torch.rand(1, 3, 16, 17))
    with pytest.raises(ValueError, match="one weight per style"):
        Objective(vgg, content, [style], TransferConfig(style_blend=(1.0, 2.0)))
    with pytest.raises(ValueError, match="at least one style"):
        Objective(vgg, content, [])


def test_blending_styles_of_different_sizes(vgg: VGG19) -> None:
    content, a, b = _images(12, (24, 24), (40, 20), (12, 36))
    result = stylize(vgg, content, [a, b], TransferConfig(steps=3, style_blend=(0.7, 0.3)))
    assert result.image.shape == content.shape


def test_noise_init_is_seeded_and_leaves_the_global_rng_alone(vgg: VGG19) -> None:
    content, style = _images(13, (16, 16), (16, 16))
    config = TransferConfig(steps=1, init="noise", seed=5)
    torch.manual_seed(0)
    expected = torch.rand(2)
    torch.manual_seed(0)
    a = stylize(vgg, content, [style], config).image
    assert torch.equal(torch.rand(2), expected)
    torch.testing.assert_close(stylize(vgg, content, [style], config).image, a)


def test_progress_sees_every_step_and_an_independent_copy(vgg: VGG19) -> None:
    content, style = _images(14, (16, 16), (16, 16))
    seen: list[tuple[LossRecord, torch.Tensor]] = []
    result = stylize(
        vgg, content, [style], TransferConfig(steps=4), on_progress=lambda r, img: seen.append((r, img))
    )
    assert [r.step for r, _ in seen] == [1, 2, 3, 4]
    assert not torch.equal(seen[0][1], seen[-1][1])
    assert result.history == tuple(r for r, _ in seen)


def test_loss_record_is_a_frozen_value() -> None:
    record = LossRecord(step=1, total=1.0, content=0.5, style=0.5, tv=0.0, elapsed=0.1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.step = 2  # type: ignore[misc]
