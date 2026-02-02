"""The objective and the optimisation loop, single-scale and coarse-to-fine."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from neural_style import (
    PRESETS,
    VGG19,
    FeatureExtractor,
    LossRecord,
    Objective,
    TransferConfig,
    gram_matrix,
    guided_gram_matrix,
    stylize,
    stylize_multiscale,
)
from neural_style.color import rgb_to_yiq, yiq_to_rgb
from neural_style.image import resize


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


def test_lbfgs_line_search_never_increases_the_loss(vgg: VGG19) -> None:
    content, style = _images(20, (32, 32), (32, 32))
    config = TransferConfig(steps=6, line_search=True, init="noise")
    history = stylize(vgg, content, [style], config).history
    totals = [r.total for r in history]
    assert all(b <= a * (1 + 1e-6) for a, b in zip(totals, totals[1:], strict=False)), totals
    assert sum(r.evaluations for r in history) > len(history)  # the search used extra evaluations
    plain = stylize(vgg, content, [style], TransferConfig(steps=6, init="noise")).history
    assert all(r.evaluations == 1 for r in plain)


def test_line_search_is_only_for_lbfgs() -> None:
    with pytest.raises(ValueError, match="L-BFGS"):
        TransferConfig(optimizer="adam", line_search=True)


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
        {"style_scale": 0.0},
        {"style_blend": (1.0, -1.0)},
        {"style_blend": (0.0, 0.0)},
        {"optimizer": "sgd"},
        {"color": "sepia"},
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
    assert objective.style_shapes == [(20, 80)]  # same pixel count as the content, same 1:4 aspect
    scaled = Objective(vgg, content, [style], TransferConfig(style_scale=2.0))
    assert scaled.style_shapes == [(40, 160)]


def test_objective_validates_its_inputs(vgg: VGG19) -> None:
    content, style = _images(10, (16, 16), (16, 16))
    with pytest.raises(ValueError, match=r"shape \(1, 3, 16, 16\)"):
        Objective(vgg, content, [style])(torch.rand(1, 3, 16, 17))
    with pytest.raises(ValueError, match=r"shape \(1, 1, 16, 16\)"):
        Objective(vgg, content, [style], TransferConfig(color="luminance"))(content)
    with pytest.raises(ValueError, match="one guidance channel per style"):
        Objective(vgg, content, [style, style], masks=[torch.ones(1, 1, 4, 4)])
    with pytest.raises(ValueError, match="one weight per style"):
        Objective(vgg, content, [style], TransferConfig(style_blend=(1.0, 2.0)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        Objective(vgg, content, [style], masks=[torch.full((1, 1, 4, 4), 2.0)])
    with pytest.raises(ValueError, match="need masks"):
        Objective(vgg, content, [style], style_masks=[None])
    with pytest.raises(ValueError, match="at least one style"):
        Objective(vgg, content, [])


def test_luminance_transfer_ignores_the_style_colours(vgg: VGG19) -> None:
    # Regression: v0.1's "preserve colours" stylised in RGB and swapped the chroma afterwards,
    # so the style's colours still shaped the result. Two styles that differ only in chroma
    # must give the same luminance-only result.
    g = torch.Generator().manual_seed(4)
    content = torch.rand((1, 3, 32, 32), generator=g)
    y = 0.3 + 0.4 * torch.rand((1, 1, 32, 32), generator=g)
    iq = 0.1 * torch.rand((1, 2, 32, 32), generator=g) - 0.05
    style, recoloured = yiq_to_rgb(torch.cat([y, iq], 1)), yiq_to_rgb(torch.cat([y, -iq], 1))
    assert not torch.allclose(style, recoloured, atol=1e-2)
    config = TransferConfig(steps=5, color="luminance")
    a = stylize(vgg, content, [style], config).image
    b = stylize(vgg, content, [recoloured], config).image
    torch.testing.assert_close(a, b, atol=1e-4, rtol=0)
    plain = TransferConfig(steps=5)
    a_rgb = stylize(vgg, content, [style], plain).image
    b_rgb = stylize(vgg, content, [recoloured], plain).image
    assert (a_rgb - b_rgb).abs().max() > 1e-2


def test_luminance_transfer_restores_the_content_chrominance(vgg: VGG19) -> None:
    content, style = _images(5, (32, 32), (32, 32))
    content = content * 0.5 + 0.25  # stay away from the gamut boundary
    out = stylize(vgg, content, [style], TransferConfig(steps=5, color="luminance")).image
    inside = ((out > 0) & (out < 1)).all(dim=1, keepdim=True).expand(-1, 2, -1, -1)
    torch.testing.assert_close(
        rgb_to_yiq(out)[:, 1:][inside], rgb_to_yiq(content)[:, 1:][inside], atol=1e-5, rtol=0
    )


def test_match_mode_recolours_the_style_to_the_content(vgg: VGG19) -> None:
    content, style = _images(6, (32, 32), (32, 32))
    content = content * 0.3 + 0.2  # dark and low-contrast
    objective = Objective(vgg, content, [style], TransferConfig(color="match"))
    prepared = objective.prepared_styles[0]
    torch.testing.assert_close(prepared.mean(dim=(2, 3)), content.mean(dim=(2, 3)), atol=2e-3, rtol=0)


def test_a_full_mask_is_the_same_as_no_mask(vgg: VGG19) -> None:
    content, style, image = _images(7, (32, 40), (24, 24), (32, 40))
    config = TransferConfig()
    plain, _ = Objective(vgg, content, [style], config)(image)
    masked, _ = Objective(vgg, content, [style], config, masks=[torch.ones(1, 1, 8, 10)])(image)
    torch.testing.assert_close(masked, plain)


def test_style_gradient_stays_inside_the_mask(vgg: VGG19) -> None:
    # With only relu1_1 (3x3 receptive field) the style loss of a region can only move pixels
    # within one pixel of that region.
    content, style, image = _images(8, (24, 24), (24, 24), (24, 24))
    mask = torch.zeros(1, 1, 24, 24)
    mask[..., :, :8] = 1
    config = TransferConfig(
        style_layers=("relu1_1",), content_layers=(), content_weight=0.0, style_weight=1.0
    )
    objective = Objective(vgg, content, [style], config, masks=[mask])
    image.requires_grad_(True)
    total, _ = objective(image)
    torch.autograd.backward(total)
    assert image.grad is not None
    assert image.grad[..., :, :8].abs().sum() > 0
    assert torch.all(image.grad[..., :, 9:] == 0)


def test_region_style_terms_are_weighted_by_their_share_of_the_image(vgg: VGG19) -> None:
    # Weighting a region by its area keeps the per-pixel balance between content and style
    # independent of how large the region is (docs/method.md, "Spatial control").
    content, style, image = _images(9, (32, 32), (32, 32), (32, 32))
    left = torch.zeros(1, 1, 32, 32)
    left[..., :8] = 1  # a quarter of the image, aligned with the pooling grid
    config = TransferConfig(
        style_layers=("relu1_1",), content_layers=(), content_weight=0.0, style_weight=1.0
    )
    _, parts = Objective(vgg, content, [style], config, masks=[left])(image)
    extractor = FeatureExtractor(vgg, ["relu1_1"])
    target = gram_matrix(extractor(style)["relu1_1"])
    guided = guided_gram_matrix(extractor(image)["relu1_1"], left)
    torch.testing.assert_close(parts["style"], 0.25 * torch.nn.functional.mse_loss(guided, target))


def test_style_masks_restrict_the_style_statistics(vgg: VGG19) -> None:
    content = _images(11, (16, 32))[0]
    half_red = torch.zeros(1, 3, 16, 32)
    half_red[:, 0, :, :16] = 1  # left half red, right half black
    left = torch.zeros(1, 1, 16, 32)
    left[..., :16] = 1
    red = torch.zeros(1, 3, 16, 32)
    red[:, 0] = 1
    config = TransferConfig(content_weight=0.0, style_layers=("relu1_1", "relu2_1"))
    full = torch.ones(1, 1, 16, 32)

    def style_loss(**kwargs: list[torch.Tensor]) -> float:
        _, parts = Objective(vgg, content, [half_red], config, masks=[full], **kwargs)(red)
        return float(parts["style"])

    # A red image matches the red half of the style far better than the whole style image.
    assert style_loss(style_masks=[left]) < 0.1 * style_loss()
    assert style_loss(style_masks=[torch.ones(1, 1, 16, 32)]) == pytest.approx(style_loss(), rel=1e-5)


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


def test_luminance_progress_frames_are_in_colour(vgg: VGG19) -> None:
    content, style = _images(15, (16, 16), (16, 16))
    frames: list[torch.Tensor] = []
    stylize(
        vgg,
        content,
        [style],
        TransferConfig(steps=2, color="luminance"),
        on_progress=lambda r, i: frames.append(i),
    )
    assert frames[0].shape[1] == 3
    assert not torch.allclose(frames[0][:, 0], frames[0][:, 1])  # not a grey image


def test_coarse_to_fine(vgg: VGG19) -> None:
    content, style = _images(16, (40, 60), (30, 30))
    result = stylize_multiscale(vgg, content, [style], sizes=[20, 40], steps=[4, 2], config=TransferConfig())
    assert result.image.shape == (1, 3, 40, 60)
    assert [(r.scale, r.step) for r in result.history] == [(0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 2)]
    elapsed = [r.elapsed for r in result.history]
    assert elapsed == sorted(elapsed)
    assert result.config.steps == 6
    broadcast = stylize_multiscale(
        vgg, content, [style], sizes=[20, 40], steps=[2], config=TransferConfig(color="luminance")
    )
    assert [r.scale for r in broadcast.history] == [0, 0, 1, 1]


def test_coarse_to_fine_starts_each_scale_from_the_previous_result(vgg: VGG19) -> None:
    content, style = _images(17, (32, 32), (32, 32))
    config = TransferConfig(init="noise")
    multi = stylize_multiscale(vgg, content, [style], sizes=[16, 32], steps=[5, 2], config=config)
    coarse = stylize(vgg, resize(content, 16), [style], dataclasses.replace(config, steps=5))
    fine = stylize(vgg, content, [style], dataclasses.replace(config, steps=2), init_image=coarse.image)
    torch.testing.assert_close(multi.image, fine.image)


def test_init_image_overrides_the_configured_init(vgg: VGG19) -> None:
    content, style, start = _images(19, (24, 24), (24, 24), (12, 12))
    config = TransferConfig(steps=1, init="noise")
    result = stylize(vgg, content, [style], config, init_image=start)
    expected, _ = Objective(vgg, content, [style], config)(resize(start, (24, 24)))
    assert result.history[0].total == pytest.approx(float(expected), rel=1e-6)


def test_coarse_to_fine_validates_its_schedule(vgg: VGG19) -> None:
    content, style = _images(18, (16, 16), (16, 16))
    with pytest.raises(ValueError, match="one steps value"):
        stylize_multiscale(vgg, content, [style], sizes=[8, 16], steps=[1, 2, 3])
    with pytest.raises(ValueError, match="at least one size"):
        stylize_multiscale(vgg, content, [style], sizes=[], steps=[1])


def test_loss_record_is_a_frozen_value() -> None:
    record = LossRecord(step=1, total=1.0, content=0.5, style=0.5, tv=0.0, elapsed=0.1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.step = 2  # type: ignore[misc]
