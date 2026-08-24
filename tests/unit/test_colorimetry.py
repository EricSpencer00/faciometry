"""Colour arithmetic, verified against closed form on synthetic images.

Every image here is built by *inverting* the colour transform: pick the L*a*b*
the region is supposed to have, run it back through CIELAB to XYZ to linear RGB
to sRGB with an implementation written independently in this file, and quantise
to 8 bits. Measuring that image has to return the L*a*b* it was built from, to
within the quantisation step. That makes the test a check on the transform
rather than a check that the code agrees with itself.

The published constants are here as anchors: sRGB white is L* 100, mid-grey 128
is L* 53.585, and pure sRGB red is (53.24, 80.09, 67.20).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from test_regions import face_points

from vitruve.derm.colorimetry import (
    D65_WHITE,
    MONK_SWATCH_LAB,
    MONK_SWATCHES_HEX,
    SE_FLOOR,
    SRGB_TO_XYZ,
    RegionColour,
    WhiteBalance,
    delta_e76,
    erythema,
    grey_card_balance,
    individual_typology_angle,
    ita_class,
    linear_to_srgb,
    monk_tone,
    paired_contrast,
    periorbital_pigmentation,
    region_colour,
    rgb_to_lab,
    sample_regions,
    srgb_to_linear,
    tone_from_regions,
)
from vitruve.derm.regions import Region, build_regions

_DELTA = 6.0 / 29.0
_XYZ_TO_SRGB = np.linalg.inv(SRGB_TO_XYZ)


# ---------------------------------------------------------------------------
# An independent inverse, so the test does not lean on the module under test
# ---------------------------------------------------------------------------


def lab_to_srgb8(lab) -> np.ndarray:
    """CIELAB to 8-bit sRGB, written out here rather than imported."""
    lab = np.asarray(lab, dtype=float)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0

    def finv(t):
        return np.where(t > _DELTA, t**3, 3.0 * _DELTA**2 * (t - 4.0 / 29.0))

    xyz = np.stack([finv(fx), finv(fy), finv(fz)], axis=-1) * D65_WHITE
    linear = np.clip(xyz @ _XYZ_TO_SRGB.T, 0.0, 1.0)
    srgb = np.where(
        linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1.0 / 2.4) - 0.055
    )
    return np.clip(np.round(srgb * 255.0), 0, 255).astype(np.uint8)


def patch(lab, height: int = 64, width: int = 64) -> np.ndarray:
    rgb = lab_to_srgb8(np.asarray(lab, dtype=float))
    return np.broadcast_to(rgb, (height, width, 3)).copy()


# ---------------------------------------------------------------------------
# The transform
# ---------------------------------------------------------------------------


def test_transfer_function_round_trips():
    c = np.linspace(0.0, 1.0, 257)
    assert linear_to_srgb(srgb_to_linear(c)) == pytest.approx(c, abs=1e-9)
    # The linear segment and the power segment must meet at the breakpoint.
    assert srgb_to_linear(np.array([0.04045])) == pytest.approx([0.04045 / 12.92])


def test_published_srgb_anchors():
    assert rgb_to_lab(np.array([[[255, 255, 255]]], dtype=np.uint8))[0, 0] == pytest.approx(
        [100.0, 0.0, 0.0], abs=1e-4
    )
    assert rgb_to_lab(np.array([[[0, 0, 0]]], dtype=np.uint8))[0, 0] == pytest.approx(
        [0.0, 0.0, 0.0], abs=1e-9
    )
    assert rgb_to_lab(np.array([[[128, 128, 128]]], dtype=np.uint8))[0, 0] == pytest.approx(
        [53.585, 0.0, 0.0], abs=1e-3
    )
    assert rgb_to_lab(np.array([[[255, 0, 0]]], dtype=np.uint8))[0, 0] == pytest.approx(
        [53.2408, 80.0925, 67.2032], abs=1e-3
    )


def test_float_and_uint8_images_agree():
    img = np.array([[[200, 150, 130]]], dtype=np.uint8)
    assert rgb_to_lab(img) == pytest.approx(rgb_to_lab(img.astype(float) / 255.0))


def test_float_images_outside_unit_range_are_refused():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        rgb_to_lab(np.array([[[2.0, 0.5, 0.5]]]))


@pytest.mark.parametrize(
    "lab", [(55.0, 12.0, 16.0), (35.0, 8.0, 10.0), (78.0, 4.0, 14.0), (22.0, 5.0, 4.0)]
)
def test_synthetic_patches_measure_back_to_the_lab_they_were_built_from(lab):
    measured = rgb_to_lab(patch(lab))[0, 0]
    assert measured == pytest.approx(np.array(lab), abs=0.5)


def test_delta_e76_is_a_euclidean_distance():
    assert delta_e76((50, 0, 0), (53, 4, 0)) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Region colour and its uncertainty
# ---------------------------------------------------------------------------


def test_region_colour_recovers_a_uniform_patch():
    lab = (58.0, 14.0, 18.0)
    img = patch(lab, 96, 96)
    colour = region_colour(rgb_to_lab(img), np.ones((96, 96), dtype=bool), region=Region.MALAR_R)
    assert colour.mean == pytest.approx(np.array(lab), abs=0.5)
    assert colour.n_pixels > 0
    assert colour.region is Region.MALAR_R


def test_uncertainty_never_falls_below_the_quantisation_floor():
    img = patch((58.0, 14.0, 18.0), 96, 96)
    colour = region_colour(rgb_to_lab(img), np.ones((96, 96), dtype=bool))
    assert colour.sd == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    # A perfectly uniform region would otherwise claim zero uncertainty over
    # nine thousand pixels, which is how a shading gradient becomes a finding.
    assert colour.se == pytest.approx((SE_FLOOR, SE_FLOOR, SE_FLOOR))


def test_cluster_standard_error_exceeds_the_pixel_one_on_correlated_data():
    """A shading ramp is spatially correlated, and the block SE has to see that."""
    h = w = 96
    rows = np.linspace(40.0, 70.0, h)
    lab = np.zeros((h, w, 3))
    lab[..., 0] = rows[:, None]
    lab[..., 1] = 10.0
    lab[..., 2] = 14.0
    img = lab_to_srgb8(lab)
    measured = rgb_to_lab(img)
    colour = region_colour(measured, np.ones((h, w), dtype=bool), block=8, trim=0.0)

    naive_se = colour.sd[0] / math.sqrt(h * w)
    assert colour.n_blocks == (h // 8) * (w // 8)
    # Blocks, not pixels: the ratio is about the block side, here eight.
    assert colour.se[0] > 5.0 * naive_se
    assert colour.se[0] == pytest.approx(colour.sd[0] / math.sqrt(colour.n_blocks), rel=0.05)


def test_small_regions_say_that_their_standard_error_is_optimistic():
    img = patch((58.0, 14.0, 18.0), 10, 10)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:6, 2:6] = True
    colour = region_colour(rgb_to_lab(img), mask)
    assert colour.n_blocks < 4
    assert any("optimistic" in n for n in colour.notes)


def test_trimming_removes_a_specular_highlight():
    img = patch((50.0, 10.0, 12.0), 64, 64)
    img[:2, :] = 255  # a blown highlight across two rows
    lab = rgb_to_lab(img)
    trimmed = region_colour(lab, np.ones((64, 64), dtype=bool), trim=0.05)
    untrimmed = region_colour(lab, np.ones((64, 64), dtype=bool), trim=0.0)
    assert trimmed.L < untrimmed.L
    assert pytest.approx(50.0, abs=0.5) == trimmed.L


def test_empty_regions_raise_rather_than_returning_a_number():
    img = patch((50.0, 10.0, 12.0), 16, 16)
    with pytest.raises(ValueError, match="no pixels"):
        region_colour(rgb_to_lab(img), np.zeros((16, 16), dtype=bool))


def test_mask_and_image_shapes_must_agree():
    img = patch((50.0, 10.0, 12.0), 16, 16)
    with pytest.raises(ValueError, match="does not match"):
        region_colour(rgb_to_lab(img), np.ones((8, 8), dtype=bool))


# ---------------------------------------------------------------------------
# White balance
# ---------------------------------------------------------------------------


def test_uncalibrated_is_the_default_and_says_so():
    wb = WhiteBalance.uncalibrated()
    assert wb.calibrated is False
    assert wb.gains == (1.0, 1.0, 1.0)
    assert any("not between images" in n for n in wb.notes)


def _cast(img: np.ndarray, gains) -> np.ndarray:
    """Apply a known illuminant cast in linear light, as a camera would."""
    lin = srgb_to_linear(img.astype(float) / 255.0) * np.asarray(gains)
    return np.clip(np.round(linear_to_srgb(lin) * 255.0), 0, 255).astype(np.uint8)


def test_a_grey_card_removes_a_known_cast():
    neutral = np.full((32, 32, 3), 160, dtype=np.uint8)
    skin = patch((58.0, 14.0, 18.0), 32, 32)
    cast = (1.25, 1.0, 0.78)
    card_cast, skin_cast = _cast(neutral, cast), _cast(skin, cast)

    # Uncalibrated, the cast moves the measured colour a long way.
    naive = rgb_to_lab(skin_cast)[0, 0]
    assert delta_e76(naive, (58.0, 14.0, 18.0)) > 5.0

    balance = grey_card_balance(card_cast)
    corrected = rgb_to_lab(skin_cast, balance=balance)[0, 0]
    assert balance.calibrated is True
    assert delta_e76(corrected, (58.0, 14.0, 18.0)) < 1.5
    # The gains are the inverse of the cast, up to the overall level.
    ratio = np.asarray(balance.gains) * np.asarray(cast)
    assert ratio / ratio[1] == pytest.approx(np.ones(3), abs=0.03)


def test_a_grey_card_neutralises_itself():
    neutral = _cast(np.full((32, 32, 3), 170, dtype=np.uint8), (1.3, 1.0, 0.7))
    balance = grey_card_balance(neutral)
    lab = rgb_to_lab(neutral, balance=balance)[0, 0]
    assert lab[1] == pytest.approx(0.0, abs=0.4)
    assert lab[2] == pytest.approx(0.0, abs=0.4)


def test_a_clipped_or_black_card_is_refused():
    with pytest.raises(ValueError, match="clipped"):
        grey_card_balance(np.full((8, 8, 3), 255, dtype=np.uint8))
    with pytest.raises(ValueError, match="black"):
        grey_card_balance(np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="at least 4"):
        grey_card_balance(np.full((1, 1, 3), 160, dtype=np.uint8))


def test_reflectance_anchors_exposure_and_records_it():
    card = np.full((32, 32, 3), 128, dtype=np.uint8)
    anchored = grey_card_balance(card, reflectance=0.18)
    unanchored = grey_card_balance(card)
    assert any("comparable between photographs" in n for n in anchored.notes)
    assert any("still depends on exposure" in n for n in unanchored.notes)
    assert anchored.gains != unanchored.gains


# ---------------------------------------------------------------------------
# Paired contrasts
# ---------------------------------------------------------------------------


def _colour(lab, region=None, se=(0.5, 0.5, 0.5), calibrated=False) -> RegionColour:
    return RegionColour(
        region=region,
        n_pixels=1000,
        n_blocks=16,
        mean=tuple(float(v) for v in lab),
        sd=(1.0, 1.0, 1.0),
        se=tuple(float(v) for v in se),
        calibrated=calibrated,
    )


def test_paired_contrast_is_target_minus_reference_with_quadrature_error():
    target = _colour((58.0, 18.0, 17.0), Region.MALAR_R, se=(0.6, 0.8, 0.7))
    reference = _colour((58.0, 12.0, 17.0), Region.LATERAL_CHEEK_R, se=(0.5, 0.6, 0.5))
    c = paired_contrast(target, reference, "a*")
    assert c.delta == pytest.approx(6.0)
    assert c.uncertainty == pytest.approx(math.hypot(0.8, 0.6))
    assert c.ci_low == pytest.approx(6.0 - 1.96 * c.uncertainty)
    assert c.ci_high == pytest.approx(6.0 + 1.96 * c.uncertainty)
    assert c.ratio == pytest.approx(6.0 / c.uncertainty)
    assert c.discriminable


def test_a_contrast_smaller_than_its_uncertainty_is_not_discriminable():
    target = _colour((58.0, 12.4, 17.0), Region.MALAR_R)
    reference = _colour((58.0, 12.0, 17.0), Region.LATERAL_CHEEK_R)
    c = paired_contrast(target, reference, "a*")
    assert c.delta == pytest.approx(0.4)
    assert not c.discriminable
    assert "withheld" in c.verdict


def test_contrast_uncertainty_respects_the_floor():
    target = _colour((58.0, 18.0, 17.0), se=(0.0, 0.0, 0.0))
    reference = _colour((58.0, 12.0, 17.0), se=(0.0, 0.0, 0.0))
    assert paired_contrast(target, reference, "a*").uncertainty == pytest.approx(SE_FLOOR)


def test_unknown_channels_are_refused():
    c = _colour((50.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="channel must be"):
        paired_contrast(c, c, "R")


def test_erythema_reads_the_a_axis_and_names_the_shading_confound():
    target = _colour((57.0, 19.0, 17.0), Region.MALAR_R)
    reference = _colour((59.0, 12.0, 17.0), Region.LATERAL_CHEEK_R)
    c = erythema(target, reference)
    assert c.channel == "a*"
    assert c.delta == pytest.approx(7.0)
    assert any("haemoglobin" in n for n in c.notes)
    assert any("illumination geometry" in n for n in c.notes)


def test_pigmentation_reports_both_channels_and_the_combined_magnitude():
    infra = _colour((50.0, 12.0, 21.0), Region.INFRAORBITAL_R)
    malar = _colour((56.0, 12.0, 17.0), Region.MALAR_R)
    p = periorbital_pigmentation(infra, malar)
    assert p.lightness.delta == pytest.approx(-6.0)
    assert p.yellowness.delta == pytest.approx(4.0)
    assert p.magnitude == pytest.approx(math.hypot(6.0, 4.0))
    # Propagated through a norm, not averaged.
    expected = math.hypot(6.0 * p.lightness.uncertainty, 4.0 * p.yellowness.uncertainty) / p.magnitude
    assert p.uncertainty == pytest.approx(expected)
    assert p.darker_and_yellower
    assert "darker and yellower" in p.interpretation


def test_pigmentation_distinguishes_shadow_from_pigment():
    infra = _colour((50.0, 12.0, 17.0), Region.INFRAORBITAL_R)
    malar = _colour((56.0, 12.0, 17.0), Region.MALAR_R)
    p = periorbital_pigmentation(infra, malar)
    assert not p.darker_and_yellower
    assert "shadowing" in p.interpretation


def test_pigmentation_below_its_uncertainty_says_nothing():
    infra = _colour((56.1, 12.0, 17.0), Region.INFRAORBITAL_R)
    malar = _colour((56.0, 12.0, 17.0), Region.MALAR_R)
    p = periorbital_pigmentation(infra, malar)
    assert not p.discriminable
    assert "above the measurement uncertainty" in p.interpretation


# ---------------------------------------------------------------------------
# Skin tone
# ---------------------------------------------------------------------------


def test_ita_matches_its_definition():
    assert individual_typology_angle(60.0, 20.0) == pytest.approx(math.degrees(math.atan(0.5)))
    assert individual_typology_angle(60.0, 20.0) == pytest.approx(26.565, abs=1e-3)
    assert individual_typology_angle(40.0, 20.0) == pytest.approx(-26.565, abs=1e-3)
    assert individual_typology_angle(50.0, 20.0) == pytest.approx(0.0)
    # b* of zero is a vertical asymptote, not a crash.
    assert individual_typology_angle(70.0, 0.0) == 90.0
    assert individual_typology_angle(30.0, 0.0) == -90.0


def test_ita_classes_are_monotone_in_the_angle():
    assert ita_class(60.0) == "very light"
    assert ita_class(45.0) == "light"
    assert ita_class(30.0) == "intermediate"
    assert ita_class(15.0) == "tan"
    assert ita_class(-10.0) == "brown"
    assert ita_class(-45.0) == "dark"


def test_monk_swatches_are_ten_and_ordered():
    assert len(MONK_SWATCHES_HEX) == 10
    itas = np.array(
        [individual_typology_angle(lab[0], lab[2]) for lab in MONK_SWATCH_LAB]
    )
    # ITA is strictly decreasing across the scale, which is the ordering that
    # actually holds. L* alone is not: swatches 2 and 3 differ mainly in hue,
    # and swatch 3 is very slightly the lighter of the two.
    assert np.all(np.diff(itas) < 0)
    assert np.all(np.diff(MONK_SWATCH_LAB[2:, 0]) < 0)


def test_the_swatches_are_far_enough_apart_to_be_distinguished():
    d = np.sqrt(((MONK_SWATCH_LAB[:, None, :] - MONK_SWATCH_LAB[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    assert d.min() > 2.5, "a nearest-swatch match needs the swatches to be separated"


def test_ita_bands_collapse_dark_skin_where_monk_does_not():
    """The compression argument, on the numbers rather than in a footnote."""
    classes = [ita_class(individual_typology_angle(lab[0], lab[2])) for lab in MONK_SWATCH_LAB]
    assert classes[7:] == ["dark", "dark", "dark"]
    assert classes[:4] == ["very light"] * 4
    # Six of the ten swatches share a class with a neighbour under the ITA
    # bands; all ten are separable under Monk.
    assert len(set(classes)) < len(MONK_SWATCHES_HEX)


@pytest.mark.parametrize("index", range(10))
def test_each_monk_swatch_matches_itself(index):
    hexval = MONK_SWATCHES_HEX[index]
    rgb = tuple(int(hexval[i : i + 2], 16) for i in (1, 3, 5))
    img = np.broadcast_to(np.array(rgb, dtype=np.uint8), (32, 32, 3)).copy()
    colour = region_colour(rgb_to_lab(img), np.ones((32, 32), dtype=bool))
    tone = monk_tone(colour)
    assert tone.monk == index + 1
    assert tone.delta_e == pytest.approx(0.0, abs=0.5)


def test_uncalibrated_tone_carries_the_absolute_caveat():
    colour = _colour((58.0, 14.0, 18.0), Region.MALAR_R, calibrated=False)
    tone = monk_tone(colour)
    assert any("absolute colour value" in n for n in tone.notes)
    assert any("never inferred as a demographic" in n for n in tone.notes)
    assert tone.calibrated is False


def test_a_tone_far_from_every_swatch_says_so():
    colour = _colour((70.0, -30.0, -40.0))  # nothing like skin
    tone = monk_tone(colour)
    assert tone.delta_e > 12.0
    assert any("white-balance artifact" in n for n in tone.notes)


def test_tone_prefers_the_cheek_and_returns_none_when_absent():
    colours = {
        Region.FOREHEAD: _colour((70.0, 10.0, 15.0), Region.FOREHEAD),
        Region.MALAR_R: _colour((58.0, 14.0, 18.0), Region.MALAR_R),
    }
    assert tone_from_regions(colours).region is Region.MALAR_R
    assert tone_from_regions({Region.NASAL: _colour((58.0, 20.0, 18.0))}) is None


# ---------------------------------------------------------------------------
# End to end over a synthetic face, with no weights and no network
# ---------------------------------------------------------------------------


def synthetic_face_image(
    height: int = 900,
    width: int = 800,
    base=(58.0, 13.0, 17.0),
    malar_a: float = 6.0,
    infraorbital=(-5.0, 4.0),
):
    """A face-shaped LAB image: uniform skin, a redder cheek, a darker eye band."""
    regions = build_regions(face_points())
    lab = np.zeros((height, width, 3), dtype=float)
    lab[..., 0], lab[..., 1], lab[..., 2] = base
    for region in (Region.MALAR_L, Region.MALAR_R):
        mask = regions.mask_for(region, height, width, erode=0.0)
        lab[mask, 1] += malar_a
    for region in (Region.INFRAORBITAL_L, Region.INFRAORBITAL_R):
        mask = regions.mask_for(region, height, width, erode=0.0)
        lab[mask, 0] += infraorbital[0]
        lab[mask, 2] += infraorbital[1]
    return lab_to_srgb8(lab), regions


def test_sample_regions_reads_every_available_region():
    img, regions = synthetic_face_image()
    colours = sample_regions(img, regions)
    assert set(colours) == regions.available
    for colour in colours.values():
        assert colour.calibrated is False
        assert any("not between images" in n for n in colour.notes)


def test_a_planted_erythema_is_recovered_and_a_flat_face_is_not():
    img, regions = synthetic_face_image(malar_a=6.0)
    colours = sample_regions(img, regions)
    c = erythema(colours[Region.MALAR_R], colours[Region.LATERAL_CHEEK_R])
    assert c.delta == pytest.approx(6.0, abs=0.6)
    assert c.discriminable

    flat_img, flat_regions = synthetic_face_image(malar_a=0.0)
    flat = sample_regions(flat_img, flat_regions)
    c0 = erythema(flat[Region.MALAR_R], flat[Region.LATERAL_CHEEK_R])
    assert c0.delta == pytest.approx(0.0, abs=0.3)
    assert not c0.discriminable


def test_a_planted_dark_circle_is_recovered_with_the_right_signature():
    img, regions = synthetic_face_image(infraorbital=(-5.0, 4.0))
    colours = sample_regions(img, regions)
    p = periorbital_pigmentation(colours[Region.INFRAORBITAL_L], colours[Region.MALAR_L])
    assert p.lightness.delta == pytest.approx(-5.0, abs=0.6)
    assert p.yellowness.delta == pytest.approx(4.0, abs=0.6)
    assert p.darker_and_yellower
    assert p.discriminable


def test_a_paired_contrast_survives_a_cast_that_moves_absolute_colour():
    img, regions = synthetic_face_image(malar_a=6.0)
    plain = sample_regions(img, regions)
    cast = sample_regions(_cast(img, (1.25, 1.0, 0.78)), regions)

    # The absolute colour of the cheek moves a long way under the cast.
    assert delta_e76(plain[Region.MALAR_R].mean, cast[Region.MALAR_R].mean) > 4.0
    # The paired contrast barely moves, which is the entire argument for it.
    before = erythema(plain[Region.MALAR_R], plain[Region.LATERAL_CHEEK_R]).delta
    after = erythema(cast[Region.MALAR_R], cast[Region.LATERAL_CHEEK_R]).delta
    assert after == pytest.approx(before, abs=1.0)
