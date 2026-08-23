"""The landmark vocabulary and point sets."""

from __future__ import annotations

import numpy as np
import pytest

from vitruve.core.landmarks import (
    BILATERAL_PAIRS,
    Landmark,
    MissingLandmarkError,
    PointSet,
)


def test_subject_left_is_negative_x(face):
    """The single most consequential convention in the project.

    Subject-left is the viewer's right in a mirror-free frontal photograph.
    Getting this backwards silently swaps every lateralised finding, so it is
    asserted here and again in every backend's mapping test.
    """
    for left, right in BILATERAL_PAIRS:
        if not face.has(left, right):
            continue
        assert face.get(left)[0] < 0 < face.get(right)[0], left.value


def test_bilateral_pairs_are_mirror_images(face):
    for left, right in BILATERAL_PAIRS:
        if not face.has(left, right):
            continue
        assert np.allclose(face.get(left) * [-1, 1, 1], face.get(right)), left.value


def test_missing_landmark_names_itself(face):
    ps = PointSet.from_mapping({Landmark.PUPIL_L: np.zeros(3)})
    with pytest.raises(MissingLandmarkError) as exc:
        ps.get(Landmark.GONION_R)
    assert exc.value.landmark is Landmark.GONION_R


def test_missing_reports_only_what_is_absent(face):
    absent = face.missing([Landmark.PUPIL_L, Landmark.PORION_L])
    assert absent == (Landmark.PORION_L,)


def test_rejects_an_out_of_range_index():
    with pytest.raises(ValueError, match="index out of range"):
        PointSet(index={Landmark.PUPIL_L: 9}, coords=np.zeros((2, 3)))


def test_rejects_a_dimension_that_is_not_two_or_three():
    with pytest.raises(ValueError, match="dim must be 2 or 3"):
        PointSet(index={Landmark.PUPIL_L: 0}, coords=np.zeros((2, 4)))


def test_batch_shape_is_the_leading_axes(face):
    stacked = PointSet(index=face.index, coords=np.repeat(face.coords[None], 8, axis=0))
    assert stacked.batch_shape == (8,)
    assert stacked.get(Landmark.PUPIL_L).shape == (8, 3)


def test_landmark_names_are_unique():
    values = [m.value for m in Landmark]
    assert len(values) == len(set(values))
