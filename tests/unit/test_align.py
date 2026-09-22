"""The canonical crop's interpolation must retain its pixel semantics."""

from __future__ import annotations

import numpy as np

from faciometry.pipeline.align import _bilinear_sample


def _full_frame_reference(
    image: np.ndarray, xs: np.ndarray, ys: np.ndarray
) -> np.ndarray:
    """The pre-optimisation implementation, kept as an exact-output oracle."""
    h, w = image.shape[:2]
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx = (xs - x0)[..., None]
    fy = (ys - y0)[..., None]
    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)
    full = image.astype(np.float64)
    top = full[y0c, x0c] * (1.0 - fx) + full[y0c, x1c] * fx
    bot = full[y1c, x0c] * (1.0 - fx) + full[y1c, x1c] * fx
    return np.clip(np.rint(top * (1.0 - fy) + bot * fy), 0, 255).astype(np.uint8)


def test_sampled_promotion_matches_full_frame_reference_at_image_edges():
    rng = np.random.default_rng(20260922)
    image = rng.integers(0, 256, size=(7, 9, 3), dtype=np.uint8)
    xs = np.array(
        [[-2.25, 0.0, 0.125, 7.5, 8.0, 8.75, 11.0]],
        dtype=float,
    )
    ys = np.array(
        [[-1.5, 0.0, 2.75, 5.125, 6.0, 6.5, 9.0]],
        dtype=float,
    )

    np.testing.assert_array_equal(
        _bilinear_sample(image, xs, ys), _full_frame_reference(image, xs, ys)
    )
