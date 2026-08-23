"""Shared fixtures: a synthetic face with exactly known geometry.

Every core test measures against closed-form answers rather than against
recorded output, so a test failure means the arithmetic is wrong rather than
that somebody re-baselined a snapshot.
"""

from __future__ import annotations

import numpy as np
import pytest

from vitruve.core.landmarks import Landmark as L
from vitruve.core.landmarks import PointSet

#: A bilaterally symmetric synthetic face in canonical millimetres:
#: +x is the subject's right, +y is up, +z is toward the viewer, and the
#: origin sits at the midpoint between the inner canthi.
SYNTHETIC: dict[L, tuple[float, float, float]] = {
    L.GLABELLA: (0.0, 22.0, 8.0),
    L.NASION: (0.0, 14.0, 4.0),
    L.SELLION: (0.0, 12.0, 3.0),
    L.PUPIL_L: (-31.5, 0.0, 0.0),
    L.PUPIL_R: (31.5, 0.0, 0.0),
    L.ENDOCANTHION_L: (-16.0, 0.0, 0.0),
    L.ENDOCANTHION_R: (16.0, 0.0, 0.0),
    # Outer canthi sit 30 mm lateral and 4 mm above the inner ones, which is
    # exactly atan(4/30) = 7.595 degrees of canthal tilt on each side.
    L.EXOCANTHION_L: (-46.0, 4.0, -4.0),
    L.EXOCANTHION_R: (46.0, 4.0, -4.0),
    L.PALPEBRALE_SUP_L: (-31.5, 5.0, 0.0),
    L.PALPEBRALE_SUP_R: (31.5, 5.0, 0.0),
    L.PALPEBRALE_INF_L: (-31.5, -5.0, 0.0),
    L.PALPEBRALE_INF_R: (31.5, -5.0, 0.0),
    L.PRONASALE: (0.0, -22.0, 26.0),
    L.SUBNASALE: (0.0, -32.0, 14.0),
    L.COLUMELLA: (0.0, -27.0, 20.0),
    L.ALARE_L: (-18.5, -28.0, 8.0),
    L.ALARE_R: (18.5, -28.0, 8.0),
    L.CRISTA_PHILTRI_L: (-5.5, -44.0, 12.0),
    L.CRISTA_PHILTRI_R: (5.5, -44.0, 12.0),
    L.LABIALE_SUPERIUS: (0.0, -44.0, 13.0),
    L.STOMION: (0.0, -48.0, 11.0),
    L.LABIALE_INFERIUS: (0.0, -53.0, 12.0),
    L.CHEILION_L: (-25.0, -48.0, 4.0),
    L.CHEILION_R: (25.0, -48.0, 4.0),
    L.SUBLABIALE: (0.0, -60.0, 8.0),
    L.POGONION: (0.0, -70.0, 12.0),
    L.GNATHION: (0.0, -78.0, 8.0),
    L.MENTON: (0.0, -80.0, 4.0),
    L.GONION_L: (-58.0, -52.0, -30.0),
    L.GONION_R: (58.0, -52.0, -30.0),
    L.ZYGION_L: (-70.0, -4.0, -20.0),
    L.ZYGION_R: (70.0, -4.0, -20.0),
    L.TRAGION_L: (-72.0, 0.0, -60.0),
    L.TRAGION_R: (72.0, 0.0, -60.0),
    L.CERVICALE: (0.0, -100.0, -40.0),
    L.ORBITALE_L: (-31.5, -8.0, 2.0),
    L.ORBITALE_R: (31.5, -8.0, 2.0),
    L.SUPERCILIARE_L: (-30.0, 16.0, 2.0),
    L.SUPERCILIARE_R: (30.0, 16.0, 2.0),
    L.SUBALARE_L: (-12.0, -30.0, 12.0),
    L.SUBALARE_R: (12.0, -30.0, 12.0),
    L.TRICHION: (0.0, 62.0, -6.0),
}


@pytest.fixture
def face() -> PointSet:
    return PointSet.from_mapping({k: np.array(v, dtype=float) for k, v in SYNTHETIC.items()})


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260823)
