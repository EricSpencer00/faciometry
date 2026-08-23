"""Anatomical landmark vocabulary and the point sets carrying them.

Measurements refer to landmarks by *anatomical name* -- ``exocanthion_l``,
``gonion_r``, ``subnasale`` -- never by a backend's integer index. A landmark
backend supplies a name-to-index map; swapping a 68-point model for a 98-point
one therefore changes one mapping table and not a single measurement
definition.

The names follow standard craniofacial anthropometry (Farkas' landmark set),
with ``_l`` / ``_r`` suffixes for bilateral points. Left and right are the
*subject's* left and right, which in a mirror-free frontal photograph is the
viewer's right and left. Getting this backwards silently swaps every
lateralised finding, so it is stated here and asserted in the backend mapping
tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

import numpy as np
from numpy.typing import NDArray


class Landmark(str, Enum):
    """Named craniofacial landmarks.

    Membership is deliberately closed. A backend that cannot supply a landmark
    omits it, and every measurement depending on it is reported as unavailable
    with that landmark named -- which is far better than a plausible number
    derived from a guessed point.
    """

    # Cranium and forehead
    TRICHION = "trichion"
    GLABELLA = "glabella"

    # Orbital region
    NASION = "nasion"
    SELLION = "sellion"
    ENDOCANTHION_L = "endocanthion_l"
    ENDOCANTHION_R = "endocanthion_r"
    EXOCANTHION_L = "exocanthion_l"
    EXOCANTHION_R = "exocanthion_r"
    PUPIL_L = "pupil_l"
    PUPIL_R = "pupil_r"
    PALPEBRALE_SUP_L = "palpebrale_superius_l"
    PALPEBRALE_SUP_R = "palpebrale_superius_r"
    PALPEBRALE_INF_L = "palpebrale_inferius_l"
    PALPEBRALE_INF_R = "palpebrale_inferius_r"
    ORBITALE_L = "orbitale_l"
    ORBITALE_R = "orbitale_r"

    # Brow
    SUPERCILIARE_L = "superciliare_l"
    SUPERCILIARE_R = "superciliare_r"

    # Nose
    PRONASALE = "pronasale"
    SUBNASALE = "subnasale"
    ALARE_L = "alare_l"
    ALARE_R = "alare_r"
    SUBALARE_L = "subalare_l"
    SUBALARE_R = "subalare_r"
    COLUMELLA = "columella"

    # Mouth
    LABIALE_SUPERIUS = "labiale_superius"
    LABIALE_INFERIUS = "labiale_inferius"
    STOMION = "stomion"
    CHEILION_L = "cheilion_l"
    CHEILION_R = "cheilion_r"
    CRISTA_PHILTRI_L = "crista_philtri_l"
    CRISTA_PHILTRI_R = "crista_philtri_r"

    # Chin and jaw
    SUBLABIALE = "sublabiale"
    POGONION = "pogonion"
    GNATHION = "gnathion"
    MENTON = "menton"
    GONION_L = "gonion_l"
    GONION_R = "gonion_r"

    # Lateral face
    ZYGION_L = "zygion_l"
    ZYGION_R = "zygion_r"
    TRAGION_L = "tragion_l"
    TRAGION_R = "tragion_r"
    PORION_L = "porion_l"
    PORION_R = "porion_r"

    # Neck (profile only)
    CERVICALE = "cervicale"


#: Landmarks that only a profile view can locate reliably.
PROFILE_ONLY: frozenset[Landmark] = frozenset(
    {Landmark.CERVICALE, Landmark.PORION_L, Landmark.PORION_R}
)

#: Bilateral pairs, subject-left first. Used by the mirror-symmetry checks and
#: by the backend mapping tests that catch a left/right transposition.
BILATERAL_PAIRS: tuple[tuple[Landmark, Landmark], ...] = (
    (Landmark.ENDOCANTHION_L, Landmark.ENDOCANTHION_R),
    (Landmark.EXOCANTHION_L, Landmark.EXOCANTHION_R),
    (Landmark.PUPIL_L, Landmark.PUPIL_R),
    (Landmark.PALPEBRALE_SUP_L, Landmark.PALPEBRALE_SUP_R),
    (Landmark.PALPEBRALE_INF_L, Landmark.PALPEBRALE_INF_R),
    (Landmark.ORBITALE_L, Landmark.ORBITALE_R),
    (Landmark.SUPERCILIARE_L, Landmark.SUPERCILIARE_R),
    (Landmark.ALARE_L, Landmark.ALARE_R),
    (Landmark.SUBALARE_L, Landmark.SUBALARE_R),
    (Landmark.CHEILION_L, Landmark.CHEILION_R),
    (Landmark.CRISTA_PHILTRI_L, Landmark.CRISTA_PHILTRI_R),
    (Landmark.GONION_L, Landmark.GONION_R),
    (Landmark.ZYGION_L, Landmark.ZYGION_R),
    (Landmark.TRAGION_L, Landmark.TRAGION_R),
    (Landmark.PORION_L, Landmark.PORION_R),
)


class MissingLandmarkError(KeyError):
    """Raised when a measurement asks for a landmark the point set lacks."""

    def __init__(self, name: Landmark) -> None:
        super().__init__(name.value)
        self.landmark = name


@dataclass(frozen=True)
class PointSet:
    """Named points, batched over an arbitrary number of leading axes.

    ``coords`` has shape ``(..., n_landmarks, dim)``. The leading axes are free:
    a single point set is ``(n, dim)``, and a Monte-Carlo ensemble is
    ``(n_samples, n, dim)``. Every consumer broadcasts, so the same measurement
    code evaluates a point estimate and an uncertainty ensemble unchanged.
    """

    index: Mapping[Landmark, int]
    coords: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.coords.ndim < 2:
            raise ValueError("coords must have shape (..., n_landmarks, dim)")
        if self.coords.shape[-1] not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {self.coords.shape[-1]}")
        n = self.coords.shape[-2]
        bad = {k: v for k, v in self.index.items() if not 0 <= v < n}
        if bad:
            raise ValueError(f"index out of range for {n} points: {bad}")

    @property
    def dim(self) -> int:
        return int(self.coords.shape[-1])

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.coords.shape[:-2])

    @property
    def available(self) -> frozenset[Landmark]:
        return frozenset(self.index)

    def has(self, *names: Landmark) -> bool:
        return all(n in self.index for n in names)

    def missing(self, names: Iterable[Landmark]) -> tuple[Landmark, ...]:
        return tuple(n for n in names if n not in self.index)

    def get(self, name: Landmark) -> NDArray[np.float64]:
        """Point ``name`` with shape ``(..., dim)``."""
        try:
            i = self.index[name]
        except KeyError as exc:
            raise MissingLandmarkError(name) from exc
        return self.coords[..., i, :]

    def subset(self, names: Iterable[Landmark]) -> NDArray[np.float64]:
        """Stack of the named points, shape ``(..., len(names), dim)``."""
        return np.stack([self.get(n) for n in names], axis=-2)

    def with_coords(self, coords: NDArray[np.float64]) -> "PointSet":
        """Same naming, different coordinates -- used by transforms and sampling."""
        return PointSet(index=self.index, coords=coords)

    @classmethod
    def from_mapping(cls, points: Mapping[Landmark, NDArray[np.float64]]) -> "PointSet":
        """Build from an explicit name-to-coordinate mapping.

        Convenient for tests and for backends that emit a dict. Insertion order
        becomes index order.
        """
        if not points:
            raise ValueError("cannot build a PointSet from no points")
        names = list(points)
        stacked = np.stack([np.asarray(points[n], dtype=float) for n in names], axis=-2)
        return cls(index={n: i for i, n in enumerate(names)}, coords=stacked)
