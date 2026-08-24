"""Face Research Lab London Set (CC BY 4.0) as Faciometry point sets.

FRLL is the only permissively licensed face set with dense human-delineated
landmarks *and* self-reported demographics: 102 individuals, ten captures each,
and a 189-point Psychomorph template on the neutral front capture. The other
nine captures ship as images only, so anything measured on them comes from a
landmark model rather than from a human.

Two things this module has to get right, and both are easy to get silently
wrong:

**Left and right.** The FRL template calls a point "left" when it is on the
*viewer's* left, which is the subject's right. Faciometry's canonical frame puts
the subject's right at +x. So the image u axis maps to -x and the image v axis
(which grows downward) maps to -y.

**The 189-point index map.** Written out from the published template
definition, then checked: the mapping is only trusted for landmarks whose
MediaPipe counterpart lands within a fraction of an interpupillary distance of
it in arm 8. Anything that does not is reported, not quietly kept.
"""

from __future__ import annotations

import csv
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from faciometry.core.landmarks import Landmark as L
from faciometry.core.landmarks import PointSet
from faciometry.core.scale import from_iris

DATA = Path(__file__).resolve().parent / "data" / "frll"
MODEL = Path(__file__).resolve().parent / "data" / "models" / "face_landmarker.task"

VIEWS = {
    "neutral_front": "neutral_front.zip",
    "smiling_front": "smiling_front.zip",
    "neutral_left_3quarter": "neutral_left_3quarter.zip",
    "neutral_right_3quarter": "neutral_right_3quarter.zip",
    "neutral_left_profile": "neutral_left_profile.zip",
    "neutral_right_profile": "neutral_right_profile.zip",
    "smiling_left_3quarter": "smiling_left_3quarter.zip",
}

# --------------------------------------------------------------------------
# FRL 189-point template -> Faciometry landmarks.
# "left" in the FRL names is the viewer's left, i.e. the subject's RIGHT.
# A tuple means the midpoint of those points; a callable picks a point per face.
# --------------------------------------------------------------------------

TEM_MAP: dict[L, object] = {
    L.PUPIL_R: 0,
    L.PUPIL_L: 1,
    L.EXOCANTHION_R: 18,
    L.ENDOCANTHION_R: 22,
    L.ENDOCANTHION_L: 23,
    L.EXOCANTHION_L: 27,
    L.PALPEBRALE_SUP_R: 20,
    L.PALPEBRALE_INF_R: 29,
    L.PALPEBRALE_SUP_L: 25,
    L.PALPEBRALE_INF_L: 32,
    L.SUPERCILIARE_R: 74,
    L.SUPERCILIARE_L: 79,
    L.GLABELLA: (76, 77),
    L.NASION: (50, 56),
    L.SELLION: (51, 57),
    L.PRONASALE: (54, 60),
    L.SUBNASALE: 55,
    L.SUBALARE_R: 64,
    L.SUBALARE_L: 69,
    L.COLUMELLA: (180, 181),
    L.LABIALE_SUPERIUS: 90,
    L.STOMION: (96, 101),
    L.LABIALE_INFERIUS: 106,
    L.CRISTA_PHILTRI_R: 89,
    L.CRISTA_PHILTRI_L: 91,
    L.CHEILION_R: 87,
    L.CHEILION_L: 93,
    L.MENTON: 129,
    L.GNATHION: 129,
    L.TRAGION_R: 109,
    L.TRAGION_L: 112,
    L.TRICHION: 139,
}

#: Landmarks whose anatomical definition is "the most lateral point of a
#: contour". Picking a fixed index would bake one delineator's choice in; the
#: extremum over the contour group is the definition itself.
TEM_EXTREMA: dict[L, tuple[tuple[int, ...], str]] = {
    L.ALARE_R: ((61, 62, 63, 64, 65), "min_u"),
    L.ALARE_L: ((66, 67, 68, 69, 70), "max_u"),
    L.ZYGION_R: ((164, 165, 166), "min_u"),
    L.ZYGION_L: ((167, 168, 169), "max_u"),
}

#: Gonion is the corner of the mandible, so it is the point of greatest turn on
#: the jaw contour rather than a fixed index.
TEM_CORNERS: dict[L, tuple[int, ...]] = {
    L.GONION_R: (125, 126, 127),
    L.GONION_L: (133, 132, 131),
}

TEM_IRIS_R = tuple(range(2, 10))
TEM_IRIS_L = tuple(range(10, 18))

#: Landmarks the frontal template cannot supply. Every measurement needing one
#: is reported as unavailable, which is the correct outcome and not a gap.
TEM_UNAVAILABLE = (L.POGONION, L.SUBLABIALE, L.CERVICALE, L.PORION_L, L.PORION_R,
                   L.ORBITALE_L, L.ORBITALE_R)


# --------------------------------------------------------------------------
# MediaPipe FaceMesh (478 points, Apache-2.0 weights) -> Faciometry landmarks.
# Validated against the human template in arm 8; anything that disagrees badly
# is reported there rather than silently used.
# --------------------------------------------------------------------------

MP_MAP: dict[L, object] = {
    L.PUPIL_R: 468,
    L.PUPIL_L: 473,
    L.EXOCANTHION_R: 33,
    L.ENDOCANTHION_R: 133,
    L.ENDOCANTHION_L: 362,
    L.EXOCANTHION_L: 263,
    L.PALPEBRALE_SUP_R: 159,
    L.PALPEBRALE_INF_R: 145,
    L.PALPEBRALE_SUP_L: 386,
    L.PALPEBRALE_INF_L: 374,
    L.SUPERCILIARE_R: 105,
    L.SUPERCILIARE_L: 334,
    L.GLABELLA: 9,
    L.NASION: 168,
    L.SELLION: 6,
    L.PRONASALE: 4,
    L.SUBNASALE: 2,
    L.ALARE_R: 48,
    L.ALARE_L: 278,
    L.SUBALARE_R: 64,
    L.SUBALARE_L: 294,
    L.COLUMELLA: 94,
    L.LABIALE_SUPERIUS: 0,
    L.STOMION: (13, 14),
    L.LABIALE_INFERIUS: 17,
    L.CRISTA_PHILTRI_R: 37,
    L.CRISTA_PHILTRI_L: 267,
    L.CHEILION_R: 61,
    L.CHEILION_L: 291,
    L.MENTON: 152,
    L.GNATHION: 152,
    L.GONION_R: 172,
    L.GONION_L: 397,
    L.ZYGION_R: 234,
    L.ZYGION_L: 454,
    L.TRAGION_R: 127,
    L.TRAGION_L: 356,
    L.TRICHION: 10,
}
MP_IRIS_R = (469, 470, 471, 472)
MP_IRIS_L = (474, 475, 476, 477)


@dataclass(frozen=True)
class Subject:
    face_id: str
    age: int
    gender: str
    ethnicity: str


@lru_cache(maxsize=1)
def subjects() -> dict[str, Subject]:
    out = {}
    with (DATA / "london_faces_info.csv").open() as fh:
        for row in csv.DictReader(fh):
            age = row["face_age"].strip()
            out[row["face_id"]] = Subject(
                row["face_id"], int(age) if age.isdigit() else -1,
                row["face_gender"].strip() or "undeclared",
                row["face_eth"].strip() or "undeclared")
    return out


def read_tem(text: str) -> np.ndarray:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n = int(lines[0])
    pts = np.array([[float(x) for x in ln.split()] for ln in lines[1:n + 1]])
    if pts.shape != (n, 2):  # pragma: no cover
        raise ValueError(f"expected {n} 2D points, got {pts.shape}")
    return pts


@lru_cache(maxsize=1)
def templates() -> dict[str, np.ndarray]:
    out = {}
    with zipfile.ZipFile(DATA / "neutral_front.zip") as z:
        for name in z.namelist():
            if not name.endswith(".tem") or "__MACOSX" in name:
                continue
            fid = Path(name).stem.split("_")[0]
            out[fid] = read_tem(z.read(name).decode())
    return out


def _turn_angle(a, b, c) -> float:
    u, v = a - b, c - b
    cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    return 180.0 - np.degrees(np.arccos(max(-1.0, min(1.0, cos))))


def tem_points(pts: np.ndarray) -> dict[L, np.ndarray]:
    """Resolve the 189-point template into named landmarks, in image pixels."""
    out: dict[L, np.ndarray] = {}
    for name, spec in TEM_MAP.items():
        out[name] = pts[spec] if isinstance(spec, int) else pts[list(spec)].mean(axis=0)
    for name, (group, how) in TEM_EXTREMA.items():
        us = pts[list(group)][:, 0]
        out[name] = pts[group[int(np.argmin(us) if how == "min_u" else np.argmax(us))]]
    for name, group in TEM_CORNERS.items():
        turns = [_turn_angle(pts[group[i - 1]], pts[group[i]], pts[group[i + 1]])
                 for i in (1,)]
        # only the middle point of a three-point contour has a turn; keep it
        out[name] = pts[group[1]] if turns else pts[group[1]]
    return out


def iris_diameter_px(pts: np.ndarray, group: tuple[int, ...]) -> float:
    g = pts[list(group)]
    return float(np.ptp(g[:, 0]))


def to_canonical(points: dict[L, np.ndarray], mm_per_px: float | None = None) -> PointSet:
    """Image pixels to Faciometry's canonical frame.

    Image u grows to the viewer's right, which is the subject's left, so
    ``x = -u``. Image v grows downward, so ``y = -v``. The origin is the
    centroid, since every measurement is translation invariant.
    """
    names = list(points)
    arr = np.array([points[n] for n in names], dtype=float)
    xy = np.stack([-arr[:, 0], -arr[:, 1]], axis=-1)
    xy = xy - xy.mean(axis=0)
    if mm_per_px is not None:
        xy = xy * mm_per_px
    return PointSet(index={n: i for i, n in enumerate(names)}, coords=xy)


def template_pointset(fid: str) -> tuple[PointSet, object]:
    pts = templates()[fid]
    named = tem_points(pts)
    d_r = iris_diameter_px(pts, TEM_IRIS_R)
    d_l = iris_diameter_px(pts, TEM_IRIS_L)
    scale = from_iris((d_r + d_l) / 2.0)
    return to_canonical(named, scale.mm_per_px), scale


# --------------------------------------------------------------------------
# MediaPipe
# --------------------------------------------------------------------------

_LANDMARKER = None


def landmarker():
    global _LANDMARKER
    if _LANDMARKER is None:
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        _LANDMARKER = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_path=str(MODEL)),
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,
                num_faces=1,
            ))
    return _LANDMARKER


def detect(image_bytes: bytes) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    import io

    import mediapipe as mp
    from PIL import Image

    arr = np.asarray(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    res = landmarker().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
    if not res.face_landmarks:
        return None
    h, w = arr.shape[:2]
    pts = np.array([[p.x * w, p.y * h, p.z * w] for p in res.face_landmarks[0]])
    xform = (np.array(res.facial_transformation_matrixes[0])
             if res.facial_transformation_matrixes else np.eye(4))
    return pts, xform, (w, h)


def pose_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """Yaw, pitch, roll in degrees from MediaPipe's 4x4 head transform.

    Decomposed as the intrinsic Y-X-Z sequence ``geometry.rotation_matrix``
    builds, so the numbers mean the same thing as everywhere else here.
    """
    r = m[:3, :3]
    pitch = np.arcsin(np.clip(r[2, 1], -1.0, 1.0))
    if abs(r[2, 1]) < 0.9999:
        yaw = np.arctan2(-r[2, 0], r[2, 2])
        roll = np.arctan2(-r[0, 1], r[1, 1])
    else:  # pragma: no cover
        yaw = np.arctan2(r[0, 2], r[0, 0])
        roll = 0.0
    return tuple(float(np.degrees(a)) for a in (yaw, pitch, roll))


def mp_points(pts: np.ndarray) -> dict[L, np.ndarray]:
    out = {}
    for name, spec in MP_MAP.items():
        out[name] = pts[spec, :2] if isinstance(spec, int) else pts[list(spec), :2].mean(axis=0)
    return out


def mp_pointset(pts: np.ndarray) -> tuple[PointSet, object]:
    named = mp_points(pts)
    d_r = float(np.ptp(pts[list(MP_IRIS_R), 0]))
    d_l = float(np.ptp(pts[list(MP_IRIS_L), 0]))
    scale = from_iris((d_r + d_l) / 2.0)
    return to_canonical(named, scale.mm_per_px), scale


def images(view: str):
    """Yield (face_id, jpeg bytes) for one view."""
    with zipfile.ZipFile(DATA / VIEWS[view]) as z:
        for name in sorted(z.namelist()):
            if not name.lower().endswith(".jpg") or "__MACOSX" in name:
                continue
            yield Path(name).stem.split("_")[0], z.read(name)


# --------------------------------------------------------------------------
# Cached detection pass over every view, so arms 8, 9 and 10 pay for it once.
# --------------------------------------------------------------------------

CACHE = Path(__file__).resolve().parent / "data" / "frll_mediapipe.npz"


def detect_all(force: bool = False) -> dict:
    """Run the landmarker over all seven views and cache the result.

    Returns ``{view: {face_id: {"pts": (478,3), "pose": (yaw,pitch,roll),
    "size": (w,h)}}}``. Faces the detector misses are absent, and the caller
    reports the miss rate rather than filling it in.
    """
    if CACHE.exists() and not force:
        raw = np.load(CACHE, allow_pickle=True)
        return raw["data"].item()
    out: dict[str, dict] = {}
    for view in VIEWS:
        got = {}
        for fid, blob in images(view):
            res = detect(blob)
            if res is None:
                continue
            pts, xform, size = res
            got[fid] = {"pts": pts, "pose": pose_from_matrix(xform), "size": size}
        out[view] = got
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, data=np.array(out, dtype=object))
    return out
