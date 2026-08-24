"""Arm 3 -- the control for arm 2. Pipeline noise at exactly zero pose.

Arm 2's slopes are meaningless without this. A measurement that moves 0.3% at
ten degrees of yaw has told you nothing if it also moves 0.3% between two
photographs of a motionless face.

So: the synthetic face at 0/0/0 is rendered as an image, pushed through a JPEG
encode/decode cycle, and re-measured, N times. Nothing about the face changes
between trials. What changes is what changes between two real photographs of a
still subject:

* sub-pixel framing offset -- the camera is never in exactly the same place,
  so the landmark never lands on the same pixel grid twice;
* sensor noise -- Gaussian, added before compression;
* JPEG quantisation -- 8x8 DCT blocks, chroma subsampling, the lot.

Landmark positions are recovered by an intensity-weighted centroid in a window
around each blob. The window is placed at the true position, which is
deliberate: this arm measures the noise floor of *encoding and resampling*, not
the failure rate of a detector. It is therefore a lower bound on real pipeline
noise, and every conclusion drawn from it is a lower bound too.

Four conditions separate the sources:

``lossless``   sub-pixel jitter + sensor noise, PNG round trip
``q95``/``q85``/``q75``  the same, JPEG at three qualities
``jpeg_only``  JPEG at 85, sensor noise, no jitter -- quantisation alone
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image

from evals._bootstrap import rng, write_csv, write_json
from evals.synth import face as F

from vitruve.core.landmarks import PointSet
from vitruve.core.spec import Unit, View
from vitruve.measure.registry import CATALOGUE

PX_PER_MM = 4.0          # an IPD of 63.36 mm spans 253 px, a realistic portrait crop
BLOB_SIGMA_PX = 1.6
BLOB_AMPLITUDE = 190.0
BACKGROUND = 40.0
SENSOR_NOISE_SD = 2.0    # digital numbers, 8-bit
WINDOW_PX = 6
N_TRIALS = 200

CONDITIONS = {
    "lossless":  dict(fmt="PNG",  quality=None, jitter=True),
    "q95":       dict(fmt="JPEG", quality=95,   jitter=True),
    "q85":       dict(fmt="JPEG", quality=85,   jitter=True),
    "q75":       dict(fmt="JPEG", quality=75,   jitter=True),
    "jpeg_only": dict(fmt="JPEG", quality=85,   jitter=False),
}


def _image_coords(coords: np.ndarray, view: str) -> np.ndarray:
    """Face millimetres to image pixels, y flipped because images grow downward."""
    if view == "frontal":
        u, v = coords[:, 0], coords[:, 1]
    else:  # profile camera looks down -x; the image horizontal axis is anatomical z
        u, v = coords[:, 2], coords[:, 1]
    return np.stack([u * PX_PER_MM, -v * PX_PER_MM], axis=-1)


def _canvas(view: str, pts_px: np.ndarray) -> tuple[int, int, np.ndarray]:
    pad = 30
    lo = pts_px.min(axis=0) - pad
    hi = pts_px.max(axis=0) + pad
    w, h = int(math.ceil(hi[0] - lo[0])), int(math.ceil(hi[1] - lo[1]))
    return w, h, lo


def _render(pts_px: np.ndarray, w: int, h: int, r: np.random.Generator) -> np.ndarray:
    img = np.full((h, w), BACKGROUND, dtype=np.float64)
    yy = np.arange(h)[:, None]
    xx = np.arange(w)[None, :]
    two_s2 = 2.0 * BLOB_SIGMA_PX ** 2
    for x, y in pts_px:
        x0, x1 = max(0, int(x) - 8), min(w, int(x) + 9)
        y0, y1 = max(0, int(y) - 8), min(h, int(y) + 9)
        if x1 <= x0 or y1 <= y0:
            continue
        sub_x = xx[:, x0:x1] - x
        sub_y = yy[y0:y1, :] - y
        img[y0:y1, x0:x1] += BLOB_AMPLITUDE * np.exp(-(sub_x ** 2 + sub_y ** 2) / two_s2)
    img += r.normal(0.0, SENSOR_NOISE_SD, img.shape)
    return np.clip(img, 0, 255)


def _round_trip(img: np.ndarray, fmt: str, quality: int | None) -> np.ndarray:
    pil = Image.fromarray(img.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    if fmt == "JPEG":
        pil.save(buf, format="JPEG", quality=quality, subsampling=0)
    else:
        pil.save(buf, format="PNG")
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("L"), dtype=np.float64)


def _recover(img: np.ndarray, pts_px: np.ndarray) -> np.ndarray:
    """Intensity-weighted centroid in a window at each nominal position."""
    h, w = img.shape
    out = np.empty_like(pts_px)
    for i, (x, y) in enumerate(pts_px):
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - WINDOW_PX), min(w, xi + WINDOW_PX + 1)
        y0, y1 = max(0, yi - WINDOW_PX), min(h, yi + WINDOW_PX + 1)
        patch = img[y0:y1, x0:x1] - BACKGROUND
        patch = np.clip(patch, 0.0, None)
        total = patch.sum()
        if total <= 1e-9:
            out[i] = (x, y)
            continue
        gx = np.arange(x0, x1)[None, :]
        gy = np.arange(y0, y1)[:, None]
        out[i] = ((patch * gx).sum() / total, (patch * gy).sum() / total)
    return out


def _back_to_face(pts_px: np.ndarray, view: str, origin: np.ndarray) -> np.ndarray:
    """Pixels back to face millimetres, into the 3D frame with the lost axis zeroed."""
    u = (pts_px[:, 0] + origin[0]) / PX_PER_MM
    v = -(pts_px[:, 1] + origin[1]) / PX_PER_MM
    out = np.zeros((pts_px.shape[0], 3))
    if view == "frontal":
        out[:, 0], out[:, 1] = u, v
    else:
        out[:, 2], out[:, 1] = u, v
    return out


def _trial(view: str, cond: dict, r: np.random.Generator) -> np.ndarray:
    coords = F.rotated(0.0, 0.0, 0.0)
    coords = F.camera_frontal_ortho(coords) if view == "frontal" else F.camera_profile_ortho(coords)
    pts = _image_coords(coords, view)
    w, h, lo = _canvas(view, pts)
    local = pts - lo
    if cond["jitter"]:
        local = local + r.uniform(-0.5, 0.5, size=2)
    img = _render(local, w, h, r)
    img = _round_trip(img, cond["fmt"], cond["quality"])
    rec = _recover(img, local)
    return _back_to_face(rec, view, lo)


def run() -> dict:
    r = rng(3)
    per_condition: dict[str, dict[str, list[float]]] = {}
    residuals: dict[str, list[float]] = {}

    for cname, cond in CONDITIONS.items():
        samples: dict[str, list[float]] = {s.id: [] for s in CATALOGUE}
        res: list[float] = []
        for _ in range(N_TRIALS):
            frontal = _trial("frontal", cond, r)
            profile = _trial("profile", cond, r)
            truth_f = F.camera_frontal_ortho(F.COORDS.copy())
            res.extend(np.linalg.norm(frontal[:, :2] - truth_f[:, :2], axis=-1).tolist())
            for spec in CATALOGUE:
                coords = profile if spec.view is View.PROFILE else frontal
                ps = PointSet(index=dict(F.INDEX), coords=coords)
                try:
                    samples[spec.id].append(float(np.asarray(spec.formula.eval(ps))))
                except Exception:
                    samples[spec.id].append(float("nan"))
        per_condition[cname] = samples
        residuals[cname] = res

    rows = []
    for spec in CATALOGUE:
        row = {"id": spec.id, "unit": spec.unit.value, "view": spec.view.value}
        for cname in CONDITIONS:
            v = np.array(per_condition[cname][spec.id], dtype=float)
            v = v[np.isfinite(v)]
            mean, sd = float(np.mean(v)), float(np.std(v, ddof=1))
            row[f"{cname}_mean"] = mean
            row[f"{cname}_sd"] = sd
            # The noise metric matches arm 2's deviation metric so the two are
            # directly comparable: degrees for angles, fractional otherwise.
            row[f"{cname}_noise"] = sd if spec.unit is Unit.DEGREES else (
                sd / abs(mean) if abs(mean) > 1e-12 else float("inf"))
        rows.append(row)

    resid_summary = {
        c: {"rms_px": float(np.sqrt(np.mean(np.square(v)))),
            "rms_mm": float(np.sqrt(np.mean(np.square(v)))),
            "p95_mm": float(np.percentile(np.abs(v), 95))}
        for c, v in residuals.items()
    }
    # residuals were computed in mm already
    for c in resid_summary:
        resid_summary[c].pop("rms_px")

    payload = {
        "arm": "3 -- encode/decode control for arm 2",
        "question": "how much does a measurement move between two photographs of a "
                    "motionless face, with no pose change at all",
        "design": {
            "n_trials": N_TRIALS, "px_per_mm": PX_PER_MM,
            "blob_sigma_px": BLOB_SIGMA_PX, "sensor_noise_sd_dn": SENSOR_NOISE_SD,
            "window_px": WINDOW_PX, "conditions": CONDITIONS,
            "caveat": "the recovery window is centred on the true position, so this is "
                      "the noise floor of encoding and resampling only and is a lower "
                      "bound on the noise of a real detector",
        },
        "landmark_residuals_mm": resid_summary,
        "noise": rows,
        "summary": {
            "n_measurements": len(rows),
            "median_q85_noise_ratios_and_lengths": float(np.median(
                [r["q85_noise"] for r in rows if r["unit"] != "deg" and math.isfinite(r["q85_noise"])])),
            "median_q85_noise_degrees": float(np.median(
                [r["q85_noise"] for r in rows if r["unit"] == "deg"])),
            "worst_q85": max(rows, key=lambda r: r["q85_noise"] if math.isfinite(r["q85_noise"]) else -1)["id"],
            "worst_q85_noise": max(r["q85_noise"] for r in rows if math.isfinite(r["q85_noise"])),
            "landmark_rms_mm_q85": resid_summary["q85"]["rms_mm"],
            "landmark_rms_mm_lossless": resid_summary["lossless"]["rms_mm"],
        },
    }
    write_json("arm03_encode_control", payload)
    write_csv("arm03_noise", rows)
    return payload["summary"]


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=1, default=str))
