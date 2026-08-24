"""Arm 11 -- how wrong is the pose estimator, without a labelled benchmark.

``core.sensitivity.POSE_ESTIMATOR_MAE_DEG = 3.97`` is 6DRepNet's published
AFLW2000-3D mean absolute error, and it is load-bearing for every verdict the
project prints: ``gated_pose`` adds ``3.97 * sqrt(pi/2) = 4.977`` degrees to
every axis before the discriminability gate, which arm 5 shows is 90.9 percent
of the gated pose at half a degree of true pose and 38.3 percent at eight.
Confirming it directly needs AFLW2000-3D, which requires registration, so the
previous run listed it as the largest unclosed hole and stopped there.

It does not have to stay entirely open. Three of the four parts below need no
labels at all, and the fourth needs only a human's landmarks:

11a **Known in-plane rotation.** Rotating a photograph about its centre by
    theta degrees rotates the camera about its optical axis by exactly theta,
    so the *change* in true roll is known to the precision of an affine warp.
    Sweeping theta over +/-15 degrees on all 102 FRLL front captures gives a
    direct error measurement on the roll axis with exact ground truth. It
    bounds the estimator's roll *increment* error, not any constant roll bias,
    and it is the positive control for everything else here: if the fitted
    slope of estimated roll on applied roll were not close to 1, no other
    number in this arm would mean anything.

11b **Mirror antisymmetry.** Under a horizontal flip the true pose becomes
    (-yaw, pitch, -roll). Half the sum of the two estimates is therefore pure
    estimator error on yaw and roll, and half the difference is pure estimator
    error on pitch. Any error component that is itself mirror-symmetric
    cancels, so this is a **lower** bound, and it needs no labels whatsoever.

11c **Roll against a human's landmarks.** The FRL 189-point template is placed
    by a person and includes both pupil centres, so the inclination of the
    interpupillary line is an independent measurement of image roll made
    without a network. Errors add in quadrature, so the spread of the
    disagreement is an **upper** bound on the estimator's roll spread. It
    carries the subject's own ocular asymmetry inside it, which is why it is an
    upper bound rather than an estimate.

11d **Two estimators on one photograph.** 6DRepNet against MediaPipe's own
    facial transformation matrix, on the same crops. Neither is a reference, so
    the disagreement upper-bounds both.

**What this arm cannot do.** None of it reproduces 3.97 on AFLW2000-3D. FRLL
front captures are near-frontal studio portraits and AFLW2000-3D is not, and a
near-frontal figure is the *easy* end of the range the published number
averages over -- the same backend's docstring puts near-frontal performance
around 2.8 and extreme pose around 13.3. Nor does 11a bound yaw or pitch: there
is no image-space operation that induces a known out-of-plane rotation. What
comes out is a bound on the roll axis in the frontal regime, a floor on yaw and
pitch from the mirror, and a statement about whether 4.977 degrees of inflation
is the right order for the photographs Vitruve actually gates.

The sign of 6DRepNet's yaw and roll is recorded as unverified in
``models/pose_sixdrepnet.py``. Nothing here assumes it: every comparison is a
fitted slope, which is allowed to come out at -1, and the residual about that
fit is the error.
"""

from __future__ import annotations

import io
import json
import math

import numpy as np

from evals._bootstrap import RESULTS, write_csv, write_json
from evals import frll

from vitruve.core.sensitivity import POSE_ESTIMATOR_MAE_DEG, POSE_ESTIMATOR_SD_DEG

#: Applied in-plane rotations, degrees. Chosen to bracket the +/-3.9 degree
#: 95th percentile roll arm 8 measured on these same captures, and to reach
#: three times it.
APPLIED_ROLL_DEG = (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0)

VIEW = "neutral_front"

CACHE = RESULTS.parent / "data" / "arm11_pose.npz"


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def _rgb(blob: bytes) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))


def _rotate(img: np.ndarray, deg: float) -> np.ndarray:
    """Rotate about the image centre, keeping the canvas.

    FRLL captures are 1350x1350 with the head centred and a wide margin, so no
    part of the face leaves the frame at 15 degrees. The border is replicated
    rather than filled black, because a hard black wedge at the corner is a
    feature a detector can see and this arm is not about detector robustness.
    """
    if deg == 0.0:
        return img
    import cv2
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _mediapipe_pose(img: np.ndarray) -> tuple[float, float, float] | None:
    import mediapipe as mp
    res = frll.landmarker().detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
    if not res.facial_transformation_matrixes:
        return None
    return frll.pose_from_matrix(np.array(res.facial_transformation_matrixes[0]))


def _collect(force: bool = False) -> dict:
    """Every (subject, condition) pose from both estimators. Cached."""
    if CACHE.exists() and not force:
        return np.load(CACHE, allow_pickle=True)["data"].item()

    from vitruve.models import detect_yunet, pose_sixdrepnet
    from vitruve.models.weights import WeightsUnavailable

    try:
        det = detect_yunet.build()
        pose = pose_sixdrepnet.build()
    except WeightsUnavailable as exc:
        # Re-raised as FileNotFoundError so run_all reports this arm as "not
        # run", with the reason, rather than as a failure. Missing weights are
        # a missing prerequisite, not a broken arm.
        raise FileNotFoundError(
            f"arm 11 needs the YuNet and 6DRepNet weights: {exc}") from exc

    out: dict[str, dict[str, dict]] = {}
    for fid, blob in frll.images(VIEW):
        img0 = _rgb(blob)
        rows: dict[str, dict] = {}
        conditions = [(f"roll{deg:+g}", _rotate(img0, deg)) for deg in APPLIED_ROLL_DEG]
        conditions.append(("mirror", np.ascontiguousarray(img0[:, ::-1])))
        for name, img in conditions:
            box = det.detect_largest(img)
            if box is None:
                rows[name] = {"detected": False}
                continue
            hp = pose.estimate(img, box)
            mp_pose = _mediapipe_pose(img)
            rows[name] = {
                "detected": True,
                "six": (hp.yaw_deg, hp.pitch_deg, hp.roll_deg),
                "mediapipe": mp_pose,
            }
        out[fid] = rows
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, data=np.array(out, dtype=object))
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Least squares y = a + b x. Returns (a, b, residuals)."""
    a_mat = np.stack([np.ones_like(x), x], axis=-1)
    coef, *_ = np.linalg.lstsq(a_mat, y, rcond=None)
    return float(coef[0]), float(coef[1]), y - a_mat @ coef


def _template_roll_deg() -> dict[str, float]:
    """Inclination of the human-placed interpupillary line, in image degrees.

    Read from the raw ``.tem`` pixel coordinates, so nothing in
    ``evals.frll.to_canonical`` or in the scale prior can touch it. Image v
    grows downward, so the y term is negated to make the angle a normal
    mathematical one; whether that matches the estimator's sign is left to the
    fit rather than asserted.
    """
    out = {}
    for fid, pts in frll.templates().items():
        r, l = pts[frll.TEM_MAP[frll.L.PUPIL_R]], pts[frll.TEM_MAP[frll.L.PUPIL_L]]
        out[fid] = math.degrees(math.atan2(-(l[1] - r[1]), (l[0] - r[0])))
    return out


# ---------------------------------------------------------------------------
# The four parts
# ---------------------------------------------------------------------------

def known_rotation(data: dict) -> tuple[list[dict], dict]:
    """11a -- estimated roll against an exactly known applied rotation."""
    rows, resid_six, resid_mp, slopes_six, slopes_mp = [], [], [], [], []
    leak_yaw, leak_pitch = [], []
    for fid, conds in data.items():
        applied, six, mp_ = [], [], []
        for deg in APPLIED_ROLL_DEG:
            rec = conds.get(f"roll{deg:+g}", {})
            if not rec.get("detected"):
                continue
            applied.append(deg)
            six.append(rec["six"])
            mp_.append(rec["mediapipe"] if rec["mediapipe"] else (np.nan,) * 3)
        if len(applied) < 4:
            continue
        x = np.array(applied, dtype=float)
        s = np.array(six, dtype=float)
        m = np.array(mp_, dtype=float)
        a_s, b_s, r_s = _fit(x, s[:, 2])
        slopes_six.append(b_s)
        resid_six.extend(np.abs(r_s).tolist())
        if np.isfinite(m[:, 2]).all():
            _, b_m, r_m = _fit(x, m[:, 2])
            slopes_mp.append(b_m)
            resid_mp.extend(np.abs(r_m).tolist())
        else:
            b_m = float("nan")
        # A pure in-plane rotation leaves the true yaw and pitch unchanged.
        # Whatever they do here is estimator error, uncontaminated by anatomy.
        leak_yaw.append(float(np.ptp(s[:, 0])))
        leak_pitch.append(float(np.ptp(s[:, 1])))
        rows.append({
            "face_id": fid, "n_conditions": len(applied),
            "six_roll_intercept": a_s, "six_roll_slope": b_s,
            "six_roll_residual_mae": float(np.mean(np.abs(r_s))),
            "six_yaw_range_over_sweep": float(np.ptp(s[:, 0])),
            "six_pitch_range_over_sweep": float(np.ptp(s[:, 1])),
            "mediapipe_roll_slope": b_m,
        })
    summary = {
        "n_subjects": len(rows),
        "applied_roll_deg": list(APPLIED_ROLL_DEG),
        "six_roll_slope_median": float(np.median(slopes_six)),
        "six_roll_slope_iqr": [float(np.percentile(slopes_six, 25)),
                               float(np.percentile(slopes_six, 75))],
        "six_roll_residual_mae_deg": float(np.mean(resid_six)),
        "six_roll_residual_p95_deg": float(np.percentile(resid_six, 95)),
        "mediapipe_roll_slope_median": (float(np.median(slopes_mp)) if slopes_mp
                                        else float("nan")),
        "mediapipe_roll_residual_mae_deg": (float(np.mean(resid_mp)) if resid_mp
                                            else float("nan")),
        "six_yaw_range_over_a_pure_roll_sweep_median_deg": float(np.median(leak_yaw)),
        "six_pitch_range_over_a_pure_roll_sweep_median_deg": float(np.median(leak_pitch)),
        "positive_control_passes": bool(abs(abs(float(np.median(slopes_six))) - 1.0) < 0.15),
    }
    return rows, summary


def mirror_antisymmetry(data: dict) -> tuple[list[dict], dict]:
    """11b -- a label-free lower bound from the horizontal mirror."""
    rows = []
    for fid, conds in data.items():
        a, b = conds.get("roll+0"), conds.get("mirror")
        if not (a and b and a.get("detected") and b.get("detected")):
            continue
        y0, p0, r0 = a["six"]
        y1, p1, r1 = b["six"]
        rows.append({
            "face_id": fid,
            "yaw": y0, "yaw_mirrored": y1,
            "pitch": p0, "pitch_mirrored": p1,
            "roll": r0, "roll_mirrored": r1,
            # (est(x) + est(mirror x)) / 2 is exactly the error component that
            # survives the mirror, for the two channels that should negate.
            "yaw_antisymmetry_error": (y0 + y1) / 2.0,
            "roll_antisymmetry_error": (r0 + r1) / 2.0,
            # pitch should be invariant, so half the difference plays the same
            # role for it.
            "pitch_symmetry_error": (p0 - p1) / 2.0,
        })
    if not rows:
        return rows, {"n": 0}
    def stat(key):
        v = np.array([r[key] for r in rows])
        return {"mae_deg": float(np.mean(np.abs(v))),
                "sd_deg": float(np.std(v, ddof=1)),
                "p95_abs_deg": float(np.percentile(np.abs(v), 95))}
    return rows, {
        "n": len(rows),
        "yaw": stat("yaw_antisymmetry_error"),
        "pitch": stat("pitch_symmetry_error"),
        "roll": stat("roll_antisymmetry_error"),
        "note": "a lower bound: any error that is itself mirror-symmetric cancels here",
    }


def roll_against_human(data: dict) -> tuple[list[dict], dict]:
    """11c -- estimated roll against the human template's interpupillary line."""
    tem = _template_roll_deg()
    rows = []
    for fid, conds in data.items():
        rec = conds.get("roll+0")
        if not (rec and rec.get("detected")) or fid not in tem:
            continue
        rows.append({"face_id": fid, "template_roll_deg": tem[fid],
                     "six_roll_deg": rec["six"][2],
                     "mediapipe_roll_deg": (rec["mediapipe"][2] if rec["mediapipe"]
                                            else float("nan"))})
    if len(rows) < 10:
        return rows, {"n": len(rows), "reason": "too few subjects"}
    x = np.array([r["template_roll_deg"] for r in rows])
    out = {"n": len(rows), "template_roll_sd_deg": float(np.std(x, ddof=1))}
    for key, label in (("six_roll_deg", "sixdrepnet"), ("mediapipe_roll_deg", "mediapipe")):
        y = np.array([r[key] for r in rows])
        ok = np.isfinite(y)
        if ok.sum() < 10:
            out[label] = {"n": int(ok.sum()), "reason": "too few estimates"}
            continue
        a, b, res = _fit(x[ok], y[ok])
        out[label] = {
            "n": int(ok.sum()),
            "intercept_deg": a, "slope": b,
            "pearson_r": float(np.corrcoef(x[ok], y[ok])[0, 1]),
            "residual_sd_deg": float(np.std(res, ddof=1)),
            "residual_mae_deg": float(np.mean(np.abs(res))),
            "note": "upper bound: the human's line carries the subject's own ocular "
                    "asymmetry as well as the camera",
        }
    return rows, out


def estimator_disagreement(data: dict) -> tuple[list[dict], dict]:
    """11d -- 6DRepNet against MediaPipe on the same unrotated crop."""
    rows = []
    for fid, conds in data.items():
        rec = conds.get("roll+0")
        if not (rec and rec.get("detected") and rec.get("mediapipe")):
            continue
        s, m = rec["six"], rec["mediapipe"]
        rows.append({"face_id": fid,
                     **{f"six_{k}": s[i] for i, k in enumerate(("yaw", "pitch", "roll"))},
                     **{f"mediapipe_{k}": m[i] for i, k in enumerate(("yaw", "pitch", "roll"))}})
    if len(rows) < 10:
        return rows, {"n": len(rows)}
    out = {"n": len(rows),
           "note": "neither estimator is a reference, so this upper-bounds both; a "
                   "fitted slope absorbs the unverified sign convention"}
    for k in ("yaw", "pitch", "roll"):
        a = np.array([r[f"six_{k}"] for r in rows])
        b = np.array([r[f"mediapipe_{k}"] for r in rows])
        i, sl, res = _fit(b, a)
        out[k] = {"raw_difference_sd_deg": float(np.std(a - b, ddof=1)),
                  "raw_difference_mae_deg": float(np.mean(np.abs(a - b))),
                  "slope": sl, "intercept_deg": i,
                  "residual_sd_deg": float(np.std(res, ddof=1)),
                  "pearson_r": float(np.corrcoef(a, b)[0, 1])}
    return rows, out


def run() -> dict:
    data = _collect()
    rot_rows, rot = known_rotation(data)
    mir_rows, mir = mirror_antisymmetry(data)
    hum_rows, hum = roll_against_human(data)
    dis_rows, dis = estimator_disagreement(data)

    # What the arm is for: is 4.977 degrees of inflation the right order for
    # the photographs the gate actually sees?
    bounds = {
        "roll_upper_bound_sd_deg": hum.get("sixdrepnet", {}).get("residual_sd_deg"),
        "roll_increment_mae_deg": rot["six_roll_residual_mae_deg"],
        "yaw_lower_bound_mae_deg": mir["yaw"]["mae_deg"],
        "pitch_lower_bound_mae_deg": mir["pitch"]["mae_deg"],
        "roll_lower_bound_mae_deg": mir["roll"]["mae_deg"],
        "declared_mae_deg": POSE_ESTIMATOR_MAE_DEG,
        "declared_sd_deg": POSE_ESTIMATOR_SD_DEG,
    }

    payload = {
        "arm": "11 -- pose estimator error without a labelled benchmark",
        "question": "POSE_ESTIMATOR_MAE_DEG = 3.97 is a literature value that gates "
                    "every measurement. How much of it can be checked with no "
                    "registration-gated data at all",
        "design": {
            "dataset": "FRLL neutral_front, 102 subjects, CC BY 4.0",
            "estimators": ["6DRepNet (the one src/ uses and the one 3.97 refers to)",
                           "MediaPipe facial transformation matrix"],
            "detector": "YuNet, the detector the pipeline uses",
            "applied_roll_deg": list(APPLIED_ROLL_DEG),
            "parts": {
                "11a": "known in-plane rotation; exact ground truth on the roll axis, "
                       "and the positive control for the whole arm",
                "11b": "horizontal mirror; label-free lower bound on all three axes",
                "11c": "human-placed interpupillary line; upper bound on roll",
                "11d": "two estimators on one photograph; upper bound on both",
            },
            "cannot_do": "no image-space operation induces a known out-of-plane "
                         "rotation, so yaw and pitch get a floor and not a bound; and "
                         "FRLL is near-frontal, which is the easy end of the range the "
                         "published 3.97 averages over",
        },
        "known_rotation": rot_rows,
        "known_rotation_summary": rot,
        "mirror": mir_rows,
        "mirror_summary": mir,
        "roll_against_human": hum_rows,
        "roll_against_human_summary": hum,
        "estimator_disagreement": dis_rows,
        "estimator_disagreement_summary": dis,
        "summary": {
            "positive_control_roll_slope_median": rot["six_roll_slope_median"],
            "positive_control_passes": rot["positive_control_passes"],
            "bounds_deg": bounds,
            "n_subjects": rot["n_subjects"],
            "roll_error_is_smaller_than_declared_sd": bool(
                bounds["roll_upper_bound_sd_deg"] is not None
                and bounds["roll_upper_bound_sd_deg"] < POSE_ESTIMATOR_SD_DEG),
        },
    }
    write_json("arm11_pose_estimator", payload)
    write_csv("arm11_known_rotation", rot_rows)
    write_csv("arm11_mirror", mir_rows)
    write_csv("arm11_roll_against_human", hum_rows)
    write_csv("arm11_estimator_disagreement", dis_rows)
    return payload["summary"]


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=str))
