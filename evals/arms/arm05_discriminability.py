"""Arm 5 -- discriminability. The project's headline number.

The question Faciometry is built around: does a measurement vary more between
people than it does between photographs of one person? The ratio is
between-subject spread over total measurement error, and below 1.0 the number
is withheld.

5a  Reproduce Kleinberg and Vanezis (2007) numerically. An index whose
    between-subject relative spread is 1.2 percent, moving 8 to 19 percent at
    ten degrees of yaw, must come out below 1. Computed three ways: from
    Kleinberg's own reported movement, from Faciometry's a-priori sensitivities,
    and from the slopes measured in arm 2.

5b  Can Faciometry's own geometry produce an 8 to 19 percent movement at ten
    degrees of yaw at all? If it cannot, the a-priori model is not merely
    imprecise, it is describing a different phenomenon.

5c  The full table: every measurement's discriminability ratio at 0.5, 3 and 8
    degrees of pose, as the code ships and with the measured slopes
    substituted.

5d  Crossover: the pose angle at which the pose effect equals the pipeline
    noise floor measured in arm 3. Below it, arm 2's slope is unobservable.
"""

from __future__ import annotations

import json
import math

import numpy as np

from evals._bootstrap import RESULTS, write_csv, write_json
from evals.synth import face as F

from faciometry.core.landmarks import PointSet
from faciometry.core.sensitivity import (
    POSE_ESTIMATOR_SD_DEG, PoseSensitivity, discriminability, gated_pose,
)
from faciometry.core.spec import Unit, View, assess_discriminability
from faciometry.measure.registry import CATALOGUE, BY_ID

POSES_DEG = (0.5, 3.0, 8.0)
KLEINBERG_RSD = 0.012
KLEINBERG_MOVEMENT = (0.08, 0.19)


def _load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def _measured_slopes(condition: str) -> dict[tuple[str, str], float]:
    d = _load("arm02_pose_sweep")
    return {(r["id"], r["axis"]): r["measured_secant_10deg"]
            for r in d["slopes"] if r["condition"] == condition}


def _pipeline_noise(condition: str = "q85") -> dict[str, float]:
    d = _load("arm03_encode_control")
    return {r["id"]: r[f"{condition}_noise"] for r in d["noise"]}


# ---------------------------------------------------------------------------
# 5a
# ---------------------------------------------------------------------------

def kleinberg_reproduction(slopes: dict) -> list[dict]:
    rows = []

    def add(label, pose_error, note):
        d = discriminability(between_subject_sd=KLEINBERG_RSD, pose_error=pose_error,
                             landmark_error=0.0)
        rows.append({"source": label, "pose_error_at_10deg_yaw": pose_error,
                     "between_subject_sd": KLEINBERG_RSD, "ratio": d.ratio,
                     "informative": d.informative, "verdict": d.verdict, "note": note})

    for m in KLEINBERG_MOVEMENT:
        add(f"Kleinberg & Vanezis 2007 measured index movement ({m:.0%})", m,
            "the published number, used directly")
    from faciometry.core.sensitivity import KLEINBERG_WORST, TRANSVERSE_WIDTH
    add("faciometry a-priori KLEINBERG_WORST", KLEINBERG_WORST.error_at(10, 0, 0),
        "sensitivity.KLEINBERG_WORST.error_at(10,0,0)")
    add("faciometry a-priori KLEINBERG_WORST, pose-gated",
        KLEINBERG_WORST.error_at(gated_pose(10.0), 0, 0),
        f"as assess_discriminability applies it, inflating 10 deg to "
        f"{gated_pose(10.0):.2f} deg")
    add("faciometry a-priori TRANSVERSE_WIDTH", TRANSVERSE_WIDTH.error_at(10, 0, 0),
        "a depth-matched transverse width, cos(yaw) only")

    # Five measurements have a near-zero value by construction -- the three
    # asymmetries sit on a 0.5 mm height difference between the outer canthi
    # and a 1.0 degree tilt difference, and the two commissure heights on the
    # 0.8 mm drop of one mouth corner, with the other corner exactly level. A
    # *relative* movement is enormous for them without the measurement being
    # comparable to one of Kleinberg's indices, which are ratios of two spans
    # of similar size. They are reported separately rather than allowed to be
    # the headline; their zero-pose values are printed beside them so the
    # exclusion can be checked rather than taken.
    degenerate = {"ocular_height_asymmetry", "mouth_corner_asymmetry",
                  "canthal_tilt_asymmetry", "commissure_height_l", "commissure_height_r"}
    for cond in ("ortho", "persp_1.0m", "persp_0.5m"):
        s = _measured_slopes(cond)
        ranked = sorted(
            ((mid, v) for (mid, ax), v in s.items()
             if ax == "yaw" and math.isfinite(v) and BY_ID[mid].unit is not Unit.DEGREES),
            key=lambda kv: -kv[1])
        worst = ranked[0]
        add(f"measured, worst relative index under {cond}", worst[1] * 10.0,
            f"{worst[0]} at 10 deg of yaw")
        clean = [kv for kv in ranked if kv[0] not in degenerate][0]
        add(f"measured, worst non-degenerate index under {cond}", clean[1] * 10.0,
            f"{clean[0]} at 10 deg of yaw; excludes the three asymmetry measurements, "
            "whose zero-pose value is near zero by construction")
        zero_pose = {r["id"]: r["value_at_zero"] for r in _load("arm02_pose_sweep")["slopes"]
                     if r["condition"] == cond and r["axis"] == "yaw"}
        rows[-1]["top5"] = [{"id": m, "movement_at_10deg_yaw": v * 10.0,
                             "value_at_zero_pose": zero_pose.get(m),
                             "excluded_as_near_zero": m in degenerate} for m, v in ranked[:5]]
    return rows


# ---------------------------------------------------------------------------
# 5b
# ---------------------------------------------------------------------------

def _value(spec, coords) -> float:
    ps = PointSet(index=dict(F.INDEX), coords=coords)
    try:
        return float(np.asarray(spec.formula.eval(ps)))
    except Exception:
        return float("nan")


def can_geometry_reach_kleinberg() -> list[dict]:
    """Movement at 10 degrees, yaw alone and all three axes together."""
    rows = []
    for cond, dist in (("ortho", None), ("persp_1.0m", 1.0), ("persp_0.5m", 0.5)):
        for spec in CATALOGUE:
            def cap(y, p, r):
                c = F.rotated(y, p, r)
                if spec.view is View.PROFILE:
                    return (F.camera_profile_ortho(c) if dist is None
                            else F.camera_profile_perspective(c, dist))
                return (F.camera_frontal_ortho(c) if dist is None
                        else F.camera_frontal_perspective(c, dist))

            v0 = _value(spec, cap(0, 0, 0))
            def dev(v):
                if spec.unit is Unit.DEGREES:
                    return abs(v - v0)
                return abs((v - v0) / v0) if abs(v0) > 1e-12 else float("nan")

            rows.append({
                "condition": cond, "id": spec.id, "unit": spec.unit.value,
                "yaw10": dev(_value(spec, cap(10, 0, 0))),
                "pitch10": dev(_value(spec, cap(0, 10, 0))),
                "roll10": dev(_value(spec, cap(0, 0, 10))),
                "all_three_10": dev(_value(spec, cap(10, 10, 10))),
                "reaches_kleinberg_floor_8pct": bool(
                    spec.unit is not Unit.DEGREES
                    and dev(_value(spec, cap(10, 0, 0))) >= 0.08),
            })
    return rows


# ---------------------------------------------------------------------------
# 5c
# ---------------------------------------------------------------------------

def full_table(noise: dict[str, float]) -> list[dict]:
    ortho = _measured_slopes("ortho")
    p10 = _measured_slopes("persp_1.0m")
    rows = []
    for spec in CATALOGUE:
        le = noise[spec.id]
        if not math.isfinite(le):
            le = 0.0
        row = {
            "id": spec.id, "unit": spec.unit.value, "view": spec.view.value,
            "evidence": spec.evidence.value,
            "between_subject_spread": spec.between_subject_rsd,
            "spread_known": spec.between_subject_rsd is not None,
            "measured_pipeline_noise": noise[spec.id],
            "measured_within_person_rsd": spec.measured_within_person_rsd,
            "apriori_yaw": spec.sensitivity.yaw,
            "apriori_pitch": spec.sensitivity.pitch,
            "apriori_roll": spec.sensitivity.roll,
            "measured_yaw_ortho": ortho[(spec.id, "yaw")],
            "measured_pitch_ortho": ortho[(spec.id, "pitch")],
            "measured_roll_ortho": ortho[(spec.id, "roll")],
            "measured_yaw_persp1m": p10[(spec.id, "yaw")],
            "measured_pitch_persp1m": p10[(spec.id, "pitch")],
            "measured_roll_persp1m": p10[(spec.id, "roll")],
        }
        for cond, table in (("ortho", ortho), ("persp1m", p10)):
            sens = PoseSensitivity(
                yaw=table[(spec.id, "yaw")] if math.isfinite(table[(spec.id, "yaw")]) else 0.0,
                pitch=table[(spec.id, "pitch")] if math.isfinite(table[(spec.id, "pitch")]) else 0.0,
                roll=table[(spec.id, "roll")] if math.isfinite(table[(spec.id, "roll")]) else 0.0,
                source=f"measured, arm 2, {cond}")
            for theta in POSES_DEG:
                key = f"{cond}_{theta}"
                if spec.between_subject_rsd is None:
                    row[f"disc_measured_{key}"] = None
                    continue
                g = gated_pose(theta)
                pose_err = sens.error_at(g, g, g)
                if spec.measured_within_person_rsd is not None:
                    pose_err = max(spec.measured_within_person_rsd, pose_err)
                d = discriminability(between_subject_sd=spec.between_subject_rsd,
                                     pose_error=pose_err, landmark_error=le)
                row[f"disc_measured_{key}"] = d.ratio
                row[f"pose_err_measured_{key}"] = pose_err
        for theta in POSES_DEG:
            d = assess_discriminability(spec, yaw_deg=theta, pitch_deg=theta, roll_deg=theta,
                                        relative_landmark_error=le)
            row[f"disc_as_shipped_{theta}"] = d.ratio if d else None
            row[f"verdict_as_shipped_{theta}"] = d.verdict if d else "spread between people unknown"
            row[f"informative_as_shipped_{theta}"] = d.informative if d else None
            dm = row.get(f"disc_measured_persp1m_{theta}")
            row[f"informative_measured_{theta}"] = (dm > 1.0) if dm is not None else None
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 5d
# ---------------------------------------------------------------------------

def crossover(noise: dict[str, float]) -> list[dict]:
    ortho = _measured_slopes("ortho")
    p10 = _measured_slopes("persp_1.0m")
    rows = []
    for spec in CATALOGUE:
        n = noise[spec.id]
        for axis in ("yaw", "pitch", "roll"):
            s_o, s_p = ortho[(spec.id, axis)], p10[(spec.id, axis)]
            def cross(s):
                if not math.isfinite(s) or s <= 0 or not math.isfinite(n):
                    return None
                return n / s
            rows.append({"id": spec.id, "axis": axis, "pipeline_noise": n,
                         "slope_ortho": s_o, "slope_persp1m": s_p,
                         "crossover_deg_ortho": cross(s_o),
                         "crossover_deg_persp1m": cross(s_p)})
    return rows


def run() -> dict:
    noise = _pipeline_noise()
    kle = kleinberg_reproduction(None)
    reach = can_geometry_reach_kleinberg()
    table = full_table(noise)
    cross = crossover(noise)

    shipped_uninformative = {
        t: sorted(r["id"] for r in table if r[f"informative_as_shipped_{t}"] is False)
        for t in POSES_DEG}
    measured_uninformative = {
        t: sorted(r["id"] for r in table if r[f"informative_measured_{t}"] is False)
        for t in POSES_DEG}
    unknown = sorted(r["id"] for r in table if not r["spread_known"])

    payload = {
        "arm": "5 -- discriminability",
        "question": "does each measurement vary more between people than between "
                    "photographs of one person",
        "design": {
            "poses_deg": list(POSES_DEG),
            "pose_applied_to": "all three axes simultaneously, then inflated by "
                               f"gated_pose (+{POSE_ESTIMATOR_SD_DEG:.2f} deg per axis) "
                               "exactly as assess_discriminability does",
            "landmark_error_source": "measured pipeline noise from arm 3, JPEG q85",
            "kleinberg_rsd": KLEINBERG_RSD,
            "kleinberg_movement": list(KLEINBERG_MOVEMENT),
        },
        "kleinberg_reproduction": kle,
        "geometry_reach": reach,
        "table": table,
        "crossover": cross,
        "summary": {
            "n_measurements": len(table),
            "n_spread_unknown": len(unknown),
            "spread_unknown": unknown,
            "n_uninformative_as_shipped": {str(k): len(v) for k, v in shipped_uninformative.items()},
            "n_uninformative_measured": {str(k): len(v) for k, v in measured_uninformative.items()},
            "uninformative_as_shipped": {str(k): v for k, v in shipped_uninformative.items()},
            "uninformative_measured": {str(k): v for k, v in measured_uninformative.items()},
            "kleinberg_reproduced": all(
                r["ratio"] < 1.0 for r in kle if r["source"].startswith("Kleinberg")),
            "n_reaching_8pct_at_10deg_yaw": {
                c: sum(1 for r in reach if r["condition"] == c and r["reaches_kleinberg_floor_8pct"])
                for c in ("ortho", "persp_1.0m", "persp_0.5m")},
            "reaching_8pct_persp_0.5m": sorted(
                r["id"] for r in reach
                if r["condition"] == "persp_0.5m" and r["reaches_kleinberg_floor_8pct"]),
            "gate_is_dominated_by_estimator_noise": {
                str(t): {
                    "true_pose_deg": t,
                    "gated_pose_deg": gated_pose(t),
                    "fraction_that_is_estimator_uncertainty": POSE_ESTIMATOR_SD_DEG / gated_pose(t),
                } for t in POSES_DEG},
            "tightest_crossover": min(
                (r for r in cross if r["crossover_deg_persp1m"]),
                key=lambda r: r["crossover_deg_persp1m"]),
        },
    }
    write_json("arm05_discriminability", payload)
    write_csv("arm05_table", table)
    write_csv("arm05_kleinberg", kle)
    write_csv("arm05_geometry_reach", reach)
    write_csv("arm05_crossover", cross)
    return payload["summary"]


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=str))
