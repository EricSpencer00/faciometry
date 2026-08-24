"""Arm 9 -- test-retest repeatability on real photographs of the same people.

``MeasurementSpec.measured_within_person_rsd`` is populated for two of the 45
measurements. For the other 43 the within-person spread is *derived* from a
projection model, and arm 2 showed that model is wrong in both directions. This
arm measures it instead.

FRLL photographed each of 102 people ten times. Seven of those captures were
downloaded; MediaPipe finds a face in five of them for every subject and in
about one in eight full profiles, so the profile views are excluded and that
exclusion is itself a result (see ``detection``).

Four conditions, in increasing order of how much the photograph is allowed to
differ:

``expression``  neutral front against smiling front. Same camera, same pose,
                different face. This isolates the term Kramer (2016) found
                dominating fWHR variance and that no projection model predicts.
``frontal``     the two frontal captures, as above, reported as the number a
                pipeline that gates hard on pose would actually see.
``three_quarter`` the three 3/4 captures, roughly +/-35 degrees of yaw. Pose
                dominates here.
``all_usable``  all five captures: what "another photograph of this person"
                means without a pose gate.

The deliverable is ``discriminability_measured``: between-person SD over
within-person SD, computed entirely from photographs, with no model of
projection anywhere in it.
"""

from __future__ import annotations

import json
import math

import numpy as np

from evals._bootstrap import rng, write_csv, write_json
from evals import frll

from faciometry.core.spec import Unit, View
from faciometry.measure.registry import CATALOGUE, satisfiable

CONDITIONS = {
    "expression": ("neutral_front", "smiling_front"),
    "frontal": ("neutral_front", "smiling_front"),
    "three_quarter": ("neutral_left_3quarter", "neutral_right_3quarter",
                      "smiling_left_3quarter"),
    "all_usable": ("neutral_front", "smiling_front", "neutral_left_3quarter",
                   "neutral_right_3quarter", "smiling_left_3quarter"),
}
EXCLUDED_VIEWS = ("neutral_left_profile", "neutral_right_profile")


def _measure(ps, specs) -> dict[str, float]:
    out = {}
    for s in specs:
        try:
            v = float(np.asarray(s.formula.eval(ps)))
        except Exception:
            continue
        if math.isfinite(v):
            out[s.id] = v
    return out


def run(n_boot: int = 2000) -> dict:
    r = rng(9)
    det = frll.detect_all()
    subs = frll.subjects()

    # measurements evaluable from a 2D frontal landmark set
    probe = frll.mp_pointset(next(iter(det["neutral_front"].values()))["pts"])[0]
    specs = [s for s in satisfiable(frozenset(probe.index)) if s.view is not View.PROFILE]

    values: dict[str, dict[str, dict[str, float]]] = {}   # view -> fid -> id -> value
    poses: dict[str, dict[str, tuple]] = {}
    for view, recs in det.items():
        values[view] = {}
        poses[view] = {}
        for fid, rec in recs.items():
            ps, _ = frll.mp_pointset(rec["pts"])
            values[view][fid] = _measure(ps, specs)
            poses[view][fid] = rec["pose"]

    detection = {v: {"n_detected": len(det[v]), "rate": len(det[v]) / 102,
                     "excluded": v in EXCLUDED_VIEWS} for v in det}

    rows = []
    for s in specs:
        row = {"id": s.id, "unit": s.unit.value, "evidence": s.evidence.value,
               "spec_measured_within_person_rsd": s.measured_within_person_rsd,
               "spec_between_subject_rsd": s.between_subject_rsd}
        # between-person spread from the neutral front capture
        between = np.array([values["neutral_front"][f][s.id]
                            for f in values["neutral_front"] if s.id in values["neutral_front"][f]])
        b_mean, b_sd = float(between.mean()), float(between.std(ddof=1))
        row["between_person_mean"] = b_mean
        row["between_person_sd"] = b_sd
        row["between_person_rsd"] = b_sd / abs(b_mean) if abs(b_mean) > 1e-12 else float("nan")

        for cname, views in CONDITIONS.items():
            per_subject = []
            for fid in subs:
                vals = [values[v][fid][s.id] for v in views
                        if fid in values[v] and s.id in values[v][fid]]
                if len(vals) < 2:
                    continue
                a = np.array(vals)
                per_subject.append((float(a.mean()), float(a.std(ddof=1))))
            if not per_subject:
                continue
            arr = np.array(per_subject)
            # Pooled within-person SD: root mean of the within-subject variances,
            # which is the right pooling for a spread rather than averaging SDs.
            within_sd = float(np.sqrt(np.mean(arr[:, 1] ** 2)))
            if s.unit is Unit.DEGREES:
                within = within_sd
                between_for_ratio = b_sd
            else:
                cvs = arr[:, 1] / np.abs(arr[:, 0])
                within = float(np.sqrt(np.mean(cvs ** 2)))
                between_for_ratio = row["between_person_rsd"]
            row[f"{cname}_within_person"] = within
            row[f"{cname}_within_person_sd_units"] = within_sd
            row[f"{cname}_n_subjects"] = int(arr.shape[0])
            row[f"{cname}_discriminability"] = (between_for_ratio / within
                                                if within > 0 else float("inf"))
            boot = []
            for _ in range(n_boot // 4):
                idx = r.integers(0, arr.shape[0], arr.shape[0])
                w = (float(np.sqrt(np.mean((arr[idx, 1] / np.abs(arr[idx, 0])) ** 2)))
                     if s.unit is not Unit.DEGREES
                     else float(np.sqrt(np.mean(arr[idx, 1] ** 2))))
                boot.append(between_for_ratio / w if w > 0 else np.nan)
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            row[f"{cname}_discriminability_ci_low"] = float(lo)
            row[f"{cname}_discriminability_ci_high"] = float(hi)
            row[f"{cname}_informative"] = bool(lo > 1.0)
            row[f"{cname}_reportable_cv_under_10pct"] = (
                bool(within < 0.10) if s.unit is not Unit.DEGREES else None)
        rows.append(row)

    # How many captures would pass the shipped pose gate at all?
    gate = []
    for view in det:
        for s in specs:
            p = np.array([poses[view][f] for f in poses[view]])
            if p.size == 0:
                continue
            worst = np.max(np.abs(p[:, :2]), axis=1)
            gate.append({
                "view": view, "id": s.id, "tolerance_deg": s.pose_tolerance_deg,
                "n": int(p.shape[0]),
                "frac_within_tolerance": float(np.mean(worst <= s.pose_tolerance_deg)),
                "frac_within_tolerance_pose_gated": float(np.mean(
                    (np.abs(p[:, 0]) + 4.977 <= s.pose_tolerance_deg)
                    & (np.abs(p[:, 1]) + 4.977 <= s.pose_tolerance_deg))),
            })

    front_gate = [g for g in gate if g["view"] == "neutral_front"]
    payload = {
        "arm": "9 -- test-retest repeatability",
        "question": "how much does a measurement move between two photographs of the "
                    "same person, measured rather than derived",
        "design": {"conditions": {k: list(v) for k, v in CONDITIONS.items()},
                   "excluded_views": list(EXCLUDED_VIEWS),
                   "landmarker": "MediaPipe FaceLandmarker v1 float16, Apache-2.0",
                   "scale": "iris diameter, core.scale.from_iris",
                   "n_bootstrap": n_boot // 4},
        "detection": detection,
        "table": rows,
        "pose_gate": gate,
        "summary": {
            "n_measurements": len(rows),
            "n_specs_with_measured_within_person_in_src": sum(
                1 for r_ in rows if r_["spec_measured_within_person_rsd"] is not None),
            "median_expression_cv": float(np.median(
                [r_["expression_within_person"] for r_ in rows if r_["unit"] != "deg"])),
            "median_all_usable_cv": float(np.median(
                [r_["all_usable_within_person"] for r_ in rows if r_["unit"] != "deg"])),
            "n_cv_over_10pct_expression": sum(
                1 for r_ in rows if r_.get("expression_reportable_cv_under_10pct") is False),
            "n_cv_over_10pct_all_usable": sum(
                1 for r_ in rows if r_.get("all_usable_reportable_cv_under_10pct") is False),
            "n_informative_expression": sum(1 for r_ in rows if r_.get("expression_informative")),
            "n_informative_all_usable": sum(1 for r_ in rows if r_.get("all_usable_informative")),
            "uninformative_expression": sorted(
                r_["id"] for r_ in rows if not r_.get("expression_informative")),
            "uninformative_all_usable": sorted(
                r_["id"] for r_ in rows if not r_.get("all_usable_informative")),
            "mean_frac_front_captures_within_tolerance": float(np.mean(
                [g["frac_within_tolerance"] for g in front_gate])),
            "mean_frac_front_captures_within_tolerance_pose_gated": float(np.mean(
                [g["frac_within_tolerance_pose_gated"] for g in front_gate])),
        },
    }
    write_json("arm09_test_retest", payload)
    write_csv("arm09_table", rows)
    write_csv("arm09_pose_gate", gate)
    return payload["summary"]


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=str))
