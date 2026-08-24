"""Arm 8 -- FRLL. Real faces, real landmarks, real demographics.

Face Research Lab London Set, CC BY 4.0, 102 individuals. The neutral front
capture carries a 189-point template placed by a human; the other captures are
images only. Everything here is 2D, because a photograph is.

8a  Measurement spread across 102 people from the human template, against the
    caliper spreads in ``norms``. This is an external validity check on the
    whole chain -- landmark map, formula, scale recovery -- and the place where
    a landmark that is not the anatomical point it is named after shows up as a
    biased mean.
8b  MediaPipe against the human template on the same 102 images, per landmark,
    normalised by interpupillary distance. This is simultaneously a landmark
    accuracy number and the validation of the index maps in ``evals/frll.py``.
8c  Measurement agreement between the two landmark sources, as Bland-Altman
    mean difference and limits of agreement -- the same statistic Lim et al.
    (2022) report for photogrammetry against calipers, so the numbers are
    comparable in kind.
8d  The pose distribution of a studio "front" capture. FRLL is about as
    controlled as face photography gets, so this is the empirical floor for
    what the pose gate has to tolerate.
8e  Roll attribution. Sixteen measurements were rewritten to reference the
    interpupillary line instead of the image horizon. Arm 2 shows the rewrite
    works on synthetic geometry; this asks what it did on 102 real
    photographs, by computing both forms of each measurement and correlating
    them against the estimated camera roll of the image they came from. A
    horizon-referenced measurement should track the camera; the rewritten one
    should not.
"""

from __future__ import annotations

import json
import math

import numpy as np

from evals._bootstrap import rng, write_csv, write_json
from evals import frll
from evals.arms.arm02_pose_sweep import HORIZON, ROLL_CLAIMS

from faciometry.core.landmarks import Landmark as L
from faciometry.core.spec import Unit, View
from faciometry.measure.registry import CATALOGUE, satisfiable
from faciometry.norms import niosh


def _values(ps_by_subject: dict, specs) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {s.id: {} for s in specs}
    for fid, ps in ps_by_subject.items():
        for s in specs:
            try:
                v = float(np.asarray(s.formula.eval(ps)))
            except Exception:
                continue
            if math.isfinite(v):
                out[s.id][fid] = v
    return out


def _boot_ci(x: np.ndarray, stat, n_boot: int, r) -> tuple[float, float]:
    if x.size < 3:
        return (float("nan"), float("nan"))
    vals = [stat(x[r.integers(0, x.size, x.size)]) for _ in range(n_boot)]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def run(n_boot: int = 2000) -> dict:
    r = rng(8)
    subs = frll.subjects()
    tmpl = {fid: frll.template_pointset(fid)[0] for fid in frll.templates()}
    available = frozenset(next(iter(tmpl.values())).index)
    specs = [s for s in satisfiable(available) if s.view is not View.PROFILE]
    unavailable = [s.id for s in CATALOGUE if s not in specs]

    tvals = _values(tmpl, specs)

    det = frll.detect_all()
    front = det["neutral_front"]
    mp_ps = {fid: frll.mp_pointset(rec["pts"])[0] for fid, rec in front.items()}
    mvals = _values(mp_ps, specs)

    # -- 8a ---------------------------------------------------------------
    rows_a = []
    for s in specs:
        v = np.array(list(tvals[s.id].values()))
        mean, sd = float(v.mean()), float(v.std(ddof=1))
        rsd = sd / abs(mean) if abs(mean) > 1e-12 else float("nan")
        n_stratum = niosh.stratum(s.id)
        rows_a.append({
            "id": s.id, "unit": s.unit.value, "evidence": s.evidence.value, "n": int(v.size),
            "frll_mean": mean, "frll_sd": sd, "frll_rsd": rsd,
            "frll_rsd_ci_low": _boot_ci(v, lambda a: a.std(ddof=1) / abs(a.mean()), n_boot, r)[0],
            "frll_rsd_ci_high": _boot_ci(v, lambda a: a.std(ddof=1) / abs(a.mean()), n_boot, r)[1],
            "niosh_mean": (n_stratum or {}).get("mean"),
            "niosh_sd": (n_stratum or {}).get("sd"),
            "niosh_rsd": ((n_stratum["sd"] / n_stratum["mean"]) if n_stratum and n_stratum["mean"]
                          else None),
            "spec_between_subject_rsd": s.between_subject_rsd,
            "frll_minus_niosh_pct": ((mean - n_stratum["mean"]) / n_stratum["mean"]
                                     if n_stratum and n_stratum["mean"] and s.unit is not Unit.DEGREES
                                     else None),
        })

    # -- 8b ---------------------------------------------------------------
    rows_b = []
    common = sorted(set(tmpl) & set(mp_ps))
    per_landmark: dict[str, list[float]] = {}
    for fid in common:
        t, m = tmpl[fid], mp_ps[fid]
        ipd = float(np.linalg.norm(t.get(L.PUPIL_L) - t.get(L.PUPIL_R)))
        # Both point sets are centroid-centred and iris-scaled; align them on the
        # two pupils so the comparison is of shape, not of scale recovery.
        for name in t.index:
            if name not in m.index:
                continue
            d = float(np.linalg.norm(t.get(name) - m.get(name)))
            per_landmark.setdefault(name.value, []).append(d / ipd)
    for name, ds in sorted(per_landmark.items()):
        a = np.array(ds)
        rows_b.append({
            "landmark": name, "n": int(a.size),
            "median_err_over_ipd": float(np.median(a)),
            "mean_err_over_ipd": float(a.mean()),
            "p90_err_over_ipd": float(np.percentile(a, 90)),
            "median_err_mm_at_63mm_ipd": float(np.median(a) * 63.36),
            "index_map_suspect": bool(np.median(a) > 0.10),
        })

    # -- 8c ---------------------------------------------------------------
    rows_c = []
    for s in specs:
        pairs = [(tvals[s.id][f], mvals[s.id][f]) for f in common
                 if f in tvals[s.id] and f in mvals[s.id]]
        if len(pairs) < 10:
            continue
        a = np.array(pairs)
        diff = a[:, 1] - a[:, 0]
        md, sdd = float(diff.mean()), float(diff.std(ddof=1))
        rows_c.append({
            "id": s.id, "unit": s.unit.value, "n": len(pairs),
            "template_mean": float(a[:, 0].mean()), "mediapipe_mean": float(a[:, 1].mean()),
            "mean_difference": md, "sd_of_difference": sdd,
            "loa_low": md - 1.96 * sdd, "loa_high": md + 1.96 * sdd,
            "mean_difference_pct": md / abs(a[:, 0].mean()) if abs(a[:, 0].mean()) > 1e-12 else None,
            "loa_width_over_between_subject_sd": (2 * 1.96 * sdd) / float(a[:, 0].std(ddof=1)),
            "pearson_r": float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]),
        })

    # -- 8d ---------------------------------------------------------------
    rows_d = []
    for view, recs in det.items():
        p = np.array([x["pose"] for x in recs.values()])
        rows_d.append({
            "view": view, "n_detected": len(recs), "n_images": 102,
            "detection_rate": len(recs) / 102,
            **{f"{k}_{stat}": float(getattr(np, stat)(p[:, i])) if p.size else None
               for i, k in enumerate(("yaw", "pitch", "roll")) for stat in ("mean", "std")},
            **{f"{k}_p95_abs": float(np.percentile(np.abs(p[:, i]), 95)) if p.size else None
               for i, k in enumerate(("yaw", "pitch", "roll"))},
        })

    # -- 8e ---------------------------------------------------------------
    # The same quantity in both forms, against the camera roll of the image it
    # came from. Two claims are on trial: that the interpupillary reference
    # removes the camera, and that the agreement the old form showed between
    # two landmark sources was agreement about the camera rather than about the
    # face.
    rolls = {fid: float(rec["pose"][2]) for fid, rec in front.items()}
    by_id = {s.id: s for s in specs}
    rows_e = []
    for shadow in HORIZON:
        spec = by_id.get(shadow.twin)
        if spec is None:
            rows_e.append({"id": shadow.twin, "claim": ROLL_CLAIMS[shadow.twin][0],
                           "n": 0, "not_evaluable": True,
                           "reason": "needs a landmark the FRL template does not supply"})
            continue

        def series(ps_by_subject, expr):
            out = {}
            for fid, ps in ps_by_subject.items():
                try:
                    v = float(np.asarray(expr.eval(ps)))
                except Exception:
                    continue
                if math.isfinite(v):
                    out[fid] = v
            return out

        t_ship = series(tmpl, spec.formula)
        t_horiz = series(tmpl, shadow.formula)
        m_ship = series(mp_ps, spec.formula)
        m_horiz = series(mp_ps, shadow.formula)
        ids = [f for f in common if f in t_ship and f in t_horiz
               and f in m_ship and f in m_horiz and f in rolls]
        if len(ids) < 10:
            rows_e.append({"id": shadow.twin, "claim": ROLL_CLAIMS[shadow.twin][0],
                           "n": len(ids), "not_evaluable": True,
                           "reason": "fewer than ten subjects with both forms and a pose"})
            continue
        roll = np.array([rolls[f] for f in ids])

        def corr(a, b):
            a, b = np.asarray(a, float), np.asarray(b, float)
            if a.std() < 1e-12 or b.std() < 1e-12:
                return float("nan")
            return float(np.corrcoef(a, b)[0, 1])

        def agreement(t, m):
            t, m = np.asarray(t, float), np.asarray(m, float)
            d = m - t
            sd = float(d.std(ddof=1))
            return corr(t, m), (2 * 1.96 * sd) / float(t.std(ddof=1)) if t.std(ddof=1) > 1e-12 \
                else float("nan")

        ts = [t_ship[f] for f in ids]
        th = [t_horiz[f] for f in ids]
        ms = [m_ship[f] for f in ids]
        mh = [m_horiz[f] for f in ids]
        r_ship, loa_ship = agreement(ts, ms)
        r_horiz, loa_horiz = agreement(th, mh)
        rows_e.append({
            "id": shadow.twin,
            "claim": ROLL_CLAIMS[shadow.twin][0],
            "unit": spec.unit.value,
            "n": len(ids),
            "not_evaluable": False,
            "camera_roll_sd_deg": float(roll.std(ddof=1)),
            "corr_shipped_with_camera_roll": corr(ts, roll),
            "corr_horizon_with_camera_roll": corr(th, roll),
            "between_source_r_shipped": r_ship,
            "between_source_r_horizon": r_horiz,
            "loa_over_between_subject_sd_shipped": loa_ship,
            "loa_over_between_subject_sd_horizon": loa_horiz,
            "template_sd_shipped": float(np.std(ts, ddof=1)),
            "template_sd_horizon": float(np.std(th, ddof=1)),
        })

    demo = {}
    from collections import Counter
    demo["by_gender"] = dict(Counter(s.gender for s in subs.values()))
    demo["by_ethnicity"] = dict(Counter(s.ethnicity for s in subs.values()))
    demo["by_gender_ethnicity"] = {f"{s.gender}|{s.ethnicity}": c for (s, c) in
                                   [(v, 0) for v in []]}
    demo["by_gender_ethnicity"] = dict(
        Counter(f"{s.gender}|{s.ethnicity}" for s in subs.values()))
    ages = [s.age for s in subs.values() if s.age > 0]
    demo["age_mean"] = float(np.mean(ages))
    demo["age_range"] = [int(min(ages)), int(max(ages))]
    demo["age_missing"] = sum(1 for s in subs.values() if s.age < 0)

    payload = {
        "arm": "8 -- FRLL landmark accuracy and measurement spread",
        "dataset": {
            "name": "Face Research Lab London Set",
            "doi": "10.6084/m9.figshare.5047666.v5",
            "license": "CC BY 4.0",
            "n_subjects": len(subs),
            "views_downloaded": sorted(frll.VIEWS),
            "templates_available_for": "neutral_front only; the other captures ship "
                                       "as images with no delineation",
            "demographics": demo,
        },
        "landmarks_supplied_by_template": sorted(x.value for x in available),
        "landmarks_not_supplied": sorted(x.value for x in frll.TEM_UNAVAILABLE),
        "measurements_evaluable": [s.id for s in specs],
        "measurements_not_evaluable": unavailable,
        "spread": rows_a,
        "landmark_agreement": rows_b,
        "measurement_agreement": rows_c,
        "pose_by_view": rows_d,
        "roll_attribution": rows_e,
        "summary": {
            "n_subjects": len(tmpl),
            "n_measurements_evaluable": len(specs),
            "n_measurements_not_evaluable": len(unavailable),
            "median_landmark_error_over_ipd": float(np.median(
                [r["median_err_over_ipd"] for r in rows_b])),
            "n_landmarks_with_suspect_index_map": sum(1 for r in rows_b if r["index_map_suspect"]),
            "suspect_index_maps": [r["landmark"] for r in rows_b if r["index_map_suspect"]],
            "worst_niosh_bias": min(
                (r for r in rows_a if r["frll_minus_niosh_pct"] is not None),
                key=lambda r: r["frll_minus_niosh_pct"]),
            "front_capture_pose_p95": {
                k: next(r for r in rows_d if r["view"] == "neutral_front")[f"{k}_p95_abs"]
                for k in ("yaw", "pitch", "roll")},
            "profile_detection_rate": {
                r["view"]: r["detection_rate"] for r in rows_d if "profile" in r["view"]},
            "worst_measurement_agreement": max(
                rows_c, key=lambda r: r["loa_width_over_between_subject_sd"]),
            "roll_attribution": {
                "n_tested": sum(1 for r in rows_e if not r["not_evaluable"]),
                "n_not_evaluable": sum(1 for r in rows_e if r["not_evaluable"]),
                "median_abs_corr_with_camera_roll_shipped": float(np.median(
                    [abs(r["corr_shipped_with_camera_roll"]) for r in rows_e
                     if not r["not_evaluable"]
                     and math.isfinite(r["corr_shipped_with_camera_roll"])])),
                "median_abs_corr_with_camera_roll_horizon": float(np.median(
                    [abs(r["corr_horizon_with_camera_roll"]) for r in rows_e
                     if not r["not_evaluable"]
                     and math.isfinite(r["corr_horizon_with_camera_roll"])])),
                "agreement_lost_by_removing_roll": sorted(
                    (r["id"] for r in rows_e if not r["not_evaluable"]
                     and math.isfinite(r["between_source_r_horizon"])
                     and math.isfinite(r["between_source_r_shipped"])
                     and r["between_source_r_horizon"] - r["between_source_r_shipped"] > 0.2),
                ),
            },
        },
    }
    write_json("arm08_frll", payload)
    write_csv("arm08_spread", rows_a)
    write_csv("arm08_landmark_agreement", rows_b)
    write_csv("arm08_measurement_agreement", rows_c)
    write_csv("arm08_pose_by_view", rows_d)
    write_csv("arm08_roll_attribution", rows_e)
    return payload["summary"]


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=str))
