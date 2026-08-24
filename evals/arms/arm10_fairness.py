"""Arm 10 -- fairness stratification, and a straight statement of its limits.

FRLL is 102 people and 68 percent white. Any per-group number here rests on
cells of four to thirty-five people, so every estimate is reported with a
bootstrap interval and the intervals are wide enough that most pairwise
comparisons do not separate. That is the finding. This arm is not powered to
settle whether Vitruve measures some groups worse than others; it is powered to
say how far from settling it we are, and to make the gap visible instead of
absent.

Three quantities are stratified by self-reported ethnicity and by self-reported
gender:

* landmark agreement between MediaPipe and the human template (arm 8b),
* between-person measurement spread (arm 8a),
* within-person repeatability across captures (arm 9).

A fourth number is the one that matters most for a normative report: the mean
percentile a group would receive against the pooled NIOSH stratum. If a group's
mean percentile is far from 0.5, the report is telling every member of that
group the same thing about themselves, and that is a property of the reference
population rather than of them.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict

import numpy as np
from scipy import stats

from evals._bootstrap import rng, write_csv, write_json
from evals import frll

from vitruve.core.landmarks import Landmark as L
from vitruve.core.spec import Unit, View
from vitruve.measure.registry import satisfiable
from vitruve.norms import niosh

MIN_CELL = 4
N_BOOT = 2000

VIEWS_USABLE = ("neutral_front", "smiling_front", "neutral_left_3quarter",
                "neutral_right_3quarter", "smiling_left_3quarter")


def _ci(x, stat, r, n_boot=N_BOOT):
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return (float("nan"), float("nan"))
    vals = [stat(x[r.integers(0, x.size, x.size)]) for _ in range(n_boot)]
    return tuple(float(v) for v in np.percentile(vals, [2.5, 97.5]))


def run() -> dict:
    r = rng(10)
    subs = frll.subjects()
    det = frll.detect_all()
    tmpl = {fid: frll.template_pointset(fid)[0] for fid in frll.templates()}
    probe = next(iter(tmpl.values()))
    specs = [s for s in satisfiable(frozenset(probe.index)) if s.view is not View.PROFILE]

    def strata(kind):
        out = defaultdict(list)
        for fid, s in subs.items():
            out[s.ethnicity if kind == "ethnicity" else s.gender].append(fid)
        return {k: v for k, v in out.items() if len(v) >= MIN_CELL}

    # measurement values per source
    tvals = {fid: {s.id: float(np.asarray(s.formula.eval(ps))) for s in specs}
             for fid, ps in tmpl.items()}
    mvals = {}
    for view in VIEWS_USABLE:
        mvals[view] = {}
        for fid, rec in det[view].items():
            ps, _ = frll.mp_pointset(rec["pts"])
            vals = {}
            for s in specs:
                try:
                    v = float(np.asarray(s.formula.eval(ps)))
                except Exception:
                    continue
                if math.isfinite(v):
                    vals[s.id] = v
            mvals[view][fid] = vals

    # landmark agreement per subject
    lm_err = {}
    for fid, t in tmpl.items():
        rec = det["neutral_front"].get(fid)
        if rec is None:
            continue
        m, _ = frll.mp_pointset(rec["pts"])
        ipd = float(np.linalg.norm(t.get(L.PUPIL_L) - t.get(L.PUPIL_R)))
        ds = [float(np.linalg.norm(t.get(n) - m.get(n))) / ipd
              for n in t.index if n in m.index]
        lm_err[fid] = float(np.median(ds))

    rows = []
    for kind in ("ethnicity", "gender"):
        for group, fids in sorted(strata(kind).items()):
            errs = np.array([lm_err[f] for f in fids if f in lm_err])
            lo, hi = _ci(errs, np.median, r)
            base = {"stratum_kind": kind, "group": group, "n_subjects": len(fids),
                    "landmark_median_err_over_ipd": float(np.median(errs)),
                    "landmark_err_ci_low": lo, "landmark_err_ci_high": hi}
            for s in specs:
                between = np.array([tvals[f][s.id] for f in fids if f in tvals])
                if between.size < MIN_CELL:
                    continue
                mean = float(between.mean())
                if s.unit is Unit.DEGREES:
                    spread = float(between.std(ddof=1))
                    stat = lambda a: a.std(ddof=1)
                else:
                    spread = float(between.std(ddof=1) / abs(mean))
                    stat = lambda a: a.std(ddof=1) / abs(a.mean())
                slo, shi = _ci(between, stat, r)
                cvs = []
                for f in fids:
                    vals = [mvals[v][f][s.id] for v in VIEWS_USABLE
                            if f in mvals[v] and s.id in mvals[v][f]]
                    if len(vals) < 2:
                        continue
                    a = np.array(vals)
                    cvs.append(a.std(ddof=1) if s.unit is Unit.DEGREES
                               else a.std(ddof=1) / abs(a.mean()))
                cv = float(np.sqrt(np.mean(np.square(cvs)))) if cvs else float("nan")
                clo, chi = _ci(np.array(cvs), lambda a: float(np.sqrt(np.mean(np.square(a)))), r)
                pct = None
                st = niosh.stratum(s.id)
                if st and st["sd"]:
                    pct = float(np.mean(stats.norm.cdf((between - st["mean"]) / st["sd"])))
                rows.append({**base, "id": s.id, "unit": s.unit.value, "n": int(between.size),
                             "mean": mean, "between_spread": spread,
                             "between_spread_ci_low": slo, "between_spread_ci_high": shi,
                             "within_person_cv": cv,
                             "within_person_cv_ci_low": clo, "within_person_cv_ci_high": chi,
                             "discriminability": spread / cv if cv and cv > 0 else None,
                             "mean_percentile_vs_pooled_niosh": pct})
    # Pairwise overlap: does any group pair separate on landmark error?
    pairs = []
    for kind in ("ethnicity", "gender"):
        groups = sorted(strata(kind))
        for i, a in enumerate(groups):
            for b in groups[i + 1:]:
                ea = np.array([lm_err[f] for f in strata(kind)[a] if f in lm_err])
                eb = np.array([lm_err[f] for f in strata(kind)[b] if f in lm_err])
                u = stats.mannwhitneyu(ea, eb, alternative="two-sided")
                pairs.append({"stratum_kind": kind, "a": a, "b": b,
                              "n_a": int(ea.size), "n_b": int(eb.size),
                              "median_a": float(np.median(ea)), "median_b": float(np.median(eb)),
                              "mannwhitney_u": float(u.statistic), "p": float(u.pvalue),
                              "separates_at_5pct": bool(u.pvalue < 0.05)})

    ethn_rows = [r_ for r_ in rows if r_["stratum_kind"] == "ethnicity"]
    payload = {
        "arm": "10 -- fairness stratification",
        "question": "does measurement quality differ by self-reported ethnicity or "
                    "gender, and is this sample capable of answering that",
        "design": {"min_cell": MIN_CELL, "n_bootstrap": N_BOOT,
                   "views": list(VIEWS_USABLE),
                   "power_statement": "FRLL is 102 people and 68 percent white. Cells "
                                      "run from 4 to 35 subjects. This arm cannot "
                                      "settle the question and does not claim to."},
        "cell_sizes": {k: {g: len(v) for g, v in strata(k).items()}
                       for k in ("ethnicity", "gender")},
        "table": rows,
        "pairwise_landmark_error": pairs,
        "summary": {
            "n_rows": len(rows),
            "cells": {k: {g: len(v) for g, v in strata(k).items()}
                      for k in ("ethnicity", "gender")},
            "landmark_error_by_group": {
                f"{r_['stratum_kind']}:{r_['group']}": {
                    "median": r_["landmark_median_err_over_ipd"],
                    "ci": [r_["landmark_err_ci_low"], r_["landmark_err_ci_high"]],
                    "n": r_["n_subjects"]}
                for r_ in {(x["stratum_kind"], x["group"]): x for x in rows}.values()},
            "n_pairwise_comparisons": len(pairs),
            "n_pairs_separating_at_5pct": sum(1 for p in pairs if p["separates_at_5pct"]),
            "pairs_separating": [f"{p['stratum_kind']}: {p['a']} vs {p['b']} (p={p['p']:.3g})"
                                 for p in pairs if p["separates_at_5pct"]],
            # The absolute offset from 0.5 is dominated by the landmark-definition
            # bias found in arm 8a, which is common to every group. The spread
            # *between* groups is the part that is about the groups.
            "group_percentile_spread": {
                mid: {
                    "min": min(v), "max": max(v), "spread": max(v) - min(v),
                    "by_group": g,
                } for mid, (v, g) in {
                    mid: (
                        [r_["mean_percentile_vs_pooled_niosh"] for r_ in ethn_rows
                         if r_["id"] == mid and r_["mean_percentile_vs_pooled_niosh"] is not None],
                        {r_["group"]: r_["mean_percentile_vs_pooled_niosh"] for r_ in ethn_rows
                         if r_["id"] == mid and r_["mean_percentile_vs_pooled_niosh"] is not None},
                    ) for mid in sorted({r_["id"] for r_ in ethn_rows})
                }.items() if v},
            "worst_group_percentile_offset": max(
                (r_ for r_ in ethn_rows if r_["mean_percentile_vs_pooled_niosh"] is not None),
                key=lambda r_: abs(r_["mean_percentile_vs_pooled_niosh"] - 0.5)),
        },
    }
    write_json("arm10_fairness", payload)
    write_csv("arm10_table", rows)
    write_csv("arm10_pairwise", pairs)
    return payload["summary"]


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=str))
