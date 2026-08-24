"""Arm 6 -- NIOSH 2003 external check.

Vitruve's discriminability gate divides by a between-subject spread taken from
``norms/data/niosh2003.json``. That file is a derived artifact: a build script
read a CSV and wrote summary statistics. This arm reads the same CSV
independently and checks four things.

6a  Reproduction. Recompute every stratum from the raw 3,997-subject CSV and
    compare against the vendored JSON, cell by cell.
6b  Units. NIOSH's ``INTPUPBR`` against Dodgson's ANSUR-derived interpupillary
    distance of 63.36 mm. ANSUR II records ``interpupillarybreadth`` in tenths
    of a millimetre; if a build script ever reads that column as millimetres it
    will produce a 633 mm interpupillary distance and a scale factor ten times
    too small. The check here is what NIOSH's own units are and whether they
    agree with the value ``core/scale.py`` ships.
6c  Distribution shape. ``norms`` stores a mean and a standard deviation, which
    is a normal model. Whether that is defensible is testable: skewness,
    excess kurtosis, and the D'Agostino-Pearson and Anderson-Darling tests per
    stratum, plus the empirical coverage of the normal 5th and 95th
    percentiles.
6d  Pooling. ``niosh.spread()`` falls back to ``both|pooled`` for a subject who
    has not declared a sex. Pooling two populations with different means
    inflates the standard deviation, and that standard deviation is the
    numerator of the discriminability ratio -- so the fallback makes every
    measurement look *more* discriminative, not less.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
from scipy import stats

from evals._bootstrap import write_csv, write_json

from vitruve.norms import niosh
from vitruve.core.scale import IPD_PRIORS

CSV_PATHS = (
    Path("/tmp/niosh/rd-10130-2020-0/datasets/RespiratorUsersData-all-subjects.csv"),
    Path(__file__).resolve().parents[1] / "data" / "RespiratorUsersData-all-subjects.csv",
)

SEX = {"1": "male", "2": "female"}
RACE = {"1": "White", "2": "Black", "3": "Hispanic", "4": "Other"}
AGE = {"1": "17-29", "2": "30-44", "3": "45+"}

COLUMN_TO_MEASUREMENT = {
    "INTPUPBR": "interpupillary_distance",
    "BIZYGOBR": "bizygomatic_width",
    "BIGONLBR": "bigonial_width",
    "NOSEBR": "nose_breadth",
    "LIPLTH": "mouth_width",
    "SUBNASAL": "nose_height",
    "MENSELL": "face_height_sellion_menton",
}
RATIOS = {
    "jaw_cheekbone_ratio": ("BIGONLBR", "BIZYGOBR"),
    "nose_mouth_width_ratio": ("NOSEBR", "LIPLTH"),
}

DEFAULT_LINEAR_RSD = 0.06


def _num(raw: str) -> float | None:
    s = raw.strip().strip('"').replace(",", "")
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return None if v <= -9000 else v


def _rows() -> list[dict]:
    for p in CSV_PATHS:
        if p.exists():
            rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
            return [r for r in rows if r.get("SUBNO", "").strip()]
    raise FileNotFoundError(f"NIOSH CSV not found in any of {CSV_PATHS}")


def _buckets(rows, selector):
    out: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        v = selector(r)
        if v is None:
            continue
        sex = SEX.get(r["SEX"].strip(), "unknown")
        race = RACE.get(r["RACEGRP"].strip(), "unknown")
        out.setdefault(("both", "pooled"), []).append(v)
        out.setdefault((sex, "pooled"), []).append(v)
        out.setdefault((sex, race), []).append(v)
    return out


def _selectors():
    sels = {mid: (lambda r, c=col: _num(r[c])) for col, mid in COLUMN_TO_MEASUREMENT.items()}
    for mid, (n, d) in RATIOS.items():
        def sel(r, n=n, d=d):
            a, b = _num(r[n]), _num(r[d])
            return a / b if a is not None and b else None
        sels[mid] = sel
    return sels


def reproduce(rows) -> tuple[list[dict], dict]:
    vendored = json.loads(
        (Path(niosh.__file__).parent / "data" / "niosh2003.json").read_text())
    out, mismatches = [], 0
    for mid, sel in _selectors().items():
        buckets = _buckets(rows, sel)
        for (sex, race), vals in sorted(buckets.items()):
            if len(vals) < 30:
                continue
            key = f"{sex}|{race}"
            mine = {"n": len(vals), "mean": round(statistics.fmean(vals), 4),
                    "sd": round(statistics.stdev(vals), 4)}
            theirs = vendored["strata"].get(mid, {}).get(key)
            ok = theirs is not None and all(
                abs(mine[k] - theirs[k]) <= (0 if k == "n" else 1e-4) for k in mine)
            if not ok:
                mismatches += 1
            out.append({"id": mid, "stratum": key, **{f"recomputed_{k}": v for k, v in mine.items()},
                        **{f"vendored_{k}": (theirs or {}).get(k) for k in mine},
                        "matches": bool(ok),
                        "rsd": mine["sd"] / mine["mean"] if mine["mean"] else None})
    meta = {"n_cells": len(out), "n_mismatched": mismatches,
            "vendored_n_subjects": vendored["n_subjects"], "recomputed_n_subjects": len(rows),
            "vendored_measurements": sorted(vendored["strata"]),
            "recomputed_measurements": sorted(_selectors())}
    return out, meta


def units(rows) -> dict:
    """Is INTPUPBR millimetres, and does it agree with the prior in scale.py?"""
    ipd = [v for v in (_num(r["INTPUPBR"]) for r in rows) if v is not None]
    others = {c: [v for v in (_num(r[c]) for r in rows) if v is not None]
              for c in ("BIZYGOBR", "BIGONLBR", "NOSEBR", "HEADCIRC", "STATURE")}
    integral = {c: float(np.mean([float(v).is_integer() for v in vs]))
                for c, vs in others.items()}
    integral["INTPUPBR"] = float(np.mean([float(v).is_integer() for v in ipd]))
    prior_mean, prior_sd = IPD_PRIORS[None]
    male = [_num(r["INTPUPBR"]) for r in rows if r["SEX"].strip() == "1"]
    female = [_num(r["INTPUPBR"]) for r in rows if r["SEX"].strip() == "2"]
    male = [v for v in male if v is not None]
    female = [v for v in female if v is not None]
    return {
        "niosh_INTPUPBR_mean": float(np.mean(ipd)),
        "niosh_INTPUPBR_sd": float(np.std(ipd, ddof=1)),
        "niosh_INTPUPBR_min": float(np.min(ipd)),
        "niosh_INTPUPBR_max": float(np.max(ipd)),
        "niosh_INTPUPBR_male_mean": float(np.mean(male)),
        "niosh_INTPUPBR_female_mean": float(np.mean(female)),
        "vitruve_IPD_PRIORS_pooled": [prior_mean, prior_sd],
        "vitruve_IPD_PRIORS_male": list(IPD_PRIORS["male"]),
        "vitruve_IPD_PRIORS_female": list(IPD_PRIORS["female"]),
        "difference_pooled_mm": float(np.mean(ipd)) - prior_mean,
        "difference_pooled_pct": (float(np.mean(ipd)) - prior_mean) / prior_mean,
        "fraction_of_values_that_are_integers": integral,
        "conclusion": (
            "NIOSH INTPUPBR is millimetres: the mean is "
            f"{np.mean(ipd):.2f} against Dodgson's ANSUR 63.36 mm, the values carry "
            "a decimal place where every other facial column in the same file is a "
            "whole number of millimetres, and the range is "
            f"{np.min(ipd):.1f} to {np.max(ipd):.1f}. ANSUR II's "
            "interpupillarybreadth column is tenths of a millimetre, so the same "
            "quantity there reads about 633; reading it as millimetres would give "
            "a face ten times too large and, because Vitruve recovers scale from "
            "interpupillary distance, a mm-per-pixel ten times too small."
        ),
        "ansur_ii_cross_check": "NOT RUN -- see docs/EVALUATION.md, arm 6",
    }


def shape(rows) -> list[dict]:
    out = []
    for mid, sel in _selectors().items():
        buckets = _buckets(rows, sel)
        for (sex, race), vals in sorted(buckets.items()):
            if len(vals) < 100:
                continue
            a = np.asarray(vals, dtype=float)
            mu, sd = a.mean(), a.std(ddof=1)
            z = (a - mu) / sd
            try:
                k2, p_norm = stats.normaltest(a)
            except Exception:
                k2, p_norm = float("nan"), float("nan")
            ad = stats.anderson(a, dist="norm")
            out.append({
                "id": mid, "stratum": f"{sex}|{race}", "n": len(vals),
                "mean": float(mu), "sd": float(sd),
                "rsd": float(sd / mu) if mu else None,
                "skew": float(stats.skew(a)),
                "excess_kurtosis": float(stats.kurtosis(a)),
                "dagostino_k2": float(k2), "dagostino_p": float(p_norm),
                "anderson_stat": float(ad.statistic),
                "anderson_crit_5pct": float(ad.critical_values[2]),
                "rejects_normal_at_5pct": bool(ad.statistic > ad.critical_values[2]),
                "empirical_below_normal_p05": float(np.mean(z < -1.6449)),
                "empirical_above_normal_p95": float(np.mean(z > 1.6449)),
                "empirical_p05_p95_coverage": float(np.mean(np.abs(z) < 1.6449)),
            })
    return out


def pooling(rows) -> list[dict]:
    out = []
    for mid, sel in _selectors().items():
        buckets = _buckets(rows, sel)
        pooled = buckets.get(("both", "pooled"), [])
        male = buckets.get(("male", "pooled"), [])
        female = buckets.get(("female", "pooled"), [])
        if not (pooled and male and female):
            continue
        p, m, f = (np.asarray(x, dtype=float) for x in (pooled, male, female))
        # eta squared for sex: between-group sum of squares over total
        grand = p.mean()
        ss_between = len(m) * (m.mean() - grand) ** 2 + len(f) * (f.mean() - grand) ** 2
        ss_total = float(((p - grand) ** 2).sum())
        within_sd = math.sqrt(
            ((len(m) - 1) * m.var(ddof=1) + (len(f) - 1) * f.var(ddof=1))
            / (len(m) + len(f) - 2))
        out.append({
            "id": mid,
            "pooled_mean": float(p.mean()), "pooled_sd": float(p.std(ddof=1)),
            "pooled_rsd": float(p.std(ddof=1) / p.mean()),
            "male_rsd": float(m.std(ddof=1) / m.mean()),
            "female_rsd": float(f.std(ddof=1) / f.mean()),
            "within_sex_sd": within_sd,
            "within_sex_rsd": within_sd / float(p.mean()),
            "pooled_over_within_sex": float(p.std(ddof=1)) / within_sd,
            "eta_squared_sex": ss_between / ss_total,
            "vitruve_spread_default": niosh.spread(mid),
            "vitruve_spread_male": niosh.spread(mid, sex="male"),
            "vitruve_spread_female": niosh.spread(mid, sex="female"),
            "default_over_within_sex": (niosh.spread(mid) or float("nan")) / (within_sd / float(p.mean())),
        })
    return out


def run() -> dict:
    rows = _rows()
    rep, meta = reproduce(rows)
    un = units(rows)
    sh = shape(rows)
    po = pooling(rows)

    linear = [r for r in sh if r["id"] not in RATIOS]
    rsds = [r["rsd"] for r in linear if r["stratum"] != "both|pooled"]

    payload = {
        "arm": "6 -- NIOSH 2003 external check",
        "question": "are the between-subject spreads Vitruve divides by reproducible "
                    "from the raw survey, in the units claimed, and normal in shape",
        "reproduction": rep,
        "reproduction_meta": meta,
        "units": un,
        "shape": sh,
        "pooling": po,
        "summary": {
            "n_subjects": len(rows),
            "n_strata_cells": meta["n_cells"],
            "n_mismatched_cells": meta["n_mismatched"],
            "vendored_json_reproduces_exactly": meta["n_mismatched"] == 0,
            "ipd_mean_mm": un["niosh_INTPUPBR_mean"],
            "ipd_vs_scale_py_prior_pct": un["difference_pooled_pct"],
            "n_strata_tested_for_normality": len(sh),
            "n_rejecting_normal_at_5pct": sum(1 for r in sh if r["rejects_normal_at_5pct"]),
            "max_abs_skew": max(abs(r["skew"]) for r in sh),
            "worst_skew_stratum": max(sh, key=lambda r: abs(r["skew"]))["id"] + " " +
                                  max(sh, key=lambda r: abs(r["skew"]))["stratum"],
            "within_sex_linear_rsd_min": min(rsds),
            "within_sex_linear_rsd_max": max(rsds),
            "within_sex_linear_rsd_median": float(np.median(rsds)),
            "default_linear_rsd_in_published_py": DEFAULT_LINEAR_RSD,
            "default_rsd_inside_measured_range": bool(min(rsds) <= DEFAULT_LINEAR_RSD <= max(rsds)),
            "max_pooled_over_within_sex": max(r["pooled_over_within_sex"] for r in po),
            "worst_pooling_inflation": max(po, key=lambda r: r["pooled_over_within_sex"])["id"],
            "max_eta_squared_sex": max(r["eta_squared_sex"] for r in po),
        },
    }
    write_json("arm06_niosh", payload)
    write_csv("arm06_reproduction", rep)
    write_csv("arm06_shape", sh)
    write_csv("arm06_pooling", po)
    return payload["summary"]


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=str))
