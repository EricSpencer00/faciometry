"""Arm 7 -- negative control on the normative model.

Shuffle identities against measurements and the percentile output must not
notice. If it does, the normative model is carrying something other than
(value, stratum) -- which for a mean-and-sd model would mean the reference
distribution has memorised the individuals it was built from.

There is no percentile function in ``src/`` yet: ``norms.niosh`` exposes
``stratum()`` and ``spread()`` and stops there. So this arm implements the
percentile the norms module *implies* -- a normal CDF against the narrowest
available stratum -- and tests that. Where the shipped code later grows a
percentile function, this arm is the test it has to pass.

Five conditions:

7a  ``identity``       every subject against their own (sex, ancestry) stratum.
                       Positive control on calibration: should be uniform, and
                       departs from uniform exactly as far as the normal model
                       departs from the empirical distribution (arm 6).
7b  ``shuffle_within`` measurement values permuted inside each stratum. The
                       percentile *distribution* must be bit-identical: it is a
                       permutation of the same set of numbers. Any change is a
                       leak.
7c  ``shuffle_across`` demographic labels permuted across the whole sample.
                       Non-uniformity here is not a leak -- it is the proof
                       that stratification carries information, i.e. the
                       positive control that the measurement has power.
7d  ``pooled``         every subject against ``both|pooled``, which is what
                       ``niosh.spread()`` falls back to when the subject has not
                       declared a sex. The male and female percentile
                       distributions should then separate, and how far they
                       separate is how much a declared sex is worth.
7e  ``loo``            leave-one-out strata, since a subject who contributed to
                       the reference distribution is being compared against
                       themselves.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from evals._bootstrap import rng, write_csv, write_json

from faciometry.norms import niosh

from .arm06_niosh import COLUMN_TO_MEASUREMENT, RACE, RATIOS, SEX, _num, _rows


def _subject_values() -> dict[str, list[tuple[str, str, float]]]:
    """measurement id -> [(sex, ancestry, value)]"""
    rows = _rows()
    out: dict[str, list[tuple[str, str, float]]] = {}
    for col, mid in COLUMN_TO_MEASUREMENT.items():
        out[mid] = [
            (SEX.get(r["SEX"].strip(), "unknown"), RACE.get(r["RACEGRP"].strip(), "unknown"), v)
            for r in rows if (v := _num(r[col])) is not None
        ]
    for mid, (n, d) in RATIOS.items():
        vals = []
        for r in rows:
            a, b = _num(r[n]), _num(r[d])
            if a is None or not b:
                continue
            vals.append((SEX.get(r["SEX"].strip(), "unknown"),
                         RACE.get(r["RACEGRP"].strip(), "unknown"), a / b))
        out[mid] = vals
    return out


def _percentile(mid: str, sex: str, anc: str, value: float, *, pooled: bool = False) -> float | None:
    s = niosh.stratum(mid, sex=None if pooled else sex, ancestry=None if pooled else anc)
    if not s or not s["sd"]:
        return None
    return float(stats.norm.cdf((value - s["mean"]) / s["sd"]))


def _uniformity(p: np.ndarray) -> dict:
    ks = stats.kstest(p, "uniform")
    counts, _ = np.histogram(p, bins=10, range=(0, 1))
    chi = stats.chisquare(counts)
    return {
        "n": int(p.size),
        "mean": float(p.mean()),
        "sd": float(p.std(ddof=1)),
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "chi2_decile": float(chi.statistic),
        "chi2_p": float(chi.pvalue),
        "decile_counts": counts.tolist(),
        "expected_per_decile": float(p.size / 10),
        "uniform_at_1pct": bool(ks.pvalue > 0.01),
    }


def run() -> dict:
    data = _subject_values()
    r = rng(7)
    rows, detail = [], {}

    for mid, recs in data.items():
        if not niosh.covers(mid):
            continue
        sexes = [x[0] for x in recs]
        ancs = [x[1] for x in recs]
        vals = np.array([x[2] for x in recs])

        def percentiles(v, sx, an, pooled=False):
            out = [_percentile(mid, a, b, c, pooled=pooled) for a, b, c in zip(sx, an, v)]
            return np.array([x for x in out if x is not None])

        ident = percentiles(vals, sexes, ancs)

        # 7b -- permute values inside each stratum
        shuffled = vals.copy()
        keys = [f"{a}|{b}" for a, b in zip(sexes, ancs)]
        # sorted(), not set(): iterating a set of strings walks them in
        # PYTHONHASHSEED order, so the draws from `r` came off the stream in a
        # different sequence on every process and this arm was the one part of
        # the harness whose "a rerun reproduces byte-identical output" claim
        # was false. The leak-free conclusion never depended on which
        # permutation was drawn; the published across-shuffle KS numbers did.
        for k in sorted(set(keys)):
            idx = np.array([i for i, kk in enumerate(keys) if kk == k])
            shuffled[idx] = vals[r.permutation(idx)]
        within = percentiles(shuffled, sexes, ancs)

        # 7c -- permute demographic labels across the whole sample
        perm = r.permutation(len(vals))
        across = percentiles(vals, [sexes[i] for i in perm], [ancs[i] for i in perm])

        # 7d -- pooled stratum, the undeclared-sex fallback
        pooled = percentiles(vals, sexes, ancs, pooled=True)
        pooled_m = percentiles(vals[np.array(sexes) == "male"],
                               ["male"] * sum(s == "male" for s in sexes),
                               [a for a, s in zip(ancs, sexes) if s == "male"], pooled=True)
        pooled_f = percentiles(vals[np.array(sexes) == "female"],
                               ["female"] * sum(s == "female" for s in sexes),
                               [a for a, s in zip(ancs, sexes) if s == "female"], pooled=True)
        strat_m = percentiles(vals[np.array(sexes) == "male"],
                              ["male"] * sum(s == "male" for s in sexes),
                              [a for a, s in zip(ancs, sexes) if s == "male"])
        strat_f = percentiles(vals[np.array(sexes) == "female"],
                              ["female"] * sum(s == "female" for s in sexes),
                              [a for a, s in zip(ancs, sexes) if s == "female"])

        # 7e -- leave-one-out strata
        loo = []
        for k in sorted(set(keys)):
            idx = np.array([i for i, kk in enumerate(keys) if kk == k])
            if idx.size < 30:
                continue
            sub = vals[idx]
            n = sub.size
            total, total_sq = sub.sum(), np.square(sub).sum()
            mu = (total - sub) / (n - 1)
            var = (total_sq - np.square(sub)) / (n - 1) - np.square(mu)
            var = np.maximum(var, 1e-12) * (n - 1) / (n - 2)
            loo.extend(stats.norm.cdf((sub - mu) / np.sqrt(var)).tolist())
        loo = np.array(loo)

        conds = {"identity": ident, "shuffle_within": within, "shuffle_across": across,
                 "pooled": pooled, "loo": loo}
        detail[mid] = {c: _uniformity(v) for c, v in conds.items() if v.size}
        detail[mid]["identity_vs_within_max_abs_diff"] = float(
            np.max(np.abs(np.sort(ident) - np.sort(within)))) if ident.size == within.size else None
        detail[mid]["sex_separation_stratified_ks"] = float(
            stats.ks_2samp(strat_m, strat_f).statistic)
        detail[mid]["sex_separation_pooled_ks"] = float(
            stats.ks_2samp(pooled_m, pooled_f).statistic)
        detail[mid]["pooled_male_mean_percentile"] = float(pooled_m.mean())
        detail[mid]["pooled_female_mean_percentile"] = float(pooled_f.mean())

        rows.append({
            "id": mid,
            "identity_ks": detail[mid]["identity"]["ks_stat"],
            "identity_uniform": detail[mid]["identity"]["uniform_at_1pct"],
            "within_shuffle_ks": detail[mid]["shuffle_within"]["ks_stat"],
            "within_shuffle_identical_to_identity": bool(
                detail[mid]["identity_vs_within_max_abs_diff"] is not None
                and detail[mid]["identity_vs_within_max_abs_diff"] < 1e-12),
            "leaks_identity": bool(
                detail[mid]["identity_vs_within_max_abs_diff"] is not None
                and detail[mid]["identity_vs_within_max_abs_diff"] >= 1e-12),
            "across_shuffle_ks": detail[mid]["shuffle_across"]["ks_stat"],
            "stratification_carries_information": bool(
                detail[mid]["shuffle_across"]["ks_stat"]
                > 2 * detail[mid]["identity"]["ks_stat"]),
            "pooled_ks": detail[mid]["pooled"]["ks_stat"],
            "sex_separation_stratified_ks": detail[mid]["sex_separation_stratified_ks"],
            "sex_separation_pooled_ks": detail[mid]["sex_separation_pooled_ks"],
            "pooled_male_mean_percentile": detail[mid]["pooled_male_mean_percentile"],
            "pooled_female_mean_percentile": detail[mid]["pooled_female_mean_percentile"],
            "loo_ks": detail[mid]["loo"]["ks_stat"],
            "loo_minus_identity_ks": detail[mid]["loo"]["ks_stat"] - detail[mid]["identity"]["ks_stat"],
        })

    payload = {
        "arm": "7 -- negative control on the normative model",
        "question": "does shuffling identities against measurements change the "
                    "percentile output; if it does, the normative model leaks",
        "design": {
            "percentile_model": "normal CDF against the narrowest NIOSH stratum, "
                                "implemented here because src/ has no percentile "
                                "function yet",
            "conditions": ["identity", "shuffle_within", "shuffle_across", "pooled", "loo"],
        },
        "per_measurement": detail,
        "table": rows,
        "summary": {
            "n_measurements": len(rows),
            "n_leaking_identity": sum(1 for r_ in rows if r_["leaks_identity"]),
            "leak_free": all(r_["within_shuffle_identical_to_identity"] for r_ in rows),
            "n_uniform_under_correct_strata_at_1pct": sum(1 for r_ in rows if r_["identity_uniform"]),
            "max_identity_ks": max(r_["identity_ks"] for r_ in rows),
            "max_across_shuffle_ks": max(r_["across_shuffle_ks"] for r_ in rows),
            "n_where_stratification_carries_information": sum(
                1 for r_ in rows if r_["stratification_carries_information"]),
            "max_sex_separation_pooled_ks": max(r_["sex_separation_pooled_ks"] for r_ in rows),
            "max_sex_separation_stratified_ks": max(r_["sex_separation_stratified_ks"] for r_ in rows),
            "max_loo_minus_identity_ks": max(abs(r_["loo_minus_identity_ks"]) for r_ in rows),
        },
    }
    write_json("arm07_negative_control", payload)
    write_csv("arm07_table", rows)
    return payload["summary"]


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=str))
