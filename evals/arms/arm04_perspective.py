"""Arm 4 -- perspective sweep, and a positive control on the distance model.

``core.scale.magnification_distortion`` says a feature 50 mm in front of the
eye plane is enlarged relative to it by ``K = 50 mm / distance``, and
``spec.decide_reportability`` withholds below 0.6 m on the strength of it. This
arm renders the synthetic face through a real pinhole camera at eight distances
and checks whether that closed form is what actually happens.

Two parts.

4a  A purpose-built probe: two transverse segments of identical true length,
    one exactly on the eye plane and one exactly 50 mm in front of it. Their
    apparent length ratio is the quantity the closed form predicts, with no
    anatomy in the way. The exact pinhole answer is ``K / (1 - K)``, not ``K``,
    so the two must diverge at close range and the size of that divergence is
    the result.

4b  All 45 measurements, orthographic against each perspective distance. For a
    quantity that is a ratio of two spans at depths z1 and z2, the first-order
    fractional distortion is ``(z1 - z2) / d``, so ``fractional_change * d``
    should be a constant with units of millimetres -- the effective depth
    straddle. Testing that it is constant across eight distances is a stronger
    check than testing any single number.
"""

from __future__ import annotations

import math

import numpy as np

from evals._bootstrap import write_csv, write_json
from evals.synth import face as F

from vitruve.core.landmarks import PointSet
from vitruve.core.scale import magnification_distortion
from vitruve.core.spec import Unit, View
from vitruve.measure.registry import CATALOGUE

DISTANCES_M = (0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
PLANE_DEPTH_MM = 50.0


def probe() -> list[dict]:
    """Two equal segments, one on the eye plane and one 50 mm in front of it."""
    rows = []
    half = 20.0
    z_ref = F.EYE_PLANE_Z
    for d in DISTANCES_M:
        d_mm = d * 1000.0
        def project(z: float) -> float:
            m = d_mm / (d_mm - (z - z_ref))
            return 2 * half * m
        near = project(z_ref + PLANE_DEPTH_MM)
        far = project(z_ref)
        measured = near / far - 1.0
        icao = magnification_distortion(d, PLANE_DEPTH_MM)
        exact = icao / (1.0 - icao)
        rows.append({
            "distance_m": d,
            "measured_magnification": measured,
            "icao_closed_form": icao,
            "exact_pinhole": exact,
            "measured_minus_icao": measured - icao,
            "icao_relative_error": (icao - measured) / measured,
            "measured_matches_exact_to": abs(measured - exact),
        })
    return rows


def _value(spec, coords) -> float:
    ps = PointSet(index=dict(F.INDEX), coords=coords)
    try:
        return float(np.asarray(spec.formula.eval(ps)))
    except Exception:
        return float("nan")


def _landmark_depth_span(spec) -> float:
    """Peak-to-peak depth of the landmarks a measurement reads, in millimetres.

    For the frontal camera that is the z span; for the profile camera, which
    looks down the x axis, it is the x span.
    """
    axis = 0 if spec.view is View.PROFILE else 2
    zs = [F.FACE[name][axis] for name in spec.landmarks]
    return max(zs) - min(zs)


def catalogue() -> tuple[list[dict], list[dict]]:
    rows, curves = [], []
    base3d = F.rotated(0.0, 0.0, 0.0)
    for spec in CATALOGUE:
        ortho = (F.camera_profile_ortho(base3d) if spec.view is View.PROFILE
                 else F.camera_frontal_ortho(base3d))
        v_ortho = _value(spec, ortho)
        row = {"id": spec.id, "unit": spec.unit.value, "view": spec.view.value,
               "value_orthographic": v_ortho,
               "landmark_depth_span_mm": _landmark_depth_span(spec)}
        straddles = []
        for d in DISTANCES_M:
            proj = (F.camera_profile_perspective(base3d, d) if spec.view is View.PROFILE
                    else F.camera_frontal_perspective(base3d, d))
            v = _value(spec, proj)
            if spec.unit is Unit.DEGREES:
                dev = v - v_ortho
            else:
                dev = (v - v_ortho) / v_ortho if abs(v_ortho) > 1e-12 else float("nan")
            row[f"dev_{d}m"] = dev
            curves.append({"id": spec.id, "distance_m": d, "value": v, "deviation": dev})
            if spec.unit is not Unit.DEGREES and math.isfinite(dev):
                straddles.append(dev * d * 1000.0)
        if straddles:
            arr = np.array(straddles)
            row["implied_depth_straddle_mm_mean"] = float(arr.mean())
            row["implied_depth_straddle_mm_spread"] = float(arr.max() - arr.min())
            row["straddle_constant_across_distance"] = bool(
                abs(arr.max() - arr.min()) <= 0.10 * max(abs(arr.mean()), 1e-9))
            row["icao_predicts"] = bool(abs(arr.mean()) > 1.0)
        rows.append(row)
    return rows, curves


def run() -> dict:
    pr = probe()
    rows, curves = catalogue()

    worst = max(rows, key=lambda r: abs(r.get("dev_0.3m", 0.0)) if math.isfinite(r.get("dev_0.3m", 0.0)) else -1)
    icao_err_03 = next(p for p in pr if p["distance_m"] == 0.3)

    payload = {
        "arm": "4 -- perspective sweep",
        "question": "does the ICAO closed form K = 50mm/d describe what a real pinhole "
                    "camera does to these measurements",
        "design": {"distances_m": list(DISTANCES_M), "plane_depth_mm": PLANE_DEPTH_MM,
                   "camera": "pinhole, scaled so the eye plane is left where orthographic "
                             "projection would put it"},
        "probe": pr,
        "catalogue": rows,
        "summary": {
            "icao_understates_at_0.3m_by": icao_err_03["icao_relative_error"],
            "icao_understates_at_1.0m_by": next(
                p for p in pr if p["distance_m"] == 1.0)["icao_relative_error"],
            "exact_pinhole_matches_measured_to": max(p["measured_matches_exact_to"] for p in pr),
            "worst_measurement_at_0.3m": worst["id"],
            "worst_measurement_at_0.3m_deviation": worst.get("dev_0.3m"),
            "n_with_constant_straddle": sum(
                1 for r in rows if r.get("straddle_constant_across_distance")),
            "n_ratio_or_length": sum(1 for r in rows if r["unit"] != "deg"),
            "at_1.0m_over_1pct": sorted(
                r["id"] for r in rows
                if math.isfinite(r.get("dev_1.0m", float("nan"))) and abs(r["dev_1.0m"]) > 0.01),
            "at_0.5m_over_1pct": sorted(
                r["id"] for r in rows
                if math.isfinite(r.get("dev_0.5m", float("nan"))) and abs(r["dev_0.5m"]) > 0.01),
        },
    }
    write_json("arm04_perspective", payload)
    write_csv("arm04_probe", pr)
    write_csv("arm04_catalogue", rows)
    write_csv("arm04_curves", curves)
    return payload["summary"]


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=1, default=str))
