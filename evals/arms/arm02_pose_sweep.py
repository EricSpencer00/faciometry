"""Arm 2 -- pose sweep. How far each measurement moves per degree of head pose.

The synthetic face is rotated through yaw, pitch and roll from -30 to +30
degrees in 2.5-degree steps, re-projected, and re-measured. The output is a
measured per-measurement per-axis slope set against the a-priori value in
``core/sensitivity.py``.

Four conditions, because "the" pose sensitivity turns out not to be a single
number:

``3d``          the rotated face with no projection at all. A distance or an
                angle is rotation invariant, so any non-zero slope here is
                frame dependence -- the measurement is defined against an image
                axis rather than against the face.
``ortho``       orthographic projection, which is the regime the defaults in
                ``sensitivity.py`` were derived in.
``persp_1.0m``  a pinhole camera at the ICAO portrait distance.
``persp_0.5m``  a pinhole camera at arm's length, which is what people submit.
``ortho_asym``  orthographic, on a face given a few millimetres of left-right
                depth asymmetry, since a perfectly symmetric face makes every
                transverse width scale by exactly cos(yaw) and hides the effect
                that depth offsets have.

A second question rides along, because sixteen measurements in the catalogue
are now *claimed* to be roll-robust rather than assumed to be. Each of them is
read against the interpupillary line or the interpupillary axis instead of the
image frame, and the claim splits in two:

* **exact** -- both the measured direction and its reference rotate with the
  head, so roll cancels in the geometry. The two canthal tilts, the three
  asymmetries, the two brow-apex offsets and the nasal dorsal angle.
* **cos(roll)** -- the reference line rotates with the head but the sign axis
  the offset is read along is still the frame's ``y``, so the answer keeps its
  magnitude and loses a cosine. The four margin-reflex distances, the two
  Cupid's-bow heights and the two commissure heights.

Both are checked against the measured slope rather than taken on trust, and
each of the sixteen is swept a second time in the **discarded
horizon-referenced form** -- the version the catalogue used before the roll
defect was found. That is the positive control: without it, "the slope is zero"
is equally consistent with roll cancelling and with the sweep being unable to
see roll at all. The horizon twins must move, and on the two sides of the face
they must move in opposite directions.

The comparison metric is chosen to match how the a-priori numbers were built.
``cosine_yaw_sensitivity`` is not a derivative: it is ``(1 - cos(10 deg)) / 10``,
a secant through the origin evaluated at ten degrees. So the measured
counterpart is the mean of ``|v(+10) - v(0)|`` and ``|v(-10) - v(0)|``, divided
by ten -- relative to ``v(0)`` for lengths and ratios, absolute for angles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from evals._bootstrap import rng, write_csv, write_json
from evals.synth import face as F

from faciometry.core.formula import (
    Abs, AngleBetween, Axis, Const, Diff, Dist, ProjLength, Pt, Ratio, SignedTilt, Vec,
)
from faciometry.core.landmarks import Landmark as L
from faciometry.core.spec import Unit, View
from faciometry.measure.registry import BY_ID, CATALOGUE

AXES = ("yaw", "pitch", "roll")
STEP = 2.5
LIMIT = 30.0
ANGLES = [round(a * STEP, 6) for a in range(int(-LIMIT / STEP), int(LIMIT / STEP) + 1)]

#: Millimetres of left-right depth asymmetry injected into the ``ortho_asym``
#: face. Real faces carry a few millimetres; a perfectly symmetric one makes
#: every bilateral width scale by exactly cos(yaw) by construction.
ASYMMETRY_SD_MM = 2.0


def _asymmetric_face() -> np.ndarray:
    coords = F.COORDS.copy()
    r = rng(2)
    for name, i in F.INDEX.items():
        if name.value.endswith("_l"):
            coords[i, 2] += r.normal(0.0, ASYMMETRY_SD_MM)
        elif name.value.endswith("_r"):
            coords[i, 2] += r.normal(0.0, ASYMMETRY_SD_MM)
    return coords


ASYM_COORDS = _asymmetric_face()

CONDITIONS = {
    "3d":         dict(project=False, distance=None, coords=None),
    "ortho":      dict(project=True,  distance=None, coords=None),
    "persp_1.0m": dict(project=True,  distance=1.0,  coords=None),
    "persp_0.5m": dict(project=True,  distance=0.5,  coords=None),
    "ortho_asym": dict(project=True,  distance=None, coords=ASYM_COORDS),
}


# ---------------------------------------------------------------------------
# The roll claim, and the control that makes testing it meaningful.
#
# ROLL_CLAIMS says, for every measurement that references the interpupillary
# line or axis, what the geometry is claimed to do under image roll. HORIZON
# gives the same measurement in the frame-referenced form the catalogue used
# before the roll defect was found, as a positive control.
# ---------------------------------------------------------------------------

#: (1 - cos 10 deg) / 10, the secant a quantity that scales as cos(roll) must
#: return under this arm's metric.
COS_ROLL_SECANT = (1.0 - math.cos(math.radians(10.0))) / 10.0

ROLL_CLAIMS: dict[str, tuple[str, str]] = {
    "canthal_tilt_l": ("exact", "tilt and reference axis both rotate with the head"),
    "canthal_tilt_r": ("exact", "tilt and reference axis both rotate with the head"),
    "canthal_tilt_asymmetry": ("exact", "difference of two exactly invariant tilts"),
    "ocular_height_asymmetry": ("exact", "inclination of a landmark line against a landmark axis"),
    "mouth_corner_asymmetry": ("exact", "inclination of a landmark line against a landmark axis"),
    "brow_apex_lateral_offset_l": ("exact", "projection onto an axis that rotates with it"),
    "brow_apex_lateral_offset_r": ("exact", "projection onto an axis that rotates with it"),
    "nasal_dorsal_deviation": ("exact", "angle between two landmark-derived directions"),
    "margin_reflex_distance_1_l": ("cos(roll)", "line rotates, but the sign axis is the frame's y"),
    "margin_reflex_distance_1_r": ("cos(roll)", "line rotates, but the sign axis is the frame's y"),
    "margin_reflex_distance_2_l": ("cos(roll)", "line rotates, but the sign axis is the frame's -y"),
    "margin_reflex_distance_2_r": ("cos(roll)", "line rotates, but the sign axis is the frame's -y"),
    "cupids_bow_peak_height_l": ("cos(roll)", "difference of two frame-y offsets from a rotating line"),
    "cupids_bow_peak_height_r": ("cos(roll)", "difference of two frame-y offsets from a rotating line"),
    "commissure_height_l": ("cos(roll)", "frame-y offset over an invariant width"),
    "commissure_height_r": ("cos(roll)", "frame-y offset over an invariant width"),
}


@dataclass(frozen=True)
class Shadow:
    """A measurement that is not in the catalogue, swept as a control.

    Carries just enough of a :class:`MeasurementSpec` for the sweep: an id, the
    unit and view that decide the metric and the camera, the formula, and the
    a-priori sensitivity of the catalogue measurement it shadows.
    """

    id: str
    twin: str
    unit: Unit
    view: View
    formula: object
    sensitivity: object


def _shadow(twin: str, formula, unit: Unit | None = None) -> Shadow:
    real = BY_ID[twin]
    return Shadow(id=f"{twin}__horizon", twin=twin, unit=unit or real.unit,
                  view=real.view, formula=formula, sensitivity=real.sensitivity)


def _mrd_horizon(pupil: L, lid: L, axis: str) -> object:
    return ProjLength(Pt(pupil), Pt(lid), Axis(axis))


HORIZON: tuple[Shadow, ...] = (
    _shadow("canthal_tilt_l",
            SignedTilt(Pt(L.ENDOCANTHION_L), Pt(L.EXOCANTHION_L), Axis("-x"))),
    _shadow("canthal_tilt_r",
            SignedTilt(Pt(L.ENDOCANTHION_R), Pt(L.EXOCANTHION_R), Axis("x"))),
    _shadow("canthal_tilt_asymmetry",
            Abs(Diff(SignedTilt(Pt(L.ENDOCANTHION_L), Pt(L.EXOCANTHION_L), Axis("-x")),
                     SignedTilt(Pt(L.ENDOCANTHION_R), Pt(L.EXOCANTHION_R), Axis("x"))))),
    _shadow("ocular_height_asymmetry",
            Abs(SignedTilt(Pt(L.EXOCANTHION_R), Pt(L.EXOCANTHION_L), Axis("-x")))),
    _shadow("mouth_corner_asymmetry",
            Abs(SignedTilt(Pt(L.CHEILION_R), Pt(L.CHEILION_L), Axis("-x")))),
    _shadow("brow_apex_lateral_offset_l",
            Ratio(ProjLength(Pt(L.EXOCANTHION_L), Pt(L.SUPERCILIARE_L), Axis("-x")),
                  Dist(Pt(L.ENDOCANTHION_L), Pt(L.EXOCANTHION_L)))),
    _shadow("brow_apex_lateral_offset_r",
            Ratio(ProjLength(Pt(L.EXOCANTHION_R), Pt(L.SUPERCILIARE_R), Axis("x")),
                  Dist(Pt(L.ENDOCANTHION_R), Pt(L.EXOCANTHION_R)))),
    _shadow("nasal_dorsal_deviation",
            Diff(Const(90.0),
                 AngleBetween(Vec(Pt(L.SELLION), Pt(L.PRONASALE)), Axis("x")))),
    _shadow("margin_reflex_distance_1_l", _mrd_horizon(L.PUPIL_L, L.PALPEBRALE_SUP_L, "y")),
    _shadow("margin_reflex_distance_1_r", _mrd_horizon(L.PUPIL_R, L.PALPEBRALE_SUP_R, "y")),
    _shadow("margin_reflex_distance_2_l", _mrd_horizon(L.PUPIL_L, L.PALPEBRALE_INF_L, "-y")),
    _shadow("margin_reflex_distance_2_r", _mrd_horizon(L.PUPIL_R, L.PALPEBRALE_INF_R, "-y")),
    _shadow("cupids_bow_peak_height_l",
            ProjLength(Pt(L.LABIALE_SUPERIUS), Pt(L.CRISTA_PHILTRI_L), Axis("y"))),
    _shadow("cupids_bow_peak_height_r",
            ProjLength(Pt(L.LABIALE_SUPERIUS), Pt(L.CRISTA_PHILTRI_R), Axis("y"))),
    _shadow("commissure_height_l",
            Ratio(ProjLength(Pt(L.STOMION), Pt(L.CHEILION_L), Axis("y")),
                  Dist(Pt(L.CHEILION_L), Pt(L.CHEILION_R)))),
    _shadow("commissure_height_r",
            Ratio(ProjLength(Pt(L.STOMION), Pt(L.CHEILION_R), Axis("y")),
                  Dist(Pt(L.CHEILION_L), Pt(L.CHEILION_R)))),
)


def _capture(cond: dict, spec, yaw: float, pitch: float, roll: float):
    coords = F.COORDS if cond["coords"] is None else cond["coords"]
    c = F.rotated(yaw, pitch, roll, coords=coords)
    if not cond["project"]:
        return c
    d = cond["distance"]
    if spec.view is View.PROFILE:
        return F.camera_profile_ortho(c) if d is None else F.camera_profile_perspective(c, d)
    return F.camera_frontal_ortho(c) if d is None else F.camera_frontal_perspective(c, d)


def _value(spec, coords) -> float:
    from faciometry.core.landmarks import PointSet
    ps = PointSet(index=dict(F.INDEX), coords=coords)
    try:
        v = float(np.asarray(spec.formula.eval(ps)))
    except Exception:
        return float("nan")
    return v


def _deviation(spec, v: float, v0: float) -> float:
    """Absolute degrees for an angle, fraction of the zero-pose value otherwise."""
    if spec.unit is Unit.DEGREES:
        return v - v0
    if abs(v0) < 1e-12:
        return float("nan")
    return (v - v0) / v0


def sweep(specs=None) -> tuple[list[dict], list[dict]]:
    specs = CATALOGUE if specs is None else specs
    curves: list[dict] = []
    slopes: list[dict] = []
    for cname, cond in CONDITIONS.items():
        for spec in specs:
            base = _value(spec, _capture(cond, spec, 0.0, 0.0, 0.0))
            for axis in AXES:
                devs: dict[float, float] = {}
                raws: dict[float, float] = {}
                for a in ANGLES:
                    kw = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, axis: a}
                    v = _value(spec, _capture(cond, spec, **kw))
                    dev = _deviation(spec, v, base)
                    devs[a] = dev
                    raws[a] = v
                    curves.append({"condition": cname, "id": spec.id, "axis": axis,
                                   "angle_deg": a, "value": v, "deviation": dev})

                def secant(theta: float) -> float:
                    plus, minus = devs.get(theta), devs.get(-theta)
                    vals = [abs(x) for x in (plus, minus) if x is not None and math.isfinite(x)]
                    return sum(vals) / len(vals) / theta if vals else float("nan")

                def secant_abs(theta: float) -> float:
                    """The same secant in the measurement's own units.

                    The relative form is nan for a quantity whose zero-pose
                    value is exactly zero by construction --
                    ``commissure_height_r`` on this face is the case -- and a
                    nan there would read as "not measured" rather than as "the
                    denominator is zero", which is a different statement.
                    """
                    vals = [abs(raws[t] - base) for t in (theta, -theta) if t in raws
                            and math.isfinite(raws[t]) and math.isfinite(base)]
                    return sum(vals) / len(vals) / theta if vals else float("nan")

                finite = {a: d for a, d in devs.items() if math.isfinite(d)}
                xs = np.array([abs(a) for a in finite], dtype=float)
                ys = np.array([abs(d) for d in finite.values()], dtype=float)
                lsq = float(np.sum(xs * ys) / np.sum(xs * xs)) if np.sum(xs * xs) > 0 else float("nan")
                apriori = getattr(spec.sensitivity, axis)
                s10 = secant(10.0)
                slopes.append({
                    "condition": cname,
                    "id": spec.id,
                    "unit": spec.unit.value,
                    "view": spec.view.value,
                    "axis": axis,
                    "value_at_zero": base,
                    "measured_secant_5deg": secant(5.0),
                    "measured_secant_10deg": s10,
                    "measured_secant_10deg_absolute": secant_abs(10.0),
                    "measured_secant_20deg": secant(20.0),
                    "measured_secant_30deg": secant(30.0),
                    "lsq_slope_through_origin": lsq,
                    "max_abs_deviation_30deg": float(np.max(ys)) if ys.size else float("nan"),
                    "apriori_sensitivity": apriori,
                    "apriori_source": spec.sensitivity.source,
                    "measured_over_apriori": (s10 / apriori) if apriori > 0 and math.isfinite(s10)
                                             else None,
                    "deviation_at_plus10": devs.get(10.0),
                    "deviation_at_minus10": devs.get(-10.0),
                    "asymmetric_in_sign": bool(
                        math.isfinite(devs.get(10.0, float("nan")))
                        and math.isfinite(devs.get(-10.0, float("nan")))
                        and abs(abs(devs[10.0]) - abs(devs[-10.0]))
                        > 0.1 * max(abs(devs[10.0]), abs(devs[-10.0]), 1e-12)
                    ),
                })
    return curves, slopes


def _recommendations(slopes: list[dict]) -> list[dict]:
    """Where the measured value and the a-priori value disagree, and by how much.

    The recommendation is made against ``ortho``, because that is the regime
    the a-priori numbers claim to describe, and reported alongside the
    perspective conditions, because that is the regime a real photograph is in.
    """
    by = {(r["condition"], r["id"], r["axis"]): r for r in slopes}
    out, skipped = [], []
    for (cond, mid, axis), r in by.items():
        if cond != "ortho":
            continue
        measured, apriori = r["measured_secant_10deg"], r["apriori_sensitivity"]
        if not math.isfinite(measured):
            # Not silently dropped: a relative slope is undefined when the
            # zero-pose value is zero, which is a statement about the
            # measurement rather than a gap in the sweep.
            skipped.append({
                "id": mid, "axis": axis, "apriori": apriori,
                "reason": "relative slope undefined: the zero-pose value is "
                          f"{r['value_at_zero']:.3g}",
                "absolute_secant_10deg": r["measured_secant_10deg_absolute"],
            })
            continue
        p10 = by[("persp_1.0m", mid, axis)]["measured_secant_10deg"]
        p05 = by[("persp_0.5m", mid, axis)]["measured_secant_10deg"]
        asym = by[("ortho_asym", mid, axis)]["measured_secant_10deg"]
        d3 = by[("3d", mid, axis)]["measured_secant_10deg"]
        worst = max(x for x in (measured, p10, p05, asym) if math.isfinite(x))
        # "Disagree" means off by more than a factor of two, or an a-priori of
        # zero against a measured effect that is not numerically zero.
        if apriori == 0.0:
            disagree = worst > 1e-6
            factor = float("inf") if disagree else 0.0
        else:
            factor = worst / apriori
            disagree = factor > 2.0 or factor < 0.5
        out.append({
            "id": mid, "axis": axis, "apriori": apriori,
            "apriori_source": r["apriori_source"],
            "measured_ortho": measured, "measured_persp_1.0m": p10,
            "measured_persp_0.5m": p05, "measured_ortho_asym": asym,
            "measured_3d_frame_dependence": d3,
            "worst_measured": worst,
            "worst_over_apriori": factor,
            "disagrees": bool(disagree),
            "direction": "a-priori too optimistic" if factor > 1 else "a-priori too pessimistic",
        })
    out.sort(key=lambda r: (-(r["worst_over_apriori"] if math.isfinite(r["worst_over_apriori"]) else 1e18), r["id"]))
    return out, skipped


def roll_invariance(slopes: list[dict], control_slopes: list[dict]) -> list[dict]:
    """Test the roll claim on every interpupillary-referenced measurement.

    Two-sided. The measurement itself must behave as claimed, *and* its
    discarded horizon-referenced twin must move -- otherwise a zero slope is
    equally consistent with the reference working and with the sweep having no
    roll in it at all.
    """
    real = {(r["id"], r["condition"]): r for r in slopes if r["axis"] == "roll"}
    ctrl = {(r["id"], r["condition"]): r for r in control_slopes if r["axis"] == "roll"}
    rows = []
    for mid, (claim, reason) in ROLL_CLAIMS.items():
        got = {c: real[(mid, c)]["measured_secant_10deg"] for c in CONDITIONS}
        got_abs = {c: real[(mid, c)]["measured_secant_10deg_absolute"] for c in CONDITIONS}
        expected = 0.0 if claim == "exact" else COS_ROLL_SECANT
        # The claim is a statement about the geometry, so it is tested in the
        # regimes where the geometry is all there is: the unprojected 3D face
        # and the orthographic camera. Perspective is reported beside it,
        # because that is what breaks the claim and by how much is the finding.
        tested = [got["3d"], got["ortho"]]
        tested_abs = [got_abs["3d"], got_abs["ortho"]]
        zero_valued = abs(real[(mid, "ortho")]["value_at_zero"]) < 1e-12
        if claim == "exact":
            holds = all(
                (math.isfinite(v) and abs(v) <= 1e-9) if math.isfinite(v) else abs(a) <= 1e-9
                for v, a in zip(tested, tested_abs))
        elif zero_valued:
            # cos(roll) on a quantity whose zero-pose value is exactly zero by
            # construction predicts zero movement, and the relative metric is
            # undefined rather than failed. commissure_height_r is the case:
            # the right mouth corner is level with the stomion on this face.
            holds = all(math.isfinite(a) and abs(a) <= 1e-9 for a in tested_abs)
        else:
            holds = all(math.isfinite(v) and abs(v - expected) <= 0.02 * expected
                        for v in tested)
        c_ortho = ctrl[(f"{mid}__horizon", "ortho")]["measured_secant_10deg"]
        c_ortho_abs = ctrl[(f"{mid}__horizon", "ortho")]["measured_secant_10deg_absolute"]
        c_signal = c_ortho if math.isfinite(c_ortho) else c_ortho_abs
        real_signal = got["ortho"] if math.isfinite(got["ortho"]) else got_abs["ortho"]
        rows.append({
            "id": mid,
            "claim": claim,
            "claim_reason": reason,
            "unit": real[(mid, "ortho")]["unit"],
            "expected_roll_secant_10deg": expected,
            "measured_3d": got["3d"],
            "measured_ortho": got["ortho"],
            "measured_persp_1.0m": got["persp_1.0m"],
            "measured_persp_0.5m": got["persp_0.5m"],
            "measured_ortho_asym": got["ortho_asym"],
            "measured_ortho_absolute": got_abs["ortho"],
            "measured_persp_0.5m_absolute": got_abs["persp_0.5m"],
            "claim_holds": bool(holds),
            "zero_valued_at_zero_pose": bool(zero_valued),
            "horizon_control_ortho": c_ortho,
            "horizon_control_ortho_absolute": c_ortho_abs,
            "horizon_control_deviation_at_plus10":
                ctrl[(f"{mid}__horizon", "ortho")]["deviation_at_plus10"],
            "control_has_power": bool(math.isfinite(c_signal) and abs(c_signal) > 1e-6),
            "control_over_measurement": (
                float("inf") if not math.isfinite(real_signal) or abs(real_signal) < 1e-15
                else abs(c_signal) / abs(real_signal)),
            # A control that moves exactly as much as the measurement it
            # controls has power but no discrimination: on this face the lid
            # margins sit directly above the pupils, so a perpendicular offset
            # from the interpupillary line and a y offset from the pupil are
            # the same number, and swapping the reference changes nothing.
            "control_discriminates": bool(
                math.isfinite(c_ortho_abs) and math.isfinite(got_abs["ortho"])
                and abs(c_ortho_abs - got_abs["ortho"])
                > 1e-6 * max(abs(c_ortho_abs), abs(got_abs["ortho"]), 1e-12)),
        })
    rows.sort(key=lambda r: (r["claim"], r["id"]))
    return rows


def run() -> dict:
    curves, slopes = sweep()
    _, control_slopes = sweep(HORIZON)
    recs, skipped = _recommendations(slopes)
    roll = roll_invariance(slopes, control_slopes)

    frame_dependent = [r for r in slopes if r["condition"] == "3d"
                       and math.isfinite(r["measured_secant_10deg"])
                       and r["measured_secant_10deg"] > 1e-9]
    ortho = [r for r in slopes if r["condition"] == "ortho"]
    p05 = [r for r in slopes if r["condition"] == "persp_0.5m"]

    payload = {
        "arm": "2 -- pose sweep",
        "question": "how far does each measurement move per degree of head pose, "
                    "and does the a-priori model in core/sensitivity.py predict it",
        "design": {
            "angles_deg": ANGLES, "axes": list(AXES),
            "conditions": {k: {kk: (vv if not isinstance(vv, np.ndarray) else "asymmetric face")
                               for kk, vv in v.items()} for k, v in CONDITIONS.items()},
            "metric": "mean of |v(+10)-v(0)| and |v(-10)-v(0)| over 10 degrees; relative "
                      "to v(0) for lengths and ratios, absolute degrees for angles. This "
                      "matches how cosine_yaw_sensitivity() builds the a-priori value.",
            "asymmetry_sd_mm": ASYMMETRY_SD_MM,
            "roll_claims": {k: v[0] for k, v in ROLL_CLAIMS.items()},
            "roll_control": "each measurement in roll_claims is swept a second time in "
                            "the frame-referenced form the catalogue used before the roll "
                            "defect was found; that twin must move, or a zero slope proves "
                            "nothing",
        },
        "slopes": slopes,
        "recommendations": recs,
        "not_comparable": skipped,
        "roll_invariance": roll,
        "roll_controls": control_slopes,
        "summary": {
            "n_rows": len(slopes),
            "n_frame_dependent_in_3d": len(frame_dependent),
            "frame_dependent": sorted({(r["id"], r["axis"]) for r in frame_dependent}),
            "n_comparable_pairs": len(recs),
            "n_not_comparable": len(skipped),
            "not_comparable": sorted({(r["id"], r["axis"]) for r in skipped}),
            "n_disagreements": sum(1 for r in recs if r["disagrees"]),
            "roll_claims_tested": len(roll),
            "roll_claims_held": sum(1 for r in roll if r["claim_holds"]),
            "roll_claims_broken": [r["id"] for r in roll if not r["claim_holds"]],
            "roll_controls_with_power": sum(1 for r in roll if r["control_has_power"]),
            "roll_controls_without_power": [r["id"] for r in roll if not r["control_has_power"]],
            "roll_controls_that_do_not_discriminate": [
                r["id"] for r in roll if not r["control_discriminates"]],
            "roll_worst_under_perspective": max(
                roll, key=lambda r: (r["measured_persp_0.5m"]
                                     if math.isfinite(r["measured_persp_0.5m"]) else -1)),
            "worst_ortho_secant10": max(
                (r for r in ortho if math.isfinite(r["measured_secant_10deg"])),
                key=lambda r: r["measured_secant_10deg"], default=None),
            "worst_persp05_secant10": max(
                (r for r in p05 if math.isfinite(r["measured_secant_10deg"])),
                key=lambda r: r["measured_secant_10deg"], default=None),
        },
    }
    write_json("arm02_pose_sweep", payload)
    write_csv("arm02_slopes", slopes)
    write_csv("arm02_curves", curves)
    write_csv("arm02_recommendations", recs)
    write_csv("arm02_roll_invariance", roll)
    write_csv("arm02_roll_controls", control_slopes)
    return payload["summary"]


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=1, default=str))
