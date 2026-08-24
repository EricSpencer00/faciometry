"""Arm 1 -- synthetic geometry control. The measurement code's own error floor.

No model, no image, no noise. A face with known coordinates goes in, and every
one of the catalogue's measurements is compared against a value computed a
second time in :mod:`evals.synth.truth` by pure-Python arithmetic written from
the anatomical definition. Anything that fails here is a bug in a formula, not
in a landmarker, and no amount of downstream care can recover it.

Three parts:

1a  primitive exactness -- each node of the formula algebra on a configuration
    whose answer is a construction parameter, checked to 1e-12. Includes the
    nodes the catalogue only started using with the second batch of
    measurements: a signed tilt read against a *landmark-derived* axis rather
    than a frame axis, and the same configuration rotated in its own plane,
    which must return the identical answer if the roll cancellation the
    catalogue now claims is real.
1b  full catalogue on the 3D synthetic face against independent truth.
1c  the ``evaluate()`` path with near-zero covariance, to confirm the
    Monte-Carlo wrapper reproduces the deterministic value rather than
    shifting it.
"""

from __future__ import annotations

import math

import numpy as np

from evals._bootstrap import rng, write_csv, write_json
from evals.synth import face as F
from evals.synth import truth as T

from vitruve.core.formula import (
    Abs, AngleAt, AngleBetween, Axis, Const, Diff, Dist, LineOffset, Mean, Mid,
    ProjLength, Pt, Ratio, Sum, Product, SignedTilt, Vec, registered_ops,
)
from vitruve.core.landmarks import Landmark as L
from vitruve.core.landmarks import PointSet
from vitruve.core.scale import ScaleEstimate, ScaleSource
from vitruve.core.spec import Unit
from vitruve.measure.evaluate import LandmarkUncertainty, Measured, evaluate
from vitruve.measure.registry import CATALOGUE, BY_ID

TOL_EXACT = 1e-12


def raw_value(spec, ps: PointSet) -> float:
    """The deterministic formula value, with no Monte Carlo in the way."""
    return float(np.asarray(spec.formula.eval(ps)))


# ---------------------------------------------------------------------------
# 1a -- primitive exactness
# ---------------------------------------------------------------------------

def _ps(**pts) -> PointSet:
    mapping = {getattr(L, k.upper()): np.array(v, dtype=float) for k, v in pts.items()}
    return PointSet.from_mapping(mapping)


def primitives() -> list[dict]:
    """One configuration per algebra node whose answer is known exactly."""
    cases: list[dict] = []

    def case(op, expr, ps, expected, why):
        got = float(np.asarray(expr.eval(ps)))
        err = abs(got - expected)
        rel = err / abs(expected) if expected else err
        cases.append({
            "op": op, "expected": expected, "got": got, "abs_error": err,
            "rel_error": rel, "pass": bool(err <= TOL_EXACT * max(1.0, abs(expected))),
            "construction": why,
        })

    p = _ps(glabella=(0, 0, 0), nasion=(3, 4, 12))
    case("dist", Dist(Pt(L.GLABELLA), Pt(L.NASION)), p, 13.0, "3-4-12 Pythagorean quadruple")

    p = _ps(glabella=(1, 2, 3), nasion=(5, 10, 15), sellion=(0, 0, 0))
    case("mid", Dist(Mid(Pt(L.GLABELLA), Pt(L.NASION)), Pt(L.SELLION)), p,
         math.dist((3, 6, 9), (0, 0, 0)), "midpoint of (1,2,3) and (5,10,15) is (3,6,9)")

    for target in (150.0, 90.0, 179.99, 0.01):
        p = _ps(sellion=(0, 0, 0), glabella=(1, 0, 0),
                nasion=(math.cos(math.radians(target)), math.sin(math.radians(target)), 0))
        case(f"angle_at@{target}", AngleAt(Pt(L.GLABELLA), Pt(L.SELLION), Pt(L.NASION)), p,
             target, "second ray placed at the target angle on the unit circle")

    p = _ps(sellion=(0, 0, 0), glabella=(1, 0, 0), nasion=(0, 0, 0),
            pronasale=(math.cos(math.radians(170.0)), math.sin(math.radians(170.0)), 0))
    case("angle_between", AngleBetween(Vec(Pt(L.SELLION), Pt(L.GLABELLA)),
                                       Vec(Pt(L.NASION), Pt(L.PRONASALE))), p,
         10.0, "170 degrees between directed lines folds to 10 between undirected ones")

    for target, axis in ((7.0, "x"), (-7.0, "x"), (7.0, "-x")):
        dx = 10.0 if not axis.startswith("-") else -10.0
        p = _ps(endocanthion_r=(0, 0, 0),
                exocanthion_r=(dx, 10.0 * math.tan(math.radians(target)), -5.0))
        case(f"signed_tilt@{target}/{axis}",
             SignedTilt(Pt(L.ENDOCANTHION_R), Pt(L.EXOCANTHION_R), Axis(axis)), p,
             target, "dy = dx tan(theta); the out-of-plane -5 must not enter the answer")

    p = _ps(glabella=(1, 2, 3), nasion=(4, 9, 3))
    case("proj_length", ProjLength(Pt(L.GLABELLA), Pt(L.NASION), Axis("y")), p, 7.0,
         "y separation is 7; x and z separations must not contribute")

    p = _ps(sellion=(0, 0, 0), menton=(0, 10, 0), pronasale=(0, 5, 3))
    case("line_offset", LineOffset(Pt(L.PRONASALE), Pt(L.SELLION), Pt(L.MENTON), Axis("z")), p,
         3.0, "line along +y, point 3 mm along +z from it; perpendicular is parallel to z")

    p = _ps(glabella=(0, 0, 0), nasion=(3, 0, 0), sellion=(0, 4, 0))
    case("ratio", Ratio(Dist(Pt(L.GLABELLA), Pt(L.NASION)),
                        Dist(Pt(L.GLABELLA), Pt(L.SELLION))), p, 0.75, "3 / 4")
    case("sum", Sum((Const(2.0), Const(3.0), Const(5.0))), p, 10.0, "2 + 3 + 5")
    case("diff", Diff(Const(9.0), Const(4.0)), p, 5.0, "9 - 4")
    case("product", Product((Const(2.0), Const(3.5))), p, 7.0, "2 * 3.5")
    case("abs", Abs(Const(-6.25)), p, 6.25, "|-6.25|")
    case("mean", Mean((Const(1.0), Const(2.0), Const(6.0))), p, 3.0, "(1 + 2 + 6) / 3")
    case("const", Const(math.pi), p, math.pi, "identity")

    # --- the nodes the second batch of measurements introduced ---

    # A signed tilt read against an axis built from two landmarks rather than
    # from the frame. The axis here is the interpupillary one.
    for target in (7.0, -7.0):
        p = _ps(pupil_r=(30.0, 0.0, 5.0), pupil_l=(-30.0, 0.0, 5.0),
                endocanthion_l=(-16.0, 0.0, 12.0),
                exocanthion_l=(-45.0, 29.0 * math.tan(math.radians(target)), 2.0))
        case(f"signed_tilt/vec_axis@{target}",
             SignedTilt(Pt(L.ENDOCANTHION_L), Pt(L.EXOCANTHION_L),
                        Vec(Pt(L.PUPIL_R), Pt(L.PUPIL_L))), p,
             target, "axis is the pupil pair, which is exactly -x here; the 10 mm "
                     "of canthal depth must not enter")

    # The same configuration rotated in its own plane. If the interpupillary
    # reference cancels roll -- which is the whole claim behind the rewritten
    # canthal tilt, the three asymmetries and the four orbital offsets -- this
    # must return the identical number, not merely a close one.
    def _roll(pt, deg):
        c, s_ = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        return (c * pt[0] - s_ * pt[1], s_ * pt[0] + c * pt[1], pt[2])

    base = {"pupil_r": (30.0, 0.0, 5.0), "pupil_l": (-30.0, 0.0, 5.0),
            "endocanthion_l": (-16.0, 0.0, 12.0),
            "exocanthion_l": (-45.0, 29.0 * math.tan(math.radians(7.0)), 2.0)}
    for roll_deg in (13.0, -21.0):
        p = _ps(**{k: _roll(v, roll_deg) for k, v in base.items()})
        case(f"signed_tilt/roll_invariance@{roll_deg}",
             SignedTilt(Pt(L.ENDOCANTHION_L), Pt(L.EXOCANTHION_L),
                        Vec(Pt(L.PUPIL_R), Pt(L.PUPIL_L))), p,
             7.0, f"the face rolled {roll_deg} deg in the image plane; a reference "
                  "that rotates with the head must return the same tilt")

    # The horizon-referenced form the catalogue discarded, on the same rolled
    # configuration, on both sides. It must move one-for-one -- otherwise the
    # roll-invariance case above would pass for a harness that cannot see roll
    # at all -- and it must move in *opposite* directions on the two sides,
    # which is the mechanism that made the old asymmetry formulas add roll
    # instead of cancelling it.
    mirror = {"pupil_r": (30.0, 0.0, 5.0), "pupil_l": (-30.0, 0.0, 5.0),
              "endocanthion_r": (16.0, 0.0, 12.0),
              "exocanthion_r": (45.0, 29.0 * math.tan(math.radians(7.0)), 2.0)}
    for roll_deg in (13.0, -21.0):
        p = _ps(**{k: _roll(v, roll_deg) for k, v in base.items()})
        case(f"signed_tilt/horizon_positive_control_l@{roll_deg}",
             SignedTilt(Pt(L.ENDOCANTHION_L), Pt(L.EXOCANTHION_L), Axis("-x")), p,
             7.0 - roll_deg,
             "the discarded frame-referenced form on the left; roll enters one for "
             "one and negatively, which is what makes the invariance case above a "
             "result rather than a null")
        p = _ps(**{k: _roll(v, roll_deg) for k, v in mirror.items()})
        case(f"signed_tilt/horizon_positive_control_r@{roll_deg}",
             SignedTilt(Pt(L.ENDOCANTHION_R), Pt(L.EXOCANTHION_R), Axis("x")), p,
             7.0 + roll_deg,
             "the same on the right, where the reference axis is +x: roll enters "
             "with the opposite sign, so a left-minus-right difference doubles it")

    # A perpendicular offset from a line that is not axis aligned, read along
    # +y. This is the margin-reflex form.
    p = _ps(pupil_r=(30.0, 0.0, 0.0), pupil_l=(-30.0, 0.0, 0.0),
            palpebrale_sup_l=(-30.0, 4.0, 0.0))
    case("line_offset/y_normal",
         LineOffset(Pt(L.PALPEBRALE_SUP_L), Pt(L.PUPIL_R), Pt(L.PUPIL_L), Axis("y")), p,
         4.0, "lid margin 4 mm above a horizontal interpupillary line")

    # A projection along a landmark-derived axis, which is the brow-apex form.
    p = _ps(pupil_r=(30.0, 0.0, 5.0), pupil_l=(-30.0, 0.0, 5.0),
            exocanthion_l=(-45.0, 1.0, 2.0), superciliare_l=(-25.0, 12.0, 18.0))
    case("proj_length/vec_axis",
         ProjLength(Pt(L.EXOCANTHION_L), Pt(L.SUPERCILIARE_L),
                    Vec(Pt(L.PUPIL_R), Pt(L.PUPIL_L))), p,
         -20.0, "the apex is 20 mm medial along an axis that points laterally, so "
                "the answer is negative; the 11 mm rise and 16 mm depth must not enter")

    # An undirected angle between a landmark-derived direction and a
    # landmark-derived axis, folded into [0, 90]. This is the nasal-dorsum form.
    p = _ps(pupil_r=(30.0, 0.0, 5.0), pupil_l=(-30.0, 0.0, 5.0),
            sellion=(0.0, 0.0, 20.0), pronasale=(0.0, -40.0, 52.0))
    case("angle_between/vec_axes",
         AngleBetween(Vec(Pt(L.SELLION), Pt(L.PRONASALE)),
                      Vec(Pt(L.PUPIL_R), Pt(L.PUPIL_L))), p,
         90.0, "a midline dorsum is exactly perpendicular to the interpupillary axis, "
               "whatever its inclination in the sagittal plane")

    # An axis with no 2D meaning must raise rather than silently return zero.
    p2 = PointSet.from_mapping({L.GLABELLA: np.array([0.0, 0.0]),
                                L.NASION: np.array([1.0, 1.0])})
    try:
        ProjLength(Pt(L.GLABELLA), Pt(L.NASION), Axis("z")).eval(p2)
        raised = False
    except ValueError:
        raised = True
    cases.append({"op": "axis_z_in_2d", "expected": 1.0, "got": float(raised),
                  "abs_error": 0.0 if raised else 1.0, "rel_error": 0.0 if raised else 1.0,
                  "pass": raised,
                  "construction": "Axis('z') on a 2D point set must raise, not return 0"})
    return cases


# ---------------------------------------------------------------------------
# 1b -- full catalogue against independent truth
# ---------------------------------------------------------------------------

def catalogue_vs_truth() -> list[dict]:
    ps = F.point_set()
    rows: list[dict] = []
    for spec in CATALOGUE:
        t = T.TRUTH[spec.id]
        got = raw_value(spec, ps)
        err = got - t.value
        denom = abs(t.value) if abs(t.value) > 1e-12 else 1.0
        row = {
            "id": spec.id,
            "unit": spec.unit.value,
            "evidence": spec.evidence.value,
            "truth": t.value,
            "measured": got,
            "abs_error": err,
            "rel_error": err / denom,
            "truth_exact_by_construction": t.exact,
            "pass": bool(abs(err) <= 1e-9 * max(1.0, abs(t.value))),
            "note": t.note,
        }
        if spec.reference_range:
            lo, hi, src = spec.reference_range
            row["reference_range"] = f"{lo} to {hi}"
            row["inside_reference_range"] = bool(lo <= got <= hi)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 1c -- evaluate() reproduces the deterministic value
# ---------------------------------------------------------------------------

_UNIT_SCALE = ScaleEstimate(mm_per_px=1.0, relative_sd=1e-12, source=ScaleSource.RULER,
                            notes=("synthetic coordinates are already millimetres",))


def evaluate_plumbing(sd_px: float = 1e-4) -> list[dict]:
    ps = F.point_set()
    unc = LandmarkUncertainty.isotropic(ps, sd_px)
    rows = []
    for spec in CATALOGUE:
        m = evaluate(spec, ps, unc, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0,
                     have_3d=True, scale=_UNIT_SCALE, n_samples=4096, seed=1)
        raw = raw_value(spec, ps)
        if isinstance(m, Measured):
            rows.append({
                "id": spec.id, "raw": raw, "mc_median": m.value,
                "shift": m.value - raw,
                "rel_shift": (m.value - raw) / abs(raw) if abs(raw) > 1e-12 else float("inf"),
                "ci_low": m.ci_low, "ci_high": m.ci_high, "sd": m.sd,
                "n_valid": m.n_valid,
                "reportability": m.verdict.reportability.value,
                "unavailable": False,
            })
        else:
            rows.append({"id": spec.id, "raw": raw, "mc_median": None, "shift": None,
                         "rel_shift": None, "unavailable": True,
                         "reason": m.reason})
    return rows


def run() -> dict:
    prim = primitives()
    cat = catalogue_vs_truth()
    plumb = evaluate_plumbing()

    payload = {
        "arm": "1 -- synthetic geometry control",
        "question": "what is the measurement code's own error, with no model involved",
        "primitives": prim,
        "catalogue": cat,
        "evaluate_plumbing": plumb,
        "line_offset_conventions": T.LINE_OFFSET_CONVENTIONS,
        "registered_ops": sorted(registered_ops()),
        "summary": {
            "n_primitive_cases": len(prim),
            "n_primitive_failures": sum(1 for c in prim if not c["pass"]),
            "n_measurements": len(cat),
            "n_measurement_failures": sum(1 for r in cat if not r["pass"]),
            "worst_abs_error": max(abs(r["abs_error"]) for r in cat),
            "worst_rel_error": max(abs(r["rel_error"]) for r in cat),
            "worst_id": max(cat, key=lambda r: abs(r["rel_error"]))["id"],
            "n_outside_reference_range": sum(
                1 for r in cat if r.get("inside_reference_range") is False),
            "outside_reference_range": [
                r["id"] for r in cat if r.get("inside_reference_range") is False],
            "n_unavailable_under_evaluate": sum(1 for r in plumb if r["unavailable"]),
            "max_mc_rel_shift": max(
                (abs(r["rel_shift"]) for r in plumb
                 if r["rel_shift"] is not None and math.isfinite(r["rel_shift"])),
                default=None),
        },
    }
    write_json("arm01_synthetic_geometry", payload)
    write_csv("arm01_catalogue_vs_truth", cat)
    write_csv("arm01_primitives", prim)
    write_csv("arm01_evaluate_plumbing", plumb)
    return payload["summary"]


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=1, default=str))
