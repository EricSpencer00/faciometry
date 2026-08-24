# Core API contract

Read this before touching anything outside `src/faciometry/core/`, `measure/`, or
`norms/`. Those packages are complete and stable; everything else builds on
them. They carry **no torch dependency** and must keep it that way.

## The idea in one paragraph

A measurement is a declaration, not code. It is evaluated over a Monte-Carlo
ensemble drawn from per-landmark covariances, producing a distribution rather
than a number. It is then judged against a single question: *does this quantity
vary more between people than it does between photographs of the same person?*
If not, it is withheld. Kleinberg and Vanezis (2007) found facial indices moving
8 to 19 percent at ten degrees of yaw against a between-subject spread of 1.2
percent, and FISWG's 2026 guidance prohibits photo-anthropometry on that basis.
Faciometry's answer is to keep the measurements and gate them honestly.

## Types you will use

```python
from faciometry.core.landmarks import Landmark, PointSet          # named points, batched
from faciometry.core.spec      import MeasurementSpec, View, Unit, Evidence, Reportability
from faciometry.core.scale     import ScaleEstimate, from_iris, from_interpupillary, from_ruler, fuse
from faciometry.core.sensitivity import PoseSensitivity, Discriminability, gated_pose
from faciometry.measure.registry import CATALOGUE, BY_ID, for_view, satisfiable
from faciometry.measure.evaluate import LandmarkUncertainty, Measured, Unavailable, evaluate
from faciometry.models.licensing import Tier, Provenance, require, LicenseViolation
from faciometry.norms import niosh, published
```

### `PointSet`
`PointSet(index: Mapping[Landmark, int], coords: ndarray)` with `coords` shaped
`(..., n_landmarks, dim)`, `dim` in `{2, 3}`. Leading axes are the Monte-Carlo
batch. `.get(name)`, `.has(*names)`, `.missing(names)`, `.subset(names)`,
`.from_mapping({Landmark: array})`. A backend emits a **name → index** map;
never leak integer indices past the backend boundary.

Canonical frame: **+x is the subject's right in image coordinates, +y is up,
+z is toward the viewer.** Subject-left landmarks carry `_l`.

### `MeasurementSpec`
Frozen. Key fields: `id, label, view, unit, evidence, formula, references,
reference_range, pose_tolerance_deg, sensitivity, between_subject_rsd,
measured_within_person_rsd`. Derived: `.landmarks` (frozenset), `.fingerprint`
(12-hex formula hash), `.needs_metric_scale`.

`CATALOGUE` holds 68 specs. Do not add measurements anywhere else.

### `evaluate(...) -> Measured | Unavailable`
```python
evaluate(spec, points, uncertainty, *, yaw_deg, pitch_deg, roll_deg,
         have_3d, scale=None, subject_distance_m=None,
         repeatability_cv=None, n_samples=2048, seed=0)
```
`Unavailable` means a landmark is missing — a different outcome from a withheld
measurement, and the report must distinguish them. `Measured` carries `value,
ci_low, ci_high, sd, verdict, discriminability, formula_fingerprint,
landmarks_used, n_valid` and a `.format()` that never hides the interval.

### `LandmarkUncertainty`
`(index, covariances)` with covariances shaped `(n_landmarks, dim, dim)`.
`LandmarkUncertainty.isotropic(ps, sd)` is a fallback. **Prefer real heatmap
second moments** — anisotropy is the whole point, because a point on a jaw
contour is well localised across the contour and poorly along it.

### `Tier` / `require(provenance, allowed)`
`PERMISSIVE < COPYLEFT < NONCOMMERCIAL < UNLICENSED`. Every backend declares a
`Provenance`. Call `require()` **before loading weights**, not after. Default
allowed tier is `PERMISSIVE`.

## Rules that are not negotiable

1. **No scalar aggregate score.** No "overall", "harmony", "attractiveness",
   or average-of-measurements field anywhere in any output. A test enforces it.
2. **Never infer demographics.** Sex and ancestry select a normative stratum,
   are user-declared and optional, and are never model output.
3. **A withheld measurement is a result.** Print the reason, never the number.
4. **Every number carries its interval.** No bare point estimates in any
   renderer.
5. **No network egress during analysis.** Weights download is a separate,
   explicit step.
6. `core/`, `measure/`, `norms/` import numpy and nothing heavier.
