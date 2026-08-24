# Evaluation

Everything below is produced by `evals/`. Run it with:

```bash
make -C evals data     # fetch FRLL (CC BY 4.0) and the MediaPipe weights, once
make evals             # run every arm, ~40 s, writes evals/results/
```

`evals/run_all.py` is the entry point; each arm also runs standalone
(`python -m evals.arms.arm02_pose_sweep`). Every arm writes JSON to
`evals/results/`, plus CSV wherever the result is a table. Everything is seeded
from a single master seed (`evals/_bootstrap.py`, `MASTER_SEED = 20260823`) and a
rerun reproduces byte-identical output. Each result file carries a provenance
block: git sha, Python and numpy versions, platform, seed.

Nothing under `src/` was modified to produce these numbers. Where a result says
`src/` is wrong, it says so and stops there; §7 lists the changes the numbers
argue for.

---

## 1. The design, stated in full before any of it ran

Ten arms. Arms 1 to 7 need no model weights and no images; arms 8 to 10 need
real photographs and a landmarker.

| # | Arm | What it establishes | Data | Depends on |
|---|-----|--------------------|------|-----------|
| 1 | Synthetic geometry control | the measurement code's own error floor, with zero model involvement | synthetic 3D face, closed-form truth | — |
| 2 | Pose sweep, ±30° in 2.5° steps, three axes | measured per-measurement per-axis sensitivity, against the a-priori values in `core/sensitivity.py` | same face, re-projected | 1 |
| 3 | Control for arm 2: 0° repeated through a JPEG cycle | separates pipeline noise from pose effect; without it arm 2's slopes are uninterpretable | same face, rendered and re-encoded | 1 |
| 4 | Perspective sweep, 0.3 to 3.0 m | positive control on `core.scale.magnification_distortion` | same face, pinhole camera | 1 |
| 5 | Discriminability | reproduces Kleinberg & Vanezis (2007), then the full 45-measurement table at 0.5, 3 and 8° of pose | arms 2 and 3 | 2, 3 |
| 6 | NIOSH 2003 external check | are the shipped between-subject spreads reproducible, in the units claimed, and normal in shape | 3,997-subject caliper survey | — |
| 7 | Negative control on the normative model | shuffled identities must not change the percentile output | NIOSH | 6 |
| 8 | FRLL landmark accuracy and spread | real faces, human-delineated landmarks, self-reported demographics | Face Research Lab London, CC BY 4.0 | — |
| 9 | Test-retest repeatability | per-measurement within-person spread, **measured** rather than assumed | FRLL, 5 usable captures per person | 8 |
| 10 | Fairness stratification | measurement quality by self-reported ethnicity and sex, with honest intervals | FRLL | 8 |

Controls are arms, not extras. Arm 3 is the control for arm 2. Arm 4's probe is
a positive control for arm 4's catalogue sweep. Arm 7 has a negative control
(within-stratum shuffle) *and* a positive control (across-stratum shuffle) that
proves the test has power. Arm 2 has a rotation-only condition that proves the
harness returns exact zeros for rotation-invariant quantities.

### The synthetic face (arms 1-5)

45 landmarks in millimetres in the canonical frame. Some measurements are exact
by construction: the pupils sit at x = ±31.68 with identical y and z, so the
interpupillary distance is 63.36 mm and not a number that happens to come out
near it. Likewise intercanthal width 32.0, nose breadth 34.0, philtrum width
11.0, bizygomatic 141.0, bigonial 117.0, palpebral fissure heights 11.0 and
10.6, canthal tilts 6.0° and 5.0° (placed by `dy = dx·tan θ`), and hence canthal
tilt asymmetry exactly 1.0°. The rest carry realistic depth — a flat face would
make arm 2 answer a question nobody asked — and their truth is recomputed in
`evals/synth/truth.py` in pure-Python `math`, written from the anatomical
definition rather than from `registry.py`, importing none of
`core.geometry`, `core.formula` or `measure.registry`.

All 45 values land inside their published reference ranges where the spec has
one, so the face is anatomically plausible as well as arithmetically convenient.

---

## 2. Arm 1 — synthetic geometry control

**Result: passes. The measurement layer contributes nothing.**

| | |
|---|---|
| primitive algebra cases | 20, 0 failures, agreement to 1e-12 |
| catalogue measurements | 45, 0 failures |
| worst absolute error | 2.8e-14 |
| worst relative error | 3.6e-15 (`mouth_corner_asymmetry`) |
| measurements outside their published reference range | 0 |
| `evaluate()` median vs deterministic value, worst relative shift | 6.0e-6 at σ = 1e-4 mm |

The 20 primitive cases cover every node of the formula algebra, including
`angle_at` at 0.01° and 179.99° where an `arccos` implementation loses
precision and the shipped `atan2` form does not, and `Axis("z")` on a 2D point
set, which must raise rather than silently return zero. It does.

**Two things arm 1 found that are not failures but are wrong.**

**1a. `e_line_upper_lip` and `e_line_lower_lip` compute neither convention in
the literature.** `LineOffset(p, a, b, normal=Axis("z"))` returns the *z
component of the vector from the line to the point*. The Ricketts reference
range shipped on those specs (−6 to −2 mm, −4 to 0 mm) is a **perpendicular**
distance. The two differ by cos of the E-line's inclination:

| | registry (z component) | perpendicular (Ricketts) | anteroposterior (along z at the lip's own height) |
|---|---|---|---|
| upper lip | −2.672 mm | −2.937 mm | −3.228 mm |
| lower lip | −0.624 mm | −0.686 mm | −0.754 mm |

The registry value reads **9.0% low** against the perpendicular convention and
**17.2% low** against the anteroposterior one. The factor is `cos` of the
E-line's inclination, which varies between subjects with chin projection, so the
bias is subject-dependent and does not cancel in a percentile.

**1b. `nasal_tip_projection_ratio` reads `ALARE_L` only**, not the mean of the
two alae. Under yaw the two alae move oppositely (arm 2 measures 0.0099/deg on
one side), so the Goode ratio inherits a lateral asymmetry it should not have.

---

## 3. Arm 2 — pose sweep, and arm 3 — its control

### 3.1 The control first, because arm 2's numbers mean nothing without it

The face at exactly 0/0/0, rendered at 4 px/mm, pushed through 200 independent
encode/decode cycles with sub-pixel framing offsets and sensor noise, landmarks
recovered by intensity-weighted centroid.

| condition | landmark RMS residual |
|---|---|
| JPEG q85 only, no framing jitter | **0.042 mm** |
| lossless, with sub-pixel framing jitter | **0.108 mm** |
| JPEG q95 / q85 / q75, with jitter | 0.106 / 0.106 / 0.111 mm |

**Compression is not the problem.** Going from lossless to JPEG q75 moves the
landmark RMS by 2.7%. Sub-pixel framing moves it by 157%. Quality 95 and quality
75 are indistinguishable.

Resulting measurement noise floor at q85 — median 0.029% for lengths and
ratios, 0.019° for angles:

| measurement | noise (relative, or degrees) |
|---|---|
| `nasolabial_angle` | 0.161° |
| `mentolabial_angle` | 0.066° |
| `nasofrontal_angle` | 0.037° |
| `canthal_tilt_asymmetry` | 0.026° |
| `ocular_height_asymmetry` | 1.78% |
| `mouth_corner_asymmetry` | 1.15% |
| `interpupillary_distance` | 0.015% |
| `bizygomatic_width` | 0.006% |

This is a **lower bound**: the recovery window is centred on the true position,
so it measures encoding and resampling only, not detector failure. Arm 9
measures the real thing on real photographs and gets numbers three orders of
magnitude larger.

### 3.2 The sweep

Five conditions, three axes, ±30° in 2.5° steps, all 45 measurements — 675 slope
rows in `evals/results/arm02_slopes.csv`. The comparison metric matches how the
a-priori numbers were built: `cosine_yaw_sensitivity()` is not a derivative, it
is `(1 − cos 10°)/10`, a secant at ten degrees. So the measured counterpart is
the mean of `|v(+10) − v(0)|` and `|v(−10) − v(0)|` over ten degrees.

**Positive control.** In the `3d` condition (rotate, do not project), every
distance and every non-axis angle returns a slope of exactly `0.00000`. Only
the 21 measurements defined against an image axis move. The harness is
measuring projection, not arithmetic drift.

**Which measurements are properties of the pose rather than of the face.** These
move under pure 3D rotation with no projection at all:

- all three canthal tilts and `canthal_tilt_asymmetry` (defined against the
  image x axis),
- `ocular_height_asymmetry` and `mouth_corner_asymmetry` (defined against the
  image y axis),
- `e_line_upper_lip`, `e_line_lower_lip`, `chin_projection`,
  `nasal_tip_projection_ratio` (defined against the image z axis).

That is legitimate — they are horizon-referenced by definition — but it means
the canonicalisation error propagates into them one-for-one, and it is not
something a "3D-first" pipeline removes.

### 3.3 The headline disagreements

86 of 135 (measurement, axis) pairs disagree with their a-priori value by more
than a factor of two. The worst, ordered by measured-over-a-priori:

| measurement | axis | a-priori | measured (ortho) | measured (0.5 m) | ratio |
|---|---|---|---|---|---|
| `ocular_height_asymmetry` | roll | 0.002 | **2.959** | 2.959 | **1480×** |
| `canthal_tilt_asymmetry` | roll | 0.002 | **1.900** | 1.894 | **950×** |
| `mouth_corner_asymmetry` | roll | 0.002 | **1.029** | 1.029 | **514×** |
| `gonial_angle_r` | roll | 0.05 | 1.147 | 1.349 | 27× |
| `gonial_angle_l` | roll | 0.05 | 1.147 | 0.860 | 23× |
| `mouth_corner_asymmetry` | yaw | 0.002 | 0.0015 | 0.102 | 51× |
| `eye_aspect_ratio_l` | yaw | 0.00015 | 0.0062 | 0.0073 | 48× |
| `ocular_height_asymmetry` | pitch | 0.002 | 0.0015 | 0.044 (asym. face) | 22× |
| `chin_projection` | pitch | 0.010 | 0.191 | 0.191 | 19× |
| `palpebral_fissure_width_{l,r}` | yaw | 0.0015 | 0.0059 | 0.0081 | 5.3× |
| `gonial_angle_{l,r}` | yaw | 0.15 | 0.447 | 0.639 | 4.3× |
| `philtrum_length` | pitch | 0.0015 | 0.0070 | 0.0039 | 4.6× |

And in the other direction, a-priori values that are far too pessimistic:

| measurement | axis | a-priori | measured (worst of five conditions) | ratio |
|---|---|---|---|---|
| `bizygomatic_width` | yaw | 0.019 (`KLEINBERG_WORST`) | 0.00152 | 0.08× |
| `bigonial_width` | yaw | 0.019 | 0.00152 | 0.08× |
| `submental_length` | yaw | 0.019 | 0.00152 | 0.08× |
| `facial_convexity_angle` | yaw | 0.15 | 0.0103 | 0.07× |
| `e_line_{upper,lower}_lip` | yaw | 0.019 | 0.0025 | 0.13× |

### 3.4 The three claims in `registry.py` that are backwards

`_CANCELLING` in `measure/registry.py` assigns 0.002/deg on all three axes to
`canthal_tilt_asymmetry`, `ocular_height_asymmetry` and
`mouth_corner_asymmetry`, with the comment *"a common-mode rotation cancels
between sides"*, and `spec.py` calls `canthal_tilt_asymmetry` *"the one canthal
quantity that image roll cannot corrupt"*.

Measured, on the synthetic face whose true asymmetry is exactly 1.0°:

| image roll | `canthal_tilt_l` | `canthal_tilt_r` | asymmetry | ocular height asym. | mouth corner asym. |
|---|---|---|---|---|---|
| 0° | 5.000 | 6.000 | **1.000** | 0.00568 | 0.01538 |
| 2° | 3.000 | 8.000 | **5.000** | 0.04057 | 0.05027 |
| 5° | 0.000 | 11.000 | **11.000** | 0.09281 | 0.10247 |
| 10° | −5.000 | 16.000 | **21.000** | 0.17924 | 0.18878 |

Roll enters `canthal_tilt_asymmetry` at **two degrees per degree** — twice the
per-side sensitivity, not zero. Two degrees of roll turns a 1° asymmetry into a
5° one. Seven degrees of roll would turn it into 15°, which is a face nobody
has.

The mechanism is in the formula, not the photograph. `canthal_tilt_l` uses
`Axis("-x")` and `canthal_tilt_r` uses `Axis("x")`. Flipping the reference axis
is what makes one tilt formula serve both sides with a consistent sign — and it
is exactly what makes an image roll **anti-symmetric** between the two sides, so
it adds in the difference instead of cancelling. The same applies to the two
`ProjLength(..., Axis("y"))` asymmetries: roll tilts the whole face, so
`y_left − y_right` picks up `2·sin(roll)·x_half`, and `x_half` is 45 mm against a
true asymmetry of 0.5 mm.

These three are currently the **most** roll-sensitive measurements in the
catalogue and are declared the least.

### 3.5 The structural finding: cosine is second order, depth offset is first

Under a symmetric face and orthographic projection, every transverse width
scales by exactly `cos(yaw)` and every ratio of parallel spans is exactly
invariant — orthographic projection of a rotation is affine. The a-priori model
is exactly right in that regime, and that regime does not exist.

Two things break it, and both are first order in the angle where the cosine is
second order:

- **Depth offset between the endpoints.** `palpebral_fissure_width` runs 29 mm
  laterally and 10 mm in depth, so its projected length changes as
  `Δx·cos θ + Δz·sin θ` — a slope of `Δz/Δx = 0.345` per radian, or 0.0060 per
  degree. Measured: 0.0059. That is **3.9× the cosine value**, and it is a
  general fact about any span whose endpoints are not co-planar.
- **Perspective.** At 0.5 m every a-priori pitch sensitivity of exactly `0.0`
  becomes 0.0005 to 0.0009 per degree for the transverse widths, and the
  transverse ratios go from an exact 0 to 0.0001-0.0025.

`sensitivity.py`'s own docstring says the cosine is *"a lower bound … Kleinberg's
measured curves have turning points away from zero, which proves the landmarks
do not lie in a single plane."* This arm is the quantity of that.

---

## 4. Arm 4 — perspective sweep

### 4.1 The probe (positive control)

Two transverse segments of identical true length, one on the eye plane and one
exactly 50 mm in front of it, through a real pinhole camera.

| distance | measured magnification | `magnification_distortion` (K = 50/d) | exact pinhole (K/(1−K)) | ICAO error |
|---|---|---|---|---|
| 0.3 m | 0.200000 | 0.166667 | 0.200000 | **−16.67%** |
| 0.4 m | 0.142857 | 0.125000 | 0.142857 | −12.50% |
| 0.5 m | 0.111111 | 0.100000 | 0.111111 | −10.00% |
| 0.7 m | 0.076923 | 0.071429 | 0.076923 | −7.14% |
| 1.0 m | 0.052632 | 0.050000 | 0.052632 | −5.00% |
| 1.5 m | 0.034483 | 0.033333 | 0.034483 | −3.33% |
| 2.0 m | 0.025641 | 0.025000 | 0.025641 | −2.50% |
| 3.0 m | 0.016949 | 0.016667 | 0.016949 | −1.67% |

The measured curve matches the **exact** pinhole form `K/(1−K)` to 9.4e-17 at
every distance. `core.scale.magnification_distortion` returns `K`, which is the
first-order term, and **understates the true magnification by exactly a factor
of K**: 16.7% at 0.3 m, 5.0% at 1.0 m. `decide_reportability` prints that number
to the user (*"implies roughly {k*100:.1f}% perspective magnification"*), so the
under-report reaches the report.

Neither model is wrong about anything else: the closed form is the right shape,
and the withhold-below-0.6 m rule is if anything too lenient.

### 4.2 The catalogue

For 32 of the 34 non-angular measurements, `fractional_change × distance` is
constant across all eight distances to within 10% — confirming the first-order
`(z₁ − z₂)/d` model. **22 of 45 measurements distort by more than 1% at 1.0 m**,
the ICAO portrait distance, at zero pose:

| measurement | 0.3 m | 0.5 m | 1.0 m | 3.0 m | implied depth straddle |
|---|---|---|---|---|---|
| `gonial_angle_r` | −28.63° | −15.86° | **−7.33°** | −2.31° | — |
| `gonial_angle_l` | +16.62° | +11.20° | **+6.13°** | +2.17° | — |
| `philtrum_length` | +17.8% | +10.1% | +4.9% | +1.6% | 49.8 mm |
| `facial_width_height_ratio` | −14.4% | −8.9% | −4.5% | −1.5% | −44.9 mm |
| `bigonial_width` | −8.5% | −5.3% | −2.7% | −0.9% | −26.9 mm |
| `facial_thirds_ratio` | +7.8% | +4.5% | +2.2% | +0.7% | +22.0 mm |
| `bizygomatic_width` | −6.8% | −4.2% | −2.2% | −0.7% | −21.3 mm |
| `eye_aspect_ratio_{l,r}` | +4.6% | +2.7% | +1.4% | +0.4% | +13.5 mm |
| `canthal_tilt_{l,r}` | +0.25° | +0.15° | +0.07° | +0.02° | — |

**The gonial angle number is the one to look at.** A perfectly symmetric jaw,
photographed in profile at 1.0 m, produces gonial angles that differ by 13.5°
between the near and far side, purely from perspective. The published
between-subject SD for that measurement is 6.05° (Saadeh et al. 2025). At the
recommended portrait distance the perspective artifact is **more than twice the
entire population spread**. (In a real profile photograph the far-side gonion is
occluded and would not be landmarked at all; the near-side error alone is
−7.33°.)

---

## 5. Arm 5 — discriminability

### 5.1 Kleinberg and Vanezis reproduced

An index with a between-subject relative spread of 1.2% at ten degrees of yaw:

| pose error at 10° yaw | source | discriminability ratio | informative? |
|---|---|---|---|
| 8% | Kleinberg & Vanezis 2007, lower end of measured index movement | **0.150** | no |
| 19% | Kleinberg & Vanezis 2007, upper end | **0.063** | no |
| 19% | Vitruve `KLEINBERG_WORST.error_at(10,0,0)` | 0.063 | no |
| 28.4% | the same, after `gated_pose` inflates 10° to 14.98° | 0.042 | no |
| 1.5% | Vitruve `TRANSVERSE_WIDTH` (a depth-matched width) | 0.790 | no |

Reproduced. Every route gives a ratio below 1, and the published-number route
gives 0.150 and 0.063.

### 5.2 Can Vitruve's own geometry produce an 8-19% movement?

Partly. Excluding the three asymmetry measurements (whose zero-pose value is
near zero by construction, so a relative movement is not comparable to one of
Kleinberg's indices), the largest movement at ten degrees of yaw is:

| condition | worst index | movement at 10° yaw |
|---|---|---|
| orthographic | `nasal_tip_projection_ratio` | 9.89% |
| perspective 1.0 m | `nasal_tip_projection_ratio` | 9.31% |
| perspective 0.5 m | `nasal_tip_projection_ratio` | 8.76% |
| perspective 0.5 m | `palpebral_fissure_width_l` | 8.10% |
| perspective 0.5 m | `eye_aspect_ratio_l` | 7.25% |

So Vitruve's geometry **reaches the bottom of Kleinberg's 8-19% band and never
the top**, and only for indices whose endpoints straddle depth planes.
Depth-matched transverse widths move 1.5%, six times less than Kleinberg's floor
and thirteen times less than his ceiling. Two conclusions follow: `TRANSVERSE_WIDTH`
is not describing what Kleinberg measured, and `KLEINBERG_WORST` — applied in
`_SENSITIVITY` to `bizygomatic_width` and `bigonial_width` — is describing a
*landmark* failure (a silhouette that is not the anatomical point), not a
projection failure. It is in the wrong term of the error budget. Arm 8 measures
the term it belongs in: 8-9 mm.

### 5.3 The pose gate is dominated by its own uncertainty assumption

`assess_discriminability` inflates every pose axis by `gated_pose`, adding
`POSE_ESTIMATOR_SD_DEG = 4.977°`:

| true pose | gated pose | fraction that is estimator uncertainty |
|---|---|---|
| 0.5° | 5.48° | **90.9%** |
| 3.0° | 7.98° | 62.4% |
| 8.0° | 12.98° | 38.3% |

Between 0.5° and 8° the true pose changes by 16× and the gated pose by 2.4×.
That is why the shipped verdicts in the table below barely move across the whole
range: the discriminability gate is currently reporting an assumption about
6DRepNet's accuracy more than it is reporting the photograph. Arm 8 measures the
real thing on a 102-subject studio set and the front-capture pose 95th
percentile is 3.9° yaw, 8.3° pitch, 3.9° roll — so a 4.977° inflation is of the
same order as the entire pose distribution it is inflating.

### 5.4 The full table

Discriminability ratio at 0.5, 3 and 8 degrees applied to all three axes, with
the landmark-error term taken from arm 3's measured pipeline noise. **as
shipped** uses the a-priori `PoseSensitivity`; **measured** substitutes arm 2's
slopes at 1.0 m. Full table with all conditions in
`evals/results/arm05_table.csv`.

| measurement | between-subject spread | ship 0.5° | ship 3° | ship 8° | meas 0.5° | meas 3° | meas 8° |
|---|---|---|---|---|---|---|---|
| `canthal_tilt_l` | 2.10° | 0.37 | 0.25 | 0.16 | 0.36 | 0.25 | 0.15 |
| `bizygomatic_width` | 5.44% | 0.45 | 0.31 | 0.19 | **6.90** | 4.74 | 2.91 |
| `e_line_upper_lip` | 6.00% | 0.49 | 0.34 | 0.21 | 1.16 | 0.86 | 0.55 |
| `e_line_lower_lip` | 6.00% | 0.50 | 0.34 | 0.21 | 1.30 | 0.90 | 0.56 |
| `chin_projection` | 6.00% | 0.50 | 0.34 | 0.21 | **0.06** | 0.04 | 0.02 |
| `submental_length` | 6.00% | 0.50 | 0.34 | 0.21 | **5.19** | 3.56 | 2.19 |
| `bigonial_width` | 9.45% | 0.78 | 0.54 | 0.33 | **10.10** | 6.93 | 4.26 |
| `facial_width_height_ratio` | 7.96% | 0.66 | 0.66 | 0.66 | 0.66 | 0.66 | 0.66 |
| `jaw_cheekbone_ratio` | 6.68% | 0.83 | 0.83 | 0.83 | 0.83 | 0.83 | 0.83 |
| `biocular_width` | 4.23% | 5.09 | 3.49 | 2.15 | 4.82 | 3.31 | 2.03 |
| `gonial_angle_l` | 6.05° | 5.91 | 4.06 | 2.49 | **0.96** | 0.66 | 0.40 |
| `palpebral_fissure_height_l` | 6.00% | 7.18 | 4.94 | 3.04 | 6.75 | 4.65 | 2.86 |
| `philtrum_width` | 6.00% | 7.18 | 4.94 | 3.04 | 6.35 | 4.37 | 2.69 |
| `philtrum_length` | 6.00% | 7.19 | 4.95 | 3.04 | **2.01** | 1.38 | **0.85** |
| `palpebral_fissure_width_l` | 6.00% | 7.21 | 4.95 | 3.04 | **1.54** | 1.06 | **0.65** |
| `upper_face_height` | 6.00% | 7.21 | 4.95 | 3.04 | 6.78 | 4.66 | 2.86 |
| `palpebral_fissure_width_r` | 6.04% | 7.26 | 4.99 | 3.07 | **1.55** | 1.06 | **0.65** |
| `interpupillary_distance` | 6.17% | 7.42 | 5.09 | 3.13 | 6.99 | 4.80 | 2.95 |
| `nasofrontal_angle` | 9.19° | 7.70 | 5.29 | 3.25 | 16.04 | 11.03 | 6.78 |
| `face_height_sellion_menton` | 6.80% | 8.16 | 5.61 | 3.45 | 7.03 | 4.83 | 2.97 |
| `intercanthal_width` | 7.23% | 8.69 | 5.97 | 3.67 | 8.07 | 5.54 | 3.41 |
| `lower_third_height` | 7.67% | 9.17 | 6.31 | 3.89 | **2.50** | 1.71 | 1.05 |
| `middle_third_height` | 8.02% | 9.64 | 6.62 | 4.07 | 9.12 | 6.26 | 3.85 |
| `nasolabial_angle` | 12.05° | 10.01 | 6.90 | 4.25 | 21.36 | 14.99 | 9.33 |
| `mouth_width` | 8.88% | 10.67 | 7.32 | 4.50 | 9.44 | 6.48 | 3.99 |
| `nose_height` | 8.91% | 10.71 | 7.36 | 4.52 | 5.45 | 3.74 | 2.30 |
| `palpebral_fissure_height_r` | 9.09% | 10.88 | 7.49 | 4.61 | 10.24 | 7.04 | 4.33 |
| `nose_breadth` | 13.32% | 16.01 | 10.99 | 6.76 | 15.28 | 10.49 | 6.45 |
| `intercanthal_biocular_ratio` | 5.43% | 61.8 | 43.6 | 27.3 | 118.4 | 90.8 | 60.0 |
| `nose_mouth_width_ratio` | 11.07% | 125.2 | 88.7 | 55.5 | 48.3 | 33.3 | 20.5 |
| 15 further measurements | **unknown** | — | — | — | — | — | — |

The 15 with no between-subject spread at all — `eye_aspect_ratio_{l,r}`,
`canthal_tilt_r`, `eye_spacing_ratio`, `facial_thirds_ratio`,
`lip_vermilion_ratio`, `nasal_tip_projection_ratio`, `gonial_angle_r`,
`facial_convexity_angle`, `mentolabial_angle`, `mentocervical_angle`,
`submental_cervical_angle`, `ocular_height_asymmetry`,
`mouth_corner_asymmetry`, `canthal_tilt_asymmetry` — cannot be judged at all
and are reported as "unknown", which is the correct behaviour.

**The left/right split is a bug.** `canthal_tilt_l` has a spread of 2.10° and is
withheld at every pose; `canthal_tilt_r` has none and is merely caveated.
`gonial_angle_l` has 6.05° and `gonial_angle_r` has none.
`palpebral_fissure_width_r` gets 6.04% from Jayaratne and `_l` gets the 6.00%
default. The same measurement on the two sides of the same face receives
different verdicts, because `norms/published.py` transcribed only the left or
only the right entry from each source. Nothing in the anatomy justifies it.

### 5.5 Crossover: where arm 2's slopes become observable

The pose angle at which the pose effect equals arm 3's pipeline noise:

| measurement | axis | noise | slope | crossover |
|---|---|---|---|---|
| `ocular_height_asymmetry` | roll | 1.78% | 2.959/deg | **0.006°** |
| `mouth_corner_asymmetry` | roll | 1.15% | 1.029/deg | 0.011° |
| `bizygomatic_width` | yaw | 0.006% | 0.0015/deg | 0.040° |
| `interpupillary_distance` | yaw | 0.015% | 0.0015/deg | 0.096° |

Every slope in arm 2 is far above the noise floor that produced it. The sweep is
measuring pose, not its own arithmetic.

---

## 6. Arms 6 and 7 — the normative model

### 6.1 Arm 6a: the vendored JSON reproduces exactly

All **99 stratum cells** recomputed from the raw 3,997-subject CSV match
`norms/data/niosh2003.json` to 1e-4 on mean and sd and exactly on n. Both
documented parsing quirks are real and are handled: `-9,999` missing values with
a thousands separator inside a quoted field, and `NECKCIRC`'s structural
missingness.

### 6.2 Arm 6b: the units question

**NIOSH `INTPUPBR` is millimetres.** Three independent confirmations:

- mean 64.37 mm, SD 3.97, range 51.5 to 79.0 — against Dodgson's ANSUR value of
  63.36 ± 3.83 that `core/scale.py` ships. Difference **+1.6%**, consistent with
  a different sampling frame (US respirator users, deliberately oversampled for
  racial minorities) and not with a unit error.
- 49.2% of `INTPUPBR` values carry a fractional part (0.5 mm resolution), while
  `BIZYGOBR`, `BIGONLBR`, `NOSEBR`, `HEADCIRC` and `STATURE` are **100%
  integers** in the same file. A tenths-of-a-millimetre column would be
  integer-valued.
- male 65.07, female 63.14 — bracketing `IPD_PRIORS` male 64.67 / female 62.31.

ANSUR II's `interpupillarybreadth` is recorded in **tenths of a millimetre**, so
the same quantity reads about 633 there. Reading it as millimetres gives a face
ten times too large and, because Vitruve recovers scale from interpupillary
distance, a millimetres-per-pixel ten times too small — every length in the
report off by an order of magnitude, and every ratio unaffected, so the error
would be invisible in exactly half the output.

**Not verified directly:** no copy of ANSUR II is vendored in this repo and the
harness does not download one, so the "tenths of a millimetre" claim is
confirmed only by its consistency with the NIOSH and Dodgson values, not by
reading the ANSUR II file. See §9.

### 6.3 Arm 6c: distribution shape

99 strata tested. **73 reject normality at 5%** by Anderson-Darling. But the
skew is modest (max |skew| 0.64, `nose_breadth male|pooled`) and the practical
consequence is small: empirical coverage of the nominal central 90% interval
runs 0.890 to 0.924 against a nominal 0.900. The normal model is statistically
rejected at n≈4000 and adequate for central percentiles; it is the tails that
misbehave, and it misbehaves worst for the right-skewed dimensions
(`nose_breadth` +0.47, `bigonial_width` +0.44).

`DEFAULT_LINEAR_RSD = 0.06` in `norms/published.py` sits inside the measured
within-sex range of **4.54% to 13.74%** but below its median of **7.53%**. That
errs toward under-stating between-subject spread, which under-states
discriminability, which withholds more. Safe direction, but it is an assumption
and the report should keep saying so.

### 6.4 Arm 6d: the undeclared-sex fallback inflates the spread

`niosh.spread()` falls back to `both|pooled` when no sex is declared. Pooling
two populations with different means inflates the SD, and that SD is the
*numerator* of the discriminability ratio:

| measurement | pooled RSD | within-sex RSD | inflation | η² (sex) |
|---|---|---|---|---|
| `face_height_sellion_menton` | 6.80% | 5.77% | **1.18×** | 0.280 |
| `bizygomatic_width` | 5.44% | 4.78% | 1.14× | 0.230 |
| `nose_height` | 8.91% | 7.96% | 1.12× | 0.202 |
| `bigonial_width` | 9.45% | 8.69% | 1.09× | 0.154 |
| `nose_mouth_width_ratio` | 11.07% | 11.07% | 1.00× | 0.001 |

So a subject who declines to declare a sex gets a discriminability ratio up to
**18% higher** than one who declares — i.e. Vitruve is more willing to print a
number for the subject it knows less about. Backwards.

### 6.5 Arm 7 — the negative control

There is no percentile function in `src/` yet; `norms.niosh` stops at
`stratum()` and `spread()`. This arm implements the percentile the norms module
implies — a normal CDF against the narrowest available stratum — and tests it.
When the shipped code grows a percentile function, this is the test it has to
pass.

| measurement | identity KS | uniform at 1%? | within-shuffle identical? | across-shuffle KS | sex sep. (stratified) | sex sep. (pooled) | LOO − identity |
|---|---|---|---|---|---|---|---|
| `interpupillary_distance` | 0.027 | no | **yes** | 0.079 | 0.051 | **0.219** | −0.0001 |
| `bizygomatic_width` | 0.030 | no | **yes** | 0.087 | 0.054 | **0.432** | +0.0004 |
| `bigonial_width` | 0.054 | no | **yes** | 0.072 | 0.052 | **0.347** | +0.0002 |
| `nose_breadth` | 0.064 | no | **yes** | 0.159 | 0.083 | 0.205 | +0.00002 |
| `mouth_width` | 0.031 | no | **yes** | 0.080 | 0.091 | 0.223 | +0.00002 |
| `nose_height` | 0.039 | no | **yes** | 0.079 | 0.062 | **0.417** | +0.00001 |
| `face_height_sellion_menton` | 0.026 | no | **yes** | 0.104 | 0.043 | **0.483** | −0.00003 |
| `jaw_cheekbone_ratio` | 0.021 | **yes** | **yes** | 0.028 | 0.025 | 0.149 | +0.0001 |
| `nose_mouth_width_ratio` | 0.031 | no | **yes** | 0.085 | 0.021 | 0.034 | −0.00003 |

**The negative control passes.** Permuting measurements within a stratum leaves
the percentile distribution bit-identical in all nine cases. The model is a pure
function of (value, stratum) and carries nothing about who the subject is.

**The positive control passes too**, which is what makes the negative one
meaningful: permuting demographic labels *across* strata raises the KS statistic
by a factor of 1.3 to 4.0 in seven of nine, so the test can see a difference
when there is one.

**Leave-one-out is irrelevant at this n:** the largest change is 0.0004.

**Two things the arm found that are not about leakage.**

- Only one of nine measurements produces uniform percentiles under correct
  strata. The largest KS is 0.064 (`nose_breadth`), so a reported percentile can
  be systematically off by up to **6.4 percentile points** purely from the
  normal-model misfit measured in arm 6c.
- Under the pooled fallback the male and female percentile distributions
  separate at KS up to **0.483** (`face_height_sellion_menton`), against 0.043
  when stratified. A pooled percentile for that measurement **is a sex readout**:
  the design rule "never infer demographics" is not violated by any model output
  here, but it is quietly violated by the percentile itself when the subject has
  not declared a sex.

---

## 7. Arms 8, 9 and 10 — real faces

Face Research Lab London Set, DOI 10.6084/m9.figshare.5047666.v5, **CC BY 4.0**,
102 individuals with self-reported age, gender and ethnicity. Obtained through
the figshare API (the web page 403s automated fetches; the API does not).

Only the `neutral_front` capture ships with the 189-point human-delineated
template; the other captures are images only. Landmarks for those come from
**MediaPipe FaceLandmarker v1 float16** (Apache-2.0, `Tier.PERMISSIVE`), used
directly because `vitruve.models` was not ready.

Demographics: 69 white, 13 black, 10 west Asian, 9 east Asian, 1 mixed; 53 male,
49 female; ages 18-47 (2 missing).

### 7.1 Arm 8b — landmark agreement, and what it says about landmark tiers

Median distance between MediaPipe and the human template on the same 102 images,
normalised by interpupillary distance and shown in millimetres at a 63.4 mm IPD:

| landmark | median error / IPD | mm |
|---|---|---|
| `trichion` | **0.451** | **28.6** |
| `superciliare_{l,r}` | 0.161-0.172 | 10.2-10.9 |
| `gonion_{r,l}` | 0.128-0.143 | **8.1-9.0** |
| `gnathion` / `menton` | 0.113 | 7.1 |
| `tragion_{l,r}` | 0.105 | 6.7 |
| `zygion_{r,l}` | 0.097-0.105 | **6.1-6.7** |
| `glabella` | 0.098 | 6.2 |
| `subalare_{l,r}` | 0.102 | 6.5 |
| ... | ... | ... |
| `exocanthion_{l,r}` | 0.043-0.050 | 2.7-3.2 |
| `pupil_{l,r}` | 0.032-0.039 | 2.0-2.5 |
| `endocanthion_{l,r}` | 0.029-0.033 | 1.8-2.1 |
| `pronasale` | 0.028 | 1.8 |
| `columella` | 0.020 | 1.3 |
| `subnasale` | **0.017** | **1.1** |

The ordering is the thesis. Interior, well-defined midline and periocular points
agree to 1-3 mm. Points on a laterally curved, self-occluding surface — gonion,
zygion, tragion, and the chin contour — disagree by 6-9 mm between two observers
looking at *the same photograph*, with no pose change involved at all.

`gonion` at 8-9 mm sits right on Lim et al. (2022)'s 9.3 mm mean difference for
bigonial breadth against calipers. Two independent observers of one image
disagree by as much as photogrammetry disagrees with a caliper. That is a
property of the landmark, not of either observer.

**Caveat that cannot be resolved here:** for `trichion` (28.6 mm) the MediaPipe
index used (10, top of the forehead mesh) is almost certainly not the hairline,
and that is a mapping error in `evals/frll.py`, not a finding about the
landmark. `superciliare` is likely a partial mapping error too. For gonion,
zygion, tragion, gnathion and menton the disagreement is consistent with the
landmark being genuinely ill-defined in 2D, but the harness cannot separate
"ill-defined landmark" from "imperfect index map" without a third observer.
`trichion` is not used by any catalogue measurement, so nothing downstream
depends on it.

### 7.2 Arm 8c — measurement agreement between two landmark sources

Bland-Altman, the same statistic Lim et al. report for photogrammetry against
calipers, so these are comparable in kind:

| measurement | template | MediaPipe | mean difference | limits of agreement | LoA width / between-subject SD | r |
|---|---|---|---|---|---|---|
| `palpebral_fissure_width_r` | 26.88 | 25.77 | −1.11 mm | [−4.21, +1.99] | **3.93** | 0.22 |
| `bizygomatic_width` | 122.5 | 130.4 | +7.94 mm | [−7.96, +23.84] | 3.69 | 0.44 |
| `jaw_cheekbone_ratio` | 0.903 | 0.796 | −0.107 | [−0.248, +0.034] | 3.55 | 0.48 |
| `biocular_width` | 87.5 | 82.2 | −5.25 mm | [−13.05, +2.54] | 3.34 | 0.52 |
| `bigonial_width` | 110.4 | 103.8 | **−6.56 mm** | **[−21.67, +8.55]** | 2.82 | 0.70 |
| `interpupillary_distance` | 62.55 | 59.41 | −3.14 mm | [−8.45, +2.17] | 3.02 | 0.64 |
| `nose_breadth` | 36.3 | 33.1 | −3.22 mm | [−7.67, +1.23] | 2.31 | 0.83 |
| `facial_thirds_ratio` | 0.797 | 0.959 | +0.161 | [+0.091, +0.232] | **1.38** | 0.94 |
| `ocular_height_asymmetry` | 0.026 | 0.026 | +0.001 | [−0.015, +0.016] | 1.54 | 0.92 |

`bigonial_width` at −6.56 mm with limits of agreement spanning 30 mm is the same
order as Lim et al.'s 9.3 mm and −0.9 to 19.6 mm — obtained here without any
calipers, purely by asking two landmark sources about the same photograph.

**For 30 of 32 measurements the limits of agreement are wider than twice the
between-subject SD.** Changing the landmark backend moves a measurement further
than the population varies. The two exceptions are `facial_thirds_ratio` and
`ocular_height_asymmetry`, and `facial_thirds_ratio` achieves it with a +20%
systematic bias — precise but not accurate, which is fine for ranking and fatal
for an absolute percentile.

### 7.3 Arm 8a — external validity against calipers

FRLL measurements scaled from the iris (`core.scale.from_iris`), against NIOSH:

| measurement | FRLL mean | NIOSH mean | bias | FRLL RSD [95% CI] | NIOSH RSD |
|---|---|---|---|---|---|
| `nose_mouth_width_ratio` | 0.727 | 0.730 | **−0.7%** | 0.075 [0.064, 0.084] | 0.111 |
| `mouth_width` | 50.0 | 50.7 | −1.5% | 0.069 [0.060, 0.076] | 0.089 |
| `nose_breadth` | 36.3 | 37.1 | −2.0% | 0.106 [0.092, 0.119] | 0.133 |
| `interpupillary_distance` | 62.5 | 64.4 | −2.8% | 0.056 [0.049, 0.063] | 0.062 |
| `bigonial_width` | 110.4 | 117.4 | −6.0% | 0.097 [0.083, 0.111] | 0.095 |
| `face_height_sellion_menton` | 111.0 | 119.9 | −7.4% | 0.068 [0.059, 0.077] | 0.068 |
| `jaw_cheekbone_ratio` | 0.903 | 0.830 | **+8.6%** | 0.088 [0.075, 0.100] | 0.067 |
| `bizygomatic_width` | 122.5 | 141.1 | **−13.2%** | 0.070 [0.059, 0.081] | 0.054 |
| `nose_height` | 40.8 | 50.4 | **−19.0%** | 0.078 [0.067, 0.088] | 0.089 |

Scale recovery from the iris is good: measurements that depend only on
well-defined points land within 3% of a 3,997-person caliper survey, from a
population prior on corneal diameter and nothing else.

The three large biases are all **landmark-definition** failures, not scale
failures. `bizygomatic_width` is 13% low because the FRL template's "cheekbone"
points are soft-tissue cheek points inside the true zygion silhouette;
`jaw_cheekbone_ratio` is 8.6% high because its denominator carries that bias;
`nose_height` is 19% low because the sellion synthesised from the nasal-bridge
points sits below the true sellion. These biases are *systematic per landmark
map* and therefore land directly in the percentile — arm 10 shows the
consequence.

### 7.4 Arm 8d — what a studio "front" capture actually looks like

FRLL is about as controlled as face photography gets: fixed rig, seated subject,
directed gaze.

| view | detected | yaw (mean ± sd, p95 abs) | pitch | roll |
|---|---|---|---|---|
| `neutral_front` | 102/102 | −0.4 ± 2.1, **3.9** | −0.8 ± 4.3, **8.3** | +0.1 ± 1.8, **3.9** |
| `smiling_front` | 102/102 | −0.5 ± 2.2, 4.6 | +0.8 ± 4.4, 8.7 | +0.3 ± 2.1, 3.9 |
| `neutral_left_3quarter` | 102/102 | −35.2 ± 3.4 | −1.4 ± 3.5 | +0.6 ± 3.4 |
| `neutral_right_3quarter` | 102/102 | +36.1 ± 3.5 | −1.4 ± 3.8 | −0.8 ± 3.2 |
| `smiling_left_3quarter` | 102/102 | −33.5 ± 3.6 | −1.7 ± 4.0 | +2.4 ± 3.6 |
| `neutral_left_profile` | **12/102** | −57.9 ± 6.9 | 0.0 ± 3.8 | −8.9 ± 27.6 |
| `neutral_right_profile` | **14/102** | +54.6 ± 15.2 | −7.2 ± 10.1 | −4.9 ± 15.1 |

Two results.

**A studio frontal capture carries 8.3° of pitch at the 95th percentile.** The
catalogue's `pose_tolerance_deg` is 6.0 for most verticals, 5.0 for the
`REQUIRES_3D` widths and 3.0 for canthal tilt. Applying the shipped tolerances
directly, 83% of these studio captures pass. Applying `gated_pose`'s +4.977°
inflation as `assess_discriminability` does, **28% pass**. The gate as
implemented would refuse nearly three quarters of the best-controlled face
photographs in existence.

**MediaPipe cannot do a 90° profile: 12-14% detection.** Where it does detect
one, it reports 55-58° of yaw for a 90° head, and the roll estimate has a
standard deviation of 15-28°. The 13 `View.PROFILE` measurements in the
catalogue — every angle in the profile section, both E-line offsets,
`chin_projection`, `submental_length`, `nasal_tip_projection_ratio` — have **no
working landmark source** in this stack. That is not a failure of this arm; it
is the current state of the pipeline, and half of the catalogue's clinical
vocabulary depends on it.

### 7.5 Arm 9 — test-retest repeatability, measured

`MeasurementSpec.measured_within_person_rsd` is populated for **2 of 45**
measurements. This is the other 30 that FRLL can reach. Within-person spread
pooled across subjects as the root mean of within-subject variances; between-
person spread from the neutral front capture; both from MediaPipe on real
photographs, with no projection model anywhere in the calculation.

Two conditions: **expression** (neutral front vs smiling front — same camera,
same pose, different face) and **all usable** (those two plus the three
3/4 captures at ±35° yaw).

| measurement | between-person | expr. CV | all CV | **D (expr)** [95% CI] | **D (all)** [95% CI] |
|---|---|---|---|---|---|
| `facial_thirds_ratio` | 9.18% | 3.58% | 3.73% | 2.57 [2.25, —] | **2.46** [2.25, —] |
| `jaw_cheekbone_ratio` | 2.59% | 0.78% | 1.61% | 3.32 [2.93, —] | **1.61** [1.50, —] |
| `lip_vermilion_ratio` | 12.52% | 8.01% | 8.55% | 1.56 [1.38, —] | **1.46** [1.37, —] |
| `eye_spacing_ratio` | 6.85% | 4.55% | 5.18% | 1.51 [1.37, —] | **1.32** [1.22, —] |
| `canthal_tilt_l` | 2.72° | 0.99° | 2.53° | 2.17 [1.82, —] | **1.15** [1.03, —] |
| `intercanthal_biocular_ratio` | 4.49% | 1.92% | 3.93% | 2.34 [2.07, —] | **1.14** [1.06, —] |
| `ocular_height_asymmetry` | — | 59.8% | 72.5% | 1.26 [1.11, —] | 1.04 [0.99, —] |
| `mouth_corner_asymmetry` | — | 66.5% | 81.6% | 1.11 [1.01, —] | 0.91 |
| `nose_breadth` | 7.65% | 5.96% | 9.66% | 1.28 | 0.79 |
| `philtrum_width` | 8.76% | 13.27% | 11.46% | 0.66 | 0.76 |
| `intercanthal_width` | 6.43% | 2.52% | 9.40% | 2.56 | 0.68 |
| `mouth_width` | 7.33% | 11.91% | 10.81% | **0.62** | 0.68 |
| `bigonial_width` | 6.37% | 1.64% | 9.87% | 3.87 | 0.65 |
| `interpupillary_distance` | 3.65% | 1.40% | 7.00% | 2.60 | **0.52** |
| `bizygomatic_width` | 4.72% | 1.43% | 9.75% | 3.30 | **0.48** |
| `face_height_sellion_menton` | 5.43% | 1.99% | 19.14% | 2.73 | **0.28** |
| `nose_height` | 4.96% | 2.22% | 19.89% | 2.23 | **0.25** |
| `palpebral_fissure_width_l` | 2.79% | 1.34% | 17.90% | 2.08 | **0.16** |
| `palpebral_fissure_width_r` | 2.70% | 1.31% | 26.59% | 2.07 | **0.10** |

Median within-person CV: **3.0% at fixed pose, 12.5% across ±35° of yaw.**

**Informative (lower CI bound above 1.0): 24 of 32 at fixed pose, 6 of 32 across
captures.**

Above the shipped `repeatability_cv > 0.10` withhold threshold: **6 of 32 at
fixed pose, 17 of 32 across captures.**

Three findings follow.

**The evidence tiers are in the wrong order.** Every `Evidence.VALIDATED_2D`
measurement — `interpupillary_distance`, `intercanthal_width`,
`biocular_width`, `nose_breadth`, `mouth_width`, `nose_height`,
`face_height_sellion_menton` — falls below D = 1 once the photograph is allowed
to vary. Meanwhile `jaw_cheekbone_ratio`, which `Evidence.REQUIRES_3D` causes to
be withheld outright from a 2D image, is the **second most discriminative
measurement in the catalogue** (D = 1.61, CI low 1.50), and it is second because
its two biased components are biased in the same direction and the ratio cancels
them. The tier is a statement about agreement with calipers; discriminability is
a statement about separating people. They are different questions and the
catalogue currently answers the second with the first.

**Expression alone is a bigger error source than the pose model predicts.**
`mouth_width` moves 11.9% and `philtrum_width` 13.3% between a neutral and a
smiling photograph *at the same pose*, which no projection model can see and
which is exactly the effect Kramer (2016) reported for fWHR. Both are
uninformative (D = 0.62 and 0.66) on that basis alone.
`measured_within_person_rsd` should exist for both.

**`facial_width_height_ratio`'s shipped 0.12 is close.** Measured across
captures: 12.52% CV, D = 0.49. The shipped value of 0.12 from Kramer is right to
within a percentage point. That is the one place where the assumed number
survives contact with data — and it is one of only two that were ever measured.

### 7.6 Arm 10 — fairness, and why this sample cannot settle it

Cells: white 69, black 13, west Asian 10, east Asian 9; male 53, female 49.

Landmark agreement (MediaPipe vs human template) by group, bootstrap 95% CI:

| group | n | median error / IPD | 95% CI |
|---|---|---|---|
| black | 13 | 0.0596 | [0.0390, 0.0647] |
| east Asian | 9 | 0.0568 | [0.0425, 0.0619] |
| west Asian | 10 | 0.0543 | [0.0359, 0.0745] |
| white | 69 | 0.0538 | [0.0484, 0.0579] |
| female | 49 | 0.0550 | [0.0462, 0.0602] |
| male | 53 | 0.0547 | [0.0484, 0.0598] |

**Zero of seven pairwise comparisons separate at 5%** (Mann-Whitney). The point
estimates run in the direction the literature would predict — black subjects
highest, white lowest — and the intervals overlap so heavily that the data
cannot distinguish that from noise. **FRLL is 102 people and 68% white. It is
not capable of settling this and this arm does not claim to.** Detecting a 20%
relative difference in landmark error between white and black subjects at this
variance would need roughly 60 per cell, which is 4.6× the black cell here.

**What FRLL *can* settle is a different and larger problem.** The mean percentile
each ethnic group receives against the **pooled** NIOSH stratum:

| measurement | white | west Asian | east Asian | black | spread |
|---|---|---|---|---|---|
| `nose_breadth` | 0.322 | 0.553 | 0.663 | 0.813 | **0.491** |
| `nose_mouth_width_ratio` | 0.383 | 0.558 | 0.656 | 0.724 | 0.341 |
| `mouth_width` | 0.373 | 0.521 | 0.545 | 0.707 | 0.334 |
| `interpupillary_distance` | 0.294 | 0.454 | 0.427 | 0.602 | 0.308 |
| `face_height_sellion_menton` | 0.171 | 0.218 | 0.307 | 0.336 | 0.166 |
| `bigonial_width` | 0.298 | 0.421 | 0.386 | 0.307 | 0.123 |

(The common offset from 0.5 is the landmark-definition bias of §7.3, shared by
every group. The **spread between groups** is the part that is about the groups.)

A pooled-stratum report tells the median black subject their nose is at the 81st
percentile and the median white subject theirs is at the 32nd — a 49-point gap
that is a property of the reference population and of nothing about the person
reading it. NIOSH stratifies by ancestry and `norms/niosh.py` will use it, but
only when the user declares an ancestry, and Vitruve correctly refuses to infer
one. **The pooled fallback is not a neutral default. It is the white default for
this measurement, and the report has to say so on the page.**

---

## 8. What these results contradict in `src/`

Ordered by how much damage the assumption does.

1. **`_CANCELLING` in `measure/registry.py` is backwards, and `spec.py` repeats
   the claim.** `canthal_tilt_asymmetry`, `ocular_height_asymmetry` and
   `mouth_corner_asymmetry` are assigned 0.002/deg on all three axes as the most
   pose-robust quantities in the catalogue. Measured roll sensitivity: **1.90,
   2.96 and 1.03 per degree** — 514× to 1480× the assumed value, and the three
   largest slopes in the whole sweep. Two degrees of roll multiplies a canthal
   tilt asymmetry by five. The cause is that flipping the reference axis between
   sides makes a common-mode roll anti-symmetric, so it adds in the difference.
   `spec.py`'s "the one canthal quantity that image roll cannot corrupt" is the
   opposite of true. (§3.4)

2. **`KLEINBERG_WORST` is in the wrong term of the error budget.**
   `_SENSITIVITY` applies it as a *pose* sensitivity to `bizygomatic_width` and
   `bigonial_width`. Their measured pose sensitivity is 0.0015/deg — 12.5× less.
   What is actually wrong with those two is landmark localisation on a
   self-occluding silhouette, and arm 8 measures that at **6-9 mm between two
   observers of the same photograph**, which is where the term belongs. As
   shipped, the two measurements are penalised for the wrong reason and by a
   number that does not vary with the thing it is multiplied by. (§3.3, §5.2, §7.1)

3. **The pose gate reports an assumption more than a photograph.** `gated_pose`
   adds 4.977° to every axis, which is 90.9% of the gated pose at 0.5° and 38.3%
   at 8°. Applied to FRLL's studio frontal captures, **72% fail the gate**.
   Between 0.5° and 8° of true pose the gated pose changes by 2.4×. (§5.3, §7.4)

4. **`Evidence.VALIDATED_2D` does not predict discriminability, and
   `Evidence.REQUIRES_3D` mispredicts it.** Measured on real photographs, all
   seven `VALIDATED_2D` measurements fall below D = 1 across captures, while
   `jaw_cheekbone_ratio` (`REQUIRES_3D`, withheld from 2D) is the second most
   discriminative thing in the catalogue at D = 1.61 [1.50, —]. Agreement with a
   caliper and separation between people are different questions. (§7.5)

5. **`e_line_upper_lip` / `e_line_lower_lip` compute neither published
   convention.** `LineOffset(..., Axis("z"))` returns the z component of the
   perpendicular, which is 9.0% below the perpendicular distance the Ricketts
   reference range is quoted in and 17.2% below the anteroposterior one. The
   factor is `cos` of the E-line inclination, so it varies between subjects with
   chin projection. (§2)

6. **`magnification_distortion` understates the true magnification by a factor
   of K.** Exact is `K/(1−K)`. −16.7% at 0.3 m, −5.0% at 1.0 m. The value is
   printed to the user in `decide_reportability`. (§4.1)

7. **Left and right of the same measurement get different verdicts.**
   `canthal_tilt_l` has a between-subject spread and is withheld;
   `canthal_tilt_r` has none and is caveated. Same for `gonial_angle_{l,r}` and
   for `palpebral_fissure_width_{l,r}` (0.0604 vs the 0.06 default).
   `norms/published.py` transcribed only one side from each source and
   `registry._rsd_for` looks up per-id. (§5.4)

8. **The undeclared-sex fallback makes Vitruve more confident, not less.**
   `both|pooled` inflates the between-subject SD by up to 1.18× (η²(sex) = 0.28
   for `face_height_sellion_menton`), and that SD is the numerator of the
   discriminability ratio. The same pooled percentile separates the sexes at
   KS = 0.483. (§6.4, §6.5)

9. **A relative `between_subject_rsd` is meaningless for a signed
   near-zero quantity.** `chin_projection` (−4 mm on the synthetic face),
   `e_line_lower_lip` (−0.62 mm) and the three asymmetry ratios all receive
   either `DEFAULT_LINEAR_RSD = 0.06` or a relative pose sensitivity. Measured
   discriminability for `chin_projection` with real slopes: **0.057**. A signed
   offset that crosses zero needs an absolute spread, not a relative one. (§5.4)

10. **The normal percentile is miscalibrated by up to 6.4 percentile points**,
    and 73 of 99 NIOSH strata reject normality. Central coverage is fine
    (0.890-0.924 against 0.900); the tails are not. (§6.3, §6.5)

11. **`nasal_tip_projection_ratio` reads one ala, not both**, so it inherits a
    lateral asymmetry under yaw that the Goode ratio should not have. (§2)

12. **13 of 45 measurements have no working landmark source.** Every
    `View.PROFILE` spec needs `pogonion`, `sublabiale` or `cervicale`, or a
    profile photograph — and the only permissive landmarker tested detects a face
    in 12-14% of true profiles. (§7.4)

Three things `src/` gets right and the data confirms:

- The measurement layer is exact. Machine precision on all 45. (§2)
- The negative control passes: the normative model carries nothing about
  identity. (§6.5)
- `facial_width_height_ratio`'s `measured_within_person_rsd = 0.12` from Kramer
  reproduces at 12.52% on 102 people. (§7.5)

---

## 9. Recommended changes to `_SENSITIVITY` (`measure/registry.py`)

Written down, not applied — `evals/` does not own `src/`. Measured values are
the arm 2 secant at ten degrees under the `persp_1.0m` condition, which is the
regime a compliant portrait is in. Full table:
`evals/results/arm02_recommendations.csv`.

**Must change — the a-priori value is wrong by more than an order of magnitude:**

| id | axis | current | recommended | basis |
|---|---|---|---|---|
| `ocular_height_asymmetry` | roll | 0.002 | **2.96** | measured, all conditions |
| `canthal_tilt_asymmetry` | roll | 0.002 | **1.90** | measured, all conditions |
| `mouth_corner_asymmetry` | roll | 0.002 | **1.03** | measured, all conditions |
| `mouth_corner_asymmetry` | yaw | 0.002 | **0.084** | measured, 1.0 m |
| `ocular_height_asymmetry` | pitch | 0.002 | **0.044** | measured, asymmetric face |
| `canthal_tilt_asymmetry` | pitch | 0.002 | **0.055** | measured, asymmetric face |
| `gonial_angle_{l,r}` | roll | 0.05 | **1.15** | measured, orthographic |
| `chin_projection` | pitch | 0.010 | **0.191** | measured, all conditions |
| `eye_aspect_ratio_{l,r}` | yaw | 0.00015 | **0.0067** | measured, 1.0 m |

The first three cannot be fixed by a number alone. `_CANCELLING` should be
deleted and its comment with it; a paired difference cancels a common-mode
rotation only when both sides are referenced to the *same* axis, and these are
not.

**Should change — wrong by 2-6×:**

| id | axis | current | recommended |
|---|---|---|---|
| `palpebral_fissure_width_{l,r}` | yaw | 0.0015 | 0.0070 |
| `gonial_angle_{l,r}` | yaw | 0.15 | 0.58 (left), 0.22 (right) — see note |
| `philtrum_length` | pitch | 0.0015 | 0.0054 |
| `lower_third_height` | pitch | 0.0015 | 0.0056 |
| `nose_height` | pitch | 0.0015 | 0.0030 |
| `facial_thirds_ratio` | pitch | 0.0 | 0.0062 |
| `philtrum_width` | yaw | 0.0015 | 0.0050 (on a face with 2 mm of depth asymmetry) |

Note on the gonial angles: their yaw sensitivity is **not symmetric between
sides** under a perspective camera (0.58 vs 0.22 at 1.0 m for a mirror-symmetric
jaw), because one gonion is nearer the lens. A single scalar per axis cannot
express that.

**Too pessimistic — the current value withholds measurements for a projection
effect that is not there:**

| id | axis | current | measured |
|---|---|---|---|
| `bizygomatic_width` | yaw | 0.019 | 0.0014 |
| `bigonial_width` | yaw | 0.019 | 0.0014 |
| `submental_length` | yaw | 0.019 | 0.0015 |
| `e_line_{upper,lower}_lip` | yaw | 0.019 | 0.0013 |
| `facial_convexity_angle` | yaw | 0.15 | 0.0103 |
| `nasofrontal_angle` | yaw | 0.15 | 0.0735 |

**Do not simply lower these.** For `bizygomatic_width` and `bigonial_width`,
dropping the pose term without adding the landmark term would flip them from
withheld to reported (arm 5: D goes from 0.45 to 6.90 and from 0.78 to 10.10),
and arm 8 shows their real error is 6-9 mm of silhouette drift. The recommendation
is **structural**: split the budget, put the measured projection slope in
`sensitivity`, and put the silhouette error in the landmark term — which the
`SELF_OCCLUDING` set now in `core/landmarks.py` is the right hook for.

Two docstrings point at a path that does not exist: `core/sensitivity.py` and
`core/spec.py` both say the empirical sweep lives at `evals/arms/pose_sweep.py`.
It is `evals/arms/arm02_pose_sweep.py`.

**Structural recommendations beyond a table of numbers:**

- `PoseSensitivity` has no distance parameter. Perspective changes the yaw
  sensitivity of `gonial_angle_r` by a factor of 3.7 between 1.0 m and 0.5 m, and
  turns exact-zero pitch sensitivities into real ones. Either take
  `subject_distance_m` in `error_at`, or state in the docstring that the values
  are the orthographic limit and inflate elsewhere.
- Three measurements are assigned `TRANSVERSE_RATIO` and are not ratios of two
  transverse spans: `eye_aspect_ratio_{l,r}` (height over width — yaw does not
  cancel, it *accumulates* as 1/cos), `facial_thirds_ratio` (two verticals — the
  yaw entry is right by accident and the pitch entry of 0.0 is wrong),
  `lip_vermilion_ratio` (two verticals).
- `measured_within_person_rsd` should be populated from arm 9 for the 30
  measurements FRLL reaches, starting with `mouth_width` (0.119) and
  `philtrum_width` (0.133), whose dominant error is expression and which no
  projection model can predict.

---

## 10. Arms not run, and why

Every arm in §1 ran. What could not be done *inside* those arms:

| Not done | Why |
|---|---|
| **ANSUR II direct unit verification** (arm 6b) | No copy of ANSUR II is vendored and the harness does not fetch one. The tenths-of-a-millimetre claim is confirmed by consistency with NIOSH (64.37 mm), Dodgson (63.36 mm) and the integer/decimal structure of the NIOSH columns, not by reading the ANSUR II file. To close it, add the ANSUR II CSV under `evals/data/` and assert `interpupillarybreadth / 10 ≈ 63`. |
| **Test-retest for the 13 profile measurements** (arm 9) | MediaPipe detects a face in 12/102 and 14/102 true profiles. There is no landmark source for `pogonion`, `sublabiale` or `cervicale` in any permissively licensed model tested. |
| **True same-pose same-expression retest** (arm 9) | FRLL has exactly one neutral front capture per person. The tightest retest available is neutral vs smiling front, which necessarily includes an expression change. The reported "fixed pose" CVs are therefore upper bounds on pure repeatability and lower bounds on the full within-person spread. |
| **Landmark NME against 300W / WFLW** (design §4 arms 1-2) | Both require registration. Arm 8b substitutes a different and arguably more relevant measurement: agreement between a model and a human on the same images, per landmark. |
| **Head-pose MAE against AFLW2000-3D** (design §4 arm 2) | Requires registration. `POSE_ESTIMATOR_MAE_DEG = 3.97` is therefore still an unverified literature value, and §5.3 shows it is load-bearing for every discriminability verdict. This is the largest unclosed hole. |
| **Separating "ill-defined landmark" from "wrong index map"** (arm 8b) | Needs a third independent observer of the same images. `trichion` at 0.451 IPD is certainly a mapping error; gonion/zygion/tragion at 0.10-0.14 IPD are consistent with either. |
| **A powered fairness result** (arm 10) | FRLL cells are 9 to 69 subjects. Roughly 60 per cell would be needed to resolve a 20% relative difference in landmark error. Reported as unresolved rather than as a null. |
| **`derm` and end-to-end pipeline arms** | Those stages do not exist yet. |
| **Offline-egress assertion** (design §4 arm 9) | Belongs in `tests/`, not `evals/`; it is a property of the analysis path, not a measurement. Note that `evals/` itself *does* fetch data — over `make -C evals data`, as an explicit separate step, never inside an arm. |

---

## 11. Files

```
evals/
  run_all.py                  entry point
  harness/run.py              shim the repository-root `make evals` calls
  Makefile                    `data` fetches FRLL + MediaPipe weights; `evals` runs
  _bootstrap.py               seeds, provenance, JSON/CSV writers
  frll.py                     FRLL loader, 189-point and 478-point landmark maps
  synth/face.py               the synthetic face and the four cameras
  synth/truth.py              independent ground truth, pure-Python math
  arms/arm01..arm10           one module per arm, each runnable standalone
  results/*.json, *.csv       machine-readable output, regenerated by run_all
  data/                       downloaded, gitignored
```
