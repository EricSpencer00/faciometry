# Evaluation

Everything below is produced by `evals/`. Run it with:

```bash
make -C evals data     # fetch FRLL (CC BY 4.0), the MediaPipe weights and the
                       # pipeline's own YuNet and 6DRepNet weights, once
make evals             # run every arm, ~45 s, writes evals/results/
```

Arms 8 to 11 run a landmarker and a pose estimator over roughly 1,500 image
passes between them and cache the result, so a *first* run costs a few minutes
and every run after it costs the 45 seconds above. Delete `evals/data/*.npz` to
force the detection passes again.

`evals/run_all.py` is the entry point; each arm also runs standalone
(`python -m evals.arms.arm02_pose_sweep`). Every arm writes JSON to
`evals/results/`, plus CSV wherever the result is a table. Everything is seeded
from a single master seed (`evals/_bootstrap.py`, `MASTER_SEED = 20260823`) and
a rerun reproduces byte-identical output: all 45 result files check out
identical across consecutive runs. `results/timings.json` is the one exception
and is excluded by design, because it records wall clock; it was split out of
`summary.json` for exactly that reason, so the reproducibility claim can be
checked with a checksum instead of qualified in prose.

Nothing under `src/` was modified to produce these numbers. Where a result says
`src/` is wrong, it says so and stops there; §8 lists what the numbers
contradict and §9 lists the changes they argue for.

**This run covers the catalogue at 68 measurements** — 48 frontal and 20
profile; 34 in millimetres, 17 in degrees, 17 dimensionless. The previous run
covered it at 45. The 23 added since are not a uniform extension of what was
there: they include the first measurements read against a *landmark-derived*
reference rather than the image frame, three `PoseSensitivity` constants whose
values were derived on paper rather than measured, and several signed
quantities that sit near zero, which turns out to be the single most consequential
structural property in the catalogue.

---

## 1. The design, stated in full before any of it ran

Eleven arms. Arms 1 to 7 need no model weights and no images; arms 8 to 11 need
real photographs, and arm 11 additionally needs the detector and pose weights
the pipeline itself uses.

| # | Arm | What it establishes | Data | Depends on |
|---|-----|--------------------|------|-----------|
| 1 | Synthetic geometry control | the measurement code's own error floor, with zero model involvement | synthetic 3D face, closed-form truth | — |
| 2 | Pose sweep, ±30° in 2.5° steps, three axes | measured per-measurement per-axis sensitivity, against the a-priori values in `core/sensitivity.py`; and whether the sixteen roll claims are true | same face, re-projected | 1 |
| 3 | Control for arm 2: 0° repeated through a JPEG cycle | separates pipeline noise from pose effect; without it arm 2's slopes are uninterpretable | same face, rendered and re-encoded | 1 |
| 4 | Perspective sweep, 0.3 to 3.0 m | positive control on `core.scale.magnification_distortion` | same face, pinhole camera | 1 |
| 5 | Discriminability | reproduces Kleinberg & Vanezis (2007), then the full 68-measurement table at 0.5, 3 and 8° of pose | arms 2 and 3 | 2, 3 |
| 6 | NIOSH 2003 external check | are the shipped between-subject spreads reproducible, in the units claimed, and normal in shape | 3,997-subject caliper survey | — |
| 7 | Negative control on the normative model | shuffled identities must not change the percentile output | NIOSH | 6 |
| 8 | FRLL landmark accuracy, spread, and roll attribution | real faces, human-delineated landmarks, self-reported demographics | Face Research Lab London, CC BY 4.0 | — |
| 9 | Test-retest repeatability | per-measurement within-person spread, **measured** rather than assumed | FRLL, 5 usable captures per person | 8 |
| 10 | Fairness stratification | measurement quality by self-reported ethnicity and sex, with honest intervals | FRLL | 8 |
| 11 | Pose estimator error without a labelled benchmark | how much of `POSE_ESTIMATOR_MAE_DEG = 3.97` can be checked with no registration-gated data | FRLL + YuNet + 6DRepNet + MediaPipe | — |

Arm 11 is new in this run. The previous one listed the unverified 3.97 as the
largest unclosed hole and stopped, on the grounds that closing it needs
AFLW2000-3D and AFLW2000-3D needs registration. That is true of the published
number and not true of the estimator: rotating a photograph in its own plane
changes the true roll by exactly the rotation applied, and mirroring it negates
the true yaw and roll, and neither operation needs a label. §7 is what that
buys and §10 is what it still does not.

Controls are arms, not extras.

- Arm 3 is the control for arm 2.
- Arm 4's probe is a positive control for arm 4's catalogue sweep.
- Arm 7 has a negative control (within-stratum shuffle) *and* a positive
  control (across-stratum shuffle) that proves the test has power.
- Arm 2 has a rotation-only condition that proves the harness returns exact
  zeros for rotation-invariant quantities, **and**, new in this run, a
  sixteen-measurement shadow catalogue holding the *discarded* horizon-referenced
  form of every measurement that now claims roll robustness. Without that
  shadow, "the roll slope is zero" is equally consistent with the reference
  working and with the sweep having no roll in it.
- Arm 11's known in-plane rotation is the positive control for the rest of arm
  11: if the fitted slope of estimated roll on applied roll were not close to
  ±1, no other number in that arm would mean anything.

### The synthetic face (arms 1-5)

45 landmarks in millimetres in the canonical frame, covering every landmark all
68 measurements read. Some measurements are exact by construction: the pupils
sit at x = ±31.68 with identical y and z, so the interpupillary distance is
63.36 mm and not a number that happens to come out near it. Likewise
intercanthal width 32.0, nose breadth 34.0, philtrum width 11.0, bizygomatic
141.0, bigonial 117.0, palpebral fissure heights 11.0 and 10.6, canthal tilts
6.0° and 5.0° (placed by `dy = dx·tan θ`), and hence canthal tilt asymmetry
exactly 1.0°.

Two properties of the face do extra work for the 23 new measurements. Both
pupils sit at the same y and the same z, so the interpupillary *line* is exactly
the x axis and the interpupillary *axis* is exactly ±x. That makes every
perpendicular offset from that line a plain y difference and every projection
onto that axis a plain x difference — which is the point, because a formula that
let the 16 mm brow depth or the 10 mm canthal depth leak into either would be
caught. Margin-reflex distances therefore come out at exactly half a fissure
height, Cupid's-bow peak height at exactly the 1.5 mm the crista philtri sits
above labiale superius, alar base : intercanthal at exactly 28/32, midface
projection at exactly 8 mm, and nasal dorsal deviation at exactly 0.

The rest carries realistic depth — a flat face would make arm 2 answer a
question nobody asked — and their truth is recomputed in `evals/synth/truth.py`
in pure-Python `math`, written from the anatomical definition rather than from
`registry.py`, importing none of `core.geometry`, `core.formula` or
`measure.registry`. All 68 values land inside their published reference ranges
where the spec has one.

---

## 2. Arm 1 — synthetic geometry control

**Result: passes on all 68. The measurement layer contributes nothing.**

| | |
|---|---|
| primitive algebra cases | **31**, 0 failures, agreement to 1e-12 |
| catalogue measurements | **68**, 0 failures |
| worst absolute error | 2.8e-14 |
| worst relative error | 3.7e-15 (`mouth_corner_asymmetry`) |
| measurements outside their published reference range | 0 |
| `evaluate()` median vs deterministic value, worst relative shift | 6.3e-6 at σ = 1e-4 mm |
| measurements `evaluate()` could not reach | 0 |

The 45-measurement result held. Every one of the 23 added since reproduces its
independently derived value to machine epsilon, including the eleven whose
answer is a construction parameter and the twelve that had to be recomputed
from the anatomical definition.

The primitive suite grew from 20 cases to 31, because the second batch of
measurements uses nodes the first did not:

- `signed_tilt` against a **landmark-derived** axis rather than a frame axis,
  which is the mechanism the whole roll fix rests on;
- the same configuration rotated in its own plane by 13° and by −21°, which
  must return the *identical* tilt — it does, to 1e-12;
- **the discarded horizon-referenced form on the same rolled configuration, on
  both sides**, which must move one-for-one and must move in opposite
  directions on the two sides. It does: on the left the tilt reads
  `θ − roll` and on the right `θ + roll`. That anti-symmetry is the entire
  mechanism of the old defect, and it is now pinned by a closed-form test
  rather than only by a sweep;
- `line_offset` with a `y` normal (the margin-reflex form), `proj_length` along
  a landmark-derived axis (the brow-apex form), and `angle_between` on two
  landmark-derived directions (the nasal-dorsum form).

Also still covered: `angle_at` at 0.01° and 179.99° where an `arccos`
implementation loses precision and the shipped `atan2` form does not, and
`Axis("z")` on a 2D point set, which must raise rather than silently return
zero. It does.

### 2.1 Four measurements compute neither published convention, not two

`LineOffset(p, a, b, normal=Axis("z"))` returns the *z component of the vector
from the line to the point*, which is not the perpendicular distance a
cephalometric reference range is quoted in. The previous run found this on the
two E-line measurements. The catalogue now contains **four** measurements built
that way, because `upper_lip_projection` and `labiomental_sulcus_depth` were
added with the same construction.

| measurement | registry (z component) | perpendicular | anteroposterior | z reads low by |
|---|---|---|---|---|
| `e_line_upper_lip` | −2.672 mm | −2.937 mm | −3.228 mm | **9.0%** / 17.2% |
| `e_line_lower_lip` | −0.624 mm | −0.686 mm | −0.754 mm | **9.0%** / 17.2% |
| `labiomental_sulcus_depth` | −5.583 mm | −6.038 mm | −6.529 mm | **7.5%** / 14.5% |
| `upper_lip_projection` | +7.303 mm | +7.333 mm | +7.364 mm | 0.4% / 0.8% |

The factor is `cos` of the reference line's inclination, so the size of the
error is a property of the line and not of the point. That is why the same
coding decision costs 9.0% on the Ricketts E-line, which is inclined 24.5° on
this face, 7.5% on the labiale-inferius-to-pogonion line at 22.4°, and only
0.4% on Burstone's subnasale-pogonion line at 5.2°. The inclination varies
between subjects with chin projection, so the bias is subject-dependent and
does not cancel in a percentile.

`upper_lip_projection` is the mild case and also the safe one: no reference
range is shipped on it. `labiomental_sulcus_depth` is the new one that matters,
at 7.5%.

### 2.2 Still true: `nasal_tip_projection_ratio` reads one ala

`ALARE_L` only, not the mean of the two alae. Under yaw the two alae move
oppositely, so the Goode ratio inherits a lateral asymmetry it should not have.
Unchanged since the previous run.

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

Resulting measurement noise floor at q85, over 68 measurements — median 0.045%
for lengths and ratios, 0.020° for angles:

| measurement | noise (relative, or degrees) |
|---|---|
| `commissure_height_r` | **442%** |
| `nasolabial_angle` | 0.162° |
| `mentolabial_angle` | 0.066° |
| `nasal_tip_rotation` | 0.037° |
| `medial_canthal_angle_{l,r}` | 0.034° |
| `canthal_tilt_asymmetry` | 0.029° |
| `interpupillary_distance` | 0.015% |
| `bizygomatic_width` | 0.006% |

`commissure_height_r` at 442% is not a measurement problem, it is the
denominator: the right mouth corner is exactly level with the stomion on this
face, so the measurement's value is exactly zero and a *relative* noise figure
divides by it. It is the first appearance of a property that recurs in every
arm below.

This is a **lower bound**: the recovery window is centred on the true position,
so it measures encoding and resampling only, not detector failure. Arm 9
measures the real thing on real photographs and gets numbers three orders of
magnitude larger.

### 3.2 The sweep

Five conditions, three axes, ±30° in 2.5° steps, all 68 measurements — **1,020
slope rows** in `evals/results/arm02_slopes.csv`, plus 240 more for the shadow
catalogue in `arm02_roll_controls.csv`. The comparison metric matches how the
a-priori numbers were built: `cosine_yaw_sensitivity()` is not a derivative, it
is `(1 − cos 10°)/10`, a secant at ten degrees. So the measured counterpart is
the mean of `|v(+10) − v(0)|` and `|v(−10) − v(0)|` over ten degrees.

Each row now also carries the same secant in the measurement's own units, so a
quantity whose zero-pose value is zero is reported rather than silently
returning nan. Three (measurement, axis) pairs — all three axes of
`commissure_height_r` — have no relative slope at all and are listed in
`not_comparable` with the reason, not dropped.

**Positive control.** In the `3d` condition (rotate, do not project), every
distance and every non-axis angle returns a slope of exactly `0.00000`. 36
(measurement, axis) pairs across 20 measurements move, and every one of them is
defined against an image axis. The harness is measuring projection, not
arithmetic drift.

### 3.3 The sixteen roll claims, tested

Sixteen measurements are now read against the interpupillary line or the
interpupillary axis rather than the image frame. The catalogue claims two
different things about them, and this run states both as falsifiable
predictions before measuring:

- **exact** — both the measured direction and its reference rotate with the
  head, so roll cancels in the geometry. The two canthal tilts, the three
  asymmetries, the two brow-apex offsets, the nasal dorsal angle.
- **cos(roll)** — the reference line rotates with the head but the sign axis
  the offset is read along is still the frame's `y`, so the answer keeps its
  magnitude and loses a cosine, i.e. a secant of exactly `(1 − cos 10°)/10 =
  0.0015230`. The four margin-reflex distances, the two Cupid's-bow heights,
  the two commissure heights.

**Result: 16 of 16 hold, in the orthographic and unprojected regimes where the
claim is a statement about geometry.**

| measurement | claim | 3d | ortho | persp 1.0 m | persp 0.5 m | horizon twin (ortho) |
|---|---|---|---|---|---|---|
| `canthal_tilt_l` | exact | 4.0e-16 | 4.0e-16 | 0.00107 | 0.00217 | **1.0000** |
| `canthal_tilt_r` | exact | 1.3e-16 | 1.3e-16 | 0.00128 | 0.00260 | **1.0000** |
| `canthal_tilt_asymmetry` | exact | 3.6e-16 | 3.6e-16 | 0.00234 | 0.00477 | **1.9000** |
| `ocular_height_asymmetry` | exact | 2.2e-16 | 2.2e-16 | 2.1e-16 | 1.8e-16 | 0.9675 |
| `mouth_corner_asymmetry` | exact | 1.0e-15 | 1.0e-15 | 1.4e-15 | 2.3e-15 | 0.9119 |
| `brow_apex_lateral_offset_{l,r}` | exact | 9e-18 | 2e-17 | 0.00029 | 0.00061 | 0.0100 / 0.0095 |
| `nasal_dorsal_deviation` | exact | 0.0 | 0.0 | 0.02824 | **0.05709** | 1.0000 |
| `margin_reflex_distance_1_{l,r}` | cos(roll) | 0.00152 | 0.00152 | 0.00152 | 0.00152 | 0.00152 |
| `margin_reflex_distance_2_{l,r}` | cos(roll) | 0.00152 | 0.00152 | 0.00152 | 0.00152 | 0.00152 |
| `cupids_bow_peak_height_{l,r}` | cos(roll) | 0.00152 | 0.00152 | 0.00159 | 0.00164 | 0.0637 |
| `commissure_height_l` | cos(roll) | 0.00152 | 0.00152 | 0.00206 | 0.00328 | 0.5644 |
| `commissure_height_r` | cos(roll) | 0 (abs) | 0 (abs) | 5.9e-5 (abs) | 5.9e-5 (abs) | 0.0087 (abs) |

The eight "exact" claims return zero to between 1e-18 and 1e-15, which is
floating-point zero and not a small number. The eight "cos(roll)" claims return
0.0015230, which is `(1 − cos 10°)/10` to five significant figures.
`commissure_height_r` is checked in absolute units because its value is exactly
zero and `0 · cos(roll)` is still zero; that is the correct prediction and it
holds.

**The controls have power.** All sixteen horizon twins move. `canthal_tilt`
moves at exactly 1.000 degrees per degree of roll and `canthal_tilt_asymmetry`
at exactly 1.900, which reproduces the defect the previous run found and the
mechanism arm 1 now pins in closed form. `nasal_dorsal_deviation`'s horizon
twin moves at 1.000 degrees per degree against a measurement that does not move
at all.

**Four controls have power but no discrimination, and this is a limitation of
the synthetic face rather than a result.** For the four margin-reflex
distances, the horizon twin returns *exactly the same* 0.0015230. On this face
the lid margins sit directly above the pupils, so a perpendicular offset from
the interpupillary line and a plain y offset from the pupil are the same
number, and swapping the reference changes nothing. On a face where the lid
margin is laterally displaced from the pupil they would differ. The four are
listed in `roll_controls_that_do_not_discriminate` rather than counted as
confirmations.

**Roll invariance is exact only under orthography.** Under a pinhole camera it
degrades, and by how much is a real finding: `nasal_dorsal_deviation` picks up
0.0282 degrees of apparent deviation per degree of roll at 1.0 m and 0.0571 at
0.5 m, against a declared 0.01. Roll about the optical axis leaves depth
unchanged, so it commutes with orthographic projection exactly and with a
perspective divide not at all.

**What the interpupillary reference does not remove is pitch.**
`signed_angle_to_axis` fixes its positive-perpendicular direction from the
frame's `y`, projected orthogonal to the reference axis. Under roll that
direction rotates with the head and cancels; under pitch it does not. All three
asymmetries and both canthal tilts therefore still move in the unprojected `3d`
condition on the pitch axis, which is exactly what the registry's docstrings
claim, and is now measured rather than asserted.

### 3.4 The headline disagreements

**131 of 201 comparable (measurement, axis) pairs disagree with their a-priori
value by more than a factor of two**, against 86 of 135 last time. 55 of those
are cases where the a-priori value is exactly 0 and the measured effect is a
small but non-zero perspective term; the ones that matter are below.

Worst, ordered by measured-over-a-priori:

| measurement | axis | a-priori | ortho | persp 1.0 m | persp 0.5 m | asym. face | ratio |
|---|---|---|---|---|---|---|---|
| `commissure_height_l` | pitch | 0.010 | 0.1954 | **1.1887** | 0.2923 | 0.1847 | **119×** |
| `eye_aspect_ratio_{l,r}` | yaw | 0.00015 | 0.0061 | 0.0067 | 0.0073 | 0.0053 | **48×** |
| `cupids_bow_peak_height_l` | pitch | 0.00152 | 0.0232 | 0.0201 | 0.0171 | **0.0684** | **45×** |
| `chin_projection` | pitch | 0.010 | 0.1910 | 0.1910 | 0.1910 | 0.1910 | 19× |
| `cupids_bow_peak_height_r` | pitch | 0.00152 | 0.0232 | 0.0201 | 0.0171 | 0.0279 | 18× |
| `commissure_height_l` | yaw | 0.019 | 0.0015 | **0.2370** | 0.1158 | 0.0016 | 12.5× |
| `canthal_tilt_asymmetry` | pitch | 0.010 | 0.0018 | 0.0019 | 0.0019 | **0.0988** | 9.9× |
| `mouth_corner_asymmetry` | yaw | 0.010 | 0.0014 | 0.0703 | **0.0895** | 0.0014 | 9.0× |
| `canthal_tilt_asymmetry` | yaw | 0.010 | 0.0677 | 0.0682 | 0.0689 | 0.0524 | 6.9× |
| `alar_base_intercanthal_ratio` | yaw | 0.00015 | 0.0 | 0.00004 | 0.00008 | **0.00098** | 6.4× |
| `lower_vermilion_height` | pitch | 0.00152 | 0.0043 | 0.0066 | 0.0090 | 0.0043 | 5.9× |
| `nasal_dorsal_deviation` | roll | 0.010 | 0.0 | 0.0282 | **0.0571** | 0.0 | 5.7× |
| `upper_vermilion_height` | pitch | 0.00152 | 0.0043 | 0.0064 | 0.0086 | 0.0043 | 5.7× |
| `lower_third_height` | pitch | 0.00152 | 0.0031 | 0.0056 | 0.0082 | 0.0031 | 5.4× |
| `palpebral_fissure_width_{l,r}` | yaw | 0.00152 | 0.0059 | 0.0070 | 0.0081 | 0.0051 | 5.3× |
| `philtrum_length` | pitch | 0.00152 | 0.0070 | 0.0054 | 0.0039 | 0.0070 | 4.6× |
| `gonial_angle_l` | yaw | 0.15 | 0.4472 | 0.5760 | 0.6391 | 0.3673 | 4.3× |
| `philtrum_width` | yaw | 0.00152 | 0.0015 | 0.0016 | 0.0017 | **0.0050** | 3.3× |
| `ocular_height_asymmetry` | pitch | 0.010 | 0.0005 | 0.0005 | 0.0005 | **0.0320** | 3.2× |
| `chin_height` | roll | **0.0** | 0.00151 | 0.00277 | 0.00555 | 0.00151 | ∞ |

And in the other direction, a-priori values that are far too pessimistic:

| measurement | axis | a-priori | measured (worst of five conditions) | ratio |
|---|---|---|---|---|
| `bizygomatic_width` | yaw | 0.019 (`KLEINBERG_WORST`) | 0.00152 | 0.08× |
| `bigonial_width` | yaw | 0.019 | 0.00152 | 0.08× |
| `submental_length` | yaw | 0.019 | 0.00152 | 0.08× |
| `midface_projection` | yaw | 0.020 (`_AP_PROJECTION`) | 0.00152 | 0.08× |
| `facial_convexity_angle` | yaw | 0.15 | 0.0103 | 0.07× |
| `e_line_{upper,lower}_lip` | yaw | 0.019 | 0.0025 | 0.13× |
| `upper_lip_projection` | yaw | 0.019 | 0.0021 | 0.11× |
| `labiomental_sulcus_depth` | yaw | 0.019 | 0.0021 | 0.11× |
| `medial_canthal_angle_{l,r}` | pitch | 0.15 (unmeasured default) | 0.0710 | 0.47× |

### 3.5 The three derived constants, measured

`_MIDLINE_DEVIATION`, `_SAGITTAL_FRAME_ANGLE` and `_AP_PROJECTION` were derived
from the geometry on paper rather than measured, and their own docstrings say
the sweep is the authority that should settle them. It now has.

**`_SAGITTAL_FRAME_ANGLE` is right, and its central claim is exactly right.**

| axis | declared | ortho | persp 1.0 m | persp 0.5 m | verdict |
|---|---|---|---|---|---|
| pitch | **1.0** | **1.00000** | **1.00000** | **1.00000** | exact, in every condition |
| yaw | 0.15 | 0.0399 | 0.0514 | 0.1029 | 0.69×, mildly pessimistic |
| roll | 0.05 | 0.0396 | 0.0395 | 0.0650 | 1.3×, right |

"Pitch rotates that plane in the image and therefore enters one for one: a
subject who lifts the chin five degrees adds five degrees to the reading" is
correct to five decimal places under an orthographic camera *and* under a
pinhole camera at half a metre. This is the one derived constant that needs no
change.

**`_MIDLINE_DEVIATION` has the right structure and the wrong arithmetic on two
of three axes.**

| axis | declared | ortho | persp 1.0 m | persp 0.5 m | asym. face | verdict |
|---|---|---|---|---|---|---|
| yaw | 0.5 | **0.7909** | 0.7952 | 0.7996 | 0.7909 | 1.6×; the derivation is low |
| pitch | 0.05 | 0.0 | 0.0 | 0.0 | 0.0277 | 0.55×, adequate |
| roll | 0.01 | **0.0** | 0.0282 | **0.0571** | 0.0 | exact under orthography, 5.7× low under a real camera |

The docstring reasons "23 mm of depth against 41 mm of dorsum, about half a
degree of apparent deviation per degree of yaw". The mechanism is right and the
number is 1.6× low: measured 0.79. At the shipped `pose_tolerance_deg = 4.0`
that is 3.2 degrees of manufactured dorsal deviation on a perfectly straight
nose, against a measurement whose whole clinical range is a few degrees.

**`_AP_PROJECTION` is 1.6× low on the axis it was derived for, and 13× too
pessimistic on yaw.**

| axis | declared | ortho | persp 1.0 m | persp 0.5 m | verdict |
|---|---|---|---|---|---|
| pitch | 0.08 | **0.1259** | 0.1259 | 0.1259 | 1.6× |
| yaw | 0.02 | 0.00152 | 0.00152 | 0.00151 | 0.08× |
| roll | 0.01 | 0.0 | 0.0031 | 0.0062 | 0.62× |

The docstring says "46 mm of span against 10 mm of projection, so a degree of
pitch is about 8% of the answer". The span is 58 mm and the projection 8 mm, so
the answer is 12.6%. And the same docstring predicts that `chin_projection`
belongs on this constant: it is right, and `chin_projection`'s measured pitch
slope is **0.191**, against the 0.010 its `KLEINBERG_WORST` fallback gives it.
Both are the same geometry — a short anteroposterior offset between two points
separated by a long vertical span — and the shorter the offset the larger the
relative slope.

### 3.6 The structural finding this run adds: a signed quantity near zero has no relative sensitivity

Five measurements in the catalogue are signed differences that sit near zero:
`ocular_height_asymmetry`, `mouth_corner_asymmetry`, `canthal_tilt_asymmetry`,
`commissure_height_l` and `commissure_height_r`. Two more —
`cupids_bow_peak_height_{l,r}` — are small differences of two much larger
offsets, 1.5 mm out of 66.

Every arm sees the consequence, and it is always the same arithmetic. The
numerator moves by a fixed amount set by the geometry; the denominator is the
measurement's own small value; the relative slope is their quotient and it is
enormous.

- `commissure_height_l` is a −0.8 mm corner drop over a 52 mm mouth. The
  cheilion sits 9 mm behind the stomion in depth, so ten degrees of pitch
  swings 1.56 mm of that depth into the vertical — nearly twice the whole
  measurement. Measured pitch slope 0.195 per degree orthographically and
  **1.189 at one metre**.
- `cupids_bow_peak_height` is a 1.5 mm rise with 2 mm of depth between its two
  points, giving 0.0232 per degree of pitch against the 0.00152 that
  `VERTICAL_DISTANCE` assigns it.
- `commissure_height_r` has a value of exactly zero on this face and therefore
  no relative slope at all, no relative noise floor, and no relative
  perspective distortion. It is reported in absolute units everywhere.

This is the same defect §8 item 9 named last time for `between_subject_rsd`,
and it is broader than that: it is not only the spread that cannot be
relative for these quantities, it is the sensitivity, the noise floor and the
perspective term as well.

### 3.7 Still true: cosine is second order, depth offset is first

Under a symmetric face and orthographic projection, every transverse width
scales by exactly `cos(yaw)` and every ratio of parallel spans is exactly
invariant. The a-priori model is exactly right in that regime, and that regime
does not exist. `palpebral_fissure_width` runs 29 mm laterally and 10 mm in
depth, so its projected length changes as `Δx·cos θ + Δz·sin θ` — a slope of
`Δz/Δx = 0.345` per radian, or 0.0060 per degree. Measured: 0.0059, which is
3.9× the cosine value.

The 23 new measurements add a case the previous run did not have.
`alar_base_intercanthal_ratio` is documented as one where "the apparent width
of a mirrored pair scales as cos(yaw) whatever depth the pair sits at, so the
cosine cancels exactly rather than approximately". Measured: **exactly zero**
under orthographic yaw on the mirror-symmetric face, and **0.00098 per degree**
once each side is given a couple of millimetres of independent depth. The claim
is true for a pair that is mirror-symmetric *in depth*, and the phrase
"whatever depth the pair sits at" is doing work the geometry does not support
when the two members sit at different depths, which on a real face they do.

---

## 4. Arm 4 — perspective sweep

### 4.1 The probe (positive control)

Two transverse segments of identical true length, one on the eye plane and one
exactly 50 mm in front of it, through a real pinhole camera.

| distance | measured magnification | `magnification_distortion` (K = 50/d) | exact pinhole (K/(1−K)) | ICAO error |
|---|---|---|---|---|
| 0.3 m | 0.200000 | 0.166667 | 0.200000 | **−16.67%** |
| 0.5 m | 0.111111 | 0.100000 | 0.111111 | −10.00% |
| 1.0 m | 0.052632 | 0.050000 | 0.052632 | −5.00% |
| 3.0 m | 0.016949 | 0.016667 | 0.016949 | −1.67% |

The measured curve matches the **exact** pinhole form `K/(1−K)` to 9.4e-17 at
every distance. `core.scale.magnification_distortion` returns `K`, the
first-order term, and **understates the true magnification by exactly a factor
of K**. `decide_reportability` prints that number to the user, so the
under-report reaches the report. Unchanged from the previous run.

### 4.2 The catalogue

For 42 of the 51 non-angular measurements, `fractional_change × distance` is
constant across all eight distances to within 10%, confirming the first-order
`(z₁ − z₂)/d` model. **32 of 68 measurements distort by more than 1% at 1.0 m**,
the ICAO portrait distance, at zero pose — against 22 of 45 last time.

| measurement | 0.3 m | 0.5 m | 1.0 m | 3.0 m | implied depth straddle |
|---|---|---|---|---|---|
| `gonial_angle_r` | −28.63° | −15.86° | **−7.33°** | −2.31° | — |
| `gonial_angle_l` | +16.62° | +11.20° | **+6.13°** | +2.17° | — |
| `commissure_height_l` | −294.6% | −170.5% | **−83.1%** | −27.2% | −841 mm |
| `cupids_bow_peak_height_{l,r}` | +44.7% | +25.1% | **+12.0%** | +3.9% | +123 mm |
| `ramus_body_ratio_r` | +23.0% | +16.7% | **+8.9%** | +3.0% | +85 mm |
| `ramus_body_ratio_l` | −26.9% | −17.2% | **−8.9%** | −3.0% | −87 mm |
| `philtrum_length` | +17.8% | +10.1% | +4.9% | +1.6% | 50 mm |
| `facial_width_height_ratio` | −14.4% | −8.9% | −4.5% | −1.5% | −45 mm |
| `bizygomatic_width` | −6.8% | −4.2% | −2.2% | −0.7% | −21 mm |
| `canthal_tilt_{l,r}` | +0.25° | +0.15° | +0.07° | +0.02° | — |

Three things to look at.

**The gonial angle is still the worst.** A perfectly symmetric jaw,
photographed in profile at 1.0 m, produces gonial angles that differ by 13.5°
between the near and far side, purely from perspective, against a published
between-subject SD of 6.05°.

**`ramus_body_ratio` inherits the same problem and does it as a ratio.** The
spec says being a ratio "buys nothing here" because both endpoints sit on a
self-occluding silhouette. That is right for a different reason than the one
given: on a mirror-symmetric jaw the two sides read **+8.9% and −8.9%** at one
metre, an 18-point split with no anatomy in it at all, because one gonion is
nearer the lens than the other. It is the gonial-angle failure in a
dimensionless wrapper.

**`commissure_height_l`'s implied depth straddle is 841 mm**, which is not a
distance any part of a face spans. It is the near-zero denominator again: the
first-order model `(z₁ − z₂)/d` is a statement about a *relative* change, and
this measurement has no meaningful relative change.

---

## 5. Arm 5 — discriminability

### 5.1 Kleinberg and Vanezis reproduced

An index with a between-subject relative spread of 1.2% at ten degrees of yaw:

| pose error at 10° yaw | source | ratio | informative? |
|---|---|---|---|
| 8% | Kleinberg & Vanezis 2007, lower end | **0.150** | no |
| 19% | Kleinberg & Vanezis 2007, upper end | **0.063** | no |
| 19% | Faciometry `KLEINBERG_WORST.error_at(10,0,0)` | 0.063 | no |
| 28.4% | the same, after `gated_pose` inflates 10° to 14.98° | 0.042 | no |
| 1.5% | Faciometry `TRANSVERSE_WIDTH` | 0.790 | no |

Reproduced. Every route gives a ratio below 1.

### 5.2 Faciometry's geometry now exceeds the top of Kleinberg's band. The previous run's conclusion is overturned.

The previous run concluded that "Faciometry's geometry reaches the bottom of
Kleinberg's 8-19% band and never the top", the largest movement being
`nasal_tip_projection_ratio` at 9.89%. That was true of the catalogue at 45. It
is false at 68.

| condition | worst index (excluding the near-zero five) | movement at 10° yaw |
|---|---|---|
| orthographic | **`ramus_body_ratio_{l,r}`** | **21.1%** |
| perspective 1.0 m | **`ramus_body_ratio_r`** | **26.1%** |
| perspective 0.5 m | **`ramus_body_ratio_r`** | **30.0%** |
| orthographic | `nasal_tip_projection_ratio` | 9.89% |
| perspective 0.5 m | `brow_apex_lateral_offset_r` | 9.31% |

`ramus_body_ratio` runs from tragion at z = −55 to gnathion at z = +22, a 77 mm
depth straddle inside a single dimensionless quantity. Ten degrees of yaw moves
it by twenty-one percent orthographically and thirty percent at arm's length,
which is **above** the 19% ceiling of the worst index Kleinberg measured on a
real rotating subject.

Nine measurements now reach Kleinberg's 8% floor at 0.5 m, against five before:
`ramus_body_ratio_{l,r}`, `brow_apex_lateral_offset_{l,r}`,
`commissure_height_{l,r}`, `eye_aspect_ratio_r`, `palpebral_fissure_width_r`,
`nasal_tip_projection_ratio`.

Five measurements are excluded from this comparison and named, with their
zero-pose values printed beside them, rather than being allowed to be the
headline: the three asymmetries and the two commissure heights, whose values
are near zero by construction. `commissure_height_l` would otherwise read a
237% movement at ten degrees of yaw, which says nothing about the face.

### 5.3 The pose gate is dominated by its own uncertainty assumption

`assess_discriminability` inflates every pose axis by `gated_pose`, adding
`POSE_ESTIMATOR_SD_DEG = 4.977°`:

| true pose | gated pose | fraction that is estimator uncertainty |
|---|---|---|
| 0.5° | 5.48° | **90.9%** |
| 3.0° | 7.98° | 62.4% |
| 8.0° | 12.98° | 38.3% |

Arm 8 measures the front-capture pose 95th percentile on a 102-subject studio
set at 3.9° yaw, 8.3° pitch, 3.9° roll, so a 4.977° inflation is of the same
order as the entire pose distribution it is inflating. **Arm 11 now measures
the assumption itself**, and finds the estimator's roll error on those same
photographs bounded above at 0.70° — seven times smaller than the inflation.

### 5.4 The full table, 68 rows

Discriminability ratio at 0.5, 3 and 8 degrees applied to all three axes, with
the landmark-error term taken from arm 3's measured pipeline noise. **as
shipped** uses the a-priori `PoseSensitivity`; **measured** substitutes arm 2's
slopes at 1.0 m. Full table in `evals/results/arm05_table.csv`.

43 of 68 have a between-subject spread. **25 have none and are reported as
"unknown", which is the correct behaviour.**

| measurement | spread | ship 0.5° | ship 3° | ship 8° | meas 0.5° | meas 3° | meas 8° |
|---|---|---|---|---|---|---|---|
| `midface_projection` | 6.00% | **0.13** | 0.09 | 0.06 | 0.09 | 0.06 | 0.04 |
| `bizygomatic_width` | 5.44% | 0.45 | 0.31 | 0.19 | **6.90** | 4.74 | 2.91 |
| `e_line_upper_lip` | 6.00% | 0.49 | 0.34 | 0.21 | 1.16 | 0.86 | 0.55 |
| `e_line_lower_lip` | 6.00% | 0.49 | 0.34 | 0.21 | 1.30 | 0.90 | 0.56 |
| `upper_lip_projection` | 6.00% | 0.50 | 0.34 | 0.21 | **3.73** | 2.88 | 1.92 |
| `chin_projection` | 6.00% | 0.50 | 0.34 | 0.21 | **0.06** | 0.04 | 0.02 |
| `labiomental_sulcus_depth` | 6.00% | 0.50 | 0.34 | 0.21 | 1.47 | 1.01 | 0.62 |
| `submental_length` | 6.00% | 0.50 | 0.34 | 0.21 | **5.18** | 3.56 | 2.19 |
| `facial_width_height_ratio` | 7.96% | 0.66 | 0.66 | 0.66 | 0.66 | 0.66 | 0.66 |
| `bigonial_width` | 9.45% | 0.78 | 0.54 | 0.33 | **10.10** | 6.93 | 4.26 |
| `jaw_cheekbone_ratio` | 6.68% | 0.83 | 0.83 | 0.83 | 0.83 | 0.83 | 0.83 |
| `gonial_angle_l` | 6.05° | 0.95 | 0.65 | 0.40 | 0.96 | 0.66 | 0.40 |
| `canthal_tilt_r` | 2.10° | 1.40 | 0.96 | 0.59 | 1.12 | 0.77 | 0.47 |
| `canthal_tilt_l` | 2.10° | 1.40 | 0.96 | 0.59 | 1.11 | 0.77 | 0.47 |
| `biocular_width` | 4.23% | 5.09 | 3.49 | 2.15 | 4.81 | 3.31 | 2.03 |
| `cupids_bow_peak_height_r` | 6.00% | 5.81 | 4.42 | 2.91 | **0.53** | 0.37 | 0.23 |
| `cupids_bow_peak_height_l` | 6.00% | 6.01 | 4.51 | 2.93 | **0.54** | 0.37 | 0.23 |
| `upper_vermilion_height` | 6.00% | 6.91 | 4.85 | 3.02 | 1.71 | 1.17 | 0.72 |
| `margin_reflex_distance_1_l` | 6.00% | 7.03 | 4.89 | 3.03 | 4.88 | 3.37 | 2.08 |
| `margin_reflex_distance_{1_r,2_l,2_r}` | 6.00% | 7.09 | 4.91 | 3.03 | 4.90 | 3.38 | 2.08 |
| `lower_vermilion_height` | 6.00% | 7.14 | 4.93 | 3.04 | 1.66 | 1.14 | 0.70 |
| `philtrum_width` | 6.00% | 7.18 | 4.94 | 3.04 | 6.35 | 4.37 | 2.69 |
| `philtrum_length` | 6.00% | 7.19 | 4.95 | 3.04 | **2.01** | 1.38 | 0.85 |
| `chin_height` | 6.00% | 7.21 | 4.95 | 3.04 | 3.95 | 2.71 | 1.67 |
| `upper_face_height` | 6.00% | 7.21 | 4.95 | 3.04 | 6.78 | 4.66 | 2.86 |
| `palpebral_fissure_width_r` | 6.04% | 7.26 | 4.99 | 3.06 | **1.55** | 1.06 | 0.65 |
| `interpupillary_distance` | 6.17% | 7.42 | 5.09 | 3.13 | 6.99 | 4.80 | 2.95 |
| `nasofrontal_angle` | 9.19° | 7.70 | 5.29 | 3.25 | 16.04 | 11.03 | 6.78 |
| `palpebral_fissure_width_l` | 6.43% | 7.72 | 5.30 | 3.26 | **1.65** | 1.13 | 0.70 |
| `face_height_sellion_menton` | 6.80% | 8.16 | 5.61 | 3.45 | 7.03 | 4.83 | 2.97 |
| `intercanthal_width` | 7.23% | 8.69 | 5.97 | 3.67 | 8.07 | 5.54 | 3.41 |
| `lower_third_height` | 7.67% | 9.17 | 6.31 | 3.89 | **2.50** | 1.71 | 1.05 |
| `middle_third_height` | 8.02% | 9.64 | 6.62 | 4.07 | 9.12 | 6.26 | 3.85 |
| `nasolabial_angle` | 12.05° | 10.01 | 6.90 | 4.25 | 21.36 | 14.99 | 9.33 |
| `mouth_width` | 8.88% | 10.67 | 7.32 | 4.50 | 9.44 | 6.48 | 3.99 |
| `nose_height` | 8.91% | 10.71 | 7.36 | 4.52 | 5.45 | 3.74 | 2.30 |
| `palpebral_fissure_height_r` | 9.09% | 10.88 | 7.49 | 4.61 | 10.24 | 7.04 | 4.33 |
| `nose_breadth` | 13.32% | 16.01 | 10.99 | 6.76 | 15.28 | 10.49 | 6.45 |
| `palpebral_fissure_height_l` | 14.69% | 17.57 | 12.09 | 7.45 | 16.53 | 11.38 | 7.00 |
| `intercanthal_biocular_ratio` | 5.43% | 61.8 | 43.7 | 27.3 | 118.4 | 90.8 | 60.0 |
| `nose_mouth_width_ratio` | 11.07% | 125.2 | 88.7 | 55.5 | 48.3 | 33.3 | 20.5 |
| 25 further measurements | **unknown** | — | — | — | — | — | — |

**`midface_projection` is now the least discriminative measurement in the
catalogue as shipped**, at 0.13, and it does not recover with measured slopes.
It, `chin_projection` (0.06) and `upper_lip_projection` share a shape: a short
anteroposterior offset carrying a `DEFAULT_LINEAR_RSD = 0.06` relative spread
that is not meaningful for a signed offset.

**The left/right split is partly repaired and partly not.** `canthal_tilt_l`
and `canthal_tilt_r` now both carry 2.10° and receive the same verdict, which
the previous run flagged as a bug. `gonial_angle_l` still has 6.05° and
`gonial_angle_r` still has nothing, so the same measurement on the two sides of
one face is withheld on one side and called unknown on the other.
`palpebral_fissure_width_{r,l}` (6.04% / 6.43%) and
`palpebral_fissure_height_{r,l}` (9.09% / 14.69%) differ per side, but those
two come from a real per-side study rather than from a one-sided transcription,
and a 14.69% against 9.09% on a bilateral feature is worth a second look at the
source rather than an automatic repair.

### 5.5 Crossover: where arm 2's slopes become observable

The pose angle at which the pose effect equals arm 3's pipeline noise:

| measurement | axis | noise | slope (1.0 m) | crossover |
|---|---|---|---|---|
| `ramus_body_ratio_r` | yaw | 0.019% | 0.0261/deg | **0.007°** |
| `midface_projection` | pitch | 0.102% | 0.1259/deg | 0.008° |
| `gonial_angle_r` | roll | 0.011° | 1.275/deg | 0.009° |
| `nasal_dorsal_deviation` | yaw | 0.008° | 0.795/deg | 0.010° |
| `commissure_height_l` | pitch | 1.22% | 1.189/deg | 0.010° |

Every slope in arm 2 is far above the noise floor that produced it. The sweep
is measuring pose, not its own arithmetic.

---

## 6. Arms 6 and 7 — the normative model

### 6.1 Arm 6a: the vendored JSON reproduces exactly

All **99 stratum cells** recomputed from the raw 3,997-subject CSV match
`norms/data/niosh2003.json` to 1e-4 on mean and sd and exactly on n. Both
documented parsing quirks are real and are handled: `-9,999` missing values with
a thousands separator inside a quoted field, and `NECKCIRC`'s structural
missingness.

### 6.2 Arm 6b: the units question

**NIOSH `INTPUPBR` is millimetres.** Mean 64.37 mm, SD 3.97, against Dodgson's
ANSUR value of 63.36 that `core/scale.py` ships — a 1.6% difference consistent
with a different sampling frame and not with a unit error. 49.2% of `INTPUPBR`
values carry a fractional part while `BIZYGOBR`, `BIGONLBR`, `NOSEBR`,
`HEADCIRC` and `STATURE` are 100% integers in the same file. Male 65.07, female
63.14, bracketing `IPD_PRIORS`.

**Not verified directly:** no copy of ANSUR II is vendored and the harness does
not download one, so the "tenths of a millimetre" claim is confirmed only by
consistency. See §10.

### 6.3 Arm 6c: distribution shape

99 strata tested, **73 reject normality at 5%** by Anderson-Darling, max |skew|
0.64. Empirical coverage of the nominal central 90% interval runs 0.890 to
0.924. The normal model is statistically rejected at n≈4000 and adequate for
central percentiles; the tails misbehave.

`DEFAULT_LINEAR_RSD = 0.06` sits inside the measured within-sex range of 4.54%
to 13.74% but below its median of 7.53%, which errs toward under-stating
between-subject spread and therefore withholds more. Safe direction, but it is
an assumption — and §5.4 shows it is now applied to eleven more measurements
than it was, several of which are signed offsets for which a *relative* spread
is not a meaningful quantity at all.

### 6.4 Arm 6d: the undeclared-sex fallback inflates the spread

| measurement | pooled RSD | within-sex RSD | inflation | η² (sex) |
|---|---|---|---|---|
| `face_height_sellion_menton` | 6.80% | 5.77% | **1.18×** | 0.280 |
| `bizygomatic_width` | 5.44% | 4.78% | 1.14× | 0.230 |
| `nose_height` | 8.91% | 7.96% | 1.12× | 0.202 |
| `bigonial_width` | 9.45% | 8.69% | 1.09× | 0.154 |

A subject who declines to declare a sex gets a discriminability ratio up to 18%
higher than one who declares. Backwards.

### 6.5 Arm 7 — the negative control

**The negative control passes.** Permuting measurements within a stratum leaves
the percentile distribution bit-identical in all nine cases. The model is a
pure function of (value, stratum) and carries nothing about who the subject is.

**The positive control passes too**, which is what makes the negative one
meaningful: permuting demographic labels across strata raises the KS statistic
in six of nine, to a maximum of 0.158.

**Leave-one-out is irrelevant at this n:** the largest change is 0.0004.

Two things the arm found that are not about leakage:

- Only one of nine measurements produces uniform percentiles under correct
  strata. The largest KS is 0.064 (`nose_breadth`), so a reported percentile can
  be off by up to 6.4 percentile points from the normal-model misfit alone.
- Under the pooled fallback the male and female percentile distributions
  separate at KS up to **0.483** (`face_height_sellion_menton`), against 0.090
  when stratified. A pooled percentile for that measurement is a sex readout.

**One thing this run found about the harness rather than about the model.** Arm
7 was the single part of `evals/` whose "a rerun reproduces byte-identical
output" claim was **false**. It iterated `set(keys)` to walk the strata, and a
set of strings iterates in `PYTHONHASHSEED` order, so the permutations came off
the seeded stream in a different sequence in every process. The leak-free
conclusion never depended on which permutation was drawn — a within-stratum
shuffle is identity whatever the order — but the across-shuffle KS numbers
printed in the previous run were not reproducible, and one of the nine has
moved from "stratification carries information" to not (7 of 9 became 6 of 9).
Fixed to `sorted(set(keys))`; all 45 result files now check out identical
across consecutive runs.

---

## 7. Arms 8 to 11 — real faces

Face Research Lab London Set, DOI 10.6084/m9.figshare.5047666.v5, **CC BY 4.0**,
102 individuals with self-reported age, gender and ethnicity. Obtained through
the figshare API (the web page 403s automated fetches; the API does not).

Only the `neutral_front` capture ships with the 189-point human-delineated
template; the other captures are images only. Landmarks for those come from
**MediaPipe FaceLandmarker v1 float16** (Apache-2.0, `Tier.PERMISSIVE`).

Demographics: 69 white, 13 black, 10 west Asian, 9 east Asian, 1 mixed; 53 male,
49 female; ages 18-47 (2 missing).

### 7.1 Arm 8b — landmark agreement

Median distance between MediaPipe and the human template on the same 102
images, normalised by interpupillary distance and shown in millimetres at a
63.4 mm IPD. Unchanged from the previous run, since the landmark maps did not
change:

| landmark | median error / IPD | mm |
|---|---|---|
| `trichion` | **0.451** | **28.6** |
| `superciliare_{l,r}` | 0.161-0.172 | 10.2-10.9 |
| `gonion_{r,l}` | 0.128-0.143 | **8.1-9.0** |
| `gnathion` / `menton` | 0.113 | 7.1 |
| `zygion_{r,l}` | 0.097-0.105 | **6.1-6.7** |
| ... | ... | ... |
| `exocanthion_{l,r}` | 0.043-0.050 | 2.7-3.2 |
| `pupil_{l,r}` | 0.032-0.039 | 2.0-2.5 |
| `endocanthion_{l,r}` | 0.029-0.033 | 1.8-2.1 |
| `subnasale` | **0.017** | **1.1** |

Interior, well-defined midline and periocular points agree to 1-3 mm. Points on
a laterally curved, self-occluding surface disagree by 6-9 mm between two
observers looking at *the same photograph*, with no pose change involved.
`gonion` at 8-9 mm sits right on Lim et al. (2022)'s 9.3 mm mean difference for
bigonial breadth against calipers.

Twelve landmarks are flagged as suspect index maps at the 0.10 IPD threshold,
and the harness cannot separate "ill-defined landmark" from "imperfect index
map" without a third observer. `trichion` at 0.451 is certainly a mapping error
in `evals/frll.py`; it is not used by any catalogue measurement.

`superciliare` matters more now than it did. Two of the 23 new measurements
read it, and §7.5 shows what a 10 mm landmark does to them.

### 7.2 Arm 8c — measurement agreement, and a reversal

Bland-Altman between the two landmark sources, on the 48 of 68 measurements
FRLL can reach. **For 44 of 48 the limits of agreement are wider than twice the
between-subject SD**, against 30 of 32 last time.

| measurement | template | MediaPipe | mean difference | LoA / between-subject SD | r |
|---|---|---|---|---|---|
| `ocular_height_asymmetry` | 0.417° | 0.134° | **−0.283°** | **4.01** | **0.08** |
| `canthal_tilt_asymmetry` | 1.772° | 0.642° | −1.130° | 3.98 | 0.14 |
| `brow_apex_lateral_offset_l` | −0.521 | −0.061 | +0.461 | 3.94 | 0.21 |
| `palpebral_fissure_width_r` | 26.88 | 25.77 | −1.11 mm | 3.93 | 0.22 |
| `brow_apex_lateral_offset_r` | −0.498 | −0.045 | +0.453 | 3.75 | 0.31 |
| `bizygomatic_width` | 122.5 | 130.4 | +7.94 mm | 3.69 | 0.44 |
| `cupids_bow_peak_height_l` | 1.510 | 1.424 | −0.086 mm | 3.67 | 0.35 |
| `jaw_cheekbone_ratio` | 0.903 | 0.796 | −0.107 | 3.55 | 0.48 |
| `nasal_dorsal_deviation` | 1.623° | 1.245° | −0.379° | 3.50 | 0.50 |
| `bigonial_width` | 110.4 | 103.8 | −6.56 mm | 2.82 | 0.70 |
| ... | | | | | |
| `commissure_height_r` | −0.032 | −0.026 | +0.006 | **1.54** | **0.92** |
| `commissure_height_l` | −0.022 | −0.022 | −0.000 | **1.51** | **0.92** |
| `facial_thirds_ratio` | 0.797 | 0.959 | +0.161 | **1.38** | **0.94** |

**`ocular_height_asymmetry` was one of the two exceptions in the previous run
and is now the single worst measurement in the table.** That is not a
regression in the landmarks. It is §7.4.

### 7.3 Arm 8a — external validity against calipers

FRLL measurements scaled from the iris (`core.scale.from_iris`), against NIOSH.
Unchanged:

| measurement | FRLL mean | NIOSH mean | bias |
|---|---|---|---|
| `nose_mouth_width_ratio` | 0.727 | 0.730 | **−0.7%** |
| `mouth_width` | 50.0 | 50.7 | −1.5% |
| `nose_breadth` | 36.3 | 37.1 | −2.0% |
| `interpupillary_distance` | 62.5 | 64.4 | −2.8% |
| `bigonial_width` | 110.4 | 117.4 | −6.0% |
| `jaw_cheekbone_ratio` | 0.903 | 0.830 | **+8.6%** |
| `bizygomatic_width` | 122.5 | 141.1 | **−13.2%** |
| `nose_height` | 40.8 | 50.4 | **−19.0%** |

Scale recovery from the iris is good: measurements that depend only on
well-defined points land within 3% of a 3,997-person caliper survey, from a
population prior on corneal diameter and nothing else. The three large biases
are landmark-definition failures, not scale failures.

### 7.4 Arm 8e — roll attribution, and what the fix cost

New in this run. Every measurement that now claims roll robustness is computed
in **both** forms on all 102 real photographs — the shipped
interpupillary-referenced one and the discarded horizon-referenced one — and
each is correlated against the estimated camera roll of the image it came from.

| measurement | corr. with camera roll, shipped | horizon | between-source r, shipped | horizon | template SD, shipped | horizon |
|---|---|---|---|---|---|---|
| `canthal_tilt_l` | 0.137 | **0.643** | 0.657 | 0.824 | 2.557° | 3.317° |
| `canthal_tilt_r` | 0.191 | **−0.441** | 0.760 | 0.805 | 2.673° | 2.890° |
| `canthal_tilt_asymmetry` | −0.187 | −0.020 | **0.142** | **0.736** | 1.243° | 2.509° |
| `ocular_height_asymmetry` | −0.124 | −0.086 | **0.078** | **0.920** | 0.313° | 1.176° |
| `mouth_corner_asymmetry` | −0.130 | 0.265 | 0.755 | 0.812 | 0.956° | 1.184° |
| `nasal_dorsal_deviation` | −0.001 | −0.251 | 0.500 | 0.662 | 1.229° | 1.458° |
| `cupids_bow_peak_height_l` | −0.036 | 0.330 | 0.352 | 0.476 | 0.638 mm | 0.677 mm |
| `commissure_height_l` | 0.048 | 0.417 | 0.923 | 0.936 | 0.038 | 0.042 |

Three results.

**The two sides carry roll with opposite signs, on real photographs.** The
horizon form of `canthal_tilt_l` correlates +0.64 with camera roll and
`canthal_tilt_r` −0.44. That is the anti-symmetry arm 1 pins in closed form and
arm 2 measures on synthetic geometry, now visible in a studio dataset. It is
also why the horizon-form asymmetry has twice the spread of the shipped one.

**Most of the old between-subject spread in the two ocular asymmetries was the
photographer.** Switching `ocular_height_asymmetry` to the interpupillary
reference cut the template's between-subject SD from 1.176° to 0.313°, a factor
of 3.8, and `canthal_tilt_asymmetry` from 2.509° to 1.243°.

**And the excellent agreement the old form showed was agreement about the
camera, not about the face.** `ocular_height_asymmetry`'s between-source
Pearson r falls from **0.920 to 0.078** and `canthal_tilt_asymmetry`'s from
0.736 to 0.142 when the roll is removed. Two landmark sources agreed about that
measurement because they were both reading the same tilted camera. With the
camera gone there is nothing left they agree about: **neither landmark source
can measure ocular height asymmetry on these photographs at all.** The previous
run reported `ocular_height_asymmetry` among the two best-agreeing measurements
in the catalogue; that finding was an artefact of the definition it was
measured under, and this is what replaces it.

The correlations are attenuated by the roll estimate's own error and by FRLL's
small roll range (template roll SD 1.9°), so they are floors rather than
estimates.

### 7.5 Arm 8d — what a studio "front" capture actually looks like

| view | detected | yaw (mean ± sd, p95 abs) | pitch | roll |
|---|---|---|---|---|
| `neutral_front` | 102/102 | −0.4 ± 2.1, **3.9** | −0.8 ± 4.3, **8.3** | +0.1 ± 1.8, **3.9** |
| `neutral_left_3quarter` | 102/102 | −35.2 ± 3.4 | −1.4 ± 3.5 | +0.6 ± 3.4 |
| `neutral_left_profile` | **12/102** | −57.9 ± 6.9 | 0.0 ± 3.8 | −8.9 ± 27.6 |
| `neutral_right_profile` | **14/102** | +54.6 ± 15.2 | −7.2 ± 10.1 | −4.9 ± 15.1 |

**A studio frontal capture carries 8.3° of pitch at the 95th percentile.**
Applying the shipped tolerances directly, 85% of these studio captures pass.
Applying `gated_pose`'s +4.977° inflation as `assess_discriminability` does,
**28% pass**.

**MediaPipe cannot do a 90° profile: 12-14% detection.** The **20**
`View.PROFILE` measurements in the catalogue — up from 13, because six of the
23 additions are profile measurements — have **no working landmark source** in
this stack. That is now 29% of the catalogue.

### 7.6 Arm 9 — test-retest repeatability, measured

`MeasurementSpec.measured_within_person_rsd` is populated for **2 of 68**
measurements. This is the other 46 that FRLL can reach. Two conditions:
**expression** (neutral front vs smiling front — same camera, same pose,
different face) and **all usable** (those two plus the three 3/4 captures at
±35° yaw).

| measurement | between-person RSD | expr. CV | all CV | **D (expr)** | **D (all)** [CI low] |
|---|---|---|---|---|---|
| `facial_thirds_ratio` | 9.18% | 3.58% | 3.73% | 2.57 | **2.46** [2.25] |
| `jaw_cheekbone_ratio` | 2.59% | 0.78% | 1.61% | 3.32 | **1.61** [1.50] |
| `lip_vermilion_ratio` | 12.52% | 8.01% | 8.55% | 1.56 | **1.46** [1.37] |
| `eye_spacing_ratio` | 6.85% | 4.55% | 5.18% | 1.51 | **1.32** [1.22] |
| `canthal_tilt_r` | 61.95% | 77.0% | 168% | 2.71 | **1.24** [1.14] |
| `alar_base_intercanthal_ratio` | 7.22% | 5.77% | 5.83% | 1.25 | **1.24** [1.14] |
| `intercanthal_biocular_ratio` | 4.49% | 1.92% | 3.93% | 2.34 | **1.14** [1.06] |
| `lower_vermilion_height` | 24.33% | 17.2% | 22.4% | 1.41 | **1.09** [1.04] |
| `upper_vermilion_height` | 26.68% | 18.2% | 26.8% | 1.46 | 1.00 [0.96] |
| `nose_breadth` | 7.65% | 5.96% | 9.66% | 1.28 | 0.79 |
| `mouth_width` | 7.33% | 11.91% | 10.81% | **0.62** | 0.68 |
| `philtrum_width` | 8.76% | 13.27% | 11.46% | **0.66** | 0.76 |
| `bigonial_width` | 6.37% | 1.64% | 9.87% | 3.87 | 0.65 |
| `interpupillary_distance` | 3.65% | 1.40% | 7.00% | 2.60 | **0.52** |
| `bizygomatic_width` | 4.72% | 1.43% | 9.75% | 3.30 | **0.48** |
| `medial_canthal_angle_l` | 9.44% | **250%** | **697%** | 1.19 | **0.43** |
| `face_height_sellion_menton` | 5.43% | 1.99% | 19.14% | 2.73 | **0.28** |
| `palpebral_fissure_width_r` | 2.70% | 1.31% | 26.59% | 2.07 | **0.10** |
| `nasal_dorsal_deviation` | 70.81% | 74.7% | **1580%** | 1.18 | **0.06** |
| `brow_apex_lateral_offset_r` | 103% | 194% | **2722%** | 0.53 | **0.04** |
| `brow_apex_lateral_offset_l` | 72.3% | 207% | **4875%** | 0.35 | **0.01** |

Median within-person CV: **7.2% at fixed pose, 19.5% across ±35° of yaw**, up
from 3.0% and 12.5% at 32 measurements — the 23 additions are markedly noisier
than what was there.

**Informative (lower CI bound above 1.0): 31 of 48 at fixed pose, 8 of 48
across captures.** Above the shipped `repeatability_cv > 0.10` withhold
threshold: 14 of 48 at fixed pose, 27 of 48 across captures.

Four findings.

**The evidence tiers are still in the wrong order, and the one tier that
predicts well now has a new member.** Every `Evidence.VALIDATED_2D` measurement
falls below D = 1 once the photograph is allowed to vary, while
`jaw_cheekbone_ratio` (`REQUIRES_3D`, withheld outright from a 2D image) is
second in the catalogue at D = 1.61. The six most discriminative measurements
are all dimensionless, and `alar_base_intercanthal_ratio` — added in the second
batch and the only new measurement tagged `POSE_INVARIANT_RATIO` — comes in
sixth at 1.24 [1.14]. That tier, at least, predicts what it says it predicts.

**The four new orbital measurements are the least informative things in the
catalogue.** `brow_apex_lateral_offset_{l,r}` at D = 0.01 and 0.04 and
`nasal_dorsal_deviation` at 0.06 are not marginal, they are noise. All three
divide a small signed quantity by something, and the two brow offsets read
`superciliare`, which arm 8b measures at 10.2-10.9 mm of disagreement between
two observers of one photograph. `medial_canthal_angle_{l,r}` has a
within-person CV of 250% at fixed pose: MediaPipe's palpebrale points are not
stable enough to subtend an angle at the inner canthus.

**Expression alone is a bigger error source than the pose model predicts.**
`mouth_width` moves 11.9% and `philtrum_width` 13.3% between a neutral and a
smiling photograph at the same pose, which no projection model can see.

**`facial_width_height_ratio`'s shipped 0.12 is close.** Measured across
captures: 12.52% CV. That is still the one place where an assumed number
survives contact with data — and it is one of only two that were ever measured,
against a catalogue that is now 68.

### 7.7 Arm 10 — fairness, and why this sample cannot settle it

Cells: white 69, black 13, west Asian 10, east Asian 9; male 53, female 49.

| group | n | median landmark error / IPD | 95% CI |
|---|---|---|---|
| black | 13 | 0.0596 | [0.0390, 0.0647] |
| east Asian | 9 | 0.0568 | [0.0425, 0.0636] |
| west Asian | 10 | 0.0543 | [0.0359, 0.0717] |
| white | 69 | 0.0538 | [0.0484, 0.0579] |

**Zero of seven pairwise comparisons separate at 5%.** The point estimates run
in the direction the literature would predict and the intervals overlap so
heavily that the data cannot distinguish that from noise. **FRLL is 102 people
and 68% white. It is not capable of settling this and this arm does not claim
to.**

**What FRLL *can* settle is a different and larger problem.** The mean
percentile each ethnic group receives against the **pooled** NIOSH stratum:

| measurement | white | west Asian | east Asian | black | spread |
|---|---|---|---|---|---|
| `nose_breadth` | 0.322 | 0.553 | 0.663 | 0.813 | **0.491** |
| `nose_mouth_width_ratio` | 0.383 | 0.558 | 0.656 | 0.724 | 0.341 |
| `mouth_width` | 0.373 | 0.521 | 0.545 | 0.707 | 0.334 |
| `interpupillary_distance` | 0.294 | 0.454 | 0.427 | 0.602 | 0.308 |
| `face_height_sellion_menton` | 0.171 | 0.218 | 0.307 | 0.336 | 0.166 |

A pooled-stratum report tells the median black subject their nose is at the 81st
percentile and the median white subject theirs is at the 32nd — a 49-point gap
that is a property of the reference population. **The pooled fallback is not a
neutral default. It is the white default for this measurement, and the report
has to say so on the page.**

### 7.8 Arm 11 — the pose estimator, bounded without a labelled benchmark

New in this run, on the 102 FRLL front captures, with YuNet supplying the box
and 6DRepNet supplying the pose — the same two models the pipeline uses, so
this is a statement about the estimator whose published 3.97 is quoted, not
about a stand-in.

**11a, the positive control and the roll ground truth.** Each image is rotated
about its centre through ±5, ±10, ±15 degrees. A rotation of the image is a
rotation of the camera about its optical axis, so the change in true roll is
exactly the angle applied.

| | 6DRepNet | MediaPipe |
|---|---|---|
| fitted slope of estimated roll on applied roll, median | **−0.994** [−1.006, −0.972] | **+1.000** |
| residual mean absolute error about that fit | **0.177°** | 0.068° |
| residual 95th percentile | 0.445° | — |
| yaw range over a sweep in which true yaw never changes | **1.90°** | — |
| pitch range over the same sweep | **1.47°** | — |

The control passes: the estimator tracks a known rotation with a slope whose
magnitude is 0.994. It tracks it **negatively**, which resolves on the roll axis
the sign question `models/pose_sixdrepnet.py` records as open — 6DRepNet's roll
is anti-aligned with a positive OpenCV in-plane rotation, and independently
anti-aligned with the mathematical angle of the human-placed interpupillary
line, and the two agree. MediaPipe's roll runs the other way, so **the two
estimators in this repository carry opposite roll sign conventions**, and their
yaw and pitch correlate at r = −0.94 and −0.92 as well.

The last two rows are the interesting ones. Over a sweep in which the true yaw
and pitch do not change at all, 6DRepNet's yaw wanders 1.90° and its pitch
1.47°. That is pure estimator error with exact ground truth and no labels.

**11b, the mirror, a label-free lower bound.** Under a horizontal flip the true
pose becomes (−yaw, pitch, −roll), so half the sum of the two yaw estimates is
pure error.

| axis | MAE | SD | p95 |
|---|---|---|---|
| yaw | **0.328°** | 0.379° | 0.923° |
| pitch | **0.273°** | 0.469° | 0.693° |
| roll | **0.228°** | 0.279° | 0.564° |

A lower bound: any error that is itself mirror-symmetric cancels here.

**11c, roll against a human's landmarks, an upper bound.** The FRL template is
placed by a person and includes both pupils, so the inclination of the
interpupillary line measures image roll without a network.

| | slope | Pearson r | residual SD | residual MAE |
|---|---|---|---|---|
| 6DRepNet | −0.822 | −0.913 | **0.699°** | 0.561° |
| MediaPipe | +0.885 | +0.936 | 0.633° | 0.506° |

Errors add in quadrature, so 0.699° upper-bounds the estimator's roll spread on
these photographs. The fitted slope of 0.82 rather than 1.0 is regression
dilution from the reference's own noise — the human's line carries the
subject's ocular asymmetry as well as the camera — which is exactly why this is
an upper bound and not an estimate. 11a, with an exact reference, recovers
0.994.

**11d, two estimators on one photograph.**

| axis | raw difference MAE | raw difference SD | residual SD after a fitted slope |
|---|---|---|---|
| yaw | 3.24° | 4.08° | 0.72° |
| pitch | **6.88°** | **8.30°** | 1.58° |
| roll | 2.75° | 3.49° | 0.44° |

Two independent estimators disagree by **6.9 degrees of pitch, on average, on a
studio frontal portrait**. Almost all of it is a systematic gain and offset —
the residual after fitting a slope is 1.58° — but a consumer that takes either
number at face value is taking a six-degree position on a photograph where the
whole pitch distribution has an SD of 4.3°.

**What this establishes.** On near-frontal studio portraits, 6DRepNet's roll
error is bounded above at 0.70° of standard deviation and its roll-increment
error measured at 0.177° MAE, against the **4.977°** that `gated_pose` adds to
every axis. Its yaw and pitch error is at least 0.33° and 0.27° and at least
1.9° and 1.5° of range over a sweep that should move neither. The declared
3.97° is not refuted — it is an AFLW2000-3D average over a pose range FRLL does
not cover, and the backend's own docstring puts near-frontal performance at
about 2.8 — but the gate does not apply 3.97 to AFLW2000-3D, it applies 4.977
to whatever photograph arrives, and for the class of photograph this project is
built for that inflation is at least four and plausibly ten times the
estimator's actual scatter.

---

## 8. What these results contradict in `src/`

Ordered by how much damage the assumption does. Items marked **new** were not
in the previous run's list.

1. **A signed quantity near zero has no relative anything, and seven
   measurements in the catalogue are one.** **new.** `commissure_height_{l,r}`,
   the three asymmetries and `cupids_bow_peak_height_{l,r}` all receive a
   *relative* pose sensitivity, a *relative* between-subject spread
   (`DEFAULT_LINEAR_RSD` where the unit is millimetres) and a *relative*
   landmark term. Measured consequences: `commissure_height_l` moves 119× its
   declared pitch sensitivity and −83% at one metre of camera distance, and
   `commissure_height_r` has no relative slope at all because its value is
   exactly zero. The fix is structural, not a number: these need an absolute
   spread and an absolute error budget. (§3.6, §4.2, §5.2)

2. **`_AP_PROJECTION` is 1.6× low on the axis it was derived for, and
   `chin_projection` is 19× low because it never got the constant at all.**
   **new.** Measured pitch slopes: `midface_projection` 0.126 against a
   declared 0.08, `chin_projection` 0.191 against the 0.010 its
   `KLEINBERG_WORST` fallback gives it. The `_AP_PROJECTION` docstring predicts
   both and says the sweep should settle them. It has. (§3.5)

3. **`_MIDLINE_DEVIATION`'s yaw term is 1.6× low and its roll term is 5.7× low
   under a real camera.** **new.** Measured yaw 0.79 against a declared 0.5. At
   the shipped 4° tolerance that manufactures 3.2° of dorsal deviation on a
   straight nose. The roll term is exactly right under orthography and 5.7×
   low at half a metre, because roll commutes with orthographic projection and
   not with a perspective divide. (§3.3, §3.5)

4. **`KLEINBERG_WORST` is in the wrong term of the error budget**, now for
   eight measurements rather than two. `_SENSITIVITY` applies it as a *pose*
   sensitivity to `bizygomatic_width` and `bigonial_width`, whose measured pose
   sensitivity is 0.0015/deg — 12.5× less — and the fallback applies it to
   `upper_lip_projection`, `labiomental_sulcus_depth`, `commissure_height_{l,r}`
   and `chin_projection`. What is actually wrong with the widths is landmark
   localisation on a self-occluding silhouette, which arm 8 measures at 6-9 mm.
   (§3.4, §5.2, §7.1)

5. **The pose gate reports an assumption more than a photograph, and the
   assumption is now measured to be at least four times too large for the
   photographs it gates.** **new evidence.** `gated_pose` adds 4.977° to every
   axis. Arm 11 bounds 6DRepNet's roll error above at 0.70° SD on the same
   class of image and measures its roll-increment MAE at 0.177°. Applied to
   FRLL's studio frontal captures the gate refuses 72%. (§5.3, §7.5, §7.8)

6. **`VERTICAL_DISTANCE` has pitch and roll swapped for a profile-view
   measurement.** **new.** `chin_height` is `View.PROFILE` and carries
   `VERTICAL_DISTANCE` (pitch 0.00152, roll 0.0). Under the profile camera,
   pitch is the *in-plane* rotation and preserves lengths exactly, while roll
   swings them out of the image plane. Measured: pitch 0.0 to 2e-14, roll
   0.00151. The constant is a frontal-camera constant applied to a profile
   measurement. (§3.4)

7. **`VERTICAL_DISTANCE` is 3-45× low for any vertical that is a difference of
   two offsets or has depth between its endpoints.** **new.**
   `cupids_bow_peak_height` 0.0232 against 0.00152 (15×, and 45× on a face with
   depth asymmetry), `upper_vermilion_height` and `lower_vermilion_height`
   0.0043 (2.9×), `lower_third_height` 0.0031, `philtrum_length` 0.0070 (4.6×).
   A vertical carries `cos(pitch)` only if its two endpoints sit at the same
   depth. (§3.4)

8. **Three measurements are assigned `TRANSVERSE_RATIO` and are not ratios of
   two transverse spans**, plus a fourth whose exactness claim is
   overstated. `eye_aspect_ratio_{l,r}` measures 48× its declared yaw
   sensitivity; `facial_thirds_ratio` and `lip_vermilion_ratio` are ratios of
   two verticals. **new:** `alar_base_intercanthal_ratio`'s "the cosine cancels
   exactly rather than approximately" holds only when the mirrored pair is
   symmetric *in depth*; give each side two millimetres of independent depth
   and it reads 6.4× the declared value. (§3.7)

9. **`_CANCELLING`'s replacement values are right about roll and wrong about
   yaw and pitch.** **new.** The 0.002/deg roll claim was corrected to a
   geometry that genuinely cancels roll, and arm 2 confirms it exactly, at
   1e-15. But the replacement assigns 0.01 on *all three* axes, and measured:
   `canthal_tilt_asymmetry` yaw 0.068 (6.9×),
   `mouth_corner_asymmetry` yaw 0.090 at half a metre (9.0×),
   `canthal_tilt_asymmetry` pitch 0.099 on an asymmetric face (9.9×),
   `ocular_height_asymmetry` pitch 0.032 (3.2×). Roll was the axis that got the
   attention and the other two inherited its number. (§3.3, §3.4)

10. **Four measurements compute neither published convention, not two.** **new
    for two of them.** `LineOffset(..., Axis("z"))` returns the z component of
    the perpendicular. Against the convention the reference range is quoted in:
    `e_line_{upper,lower}_lip` −9.0%, `labiomental_sulcus_depth` −7.5%,
    `upper_lip_projection` −0.4%. The factor is `cos` of the reference line's
    inclination and varies between subjects. (§2.1)

11. **`ocular_height_asymmetry` and `canthal_tilt_asymmetry` are not measurable
    by either landmark source, and the previous run's evidence that they were
    is now known to have been evidence about the camera.** **new.**
    Between-source Pearson r falls from 0.920 to 0.078 and from 0.736 to 0.142
    when the interpupillary reference removes the roll, and the template's
    between-subject SD falls by 3.8× and 2.0×. The measurements are now
    correct and now demonstrably empty. (§7.4)

12. **`Evidence.VALIDATED_2D` does not predict discriminability, and
    `Evidence.REQUIRES_3D` mispredicts it.** All seven `VALIDATED_2D`
    measurements fall below D = 1 across captures, while `jaw_cheekbone_ratio`
    (`REQUIRES_3D`) is second in the catalogue at 1.61 [1.50]. The one tier
    that does predict is `POSE_INVARIANT_RATIO`. (§7.6)

13. **`magnification_distortion` understates the true magnification by a factor
    of K.** Exact is `K/(1−K)`. −16.7% at 0.3 m, −5.0% at 1.0 m. The value is
    printed to the user in `decide_reportability`. (§4.1)

14. **Left and right of the same measurement still get different verdicts, for
    one pair.** `canthal_tilt_{l,r}` is repaired. `gonial_angle_l` has a
    published spread and `gonial_angle_r` has none, so one side is withheld and
    the other is called unknown. (§5.4)

15. **The undeclared-sex fallback makes Faciometry more confident, not less.**
    `both|pooled` inflates the between-subject SD by up to 1.18×, and that SD
    is the numerator of the discriminability ratio. (§6.4, §6.5)

16. **The normal percentile is miscalibrated by up to 6.4 percentile points**,
    and 73 of 99 NIOSH strata reject normality. (§6.3, §6.5)

17. **`nasal_tip_projection_ratio` reads one ala, not both.** (§2.2)

18. **20 of 68 measurements have no working landmark source**, up from 13 of
    45. Every `View.PROFILE` spec needs `pogonion`, `sublabiale` or
    `cervicale`, or a profile photograph, and the only permissive landmarker
    tested detects a face in 12-14% of true profiles. Six of the 23 additions
    landed in that bucket. (§7.5)

19. **6DRepNet and MediaPipe carry opposite sign conventions on all three
    axes.** **new.** r = −0.97, −0.94, −0.92 for roll, yaw and pitch. Nothing
    downstream breaks today because `gated_pose` takes an absolute value, and
    `HeadPose.disagreement` between two estimators is documented as the check
    that has power — a raw difference between two anti-aligned estimates is
    twice the pose, not the disagreement. (§7.8)

Five things `src/` gets right and the data confirms:

- **The measurement layer is exact.** Machine precision on all 68. (§2)
- **The roll fix is real, and exact.** Sixteen roll claims, sixteen held, at
  1e-15 for the eight that claim exactness and to five significant figures for
  the eight that claim a cosine — with a positive control that proves the
  discarded form moves. (§3.3)
- **`_SAGITTAL_FRAME_ANGLE`'s central claim is exactly right.** Pitch enters
  `nasal_tip_rotation` at 1.00000 per degree in every condition tested. (§3.5)
- **The negative control passes:** the normative model carries nothing about
  identity, and the positive control proves the test can see a difference.
  (§6.5)
- **`facial_width_height_ratio`'s `measured_within_person_rsd = 0.12` from
  Kramer reproduces at 12.52% on 102 people.** (§7.6)

---

## 9. Recommended changes to `_SENSITIVITY` (`measure/registry.py`)

Written down, not applied — `evals/` does not own `src/`. Measured values are
the arm 2 secant at ten degrees under the `persp_1.0m` condition unless noted,
which is the regime a compliant portrait is in. Full table:
`evals/results/arm02_recommendations.csv`.

**Must change — the a-priori value is wrong by more than an order of
magnitude:**

| id | axis | current | recommended | basis |
|---|---|---|---|---|
| `commissure_height_l` | pitch | 0.010 | **1.19** | measured, 1.0 m; see the note below |
| `commissure_height_l` | yaw | 0.019 | **0.237** | measured, 1.0 m |
| `chin_projection` | pitch | 0.010 | **0.191** | measured, all conditions |
| `cupids_bow_peak_height_{l,r}` | pitch | 0.00152 | **0.020** | measured, 1.0 m (0.023 orthographic, 0.068 with depth asymmetry) |
| `eye_aspect_ratio_{l,r}` | yaw | 0.00015 | **0.0067** | measured, 1.0 m |
| `midface_projection` | pitch | 0.08 | **0.126** | measured, all conditions |

**Should change — wrong by 2-10×:**

| id | axis | current | recommended |
|---|---|---|---|
| `canthal_tilt_asymmetry` | yaw | 0.01 | 0.068 |
| `canthal_tilt_asymmetry` | pitch | 0.01 | 0.099 (asymmetric face) |
| `mouth_corner_asymmetry` | yaw | 0.01 | 0.090 (0.5 m) |
| `ocular_height_asymmetry` | pitch | 0.01 | 0.032 (asymmetric face) |
| `nasal_dorsal_deviation` | yaw | 0.5 | 0.795 |
| `nasal_dorsal_deviation` | roll | 0.01 | 0.057 (0.5 m) |
| `gonial_angle_{l,r}` | yaw | 0.15 | 0.58 (left), 0.22 (right) — see note |
| `upper_vermilion_height` | pitch | 0.00152 | 0.0064 |
| `lower_vermilion_height` | pitch | 0.00152 | 0.0066 |
| `palpebral_fissure_width_{l,r}` | yaw | 0.00152 | 0.0070 |
| `philtrum_length` | pitch | 0.00152 | 0.0054 |
| `lower_third_height` | pitch | 0.00152 | 0.0056 |
| `nose_height` | pitch | 0.00152 | 0.0030 |
| `facial_thirds_ratio` | pitch | 0.0 | 0.0062 |
| `philtrum_width` | yaw | 0.00152 | 0.0050 (asymmetric face) |
| `alar_base_intercanthal_ratio` | yaw | 0.00015 | 0.00098 (asymmetric face) |
| `chin_height` | pitch → roll | 0.00152 on pitch | **0.0 on pitch, 0.0028 on roll** |

**Too pessimistic — the current value withholds for a projection effect that is
not there:**

| id | axis | current | measured |
|---|---|---|---|
| `bizygomatic_width` | yaw | 0.019 | 0.0014 |
| `bigonial_width` | yaw | 0.019 | 0.0014 |
| `submental_length` | yaw | 0.019 | 0.0015 |
| `midface_projection` | yaw | 0.020 | 0.0015 |
| `e_line_{upper,lower}_lip` | yaw | 0.019 | 0.0013 |
| `upper_lip_projection` | yaw | 0.019 | 0.0015 |
| `labiomental_sulcus_depth` | yaw | 0.019 | 0.0011 |
| `facial_convexity_angle` | yaw | 0.15 | 0.0103 |
| `nasofrontal_angle` | yaw | 0.15 | 0.0735 |
| `medial_canthal_angle_{l,r}` | pitch | 0.15 | 0.053 |
| `medial_canthal_angle_{l,r}` | roll | 0.05 | 0.0 (ortho) / 0.0027 (1.0 m) |

**Do not simply lower these.** For `bizygomatic_width` and `bigonial_width`,
dropping the pose term without adding the landmark term would flip them from
withheld to reported (arm 5: D goes from 0.45 to 6.90 and from 0.78 to 10.10),
and arm 8 shows their real error is 6-9 mm of silhouette drift. The
recommendation is **structural**: split the budget, put the measured projection
slope in `sensitivity`, and put the silhouette error in the landmark term.

**Structural recommendations beyond a table of numbers, in priority order:**

- **A relative sensitivity is the wrong type for a signed near-zero quantity.**
  `commissure_height_l`'s recommended pitch slope of 1.19 per degree is
  arithmetically correct and practically useless: it says the measurement is
  destroyed by one degree of pitch, which is true, and it says it in a unit
  that will read as a 119% error on a quantity whose absolute movement is 0.016
  of a mouth-width. `commissure_height_r` cannot be given a number at all,
  because its value is zero. The seven measurements listed in §8 item 1 need an
  **absolute** error budget and an absolute between-subject spread, and until
  they have one the honest output for them is "unknown", which is what the 25
  measurements without a published spread already get.
- **`PoseSensitivity` has no distance parameter, and it now matters more.**
  Perspective changes the yaw sensitivity of `gonial_angle_r` by 3.7× between
  1.0 m and 0.5 m, turns exact-zero pitch sensitivities into real ones, and
  turns the *exactly* zero roll sensitivity of every interpupillary-referenced
  measurement into a real one — `nasal_dorsal_deviation` from 0.0 to 0.057.
  Either take `subject_distance_m` in `error_at`, or state in the docstring
  that the values are the orthographic limit.
- **`VERTICAL_DISTANCE` needs a profile-view sibling.** Its pitch and roll
  entries are swapped for a `View.PROFILE` measurement, because the profile
  camera's in-plane rotation is pitch and not roll.
- **`POSE_ESTIMATOR_SD_DEG` should not be a single scalar applied to all three
  axes.** Arm 11 measures roll error at 0.18-0.70° and yaw and pitch at 1.5-1.9°
  of range on the same photographs — a factor of ten between axes, in a
  constant that has one value. And `gated_pose`'s 4.977° is at least four times
  the largest of them for the near-frontal regime the project is built for.
- **`measured_within_person_rsd` should be populated from arm 9 for the 46
  measurements FRLL reaches**, starting with `mouth_width` (0.119) and
  `philtrum_width` (0.133), whose dominant error is expression and which no
  projection model can predict. Two of 68 is the state of the art in this
  repository and it is the finding.

Two docstrings point at a path that does not exist: `core/sensitivity.py` and
`core/spec.py` both say the empirical sweep lives at `evals/arms/pose_sweep.py`.
It is `evals/arms/arm02_pose_sweep.py`.

---

## 10. Arms not run, and why

Every arm in §1 ran, including the new arm 11. What could not be done *inside*
those arms:

| Not done | Why |
|---|---|
| **AFLW2000-3D head-pose MAE** (arm 11) | Requires registration. Arm 11 bounds the estimator on FRLL instead: an exact bound on roll increments (0.177° MAE), an upper bound on roll (0.699° SD), and label-free floors on yaw and pitch (0.33°, 0.27°). None of that reproduces 3.97 on AFLW2000-3D, because FRLL is near-frontal and AFLW2000-3D is not. **The published number remains unverified; what is no longer unknown is that it is far too large for the photographs the gate is applied to.** |
| **A known out-of-plane rotation** (arm 11) | No image-space operation induces one. A planar homography does not — the backend docstring records an attempt that moved the estimate by under two degrees for twenty-five degrees of simulated rotation. So yaw and pitch get a floor from the mirror and a range from the roll sweep, not a bound. |
| **ANSUR II direct unit verification** (arm 6b) | No copy of ANSUR II is vendored and the harness does not fetch one. The tenths-of-a-millimetre claim is confirmed by consistency with NIOSH (64.37 mm), Dodgson (63.36 mm) and the integer/decimal structure of the NIOSH columns, not by reading the ANSUR II file. |
| **Test-retest for the 20 profile measurements** (arm 9) | MediaPipe detects a face in 12/102 and 14/102 true profiles. There is no landmark source for `pogonion`, `sublabiale` or `cervicale` in any permissively licensed model tested. |
| **True same-pose same-expression retest** (arm 9) | FRLL has exactly one neutral front capture per person. The tightest retest available is neutral vs smiling front, which necessarily includes an expression change, so the "fixed pose" CVs are upper bounds on pure repeatability and lower bounds on the full within-person spread. The same limitation is what stops multi-capture pooling being validated end to end. |
| **A discriminating roll control for the four margin-reflex distances** (arm 2) | On the synthetic face the lid margins sit directly above the pupils, so the interpupillary-referenced and horizon-referenced forms are the same number. The control has power (it moves) but no discrimination (it moves identically). Needs a synthetic face whose lid margin is laterally displaced from the pupil. |
| **Landmark NME against 300W / WFLW** | Both require registration. Arm 8b substitutes agreement between a model and a human on the same images, per landmark. |
| **Separating "ill-defined landmark" from "wrong index map"** (arm 8b) | Needs a third independent observer. Twelve landmarks are flagged suspect; `trichion` at 0.451 IPD is certainly a mapping error, `superciliare` at 0.16-0.17 probably is, and `superciliare` is now load-bearing for two catalogue measurements. |
| **A powered fairness result** (arm 10) | FRLL cells are 9 to 69 subjects. Roughly 60 per cell would be needed to resolve a 20% relative difference in landmark error. Reported as unresolved rather than as a null. |
| **`derm` and end-to-end pipeline arms** | Those stages are outside the scope of this harness, which stops at the measurement and normative layers. |
| **Offline-egress assertion** | Belongs in `tests/`, not `evals/`; it is a property of the analysis path, not a measurement. Note that `evals/` itself *does* fetch data — over `make -C evals data`, as an explicit separate step, never inside an arm. |

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
  synth/truth.py              independent ground truth for all 68, pure-Python math
  arms/arm01..arm11           one module per arm, each runnable standalone
  results/*.json, *.csv       machine-readable output, regenerated by run_all
  results/timings.json        wall clock, the one file excluded from the
                              byte-identical-rerun guarantee
  data/                       downloaded, gitignored, including the two
                              detection caches (frll_mediapipe.npz,
                              arm11_pose.npz)
```
