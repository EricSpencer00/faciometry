# What this system actually establishes

Written after the build, from measured results rather than from the design.

## 1. On a single frontal photograph, with a 4%-NME landmark model, almost nothing is reportable

Running the shipped permissive stack (YuNet, SPIGA, MediaPipe, 6DRepNet) against
studio portraits from the Face Research Lab London set (CC BY 4.0, 1350x1350):

```
FRLL 001_03.jpg   1 of 45 reported (with a caveat), 23 withheld, 21 not attempted
FRLL 002_03.jpg   0 of 45 reported,                 24 withheld, 21 not attempted
```

This is the correct output, and the arithmetic behind it is short. SPIGA's
published WFLW normalised mean error is 4.06% of the inter-ocular distance. On
this image the inter-ocular span is 215 px, so mean landmark error is roughly
8.7 px. The intercanthal width spans about 110 px, so two independent endpoint
errors put roughly 11% of relative error on it. NIOSH measured the
between-person spread of comparable dimensions at 6 to 13%. The measurement
error is the same size as the thing being measured, so the number describes the
photograph.

Kleinberg and Vanezis reached the same conclusion by rotating subjects rather
than by propagating landmark error, and FISWG's 2026 guidance prohibits
photo-anthropometry for identification on that basis. Two independent routes,
one answer.

**What would change it**, in descending order of effect:

- a ruler in the facial plane, worth seven additional clean measurements in a
  synthetic clinical capture by removing the scale prior entirely
- a lower-NME landmark model, which enters linearly
- more pixels on the face, since the landmark term halves when resolution doubles
- repeat photographs, since averaging N captures divides the landmark term by
  the square root of N
- a real 3D fit, which unlocks the measurements over self-occluding surfaces

None of these is exotic. The system is not saying facial morphometry is
impossible; it is saying a single casual photograph does not carry it.

## 2. SPIGA's heatmap widths are a training artefact, not confidence

The design assumed heatmap second moments would give genuine per-landmark
uncertainty, tight on a pupil and elongated along a jaw contour. Measured on a
real portrait across all 98 channels:

```
per-landmark sd        median 10.04 px, range 9.55 to 10.70   (12% spread)
anisotropy             median 1.11, max 1.26
```

If these carried real confidence a pupil would be several times tighter than a
contour point, and contour points would be strongly elongated. They are neither.
The second moment is dominated by the width of the Gaussian target the network
was trained against.

Consequences, both of which are live in the code:

- The **shape** is kept, because it is the only per-landmark signal available,
  but the **scale** is calibrated against SPIGA's published aggregate NME. That
  calibration is provisional, flagged in the backend docstring, and disableable.
- The anisotropic covariance handling in `measure/evaluate.py` and the
  uncertainty ellipses in `report/overlay.py` are correct and exercised by
  tests, but this backend does not deliver the anisotropy they were built for.
  A model trained with a heteroscedastic or STAR-style anisotropic loss would.

Recorded here rather than quietly worked around, because a reader looking at an
ellipse in a report should know it is nearly a circle for a reason.

## 3. Three defects the build found in its own gate

Each was caught by a different layer, which is the argument for having built
them separately.

**Roll cancellation was asserted backwards.** The catalogue claimed a common
rotation "adds the same offset to both sides and cancels in the difference" at
0.002 per degree. The pose sweep measured 2.0 per degree, and the pipeline
agent independently derived the same number. Because each side was measured
against its own lateral axis, roll entered with opposite sign and *added*: two
degrees of roll manufactured four degrees of canthal tilt asymmetry on a
perfectly symmetric synthetic face. Fixed by measuring against the
interpupillary line, which rotates with the head, so roll now cancels in the
geometry rather than being declared away in a constant.

**The landmark error term was dimensionally wrong.** It divided a positional
spread in pixels by the measurement's own value, which is only sound when that
value is a length in those same pixels. For a dimensionless ratio it returned
3.45 where the propagated truth was 0.03, withholding every ratio in the
catalogue for a units artefact. The tell, found by the pipeline layer, was that
it did not change when the face was photographed at twice the resolution. It
now comes from the Monte-Carlo ensemble and halves exactly when resolution
doubles.

**The 3D requirement was keyed off the evidence tier rather than the
landmarks.** The gonial angle is tagged for its pose sensitivity, not as
requiring 3D, so it walked through the self-occlusion refusal and was briefly
the only measurement a frontal photograph reported.

## 4. The measurement layer itself contributes nothing

Arm 1 of the evaluation harness evaluates every catalogue measurement and 20
geometric primitives against independently computed ground truth: worst absolute
error 2.8e-14, worst relative error 3.6e-15, no failures. Whatever is wrong with
a Vitruve number, the arithmetic is not it.

That run covered the catalogue as it stood at 45 measurements. The 23 added
since carry equivalent closed-form checks in
`tests/unit/test_extended_catalogue.py`, written against independently derived
geometry, but they have not been through the evaluation harness itself. Arm 1
should be re-run before the next release, and this paragraph should say so
until it has been.

## 5. The permissive stack cannot measure a true profile

A full facial-analysis report is taken from a frontal and a profile photograph.
Vitruve's catalogue carries 13 profile measurements, the formulas are correct
against synthetic geometry, and the pipeline accepts a `--profile` argument.
The models are what stop it.

Measured on FRLL's 90-degree profile captures:

```
frontal          6DRepNet yaw    3.0 deg     (correct)
90 deg profile   6DRepNet yaw  -26.0 deg     (should be near -90)
```

6DRepNet is trained on 300W-LP, whose extreme-yaw coverage is thin, and it
saturates well before a true profile. SPIGA, trained on WFLW, fails there too:
on the same image the two estimators disagreed by 136 degrees. Published
estimators sit near 4 degrees of mean absolute error, so a gap that size means
one of them has failed outright rather than been imprecise, and nothing in the
image says which.

The quality gate refuses the photograph, and that is the correct outcome. A
profile angle derived from a pose estimate that is 64 degrees wrong would be a
confident number about nothing. What it costs is real: the 13 profile
measurements are unreachable with the shipped permissive stack.

**What would fix it**, in order of directness:

- a head-pose estimator with genuine full-range coverage. WHENet and DirectMHP
  both train against 360 degrees rather than the 300W-LP frontal cone.
- a landmark model that localises on a profile silhouette. This is a different
  problem from frontal landmarking, because half the anatomical points are
  occluded by definition rather than by accident.
- a three-quarter oblique instead of a true profile, which is within the range
  the current models handle and still carries the nasal and mandibular contour.

Recorded here rather than quietly dropped, because "supports profile
photographs" is exactly the kind of claim a repository makes on the strength of
an argument parser.

## 6. What is still unverified

- `POSE_ESTIMATOR_MAE_DEG = 3.97` is taken from the 6DRepNet paper and is
  load-bearing for every verdict, because it sets how much the pose estimate is
  inflated before gating. Confirming it needs AFLW2000-3D, which requires
  registration. **This is the largest hole in the evaluation.**
- 6DRepNet's absolute yaw sign is unresolved. Channel identity is confirmed by a
  mirror test; the sign is not. It is not load-bearing today because the gate
  takes an absolute value.
- The fairness stratification has no power. FRLL has 13 black subjects against
  the roughly 60 per cell a real comparison needs, and 0 of 7 pairwise
  comparisons separate. Reported as not established rather than as no difference.
- Test-retest was measured across pose rather than at fixed pose and expression,
  because FRLL carries one neutral frontal capture per person.
