# What this system actually establishes

Written after the build, from measured results rather than from the design.

## 1. On a single frontal photograph, with a 4%-NME landmark model, almost nothing is reportable

Running the shipped permissive stack (YuNet, SPIGA, MediaPipe, 6DRepNet) against
studio portraits from the Face Research Lab London set (CC BY 4.0, 1350x1350):

```
FRLL 001_03.jpg   0 of 68 reported, 35 withheld, 33 not attempted
FRLL 002_03.jpg   0 of 68 reported, 35 withheld, 33 not attempted
```

Re-measured against the catalogue at 68. At 45 the same two photographs gave
1 reported with a caveat and 0 reported. Of the 23 measurements added since,
12 cannot be attempted at all — they need a profile photograph or a landmark
this stack does not supply — and 11 are computed and then withheld. The one
measurement that used to clear the gate with a caveat no longer clears it.

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
- repeat photographs, since averaging N *independent* captures divides the
  landmark term by the square root of N. The word doing the work is
  independent, and §3 is what happened when it was assumed rather than
  measured.
- a real 3D fit, which unlocks the measurements over self-occluding surfaces

None of these is exotic. The system is not saying facial morphometry is
impossible; it is saying a single casual photograph does not carry it.

### The same conclusion arrived at from the opposite direction

The argument above propagates the landmark model's published error *into* a
measurement. A second route measures the between-person difference *directly*,
and lands in the same place harder.

Three different FRLL subjects were pooled as if they were repeat captures of
one face, to see whether the multi-capture scatter guard would notice. It did
not, and no threshold could have. After eye-normalisation the three faces'
landmarks differ by a median of **2.9 px** — 10.5 px at the worst point,
menton — against SPIGA's own claimed positional spread of about 10 px on a
215 px interocular span. Observed scatter 2.9 px against 17.1 px accounted for
by the model's own noise, a ratio of 0.17 against a limit of 2.0.

**The difference between two people is smaller than the model's uncertainty
about one.** A guard tight enough to reject three different faces would reject
honest repeat captures of the same face, because in this feature space they are
not distinguishable. Faciometry does not do face recognition and the scatter guard
is not one; the CLI help states the precondition instead of pretending to
enforce it.

That is section 1 restated in a unit that does not depend on any propagation
step. The first route says the error bar is the size of the signal. This one
says the signal between two *different people* is smaller than the error bar.

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

## 3. Five defects the build found in its own gate

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

*The fix has since been tested rather than trusted.* Sixteen measurements now
reference the interpupillary line or axis, and the catalogue makes two
different claims about them — eight are said to cancel roll exactly and eight
to keep their magnitude and lose a cosine. All sixteen hold: the exact eight
return between 1e-18 and 1e-15, which is floating-point zero, and the cosine
eight return 0.0015230, which is `(1 − cos 10°)/10` to five figures. Each is
swept a second time in the **discarded** horizon-referenced form as a positive
control, because "the slope is zero" is otherwise equally consistent with the
reference working and with the sweep having no roll in it; all sixteen twins
move, canthal tilt at exactly 1.000 degrees per degree and its asymmetry at
exactly 1.900.

What the fix cost is worth recording next to what it bought. On 102 real
photographs, `ocular_height_asymmetry` computed both ways agrees between two
independent landmark sources at r = 0.92 in the old horizon-referenced form and
r = 0.08 in the corrected one, and the between-subject spread falls by a factor
of 3.8. The old form's excellent agreement was agreement about the camera. With
the camera removed there is nothing left the two sources agree about, which
means the measurement is now correct and now demonstrably empty.

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

**Pooling repeated captures did not have to earn its reduction.** Feeding the
same photograph three times took a real report from 1 of 68 reportable to 3 of
68 and claimed 1.76 effective captures. A landmark model is deterministic, so
identical input gives identical points and identical covariances; the reduction
was a function of how many files were passed rather than of how much
independent information arrived. `multishot.combine` now estimates the shared
fraction from the captures' own scatter against the model's claimed noise, and
the estimate can only tighten the assumption, never loosen it. Three identical
copies now return exactly the single-capture result.

The limitation underneath it is the same one that stops arm 9 of the evaluation
measuring true repeatability: FRLL carries one neutral frontal capture per
person, so the tightest retest available is neutral against smiling, which
necessarily includes an expression change. There is no dataset here on which
end-to-end multi-capture pooling can be validated against genuinely independent
repeats of the same face at the same pose. The correction is a refusal to
assume independence, not a demonstration that the pooling works when
independence holds.

**The harness's own reproducibility claim was false in one arm.** `evals/`
advertises that a rerun reproduces byte-identical output. It did not: arm 7,
the negative control on the normative model, walked its strata by iterating a
Python `set` of strings, whose order depends on `PYTHONHASHSEED`, so the seeded
permutations came off the stream in a different sequence in every process. The
leak-free conclusion never depended on which permutation was drawn — a
within-stratum shuffle is the identity whatever the order — but the
across-stratum control numbers that were published were not reproducible, and
one of nine measurements has moved category now that they are. All 45 result
files now check out identical across consecutive runs, with wall-clock timings
split into their own file so the claim can be checked with a checksum rather
than qualified in prose.

## 4. The measurement layer itself contributes nothing, and everything else does

Arm 1 of the evaluation harness evaluates **all 68** catalogue measurements and
31 geometric primitives against independently computed ground truth, recomputed
in pure-Python `math` from each measurement's anatomical definition without
importing `core.geometry`, `core.formula` or `measure.registry`: worst absolute
error 2.8e-14, worst relative error 3.7e-15, **no failures**. Nothing is
outside its published reference range, and the Monte-Carlo wrapper reproduces
the deterministic value to 6e-6 at a landmark spread of 1e-4 mm. Whatever is
wrong with a Faciometry number, the arithmetic is not it.

The primitive suite grew from 20 cases to 31, because the second batch of
measurements uses nodes the first did not: a signed tilt read against a
landmark-derived axis, the same configuration rotated in its own plane, and —
as the positive control — the discarded horizon-referenced form on both sides
of the face, which must move one-for-one and in opposite directions. It does:
`θ − roll` on the left and `θ + roll` on the right. That anti-symmetry is the
whole mechanism of the roll defect in §3, and it is now pinned by a closed-form
test rather than only by a sweep.

**The exactness of the arithmetic is not a claim about the catalogue.** The
same run that confirms all 68 formulas also finds that 131 of 201 comparable
(measurement, axis) pairs disagree with the pose sensitivity declared for them
by more than a factor of two, that 32 of 68 measurements distort by more than
1% at the ICAO portrait distance, and that 20 of 68 have no working landmark
source at all. `docs/EVALUATION.md` §8 lists what those numbers contradict.

Three results from the re-run are worth stating here because they change what
the catalogue can claim.

**One structural property drives most of the damage: seven measurements are
signed quantities that sit near zero, and everything about them is expressed as
a fraction of themselves.** `commissure_height_r`'s value is exactly zero on
the reference face, so it has no relative pose slope, no relative noise floor
and no relative perspective distortion — three arms report it as "not
comparable" with the reason rather than as a number. Its left-hand twin moves
119 times its declared pitch sensitivity and −83% of itself at one metre of
camera distance, and arm 4 backs out an implied depth straddle of 841 mm, which
is not a distance any part of a face spans. These need an absolute error budget
before they need a corrected constant.

**Of the three pose-sensitivity constants that were derived on paper rather
than measured, one is exactly right and two are about 1.6× low.**
`_SAGITTAL_FRAME_ANGLE` predicts that pitch enters a frame-referenced sagittal
angle one for one; measured, it is 1.00000 per degree in every condition
including a pinhole camera at half a metre. `_MIDLINE_DEVIATION` reasons its
way to 0.5 per degree of yaw from the nasal dorsum's depth; measured, 0.79.
`_AP_PROJECTION` reasons its way to 0.08 per degree of pitch from "46 mm of
span against 10 mm of projection"; the span is 58 mm and the projection 8 mm,
so the answer is 0.126. Both derivations have the right mechanism and the wrong
arithmetic, which is the argument for measuring rather than deriving — and both
docstrings said so in advance.

**Faciometry's own geometry now exceeds the top of Kleinberg's band, which the
previous run said it never reached.** At 45 measurements the largest movement
at ten degrees of yaw was 9.9%, below Kleinberg and Vanezis's measured 8-19%
range. `ramus_body_ratio` runs from tragion to gnathion across 77 mm of depth
inside a single dimensionless quantity, and moves **21.1% orthographically and
30.0% at arm's length**. The conclusion that the project's geometry could not
reproduce the phenomenon it was built around was a property of which
measurements happened to be in the catalogue.

## 5. The permissive stack cannot measure a true profile

A full facial-analysis report is taken from a frontal and a profile photograph.
Faciometry's catalogue carries **20** profile measurements — 29% of it, up from 13
of 45, because six of the twenty-three most recent additions landed there — the
formulas are correct against synthetic geometry, and the pipeline accepts a
`--profile` argument. The models are what stop it.

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
confident number about nothing. What it costs is real: all 20 profile
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

- **`POSE_ESTIMATOR_MAE_DEG = 3.97` is still a literature value, but it is no
  longer unexamined, and it now looks far too large for the photographs it
  gates.** Confirming the published figure needs AFLW2000-3D and AFLW2000-3D
  needs registration — that has not changed. What has changed is that some of
  the estimator's error can be measured with no labels at all. Rotating a
  photograph in its own plane changes the true roll by exactly the angle
  applied; mirroring it negates the true yaw and roll. On the 102 FRLL front
  captures, with the pipeline's own YuNet box and 6DRepNet pose:

  - the estimator tracks a known in-plane rotation with a fitted slope of
    **−0.994** and a residual **mean absolute error of 0.177°**;
  - its roll error is bounded above at **0.699° of standard deviation** by
    comparison against the human-placed interpupillary line;
  - the mirror puts a floor of 0.33° on yaw, 0.27° on pitch and 0.23° on roll;
  - over a sweep in which the true yaw and pitch never change, its yaw wanders
    **1.90°** and its pitch **1.47°**.

  `gated_pose` adds **4.977°** to every axis before the discriminability gate.
  For this class of photograph that is at least four and plausibly ten times the
  estimator's actual scatter, and it is a single scalar across three axes whose
  measured errors differ by a factor of ten. The declared number is not refuted
  — it is an average over a pose range FRLL does not cover — but the gate does
  not apply it to AFLW2000-3D, it applies it to whatever photograph arrives.

- **6DRepNet's roll sign is now resolved; its yaw sign is not.** A known
  in-plane rotation moves the roll estimate with slope −0.994, and the
  human-placed interpupillary line agrees with that direction independently.
  MediaPipe's roll runs the other way, so the two estimators in this repository
  carry **opposite sign conventions on all three axes** (r = −0.97, −0.94,
  −0.92). Nothing breaks today because the gate takes an absolute value, but
  `HeadPose.disagreement` between two anti-aligned estimates is twice the pose,
  not the disagreement. A signed consumer must resolve yaw against a labelled
  benchmark first.
- **Two estimators disagree by 6.9° of pitch on a studio frontal portrait**,
  on average, against a pitch distribution whose whole standard deviation is
  4.3°. Almost all of it is a systematic gain and offset — the residual after
  fitting a slope is 1.6° — but neither can be taken at face value.
- **`measured_within_person_rsd` is populated for 2 of 68 measurements.** Arm 9
  measures it for 46 of them. Two is the state of the art in this repository
  and that is itself the finding.
- The fairness stratification has no power. FRLL has 13 black subjects against
  the roughly 60 per cell a real comparison needs, and 0 of 7 pairwise
  comparisons separate. Reported as not established rather than as no difference.
- Test-retest was measured across pose rather than at fixed pose and expression,
  because FRLL carries one neutral frontal capture per person. The same gap is
  why multi-capture pooling in §3 could be corrected but not validated: there is
  no set here of genuinely independent repeat captures of one face at one pose.
- **ANSUR II's units are confirmed only by consistency.** No copy is vendored
  and the harness does not fetch one, so "interpupillary breadth is recorded in
  tenths of a millimetre" rests on agreement with NIOSH's 64.37 mm and
  Dodgson's 63.36 mm rather than on reading the file. Getting it wrong would
  put every length in the report off by an order of magnitude and leave every
  ratio untouched, so the error would be invisible in exactly half the output.
