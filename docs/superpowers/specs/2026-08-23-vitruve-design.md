# Vitruve — open-source facial morphometrics, design

Date: 2026-08-23
Status: draft (research pending on model/weight availability)

## 1. What this is

Vitruve is a locally-hosted, open-source system that takes standardised photographs
of a face and returns a **morphometric report**: a table of named anatomical
measurements, each with a unit, an uncertainty interval, a population percentile,
and the quality caveats that apply to it — plus a dermatological findings layer
(acne, erythema, periorbital hyperpigmentation, folds) detected with YOLO.

It is the open analogue of a Qoves-style facial analysis report, rebuilt so that
every number is traceable to a landmark, a formula, a reference distribution, and
a measured error bar.

### 1.1 What it deliberately does not do

Vitruve does not emit a scalar attractiveness score, rating, rank, or "harmony
index". This is a load-bearing architectural constraint, not a disclaimer:

- A single number is the part of this product class with no defensible
  measurement basis. Phi-ratio "beauty" claims do not survive contact with the
  anthropometric literature, and models trained on rating datasets mostly
  reproduce their raters' demographics.
- A scalar rating is also the documented harm vector (looksmaxxing communities,
  body dysmorphia). The measurements are the useful part; the ranking is not.
- Percentiles against a *stratified empirical* distribution carry strictly more
  information than a rating and are falsifiable.

This is enforced by `tests/unit/test_no_aggregate_score.py`, which asserts that no
public report field is a scalar aggregate over measurements. Removing that test is
a deliberate act, and the reason is written down here.

Vitruve also does not do identification, matching, or demographic inference as a
*product* surface. Demographic attributes are used only to select a normative
stratum, are computed on-device, are user-overridable, and are never the output.

## 2. Success criteria

1. `vitruve analyze front.jpg --profile side.jpg` produces a report in < 10 s on
   an M1 Max, entirely offline after the first weight download.
2. Every reported measurement carries: value, unit, 95% CI, normative percentile
   with CI, the landmark indices it derives from, and the formula id.
3. The pose gate rejects or downgrades images whose head pose exceeds
   empirically-derived per-measurement tolerances — thresholds that come from an
   experiment in `evals/`, not from a guess.
4. Test-retest coefficient of variation is measured per metric, and any metric
   above its threshold is marked non-reportable rather than printed.
5. Landmark and pose accuracy on public benchmarks match published numbers for
   the chosen backbones (positive control: the model actually loaded and ran).
6. Measurement error is reported stratified by demographic group; gaps are
   published, not hidden.
7. No network egress during analysis, asserted by a test.

## 3. Architecture

### 3.1 Shape

A linear pipeline of typed, independently testable stages. Every stage is a
`Protocol` with one method; every stage's output is a frozen dataclass; every
stage is content-addressed and cacheable.

```
Image bytes
  │
  ├─ 1. ingest      → SourceImage      (pixels, EXIF, sha256, view hint)
  ├─ 2. detect      → FaceBox[]        YOLO face detector + 5 keypoints
  ├─ 3. align       → AlignedFace      similarity xform → canonical 512²
  ├─ 4. landmark    → Landmarks2D      98/68-pt + per-point covariance
  ├─ 5. fit3d       → FaceGeometry3D   3DMM: shape, expr, pose, camera
  ├─ 6. gate        → QualityReport    pose / blur / exposure / occlusion / FOV
  ├─ 7. measure     → Measurement[]    declarative specs over canonical frame
  ├─ 8. normalize   → Percentile[]     vs stratified normative model
  ├─ 9. derm        → Finding[]        YOLO-seg over region crops
  └─ 10. report     → Report           overlays + tables + prose + run manifest
```

Frontal and profile images run stages 1-7 independently and are joined at stage 7
(some measurements are frontal-only, some profile-only, a few need both).

### 3.2 The measurement registry is the core

The centre of the system is not the models — it is a declarative registry of
`MeasurementSpec`s:

```python
@dataclass(frozen=True)
class MeasurementSpec:
    id: str                      # "gonial_angle_left"
    label: str                   # "Gonial angle (left)"
    view: View                   # FRONTAL | PROFILE | EITHER | BOTH
    dimension: Dimension         # ANGLE | RATIO | LENGTH_MM | LENGTH_PX
    landmarks: tuple[str, ...]   # anatomical names, not indices
    formula: Formula             # composable expression over named points
    frame: Frame                 # CANONICAL_3D | IMAGE_2D
    normative_key: str | None
    references: tuple[str, ...]  # literature the definition comes from
```

Consequences that fall out of this choice:

- Adding a metric is a data change, not a code change.
- Every metric can be unit-tested against synthetic geometry with a known answer.
- The report can print, for any number, exactly which points and which formula
  produced it. No unexplained numbers.
- Landmarks are referenced by *anatomical name* (`exocanthion_l`, `gonion_r`,
  `subnasale`), and each landmark backend supplies a name→index map. Swapping a
  68-pt for a 98-pt backend does not touch a single measurement definition.

`Formula` is a small closed expression algebra — `Distance`, `Angle3`,
`AngleToPlane`, `Ratio`, `Projection`, `Midpoint`, `Const`, `Div`, `Mul` — not
arbitrary Python. Closed so it can be serialised, diffed, hashed into the
provenance record, and evaluated under Monte-Carlo uncertainty propagation.

### 3.3 Geometry: 3D-first

Every angle-valued measurement is computed in a **canonical 3D frame** derived
from the fitted 3DMM, with the Frankfort horizontal as the reference plane and
scale recovered from interpupillary distance. Never from raw 2D projections.

The reason is the single largest error source in this product class: a 2D ratio
is a projection, and 10° of yaw measurably biases bizygomatic width, canthal
tilt, and every mandibular angle. Systems that measure on the raw photo are
reporting the photographer's pose as if it were the subject's anatomy.

Measurements that are genuinely 2D (e.g. some skin-region areas) are tagged
`frame=IMAGE_2D` and their uncertainty is inflated as a function of the estimated
pose. The tag is in the type, so this cannot be forgotten.

**Scale.** Pixel ratios are scale-free and reported directly. Millimetre values
require a reference; Vitruve uses interpupillary distance against a sex-stratified
population mean, and every `LENGTH_MM` value carries the resulting inflated CI.
An `--ipd-mm` flag lets a user supply a measured value and collapse that term.

**Perspective.** Camera-to-subject distance changes apparent proportions
substantially at close range. Vitruve estimates effective focal length from EXIF
where available and from the 3DMM camera fit otherwise, and the gate warns when
the implied subject distance is below the clinical protocol minimum.

### 3.4 Uncertainty is first-class

Landmark backends emit heatmaps; Vitruve converts each to a per-point 2×2
covariance. Uncertainty propagates through `Formula` evaluation by Monte Carlo
(N samples, seeded, N configurable), producing a distribution per measurement, not
a point. The report prints the median and a 95% interval.

This is what separates a real instrument from a facial-harmony calculator: a
number whose interval spans two population deciles is reported as such rather
than printed to three decimal places.

### 3.5 Normative model

A separately-versioned artifact (`vitruve-norms-<version>.parquet`) mapping
`(measurement_id, sex, age_band, ancestry_group) → (n, mean, sd, quantile grid)`.

- Built by a reproducible pipeline in `scripts/build_norms.py` over public face
  datasets with demographic labels, running the *same* measurement code path as
  inference. It is the empirical distribution of the pipeline's own output, so
  systematic bias in the pipeline cancels in the percentile.
- If the subject's stratum has `n` below threshold, Vitruve falls back to the
  pooled distribution, widens the interval, and says so in the report.
- Norms are data, downloaded and hash-pinned like weights. Users can build and
  point at their own — the whole point of stratification is that the reference
  population should be stated, not assumed.

### 3.6 Where YOLO fits, and where it does not

YOLO is used where the task is genuinely multi-instance detection:

- **Face detection + 5 coarse keypoints** (stage 2). Fast, robust to pose, gives
  the alignment seed.
- **Dermatological findings** (stage 9). Acne lesions, erythematous regions,
  periorbital hyperpigmentation, nasolabial and glabellar folds, enlarged pores.
  Detection and instance segmentation over region crops (T-zone, periorbital,
  malar, perioral) rather than the whole face, so small lesions survive
  downsampling. This is the classic small-object regime and it is exactly what
  YOLO-seg is for.

YOLO is *not* used for dense landmarks. Keypoint regression at 68-98 points with
calibrated per-point uncertainty is a different problem, better served by a
heatmap landmark network, and the uncertainty is not optional here (§3.4).

Backends sit behind `Detector`, `Landmarker`, `Fitter3D`, `DermDetector`
protocols. Concrete implementations are selected by config, so the license and
accuracy trade-offs are a deployment decision rather than a rewrite.

### 3.7 Deployment (local)

- `uv` project, Python 3.11, `pyproject.toml` with optional extras per backend.
- Device selection `mps > cuda > cpu`, chosen once and threaded through.
- Weights and norms fetched on first run from Hugging Face into
  `~/.cache/vitruve/`, each pinned by sha256 recorded in `assets/weights.lock.json`.
  A mismatch is a hard failure.
- `vitruve serve` → FastAPI on 127.0.0.1 with a local web UI (upload, capture
  guidance overlay, report view, export). Binds loopback only by default.
- `vitruve analyze` → CLI, JSON + HTML + PDF output.
- Every run writes `run.json`: git sha, package version, model ids and hashes,
  device, per-stage timings, seed, and the full provenance of each measurement.

### 3.8 Privacy

Images are processed on-device and are not persisted unless the user opts in.
EXIF (including GPS) is stripped from anything written to disk or embedded in a
report. An integration test runs the whole pipeline with outbound network
blocked and asserts success.

## 4. Validation — the full experimental design

This is stated in full before any of it runs, and all arms are run.

| # | Arm | What it establishes | Data |
|---|-----|--------------------|------|
| 1 | Landmark NME vs published | positive control: weights loaded and run correctly | 300W / WFLW test |
| 2 | Head-pose MAE vs published | positive control for the 3D fit | AFLW2000-3D |
| 3 | Synthetic-geometry measurement error | the measurement code's own error floor, independent of any model | rendered mesh, known GT |
| 4 | Yaw/pitch/roll sweep | per-measurement pose sensitivity → **derives the gate thresholds** | synthetic re-render of fixed identity |
| 4c | **Control for 4**: 0° repeated with re-encode | separates pipeline noise from pose effect | same |
| 5 | Camera-distance sweep | perspective sensitivity → distance warning threshold | synthetic |
| 6 | Test-retest repeatability | per-measurement CV → which metrics are reportable at all | multi-image-per-identity dataset |
| 7 | Fairness stratification | NME and CV gaps by ancestry / sex / skin type | FairFace, Fitzpatrick-labelled |
| 8 | **Negative control**: shuffled identities | percentile output must go uniform; if it does not, the normative model leaks | norms build set |
| 9 | Offline assertion | no egress during analysis | n/a |

Arms 1, 2, 6 and 7 depend on datasets that may require registration. Any arm
whose data cannot be obtained is reported as **not run**, with the reason, and is
not silently dropped. A number without its control is not reported as a result.

## 5. Module boundaries

| Module | Responsibility | Depends on |
|--------|---------------|------------|
| `core` | geometry primitives, landmark schema, formula algebra, types | numpy only |
| `measure` | measurement registry, evaluation, uncertainty propagation | `core` |
| `norms` | normative model load/query/build | `core`, pandas |
| `models` | backend protocols + concrete torch/YOLO implementations | torch |
| `pipeline` | stage orchestration, caching, quality gate | all of the above |
| `derm` | region crops, YOLO-seg findings, severity aggregation | `models`, `core` |
| `report` | overlays, HTML/PDF rendering, prose | `core`, no model deps |
| `api` / `cli` | transport | `pipeline`, `report` |

`core` and `measure` carry no torch dependency, so the entire measurement layer —
the part that produces the actual numbers — is testable in milliseconds without a
GPU or a model. That is the point of the seam.

## 6. What the research changed

The design above was drafted before the literature review. Six findings changed
it, and they are recorded here because each one overturned something the draft
took for granted.

### 6.1 The primary gate is discriminability, not accuracy

Kleinberg and Vanezis (2007) photographed subjects in ten-degree steps and
measured how far each facial index moved. At **ten degrees of yaw their indices
shifted by 8 to 19 percent, against a between-subject relative spread of 1.2
percent** for the tightest index. The pose artifact was larger than the entire
spread between different people. FISWG's 2026 guidance (V2.1, section 6.4.1)
prohibits photo-anthropometry for identification, citing that work directly.

So the question a measurement must answer is not "how accurate is it" but
"does it vary more between people than between photographs of one person". That
ratio is now computed for every measurement on every image, and it decides
reportability before anything else. `core/sensitivity.py` implements it;
`evals/` derives the sensitivity slopes empirically.

Consequence: on a careful handheld photograph, roughly two of forty-five
measurements report cleanly, most carry a caveat, and about nine are withheld.
That is the honest answer, and printing forty-five numbers instead would be the
bug.

### 6.2 The pose estimate is itself uncertain

Published head-pose estimators sit near 3.5 to 4 degrees mean absolute error on
AFLW2000-3D, with a label-noise floor around 2.5 to 3 degrees. A five-degree
gate is therefore at the edge of resolvability. Vitruve gates on
`|pose| + k * pose_sd` rather than on the point estimate, and runs two
independent pose sources so their disagreement is itself a quality signal.

### 6.3 Derived sensitivity models must not override measured variance

The first-order projection model only knows about rigid rotation. Kramer (2016)
decomposed the variance of the facial width-to-height ratio and found **posed
expression accounting for more of it than identity did** (eta-squared 0.58
against 0.31), with the rank ordering of individuals changing depending on which
photograph was used. Kramer et al. (2012) measured the same 66 men at 2.01 from
photographs and 1.83 from 3D scans, a gap larger than any published sex
difference in that measurement.

`MeasurementSpec.measured_within_person_rsd` now overrides the derived model
wherever a variance decomposition exists. fWHR is withheld as a result, which
is the correct outcome and the opposite of what the draft would have produced.

### 6.4 Scale: iris and interpupillary, fused, with the correlation stated

No permissively licensed metric-scale face model exists. MICA is the only
genuinely metric option and it is research-licence only, neutral-shape only,
and depends on FLAME 2020 and InsightFace weights. So every millimetre descends
from an assumption.

The ladder is: a ruler in frame (what Qoves itself asks customers for), then
iris diameter (11.84 mm, SD 0.79, adult-equivalent from age four), then
interpupillary distance (63.36 mm, SD 3.83 over 3,976 ANSUR adults, with a
further 4.7 percent downward correction at selfie distance for near fixation).
Both image-derived cues are read by the same landmark model from the same
image, so they are fused with an explicit correlation term rather than as
independent observations. Assuming independence would understate the fused
uncertainty by about 18 percent.

### 6.5 The normative model is caliper data, not photogrammetry

Farkas' international series is paywalled and not redistributable. The
substitute is better: the **NIOSH 2003 head-and-face survey** measured 3,997 US
respirator users with calipers across twenty facial dimensions and released
per-subject data as a US Government work in the public domain. It covers
bigonial and bizygomatic breadth, the two dimensions 2D photogrammetry cannot
recover, which is exactly where an independent spread is most needed.

Using a photogrammetric study as the discriminability numerator would have put
the same errors on both sides of the ratio. `scripts/build_niosh_norms.py`
derives the strata; ratios are computed per subject before aggregating, because
the numerator and denominator covary strongly within a person.

### 6.6 Licensing is a runtime constraint, not a README note

Ultralytics asserts AGPL-3.0 over models produced by its training code, not
only over the code. Every third-party "yolov8-face" checkpoint tagged MIT or
Apache-2.0 is a relabel that does not launder the obligation. InsightFace ships
MIT code with research-only weights that download automatically. FLAME and the
Basel Face Model are non-commercial and forbid redistribution, which several
popular repositories do anyway.

`models/licensing.py` therefore declares a `Provenance` per backend with an
ordered `Tier`, and refuses at load time to exceed the tier the user selected.
The default is `PERMISSIVE`: YuNet (MIT), SPIGA (BSD-3), MediaPipe (Apache-2.0,
including its bundled models), 6DRepNet (MIT). YOLO sits behind `COPYLEFT`,
where it does the one job it is genuinely best at: multi-instance lesion
detection on region crops.

### 6.7 Dermatology: measure what you cannot detect

Only acne has real box annotations, and only the Roboflow CC BY 4.0 sets are
commercially clean. Wrinkle masks exist but are non-commercial. **Dark circles,
erythema and pores have no usable public annotated data at all.**

Rather than pretend otherwise, those are measured colorimetrically as paired
within-face contrasts in CIELAB: erythema as a* elevation against a reference
region, periorbital hyperpigmentation as the infraorbital-to-malar L* and b*
difference. A paired contrast is robust to illumination and skin tone in a way
an absolute threshold is not. Skin tone is reported on the Monk scale, not
Fitzpatrick, which was designed to predict UV burn response in light skin and
compresses all dark skin into two bins.

### 6.8 What is faithful to Qoves, and what is not

Qoves' own laboratory pages state they are "firmly against the idea of
'scoring' beauty on a 1-10 scale". Their report identifies features, cites
reference ranges from Farkas, Powell, Ricketts and Naini, compares against
ancestry-matched norms, and prescribes interventions.

Vitruve keeps the measurements, the reference ranges as context, and the
user-declared ancestry stratum. It drops the interventions, the morph image,
and any comparative language implying a goal state. It adds the intervals, the
provenance, and the refusal to print numbers that cannot be distinguished from
photographic noise.
