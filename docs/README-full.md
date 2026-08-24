# Vitruve

[![CI](https://github.com/EricSpencer00/vitruve/actions/workflows/ci.yml/badge.svg)](https://github.com/EricSpencer00/vitruve/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/vitruve.svg)](https://pypi.org/project/vitruve/)
[![Python](https://img.shields.io/pypi/pyversions/vitruve.svg)](https://pypi.org/project/vitruve/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Vitruve measures a face from a photograph and reports each measurement with the
interval it is known to within. It runs on your machine and downloads nothing
during an analysis.

```
pip install 'vitruve[permissive]' && vitruve fetch-weights
vitruve analyze portrait.jpg --out report/
```

That writes `report.json`, `report.txt` and `report.html`, and `report.pdf` if
you add `--pdf`. Every value carries a 95% interval, the landmarks it came
from, and the hash of the formula that produced it.

![The capture and subject plates of the local web UI](https://raw.githubusercontent.com/EricSpencer00/vitruve/main/docs/images/capture.png)

## Two things to know before you install it

**A measurement whose error is the size of the thing it measures is withheld,
with its reason printed where the number would have gone.** Vitruve compares
each measurement's own error against how far that measurement spreads between
different people. When the photograph contributes more variance than the person
does, the report says so instead of printing a value.

**There is no attractiveness score, no harmony index, and no average over the
measurements.** A single number is the part of this product class with no
measurement basis behind it, and it is the documented harm vector.
`tests/unit/test_no_aggregate_score.py` asserts the absence across every
renderer.

On studio portraits from the Face Research Lab London set, at 1350x1350 with
the shipped landmark model, 0 to 1 of the 45 measurements evaluated cleared
the gate. That is the system working. The arithmetic is four lines, in
[docs/FINDINGS.md](docs/FINDINGS.md):

- SPIGA's published WFLW normalised mean error is 4.06% of the inter-ocular distance.
- The inter-ocular span on that image is 215 px, so mean landmark error is about 8.7 px.
- Intercanthal width spans about 110 px, and two independent endpoint errors put roughly 11% of relative error on it.
- NIOSH measured the between-person spread of comparable dimensions at 6 to 13%.

The measurement error is the same size as the quantity, so the number describes
the photograph rather than the person.

![The report, showing one reported measurement and the reasons behind the withheld ones](https://raw.githubusercontent.com/EricSpencer00/vitruve/main/docs/images/verdicts.png)

## What makes more measurements clear the gate

Four inputs move the result, and their effect sizes are known.

**Put a ruler in the plane of the face.** Millimetre values otherwise descend
from a population prior on interpupillary distance carrying about 5.5% error at
one standard deviation. Pass `--ruler-mm` with a measured length and that term
collapses to your reading error. In a synthetic clinical capture this was worth
about seven additional clean measurements.

**Use more pixels on the face.** The landmark error term enters linearly, so it
halves when the resolution on the face doubles.

**Take the photograph again.** Averaging N captures divides the landmark term by
the square root of N.

**Use a lower-NME landmark model.** The landmark model's normalised mean error
also enters linearly, and Vitruve's backend interface takes any model that
returns named points with covariances.

## Where the gate comes from

Kleinberg and Vanezis (2007) photographed subjects in ten-degree steps and
measured how far each facial index moved. At ten degrees of yaw the indices
shifted by 8 to 19 percent, against a between-subject relative standard
deviation of 1.2 percent for the tightest index. The pose artifact was larger
than the entire spread between different people.

FISWG's 2026 guidance (V2.1, section 6.4.1) prohibits photo-anthropometry for
identification, citing that evidence. Vitruve keeps the measurements and gates
them. Every empirical constant in the codebase cites its source next to the
constant, and the bibliography is [docs/REFERENCES.md](docs/REFERENCES.md).

`vitruve catalogue` prints the whole table with no photograph and no weights:

![The catalogue, with pose sensitivity and between-person spread per measurement](https://raw.githubusercontent.com/EricSpencer00/vitruve/main/docs/images/catalogue.png)

`moves @10deg` is how far a measurement shifts when the head turns ten degrees.
`between people` is how far it spreads across different people. The ratio of the
two decides whether the number describes a person or describes a photograph.
Below 1, Vitruve prints the reason and not the number:

```
Bizygomatic width: withheld
    endpoints lie on a self-occluding lateral surface; 2D photogrammetry does
    not reproduce this measurement (Lim et al. 2022: bigonial breadth mean
    difference 9.3 mm, limits of agreement -0.9 to 19.6 mm)
```

## How this compares to a consumer face report

Commercial facial-analysis services, Qoves being the best known of them, return
an aesthetic assessment: per-feature ratings and an overall figure, derived from
the same class of 2D landmark geometry. Vitruve computes the geometry, attaches
the interval, and stops there. The measurements a consumer report prints to two
decimal places, bizygomatic width and facial width-to-height ratio and canthal
tilt among them, are the ones whose pose error exceeds their between-person
spread, so those are exactly the rows Vitruve withholds.

## Download for macOS

There is a Mac app, so none of the above is required. `packaging/macos/build_app.sh`
produces `Vitruve-<version>-arm64.dmg`, about 331 MB, which installs a 1.2 GB
app. Drag it to Applications and open it; the web interface appears in your
browser.

**The app is signed with a Developer ID and it is not notarised.** Gatekeeper
reports `rejected, source=Unnotarized Developer ID`, so the first launch needs
a right-click and Open rather than a double-click. Every build writes a
`.build.txt` beside the dmg recording that artifact's signed and notarised
state; read it before you install a copy you did not build.

On first launch the app downloads about 416 MB of model weights instead of
bundling them, verifying each file against the sha256 in
`assets/weights.lock.json` and showing a progress bar. That needs a network
connection once. After it, the app runs offline.

[docs/INSTALL-MACOS.md](docs/INSTALL-MACOS.md) has the install steps and the
release procedure.

## Install

```
pip install 'vitruve[permissive,api]'
vitruve fetch-weights
vitruve doctor
```

`fetch-weights` is the only command that opens a socket. It verifies every
artifact against the sha256 pinned in `assets/weights.lock.json` and fails hard
on a mismatch.

On macOS, `make install` does the same through `uv` and then ad-hoc codesigns
the compiled extensions in the venv, which stops XProtect from stalling the
first import for minutes.

In a container:

```
docker compose run --rm fetch-weights
docker compose up
```

That serves <http://127.0.0.1:8731>. The image is CPU-only, so it has no MPS
and no CUDA and an analysis takes longer inside it than on the host. The
weights live in a named volume rather than in a layer. Source builds, the
Linux packages OpenCV needs, and the reasoning behind the container's bind
address are in [docs/INSTALL.md](docs/INSTALL.md).

## Commands

| Command | What it does |
|---|---|
| `vitruve analyze FRONTAL [FRONTAL ...]` | measure a face. Offline. Several captures of the same person in one session are pooled before measuring. |
| `vitruve catalogue` | every measurement, its evidence tier, its pose sensitivity and its between-subject spread |
| `vitruve catalogue --id gonial_angle_l` | everything known about one measurement, including its formula hash |
| `vitruve licenses --tier T` | what tier `T` obliges you to |
| `vitruve fetch-weights` | download and hash-verify weights. The only networked command. |
| `vitruve doctor` | device, weights, versions |
| `vitruve serve` | the local API and web UI |

Exit codes: `0` ran, `2` bad input, `3` the photograph did not clear the quality
gate, `4` a backend exceeded the permitted license tier.

## Licensing

Vitruve's own code is Apache-2.0. The weights are where the obligations live.

The default stack is permissive throughout: YuNet (MIT), SPIGA (BSD-3-Clause),
MediaPipe Face Landmarker (Apache-2.0, including its bundled `.task` models) and
6DRepNet (MIT). Installing `vitruve[permissive]` leaves your deployment
Apache-2.0.

YOLO sits behind an opt-in tier. **Ultralytics asserts AGPL-3.0 over the models
its training code produces, not only over its code.** Every third-party
"yolov8-face" checkpoint tagged MIT or Apache-2.0 was trained from Ultralytics
weights with the Ultralytics trainer, so the permissive tag is a relabel that
does not launder the upstream obligation. Two other traps recur in this field:
InsightFace ships MIT code with research-only pretrained models and downloads
them on first use, and FLAME and the Basel Face Model are non-commercial and
forbid redistribution, which several popular repositories breach by vendoring
the basis file.

So Vitruve treats the license as a property of the backend, declared in the
type, and refuses at load time to exceed the tier you selected.

| Tier | Stack | What it does to your deployment |
|---|---|---|
| `permissive` | YuNet, SPIGA, MediaPipe Face Landmarker, 6DRepNet | stays Apache-2.0 |
| `copyleft` | adds YOLO face detection and YOLO-seg dermatological findings | becomes AGPL-3.0, and AGPL section 13 treats network use as distribution |
| `noncommercial` | adds 3DDFA_V2, MICA, SegFormer face parsing | research and personal use only, weights not redistributable |

The permissive tier is the default and it is a complete pipeline. Read the full
obligations, including the ones inherited from training data rather than from a
code license, with:

```
vitruve licenses --tier copyleft
```

`vitruve.models.licensing` enforces this. `require(provenance, allowed)` is
called before a weight file is opened, and exceeding the tier raises
`LicenseViolation`, which the CLI turns into exit code 4.

## The local server

```
vitruve serve
```

Binds `127.0.0.1:8731` and serves a single-page UI with a capture overlay and
the report view. `POST /analyze` takes multipart, `GET /catalogue` returns the
same rows the CLI prints, `GET /licenses?tier=T` returns the obligation page,
`GET /health` reports the device and whether the weights are present.

Uploads are held in memory. Passing `--store` writes them, re-encoded from the
pixel array so they carry no metadata, and prints that it is doing so.

Binding anything other than loopback requires `--allow-remote` and prints why
that is a decision. Vitruve has no authentication and it reads faces.

## Privacy

Pixels are processed, EXIF including GPS is dropped at ingest, nothing is stored
unless you asked, and nothing leaves the machine during an analysis.
[docs/PRIVACY.md](docs/PRIVACY.md) states each claim and where to check it.
`tests/integration/test_offline.py` replaces `socket.socket`,
`socket.create_connection` and `socket.getaddrinfo` with functions that raise,
then runs the measurement path underneath. The first test in that file is the
control that proves the block blocks.

Sex and ancestry are declared by the subject or left empty. They select a
normative stratum and narrow the interpupillary prior. Nothing in Vitruve
predicts either one.

## Layout

| Module | Responsibility | Depends on |
|---|---|---|
| `core` | geometry, landmark schema, formula algebra, evidence tiers, the gate | numpy |
| `measure` | the measurement registry and Monte-Carlo evaluation | `core` |
| `norms` | normative distributions | `core` |
| `models` | backend protocols and license enforcement | torch, ONNX |
| `pipeline` | stage orchestration, quality, run manifest | the above |
| `report` | overlays, prose, one self-contained HTML file | `core`, `measure` |
| `cli`, `api` | transport | `pipeline`, `report` |

`core` and `measure` import numpy and nothing heavier, so the layer that
produces the numbers is testable in milliseconds with no GPU and no weights.
The contract they expose is [docs/CORE_API.md](docs/CORE_API.md).

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) has the dev setup and the invariants a patch
has to keep. Security reports go through [SECURITY.md](SECURITY.md) rather than
a public issue, because a report about this codebase may carry a photograph of
someone.

## License

Apache-2.0. See [LICENSE](LICENSE), and `vitruve licenses` for the weights.
