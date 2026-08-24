# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-24

First public release.

### Added

- A catalogue of craniofacial measurements, each declared as a
  `MeasurementSpec` with its formula, its landmarks, its evidence tier, its
  pose tolerance and its published between-subject spread. Adding a measurement
  touches one file.
- Monte-Carlo evaluation over per-landmark covariances, so every reported value
  carries a 95% interval rather than a point estimate.
- The reportability gate. A measurement whose own error exceeds the spread of
  that measurement between different people is withheld, and the report prints
  the reason where the number would have gone. The rule follows Kleinberg and
  Vanezis (2007) and FISWG 2026 V2.1 section 6.4.1.
- License tiers enforced at load time. `require(provenance, allowed)` runs
  before a weight file is opened and raises `LicenseViolation`, which the CLI
  reports as exit code 4. `vitruve licenses --tier T` prints the full obligation
  page for a tier, including obligations inherited from training data rather
  than from a code license.
- The permissive backend stack: YuNet (MIT), SPIGA (BSD-3-Clause), MediaPipe
  Face Landmarker (Apache-2.0) and 6DRepNet (MIT).
- Scale recovery from iris diameter, interpupillary distance, or a ruler passed
  with `--ruler-mm`. Without a ruler, millimetre values carry the population
  prior's 5.5% at one standard deviation and say so.
- `vitruve analyze`, `catalogue`, `licenses`, `fetch-weights`, `doctor` and
  `serve`.
- `vitruve fetch-weights` verifies every artifact against the sha256 pinned in
  `assets/weights.lock.json` and fails on a mismatch. It is the only command
  that opens a socket.
- `vitruve serve`: a local HTTP API and a single-page web UI with a capture
  overlay and the report view. It binds `127.0.0.1:8731` and refuses a
  non-loopback bind without `--allow-remote`.
- Reports in JSON, plain text, and one self-contained HTML file with annotated
  overlays and uncertainty ellipses. `--pdf` adds a typeset `report.pdf` and
  needs `pip install 'vitruve[pdf]'`.
- EXIF is dropped at ingest. Four camera values are kept for the perspective
  warning, and the run manifest records the names of the other tags without
  their values.
- `tests/integration/test_offline.py`, which replaces `socket.socket`,
  `socket.create_connection` and `socket.getaddrinfo` with functions that raise
  and runs the measurement path underneath. The first test in the file is the
  control that proves the block blocks.
- `tests/unit/test_no_aggregate_score.py`, which asserts that no renderer emits
  an overall, harmony, attractiveness, average or rank field.
- A Docker image built on `python:3.11-slim`, running as a non-root user, with
  weights fetched at runtime into a mounted volume.
- A macOS app, built by `packaging/macos/build_app.sh`, signed with a Developer
  ID and not notarised. Each build writes a `.build.txt` beside the dmg
  recording that artifact's signed and notarised state.
- Documentation: `docs/CORE_API.md` (the contract), `docs/FINDINGS.md` (what
  the build measured about itself), `docs/PRIVACY.md` (each claim and where to
  check it), `docs/REFERENCES.md` (a source for every empirical constant),
  `docs/EVALUATION.md` and `docs/INSTALL.md`.

[Unreleased]: https://github.com/EricSpencer00/vitruve/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/EricSpencer00/vitruve/releases/tag/v0.1.0
