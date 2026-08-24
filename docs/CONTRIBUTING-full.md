# Contributing

## The shape of the thing

`core` and `measure` are the project. They import numpy and nothing heavier,
they hold every decision about what a measurement is and whether it may be
shown, and they run in milliseconds without a GPU or a weight file. Everything
else orchestrates them.

Adding a measurement is a data change. Write a `MeasurementSpec` in
`faciometry/measure/registry.py` and nothing else moves: not the pipeline, not the
report, not the uncertainty machinery. If adding a measurement makes you edit a
second file, the seam is in the wrong place and that is worth raising before
the change.

`docs/CORE_API.md` is the contract those packages expose. Read it before
touching anything outside them.

## Rules a patch has to keep

These are enforced by tests. Each one has a reason and the reason is written
down where it is enforced.

1. **No scalar aggregate.** No overall, harmony, attractiveness, average or
   rank field in any output. A single number is the part of this product class
   with no measurement basis behind it, and it is the documented harm vector.
   `tests/unit/test_no_aggregate_score.py` walks every renderer and every
   serialised field looking for one. The evidence is in `docs/REFERENCES.md`
   under "Why there is no score": Holland (2008) on the Marquardt mask,
   Hönekopp (2006) on the private share of rating variance, and Jacobs and
   Wallach (2021) on operationalising a construct that has no ground truth to
   validate against.
2. **No inferred demographics.** Sex and ancestry are user-declared, optional,
   and select a normative stratum. They are never model output.
3. **A withheld measurement is a result.** Print the reason, never the number.
4. **Every number carries its interval.** No bare point estimates in any
   renderer.
5. **No network egress during analysis.** `faciometry fetch-weights` is the only
   command that opens a socket.
6. `core`, `measure` and `norms` import numpy and nothing heavier.

## Setup

```
make install
make test
make lint
```

`make install` installs `permissive,api,dev` and then ad-hoc codesigns the
compiled extensions in the venv. On macOS an unsigned `.dylib` is deep scanned
by XProtect on first load, and a fresh install of opencv and mediapipe is
enough of them to stall the first import for minutes. Add the `pdf` extra if
you are touching `report/pdf.py`, because its tests exercise the real reportlab
output rather than a stub. `docs/INSTALL.md` has the platform notes and the
Docker route.

`make lint` runs ruff and mypy. Ruff currently reports a backlog of findings
that predate the public release, and CI runs it with `continue-on-error`. Fix
findings in the code you are already touching. Do not run `ruff --fix` across
the tree in a PR: a sweep like that has destroyed uncommitted work in this
repository once, and it makes a review impossible to read.

## One file, one owner

Work here is often split across several people or agents at once, and the
convention that keeps that from turning into a merge conflict is that a change
names the files it owns before it starts and touches nothing else. A patch that
had to edit a file outside its own set should say so, with the reason, rather
than editing it quietly. If two changes need the same file, the second one
waits.

That is also why an unrelated lint fix in a PR is unwelcome. It puts your name
on a line somebody else is mid-way through rewriting.

## A new model backend

Declare a `Provenance` in `faciometry/models/licensing.py` with its tier, its
license id, its source, and its `inherited_from`. That last field is where the
surprises live: a checkpoint's own tag frequently says nothing about the
obligations its training data or its morphable-model basis carry. Ultralytics
asserts AGPL-3.0 over models produced by its trainer, so a "yolov8-face"
checkpoint tagged MIT is a relabel.

Call `require(provenance, allowed)` before opening the weight file, not after.

Pin the artifact in `assets/weights.lock.json` with a sha256 you produced by
fetching the URL yourself. A mismatch at load time is a hard failure and is
never downgraded to a warning, and never resolved by editing the lock file.

## Numbers in docstrings

Every empirical claim in this codebase cites the study it came from, in the
docstring next to the constant it justifies. A patch that adds a threshold adds
the citation for it, or adds a note that the value is a guess and what would
replace it. A threshold with neither is the thing this project exists to avoid.

If a number came from an experiment in `evals/`, say which arm.

## Style

Match `core/scale.py` and `core/sensitivity.py`. Module docstrings explain why
the module exists and what the alternative would have got wrong. Prose in
source files uses `--` rather than an em dash, because the source is read in a
terminal as often as in a browser.

## Tests

`pytest` runs everything. The suite has no network access by design, and
`tests/integration/test_offline.py` proves the block is real before asserting
anything under it.

A test for a gate asserts both directions: that it refuses what it should, and
that it passes what it should. A gate that only ever refuses is
indistinguishable from a broken import.

Do not commit a photograph of an identifiable person, and do not attach one to
an issue or a pull request. Tests use synthetic faces from `evals/synth/`.
`docs/EVALUATION.md` covers the arms that need a real dataset and how they get
one.

## Releasing

Maintainer step. Update `CHANGELOG.md`, bump `version` in `pyproject.toml`,
then tag:

```
git tag v0.1.1 && git push --tags
```

`.github/workflows/release.yml` builds the sdist and wheel, checks that the tag
matches the version in `pyproject.toml`, runs `twine check`, publishes to PyPI
through trusted publishing, and creates the GitHub release. There is no API
token in the repository.
