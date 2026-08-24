## What this changes

<!-- One or two sentences. What is different after this merges. -->

## Why

<!-- What made this necessary. If it fixes an issue, "Fixes #123". -->

## Checklist

- [ ] `pytest` passes.
- [ ] `ruff check src tests` reports nothing new. (There is a known backlog; do
      not fix unrelated findings in this PR.)
- [ ] Any new empirical number cites its source next to the constant, or says
      it is a guess and what would replace it.
- [ ] Any new measurement is a `MeasurementSpec` in
      `vitruve/measure/registry.py` and nothing else moved.
- [ ] Any new model backend declares a `Provenance` in
      `vitruve/models/licensing.py`, including anything inherited from its
      training data, and calls `require()` before opening the weight file.
- [ ] No photograph of an identifiable person is attached to this PR or
      committed to the repository.

## Invariants

These are enforced by tests. If this PR changes one, say which and why, because
each of them exists for a documented reason.

- [ ] No scalar aggregate score, average, rank or harmony field in any output.
- [ ] No inferred demographics. Sex and ancestry stay user-declared and
      optional.
- [ ] A withheld measurement prints its reason and not its number.
- [ ] Every number carries its interval. No bare point estimates in a renderer.
- [ ] No network egress during an analysis. `fetch-weights` is the only command
      that opens a socket.
- [ ] `core`, `measure` and `norms` import numpy and nothing heavier.

## How this was checked

<!-- The commands you ran, and what they printed. If a number changed, say
     which evals arm produced the new one. -->
