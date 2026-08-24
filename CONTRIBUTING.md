# Contributing

## If you have never used GitHub

You do not need git to help. The two most useful things anyone can do:

- **Report what went wrong.** [Open an issue](https://github.com/EricSpencer00/vitruve/issues/new/choose).
  Say what you did, what happened, and what you expected. A photo is not needed
  and please do not attach one of a real person.
- **Fix a typo or a confusing sentence.** Click the pencil icon on any page in
  this repository, edit it in the browser, and press "Propose changes". GitHub
  handles the rest.

## Setting it up to change code

You need [git](https://git-scm.com/downloads), Python 3.11 or 3.12, and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

```
git clone https://github.com/EricSpencer00/vitruve
cd vitruve
make install
make test
```

`make test` should end with everything passing. If it does not, that is a bug
worth reporting on its own.

## Three rules that are not up for negotiation

**1. No score.** Nothing may combine measurements into a single number for a
face: no overall score, no harmony index, no rating. A test enforces this
across every output format. The reasoning is in
[docs/FINDINGS.md](docs/FINDINGS.md), and the short version is that there is no
ground truth for such a number and it is the part of this idea that does
documented harm.

**2. No number without its interval.** Every value carries the range it is
known to within. There is one formatter and it cannot print a bare number.

**3. Nothing prescriptive.** The report describes a measurement. It does not
tell anyone what to do about their face. Words like "ideal", "should be" and
"deviates from" are blocked by a test.

## Adding a measurement

Add a `MeasurementSpec` to `src/vitruve/measure/registry.py` and nowhere else.
It needs a formula over named anatomical landmarks, a citation, and an honest
evidence tier.

If the landmark you need does not exist, **do not approximate it with a nearby
one.** Leave the measurement out and say which landmark blocked it. Eight
measurements are currently absent for exactly this reason, each naming the
point it needs.

## Before you open a pull request

```
make test
make lint
```

Tests should assert the thing that matters, not an incidental. Two real
examples from this repository: one test pinned the catalogue size at exactly
45, so adding any measurement anywhere failed an unrelated test. Another
searched a whole document for the digits "987" and failed because a different
measurement's interval legitimately ended in `0.987`.

Please do not run a repository-wide `ruff --fix`. One destroyed uncommitted
work here already. Fix findings in small batches by rule.

## Reporting something sensitive

Do not open a public issue for a security or privacy problem. See
[SECURITY.md](SECURITY.md).

Longer version, including the file-ownership convention:
[docs/CONTRIBUTING-full.md](docs/CONTRIBUTING-full.md).
