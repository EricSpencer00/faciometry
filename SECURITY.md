# Security policy

## Report privately

Open a [private security advisory](https://github.com/EricSpencer00/vitruve/security/advisories/new)
rather than a public issue.

This matters more here than in most repositories. Vitruve processes
photographs of faces, so a reproduction case for a bug in the ingest, report or
API layers tends to be a picture of a real person. A public issue with an
attached repro publishes that picture. If you need to send an image to
demonstrate something, say so in the advisory and it will be handled through
the advisory's private attachments.

Include the version (`vitruve --version`), the platform, the license tier, and
the smallest input that shows the problem. If the input is an image, describe
it first and attach it only if asked.

You should get an acknowledgement within a week. This is a single-maintainer
project, so a fix takes as long as it takes, and you will be told which release
carries it.

## Supported versions

The latest release on PyPI. Fixes are not backported.

## What counts

Vitruve is a local instrument, and its threat model is shaped by that.

**In scope:**

- Anything that causes an analysis to reach the network. `vitruve fetch-weights`
  is the only command that should open a socket, and
  `tests/integration/test_offline.py` asserts it. A path that egresses during
  `analyze` or `serve` is a bug in the property this project is built on.
- Anything that writes an image, a report or an EXIF value to disk when the
  user did not ask for it. See `docs/PRIVACY.md` for the three commands that
  write files and the one internal branch that materialises pixels.
- EXIF leakage. GPS coordinates, serial numbers or timestamps surviving into a
  report, a manifest or a stored PNG.
- A license-tier escape: any route that loads a backend above the tier passed
  to `--license-tier`. `require()` is called before a weight file is opened,
  and getting past it is a defect with a legal consequence for the user.
- Weight verification bypass: anything that installs an artifact whose sha256
  does not match `assets/weights.lock.json`.
- Path traversal, deserialisation, decompression bombs or memory exhaustion via
  a crafted upload to `POST /analyze`.
- A crafted image that executes code through the decode path.

**Out of scope:**

- `vitruve serve` having no authentication. That is documented, it is why the
  server binds loopback, and it is why `--allow-remote` prints a warning before
  it binds. If you find a way to bind a non-loopback address without
  `--allow-remote`, that is in scope.
- Vulnerabilities in the model weights or their upstream projects. Report those
  upstream; open an issue here so the pin can be moved.
- The accuracy of a measurement. That is a correctness question and belongs in
  a normal issue, with the reasoning.

## What Vitruve does not do

No network egress during an analysis. No storage without a flag. No inference
of sex, ancestry, age or identity. No face matching, no face recognition, no
attractiveness score. Vitruve is not an identification tool, and FISWG's 2026
guidance (V2.1, section 6.4.1) prohibits photo-anthropometry for identification
on evidence this project's measurement gate is built from.
