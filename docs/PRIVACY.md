# Privacy

Vitruve reads photographs of faces. This document says exactly what it does
with them, in enough detail that you can check each claim against the code
rather than trust it.

## What is processed

Pixels. An image is decoded to an RGB array, its orientation is normalised
from the EXIF orientation tag, and everything after that stage sees an array
and a handful of numbers about the camera.

Four EXIF values are read and kept, because the perspective warning needs
them:

| Value | Used for |
|---|---|
| `FocalLength` | camera identification in the run manifest |
| `FocalLengthIn35mmFilm` | estimating camera-to-subject distance |
| `SubjectDistance` | the same, when the camera recorded it |
| `Orientation` | rotating a phone photograph upright before detection |

The run manifest records the *names* of the other tags the file carried and
never their values. That is deliberate: a manifest that listed GPS coordinates
would defeat the point of dropping them.

## What is stored

Nothing, by default. Three commands write files, and each one is asked for.

**`vitruve analyze --out DIR`** writes `report.json`, `report.txt` and
`report.html` into `DIR`. The JSON and the text carry measurements, intervals,
reasons and the run manifest. The HTML embeds the annotated overlays, which
are crops of the face with the landmarks and their uncertainty ellipses drawn
on them. An HTML report is therefore a picture of the subject. The source
photographs are not copied into `DIR` in any format.

**`vitruve serve --store`** writes each uploaded image into `vitruve-store/`
as a PNG named by the first 16 hex digits of its sha256. The PNG is re-encoded
from the decoded pixel array rather than copied from the upload, so it carries
no metadata at all. Without `--store`, no upload reaches disk.

**`vitruve fetch-weights`** writes model files into
`~/.cache/vitruve/weights`, or under `$VITRUVE_CACHE_DIR` if you set it.
Weights only.

An HTTP upload is held in memory for the length of the request. Starlette
spools any multipart file over a megabyte to a temporary file by default, so
`create_app` raises that threshold above the upload ceiling; the test
`test_a_large_upload_does_not_spool_to_a_temporary_file` asserts it against a
part large enough to have triggered the old default.

One exception, and it is visible in the code as `runner._materialised`. The
analysis pipeline currently opens image files by path. When an image arrived
over HTTP there is no path, so the decoded pixels are re-encoded to PNG and
written into a private directory created with mode 0700, passed to the
pipeline, and unlinked in a `finally`. What is written is the metadata-free
array, never the upload. The CLI never takes this branch, because its images
are already files. When the pipeline grows an entry point that accepts arrays,
this branch stops being reachable.

## What is never inferred

Sex and ancestry are declared by the subject or left empty. They select a
normative stratum and narrow the interpupillary prior. No model in Vitruve
predicts either one, and an undeclared subject gets the pooled distribution
and a wider interval.

Vitruve does not do identification or matching. FISWG's 2026 guidance (V2.1,
section 6.4.1) prohibits photo-anthropometry for identification, and the
measurement gate here is built on the same evidence.

There is no attractiveness score, no harmony index, and no average over the
measurements anywhere in any output.

## What leaves the machine

Nothing, during an analysis.

`vitruve fetch-weights` is the only command that opens a socket. It downloads
the artifacts pinned in `assets/weights.lock.json`, verifies each against its
sha256, and fails hard on a mismatch.

`tests/integration/test_offline.py` replaces `socket.socket`,
`socket.create_connection` and `socket.getaddrinfo` with functions that raise,
and runs the measurement path underneath. The first test in that file is the
control that proves the block blocks.

The API binds `127.0.0.1` and refuses any other address unless you pass
`--allow-remote`, which prints what it means before it binds. The web UI loads
no fonts, no scripts and no stylesheets from any other origin; a CDN request
would tell somebody else that a face-analysis tool had been opened.

## Deleting things

Reports and stored images are ordinary files in directories you named. Weights
live in `~/.cache/vitruve/weights` and `rm -rf` on that costs a re-download
and nothing else.
