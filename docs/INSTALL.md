# Installing Faciometry

Pick the row that matches what you want to do.

| You want to | Use |
|---|---|
| read the catalogue, no photographs | `pip install faciometry` |
| measure faces on this machine | `pip install 'faciometry[permissive,api]'` |
| a Mac app, no terminal | [INSTALL-MACOS.md](INSTALL-MACOS.md) (signed, not notarised: first launch is right-click, Open) |
| a container | [Docker](#docker) |
| to work on Faciometry | [From source](#from-source) |

Python 3.11 or 3.12. macOS on Apple silicon and Linux on x86_64 are the two
platforms exercised in CI.

## The measurement layer alone

```
pip install faciometry
faciometry catalogue
faciometry licenses --tier copyleft
```

The base install is numpy and Pillow. `core`, `measure` and `norms` hold every
decision about what a measurement is and whether it may be shown, and they run
in milliseconds with no GPU and no weight file. `faciometry catalogue` and
`faciometry licenses` work from this install and open no sockets.

## The full local pipeline

```
pip install 'faciometry[permissive,api]'
faciometry fetch-weights
faciometry doctor
```

`permissive` brings OpenCV, MediaPipe, torch, torchvision, SPIGA and 6DRepNet.
`api` brings FastAPI and Uvicorn for `faciometry serve`. Both are permissively
licensed and neither changes the license of your deployment. `faciometry
licenses --tier permissive` prints the exact obligations.

`fetch-weights` downloads about 416 MB into `~/.cache/faciometry/weights`, or into
`$FACIOMETRY_CACHE_DIR` if you set it. It verifies every artifact against the
sha256 pinned in `assets/weights.lock.json` and fails hard on a mismatch. It is
the only command that opens a socket.

`faciometry doctor` prints the device, which weights are present and which
versions are installed. Run it before opening an issue.

### Extras

| Extra | What it adds | Effect on your license |
|---|---|---|
| `permissive` | the default backend stack | none, stays Apache-2.0 |
| `api` | FastAPI and Uvicorn, for `faciometry serve` | none |
| `pdf` | ReportLab, for `report.pdf` | none |
| `all` | the three above together | none |
| `copyleft` | Ultralytics, for YOLO detection and YOLO-seg findings | your deployment becomes AGPL-3.0 |
| `dev` | pytest, ruff, mypy, build, twine | not installed at runtime |

`all` is deliberately not `copyleft`. AGPL section 13 treats network use as
distribution, so an extra named `all` that quietly pulled it in would make a
`faciometry serve` instance oblige you to release your corresponding source
without your having chosen that. Ask for it by name:

```
pip install 'faciometry[permissive,api,copyleft]'
faciometry analyze portrait.jpg --license-tier copyleft
```

Both halves are required. Installing the extra does not raise the runtime tier,
and raising the runtime tier does not install the extra. `require(provenance,
allowed)` runs before any weight file is opened and raises `LicenseViolation`,
which the CLI turns into exit code 4.

## macOS

```
git clone https://github.com/EricSpencer00/faciometry
cd faciometry
make install
```

`make install` creates the venv with `uv`, installs `permissive,api,dev`, and
then ad-hoc codesigns every compiled extension in the venv:

```
find .venv \( -name "*.so" -o -name "*.dylib" \) -print0 \
  | xargs -0 -P 8 -n 20 codesign -s - -f
```

XProtect deep scans an unsigned `.dylib` the first time it is loaded. A fresh
install of opencv, mediapipe, torch and numpy contains enough of them that the
first `import` stalls for minutes with no output and no indication that
anything is happening. Signing them at install time pays that cost once, in a
place where you can see it.

The `mediapipe>=0.10.31,<1` upper bound is load-bearing on Apple silicon.
MediaPipe 1.0.x aborts the process when the face landmarker graph opens, with
`Check failed: service_ Service is unavailable` from inside a Metal helper
initialised for a CPU graph. It is a hard abort rather than an exception, so it
cannot be caught at the call site, and `faciometry.models.dense_mediapipe` refuses
to load that version.

`faciometry doctor` reports `mps` when Metal is available. That is where the
analysis runs.

## Linux

```
sudo apt-get install -y libgl1 libglib2.0-0
pip install 'faciometry[permissive,api]'
faciometry fetch-weights
```

`opencv-python` links against libGL at import time. Without those two packages
`import cv2` fails with an `ImportError` about `libGL.so.1` that does not name
the missing apt package.

x86_64 only. MediaPipe publishes a Linux wheel for x86_64 and for no other
Linux architecture.

## Docker

```
docker compose run --rm fetch-weights
docker compose up
```

Then open <http://127.0.0.1:8731>.

Or without compose:

```
docker build -t faciometry .
docker volume create faciometry-weights
docker run --rm -v faciometry-weights:/weights faciometry fetch-weights
docker run --rm -v faciometry-weights:/weights \
  -p 127.0.0.1:8731:8731 faciometry serve --host 0.0.0.0 --allow-remote
```

**The container is CPU-only.** There is no MPS and no CUDA inside it, so an
analysis takes noticeably longer than the same analysis from a `make install`
on an Apple silicon host. If you are measuring faces on your own laptop, the
native install is the faster route. The container is for putting Faciometry
somewhere reproducible.

The image is `linux/amd64`, for the MediaPipe wheel reason above. On Apple
silicon it runs under Rosetta, which costs more speed on top of the CPU-only
cost.

**Weights are not baked into the image.** `fetch-weights` writes them into the
`faciometry-weights` volume at runtime. The image stays small, the SPIGA
checkpoint is not redistributed inside a layer that ends up on a registry, and
every artifact is hash-checked on arrival. A container started before the
volume is populated serves `/health` with `pipeline_available: false` and
returns 503 from `/analyze`.

**Only the permissive extra is installed.** Adding `copyleft` to the Dockerfile
would make every image built from it AGPL-3.0, and AGPL section 13 treats
serving it over a network as distribution. Build a derived image if you want
that tier.

### About the 0.0.0.0 bind

The container command is `serve --host 0.0.0.0 --allow-remote`, which is the
combination the README tells you to think twice about. Inside a container the
two flags mean something narrower than they do on a laptop.

`0.0.0.0` inside the container means the container's own network namespace, not
your machine's interfaces. Nothing reaches it except through a port you
published, and a published port is a separate decision you make in the run
command. `-p 127.0.0.1:8731:8731`, which is what `docker-compose.yml` uses,
puts the service on your loopback and nowhere else. The short form `-p
8731:8731` publishes it on every interface the host has, and Docker's port
publishing bypasses the host firewall on Linux, so that form puts a service
that reads faces and has no authentication on your network.

`--allow-remote` is still required and it still prints its warning. Keeping it
required inside the container is the point: the flag is what makes the decision
appear in the run command instead of in a default.

## From source

```
git clone https://github.com/EricSpencer00/faciometry
cd faciometry
make install
make test
make lint
```

Or with pip:

```
python -m venv .venv && source .venv/bin/activate
pip install -e '.[permissive,api,dev]'
pytest
```

[CONTRIBUTING.md](../CONTRIBUTING.md) has the invariants a patch has to keep.

## Verifying an install

```
faciometry doctor
```

That prints the device, the weights present, the versions, and the catalogue
size. Two checks worth running once:

```
# the measurement layer, with no weights and no network
faciometry catalogue --id nose_breadth

# the obligation page for a tier you are not using, which needs nothing installed
faciometry licenses --tier noncommercial
```

## Uninstalling

```
pip uninstall faciometry
rm -rf ~/.cache/faciometry
```

Reports and any stored images are ordinary files in directories you named.
`docs/PRIVACY.md` lists every path Faciometry writes to.
