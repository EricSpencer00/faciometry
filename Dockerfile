# Vitruve, containerised.
#
# Two decisions are worth reading before you edit this file.
#
# 1. The weights are not baked in. `vitruve fetch-weights` writes about 416 MB
#    into $VITRUVE_CACHE_DIR, which is a volume, so the image stays small and
#    the SPIGA checkpoint is not redistributed inside a layer that ends up on a
#    registry. It also keeps the image's provenance honest: every artifact is
#    fetched at runtime and hash-checked against assets/weights.lock.json.
#
# 2. Only the permissive extra is installed. Adding `copyleft` here would make
#    every image built from this file AGPL-3.0, and AGPL section 13 treats
#    serving it over a network as distribution. If you want that tier, build a
#    derived image and accept the obligation explicitly.
#
# This is a CPU-only image. There is no MPS and no CUDA inside it, so an
# analysis takes noticeably longer than the same analysis run from `make
# install` on the host.
#
# The platform is pinned to linux/amd64. MediaPipe publishes a Linux wheel for
# x86_64 and for no other Linux architecture, so an arm64 build of this image
# cannot install the permissive stack at all. On Apple silicon this runs under
# Rosetta, which works and costs some more speed on top of the CPU-only cost.

# ---------------------------------------------------------------- build stage
FROM --platform=linux/amd64 python:3.11-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# hatchling reads the README for the long description and the LICENSE for the
# license field, so both have to be present before the wheel is built.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY web ./web
COPY assets ./assets

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# torch is installed first, and from PyTorch's CPU index rather than from PyPI.
# The PyPI wheel drags in the CUDA runtime, cuDNN and cuSPARSELt, which is
# about 2.5 GB of NVIDIA libraries that this image can never use: there is no
# GPU inside it. Installing the +cpu build first satisfies `torch>=2.2` for the
# resolver, so the vitruve install below does not go back to PyPI for it.
RUN pip install --upgrade pip build \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && python -m build --wheel --outdir /build/dist \
 && pip install "$(ls /build/dist/*.whl)[permissive,api]" \
 && python -c "import torch; assert '+cpu' in torch.__version__, torch.__version__; print(torch.__version__)" \
 && find /opt/venv -name "__pycache__" -type d -prune -exec rm -rf {} + \
 && find /opt/venv -name "*.pyc" -delete

# --------------------------------------------------------------- runtime stage
FROM --platform=linux/amd64 python:3.11-slim AS runtime

# libGL and libglib are what opencv-python links against at import time.
# Without them `import cv2` fails with an ImportError about libGL.so.1 that
# says nothing about the missing apt package.
RUN apt-get update \
 && apt-get install --no-install-recommends -y \
      libgl1 \
      libglib2.0-0 \
      curl \
 && rm -rf /var/lib/apt/lists/*

# Non-root. The container reads photographs of faces and speaks HTTP with no
# authentication of its own, so it runs with as little as it can.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin vitruve

COPY --from=build /opt/venv /opt/venv
COPY --from=build /build/assets /opt/vitruve/assets

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VITRUVE_CACHE_DIR=/weights

# Declared as a volume so a `docker run` with no -v still gets a writable
# anonymous volume rather than writing 416 MB into the container layer.
RUN mkdir -p /weights /reports && chown -R vitruve:vitruve /weights /reports
VOLUME ["/weights"]

USER vitruve
WORKDIR /reports

# 0.0.0.0 is the only address that reaches a published port from outside the
# container, and inside the container it means "this container's interfaces"
# rather than "this laptop's LAN". The network boundary is the published port:
# `-p 127.0.0.1:8731:8731` keeps it on loopback, and a bare `-p 8731:8731`
# publishes it to every interface the host has. `--allow-remote` is still
# required, because vitruve refuses a non-loopback bind without it and that
# refusal is the thing that makes the decision visible in the run command.
EXPOSE 8731

# The timeout is 10s rather than the usual 2 or 3. /health calls
# doctor.resolve_device(), which touches torch, and the first call after a cold
# start took over 5s in an emulated amd64 container on Apple silicon.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8731/health || exit 1

ENTRYPOINT ["vitruve"]
CMD ["serve", "--host", "0.0.0.0", "--allow-remote"]
