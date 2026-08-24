"""Choosing a compute device once, and keeping float64 away from it.

Two things live here because getting either wrong is quiet rather than loud.

**Selection order is `mps > cuda > cpu`**, resolved once at startup and
threaded through, so the run manifest can record a single answer. An
environment override exists (`FACIOMETRY_DEVICE`) because reproducing a reported
number sometimes means forcing the device it was produced on: MPS and CPU do
not agree bit-for-bit, and a measurement whose interval is a fraction of a
pixel wide can move in its last digit between them.

**MPS has no float64.** Not "float64 is slow" -- Metal has no 64-bit float
type at all, and PyTorch raises `TypeError: Cannot convert a MPS Tensor to
float64 dtype` the moment one appears. This collides directly with Faciometry's
geometry layer, which is float64 numpy throughout because the Monte-Carlo
ensemble multiplies small differences of large coordinates and a float32
ensemble visibly quantises the resulting interval.

So the policy, stated once and applied everywhere:

    Models run in float32 on the selected device. Everything that comes out of
    a model is cast to float64 on the host before any geometry touches it.

The cast happens at the backend boundary, in :func:`to_host`, and never later.
Casting later would mean some geometry ran at float32 precision without anyone
choosing that, and the place it would show up is the width of a confidence
interval, where nobody would recognise it as a bug.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: Environment variable that overrides automatic selection.
DEVICE_ENV_VAR = "FACIOMETRY_DEVICE"

#: Preference order. MPS first on Apple silicon because the alternative is a
#: CPU that is roughly an order of magnitude slower on the hourglass backbone.
PREFERENCE: tuple[str, ...] = ("mps", "cuda", "cpu")

VALID = frozenset({"mps", "cuda", "cpu"})


class DeviceUnavailable(RuntimeError):
    """Raised when an explicitly requested device is not usable."""


@dataclass(frozen=True)
class Device:
    """The resolved compute device and how it was arrived at.

    ``description`` is written for the run manifest, so it names the backend,
    the reason it was chosen, and the float policy. A manifest that recorded
    only "mps" would not let anyone reproduce a number a year later.
    """

    kind: str
    description: str
    explicit: bool
    supports_float64: bool

    def __post_init__(self) -> None:
        if self.kind not in VALID:
            raise ValueError(f"unknown device {self.kind!r}, expected one of {sorted(VALID)}")

    @property
    def is_cpu(self) -> bool:
        return self.kind == "cpu"

    @property
    def torch_device(self) -> Any:
        """A `torch.device`, imported lazily so `core` stays torch-free."""
        import torch

        return torch.device(self.kind)

    @property
    def model_dtype(self) -> Any:
        """The dtype models run in. Always float32; see the module docstring."""
        import torch

        return torch.float32

    def __str__(self) -> str:
        return self.kind


def _mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


AVAILABILITY = {"mps": _mps_available, "cuda": _cuda_available, "cpu": lambda: True}


def _describe(kind: str, reason: str) -> str:
    bits = [f"{kind}", reason]
    if kind == "mps":
        bits.append(
            "Metal has no float64 type, so models run float32 on device and "
            "every result is cast to float64 on the host before geometry"
        )
    else:
        bits.append("models run float32; geometry runs float64 on the host")
    return "; ".join(bits)


def resolve(requested: str | None = None, *, env: dict[str, str] | None = None) -> Device:
    """Pick a device, honouring an explicit request or `FACIOMETRY_DEVICE`.

    An explicit request that cannot be satisfied raises rather than falling
    back. A silent fallback would mean a run recorded as `mps` in one manifest
    and `cpu` in another produced different numbers for no visible reason.
    """
    environ = os.environ if env is None else env
    choice = requested if requested is not None else environ.get(DEVICE_ENV_VAR)

    if choice is not None:
        kind = choice.strip().lower()
        if kind not in VALID:
            raise DeviceUnavailable(
                f"{kind!r} is not a device Faciometry knows; expected one of {sorted(VALID)}"
            )
        if not AVAILABILITY[kind]():
            source = "requested" if requested is not None else f"{DEVICE_ENV_VAR}={choice}"
            raise DeviceUnavailable(
                f"{kind} was {source} but is not available on this machine "
                f"({platform.machine()} / {platform.system()}). Unset the override to "
                "let Faciometry choose."
            )
        return Device(
            kind=kind,
            description=_describe(kind, "selected explicitly"),
            explicit=True,
            supports_float64=(kind != "mps"),
        )

    for kind in PREFERENCE:
        if AVAILABILITY[kind]():
            skipped = [k for k in PREFERENCE[: PREFERENCE.index(kind)]]
            reason = "first available in preference order mps > cuda > cpu"
            if skipped:
                reason += f" (skipped {', '.join(skipped)})"
            return Device(
                kind=kind,
                description=_describe(kind, reason),
                explicit=False,
                supports_float64=(kind != "mps"),
            )
    raise DeviceUnavailable("no device available, which should be impossible since cpu always is")


def pin_to_cpu(requested: Device | None, reason: str) -> tuple[Device, str]:
    """Force CPU for a backend whose upstream code cannot leave it.

    Two of the four backends Faciometry loads are pinned this way, and for the same
    underlying cause: their authors wrote ``if tensor.is_cuda`` or
    ``tensor.get_device()`` as a proxy for "is this on an accelerator", which was
    true when they wrote it and stopped being true when Metal arrived. On MPS
    both tests take the CUDA branch and the forward pass dies constructing a
    tensor on a device that does not exist. Patching site-packages would fix it
    until the next release.

    A device the *user* asked for raises rather than being substituted, because
    a caller who wrote ``--device mps`` and silently got CPU would attribute the
    timing to the wrong hardware. A device that automatic selection arrived at
    is substituted, and the reason is returned for the run manifest so the
    choice is visible rather than merely correct.

    `Device.explicit` is what separates the two, which is why it exists. The
    pipeline resolves the device once and threads the resulting object through
    every backend, so by the time a pinned backend sees it there is no longer a
    ``None`` to distinguish "nobody asked" from "somebody asked for this".
    """
    if requested is not None:
        if requested.is_cpu:
            return requested, ""
        if requested.explicit:
            raise DeviceUnavailable(f"{reason} Pass the CPU device instead.")
        return resolve("cpu"), f"{requested.kind} was available but not used: {reason}"
    resolved = resolve()
    if resolved.is_cpu:
        return resolved, ""
    return resolve("cpu"), f"{resolved.kind} was available but not used: {reason}"


def to_host(tensor: Any) -> NDArray[np.float64]:
    """A device tensor to a float64 numpy array on the host.

    This is the one sanctioned crossing point between the model half of the
    system and the geometry half. The order matters: detach, then move to CPU,
    then widen. Widening on an MPS tensor raises, and the error message names
    dtypes rather than the actual problem, which is why this is a function and
    not three lines repeated in four backends.
    """
    if isinstance(tensor, np.ndarray):
        return np.asarray(tensor, dtype=np.float64)
    detached = tensor.detach() if hasattr(tensor, "detach") else tensor
    return np.asarray(detached.to("cpu").numpy(), dtype=np.float64)


def describe_environment(device: Device) -> dict[str, str]:
    """Everything about the compute environment the run manifest should carry."""
    info = {
        "device": device.kind,
        "device_description": device.description,
        "device_explicit": str(device.explicit).lower(),
        "float_policy": "models float32, geometry float64 on host",
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "python": platform.python_version(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
    except ImportError:
        info["torch"] = "not installed"
    info["numpy"] = np.__version__
    return info
