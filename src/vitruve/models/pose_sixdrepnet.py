"""6DRepNet head pose, carried as a second opinion rather than as the answer.

The pose estimate is not a reported measurement; it is the gate. Kleinberg and
Vanezis found facial indices moving 8 to 19 percent at ten degrees of yaw
against a between-subject spread of 1.2 percent, so whether a measurement is
printed at all turns on a number that is itself uncertain. Published head-pose
estimators sit around 3.5 to 4 degrees mean absolute error with a label-noise
floor near 2.5 to 3 degrees, which means a five-degree gate is at the edge of
what the instrument can resolve. `core.sensitivity` handles that by gating on
``|pose| + k * sd``, and this module's job is to supply an estimate whose
scatter is known and, more importantly, one that was produced by a model that
did not also produce the landmarks.

That independence is the point. SPIGA regresses pose from the same hourglass
features it regresses landmarks from, so if its features are wrong for a face,
its pose and its landmarks are wrong *together* and consistently, which is the
failure a single-model pipeline cannot see. 6DRepNet is a separate architecture
on separate training data; `HeadPose.disagreement` between the two is the check
that has any power.

**Rotation representation.** 6DRepNet regresses a continuous 6D rotation
representation and orthonormalises it into a matrix, rather than regressing
Euler angles directly. That matters because Euler angles are discontinuous and
a network trained to regress them has to learn to jump, which is where the
large-pose errors in older estimators came from. Euler angles are extracted
only at the end, for the gate.

**Channel identity was verified, absolute sign was not.** Under a horizontal
mirror of the input crop, the first output channel is invariant while the other
two negate, which identifies channel 0 as pitch and channels 1 and 2 as yaw and
roll. Measured on a frontal portrait: pitch -4.99 to -4.85, yaw -1.33 to +2.22,
roll -0.72 to +0.68. Which direction of turn counts as positive yaw was *not*
established: an attempt to induce a known yaw by warping a frontal photograph
through a planar homography moved the estimate by under two degrees for
twenty-five degrees of simulated rotation, so the experiment had no power and
its result is not used. Nothing downstream depends on the sign today, because
`core.sensitivity.gated_pose` takes an absolute value; a future signed
consumer must resolve this against a labelled benchmark first.

**Licensing.** The code and weights are MIT, but training used 300W-LP, which
is rendered from the Basel Face Model. That obligation is recorded on the
`Provenance` rather than in a comment, so a tier check sees it.

**This backend runs on the CPU.** `sixdrepnet.utils.normalize_vector` reads
``tensor.get_device()`` and, for anything that is not the host, builds its
epsilon on ``cuda:N``. An MPS tensor reports device 0, so the forward pass ends
in ``Torch not compiled with CUDA enabled`` on Apple silicon. The same
device-detection assumption breaks SPIGA, for the same reason, and the same
answer applies: a RepVGG-B1g2 forward on a 224-pixel crop costs about 0.03 s on
CPU, so the pin is not worth a monkey-patch of site-packages. It is recorded on
`device_note` for the run manifest.

**Packaging.** `sixdrepnet` 0.1.6 uses flat imports (``from model import
SixDRepNet``, ``import utils``) and makes them resolve by appending its own
directory to `sys.path` in `__init__.py`. Importing the package therefore
inserts modules named `model`, `utils`, `backbone` and `test` into the global
module table, where they shadow anything of the same name in the rest of the
process. This module imports the submodules deliberately and then removes both
the path entry and the shadowing `sys.modules` keys, keeping direct references
so the already-imported modules stay alive.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.sensitivity import POSE_ESTIMATOR_MAE_DEG, POSE_ESTIMATOR_SD_DEG
from . import licensing, weights
from .device import Device, pin_to_cpu, to_host
from .licensing import Provenance, Tier
from .protocols import FaceBox, HeadPose, Image

WEIGHT_KEY = "sixdrepnet_300wlp"

PROVENANCE: Provenance = licensing.SIXDREPNET

#: Input resolution and the ImageNet statistics the backbone was trained with.
INPUT_PX = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

#: The crop 6DRepNet expects is slightly looser than a tight detection box.
BOX_EXPANSION = 1.1

#: What the run manifest should say when the device was substituted.
CPU_ONLY_REASON = (
    "6DRepNet runs on the CPU: sixdrepnet.utils.normalize_vector builds a tensor "
    "on cuda:N whenever its input is not on the host, and an MPS tensor reports "
    "device 0, so the forward pass ends in 'Torch not compiled with CUDA enabled'. "
    "The forward costs about 0.03 s on CPU."
)

#: Module names the upstream package leaks into the global table.
_LEAKED = ("model", "utils", "backbone", "test", "train", "datasets", "loss", "regressor")


class _MissingDependency(RuntimeError):
    pass


def _import_sixdrepnet() -> tuple[Any, Any]:
    """Import the two submodules that matter, then undo the path pollution."""
    spec = importlib.util.find_spec("sixdrepnet")
    if spec is None or spec.origin is None:
        raise _MissingDependency(
            "sixdrepnet is not installed. Install it with `uv pip install sixdrepnet`. "
            "Without it there is no independent pose estimate, and the pose gate "
            "would be reading a number produced by the same features it is gating."
        )
    pkg_dir = os.path.dirname(spec.origin)
    pre_existing = {name: sys.modules.get(name) for name in _LEAKED}
    sys.path.insert(0, pkg_dir)
    try:
        model = importlib.import_module("sixdrepnet.model")
        utils = importlib.import_module("sixdrepnet.utils")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise _MissingDependency(f"sixdrepnet is installed but unusable: {exc}") from exc
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(pkg_dir)
        # Restore whatever occupied these names before, or clear them. The two
        # modules above hold direct references to their dependencies, so
        # removing the aliases does not unload anything they need.
        for name, previous in pre_existing.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return model, utils


def _resize_shorter_side(img: NDArray[np.uint8], size: int, cv2: Any) -> NDArray[np.uint8]:
    """Scale so the shorter side is ``size``, preserving aspect ratio.

    This reproduces `torchvision.transforms.Resize(size)` with an integer
    argument, which resizes the shorter side rather than forcing a square. It
    is followed by a centre crop, so a non-square face box loses its edges
    rather than being squashed. Squashing instead would change apparent face
    proportions, which is the one thing a pose estimator reads.
    """
    h, w = img.shape[:2]
    if h <= w:
        new_h, new_w = size, max(1, round(w * size / h))
    else:
        new_h, new_w = max(1, round(h * size / w)), size
    interp = cv2.INTER_AREA if new_h < h else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def _centre_crop(img: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    h, w = img.shape[:2]
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return img[top : top + size, left : left + size]


class SixDRepNetPose:
    """Head pose from a face crop, with the estimator's published scatter."""

    provenance = PROVENANCE

    def __init__(
        self,
        *,
        allowed_tier: Tier = Tier.PERMISSIVE,
        device: Device | None = None,
        weights_path: Path | None = None,
        box_expansion: float = BOX_EXPANSION,
    ) -> None:
        licensing.require(PROVENANCE, allowed_tier)

        import torch

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise _MissingDependency(
                "6DRepNet preprocessing needs opencv-python for the resize."
            ) from exc

        model_mod, utils_mod = _import_sixdrepnet()

        self._torch = torch
        self._cv2 = cv2
        self._utils = utils_mod
        self._device, self._device_note = pin_to_cpu(device, CPU_ONLY_REASON)
        self._box_expansion = float(box_expansion)
        self._path = weights_path or weights.resolve(WEIGHT_KEY)
        self._sha256 = weights.spec_for(WEIGHT_KEY).sha256

        # deploy=True builds the re-parameterised RepVGG blocks the released
        # checkpoint was saved from; pretrained=False because the backbone
        # weights come from that checkpoint rather than from a separate file.
        net = model_mod.SixDRepNet(
            backbone_name="RepVGG-B1g2", backbone_file="", deploy=True, pretrained=False
        )
        state = torch.load(str(self._path), map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval()
        net.to(self._device.torch_device)
        self._net = net

    @property
    def weights_path(self) -> Path:
        return self._path

    @property
    def weights_sha256(self) -> str:
        return self._sha256

    @property
    def device(self) -> Device:
        return self._device

    @property
    def device_note(self) -> str:
        """Empty unless a faster device was available and deliberately skipped."""
        return self._device_note

    def _preprocess(self, image: Image, box: object) -> Any:
        img = np.asarray(image)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"expected an (h, w, 3) RGB image, got shape {img.shape}")
        h, w = img.shape[:2]
        target = FaceBox.from_detection(box, PROVENANCE).expanded(self._box_expansion)
        x0 = max(0, round(target.x))
        y0 = max(0, round(target.y))
        x1 = min(w, round(target.x + target.w))
        y1 = min(h, round(target.y + target.h))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("the face box does not intersect the image")
        crop = img[y0:y1, x0:x1]

        # The upstream predict() converts its BGR input to RGB before
        # normalising, so RGB is what the backbone saw during training and the
        # pipeline's RGB goes through untouched.
        resized = _centre_crop(_resize_shorter_side(crop, INPUT_PX, self._cv2), INPUT_PX)
        arr = resized.astype(np.float64) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        tensor = self._torch.from_numpy(
            np.ascontiguousarray(arr.transpose(2, 0, 1)[None])
        ).to(self._torch.float32)
        return tensor.to(self._device.torch_device)

    def estimate(self, image: Image, box: object) -> HeadPose:
        torch = self._torch
        tensor = self._preprocess(image, box)
        with torch.no_grad():
            rotation = self._net(tensor)
            euler = self._utils.compute_euler_angles_from_rotation_matrices(rotation)
        angles = to_host(euler)[0] * 180.0 / np.pi
        pitch, yaw, roll = (float(a) for a in angles)
        return HeadPose(
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            # The published AFLW2000-3D mean absolute error converted to a
            # standard deviation under a normal assumption. It is a
            # population figure, not a per-image confidence, and the class
            # docstring in protocols says why that distinction matters.
            sd_deg=POSE_ESTIMATOR_SD_DEG,
            provenance=PROVENANCE,
            source=f"6DRepNet, AFLW2000-3D MAE {POSE_ESTIMATOR_MAE_DEG:.2f} deg",
        )


def build(*, allowed_tier: Tier = Tier.PERMISSIVE, **kwargs: object) -> SixDRepNetPose:
    return SixDRepNetPose(allowed_tier=allowed_tier, **kwargs)  # type: ignore[arg-type]
