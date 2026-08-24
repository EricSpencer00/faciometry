"""License tiers, enforced at load time.

Face analysis is a field where the useful weights and the usable weights are
different sets, and the difference is not visible in a `pip install`. Three
traps recur:

* **Ultralytics asserts AGPL-3.0 over the models its trainer produces**, not
  only over its code. Every third-party "yolov8-face" checkpoint tagged MIT or
  Apache-2.0 was trained from Ultralytics weights with the Ultralytics trainer.
  The permissive tag does not launder the upstream obligation.
* **InsightFace** ships MIT code with research-only pretrained models, and
  `insightface` downloads those models automatically on first use.
* **FLAME and the Basel Face Model** are non-commercial and forbid
  redistribution, so a repository that vendors the basis file is already in
  breach -- which several popular repositories do. (FLAME 2023 Open, released
  under Creative Commons Attribution, is the one exception.)

So Faciometry treats the license as a property of the backend, declared in the
type, and refuses at load time to exceed the tier the user selected. The
default tier is `PERMISSIVE`, which means the out-of-the-box pipeline carries
no copyleft and no non-commercial obligation. Choosing a higher tier is a
deliberate act that prints what it costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Tier(IntEnum):
    """Obligation tiers, ordered by how much they constrain the user.

    Ordered, so `backend.tier <= allowed` is the whole check.
    """

    PERMISSIVE = 0
    """MIT, BSD, Apache-2.0. No copyleft, no field-of-use restriction."""

    COPYLEFT = 1
    """AGPL-3.0 or GPL-3.0. Usable, but network use obliges source release."""

    NONCOMMERCIAL = 2
    """CC BY-NC, academic-only, or a research-use agreement. Not shippable."""

    UNLICENSED = 3
    """No license file at all, so all rights reserved. Never loaded by default."""


class LicenseViolation(RuntimeError):
    """Raised when a backend exceeds the permitted tier."""


@dataclass(frozen=True)
class Provenance:
    """Where a model came from and what obligations ride along with it."""

    name: str
    tier: Tier
    license_id: str
    source_url: str
    #: Obligations that come from the *training data* or a morphable-model
    #: basis rather than from the code license. This is where the surprises
    #: live, so it is a separate field and it is never empty by accident.
    inherited_from: tuple[str, ...] = ()
    note: str = ""

    @property
    def effective_tier(self) -> Tier:
        return self.tier

    def describe(self) -> str:
        parts = [f"{self.name}: {self.license_id} ({self.tier.name.lower()})"]
        if self.inherited_from:
            parts.append("inherits " + "; ".join(self.inherited_from))
        if self.note:
            parts.append(self.note)
        return " -- ".join(parts)


def require(prov: Provenance, allowed: Tier) -> None:
    """Raise unless ``prov`` is within ``allowed``."""
    if prov.tier > allowed:
        raise LicenseViolation(
            f"{prov.name} is {prov.license_id} ({prov.tier.name.lower()}), which "
            f"exceeds the permitted tier {allowed.name.lower()}. "
            f"{prov.describe()}. Re-run with --license-tier {prov.tier.name.lower()} "
            "to accept that obligation."
        )


# ---------------------------------------------------------------------------
# The catalogue of backends Faciometry knows how to load.
# ---------------------------------------------------------------------------

YUNET = Provenance(
    name="YuNet face detector",
    tier=Tier.PERMISSIVE,
    license_id="MIT",
    source_url="https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
    note="ONNX, under a megabyte, emits a box and five keypoints; "
    "WIDER Face val AP 0.834 / 0.824 / 0.708",
)

RETINAFACE_MOBILE = Provenance(
    name="RetinaFace MobileNet-0.25",
    tier=Tier.PERMISSIVE,
    license_id="MIT",
    source_url="https://github.com/biubug6/Pytorch_Retinaface",
    note="pure PyTorch, runs on MPS, box plus five keypoints",
)

SPIGA = Provenance(
    name="SPIGA landmark model",
    tier=Tier.PERMISSIVE,
    license_id="BSD-3-Clause",
    source_url="https://github.com/andresprados/SPIGA",
    note="98-point heatmaps plus head pose; WFLW NME 4.060, 300W NME 2.994. "
    "The heatmaps are why this is the primary backend -- Faciometry needs "
    "per-landmark uncertainty, and coordinate-regression models cannot give it",
)

MEDIAPIPE_FACE_LANDMARKER = Provenance(
    name="MediaPipe Face Landmarker",
    tier=Tier.PERMISSIVE,
    license_id="Apache-2.0",
    source_url="https://github.com/google-ai-edge/mediapipe",
    note="478 dense points including ten iris landmarks, and the bundled .task "
    "models are Apache-2.0 too. The iris points are what make permissive "
    "scale recovery possible; its own metric output is a constant-scale "
    "assumption, not a measurement, so Faciometry ignores it",
)

SIXDREPNET = Provenance(
    name="6DRepNet head pose",
    tier=Tier.PERMISSIVE,
    license_id="MIT",
    source_url="https://github.com/thohemp/6DRepNet",
    inherited_from=("300W-LP, which is derived from the Basel Face Model",),
    note="3DMM-free at inference; used as an independent cross-check on the "
    "pose estimate that the landmark model reports",
)

YOLO_FACE = Provenance(
    name="YOLO face detector (Ultralytics lineage)",
    tier=Tier.COPYLEFT,
    license_id="AGPL-3.0",
    source_url="https://www.ultralytics.com/license",
    inherited_from=(
        "Ultralytics, which asserts AGPL-3.0 over models produced by its "
        "training code, not only over the code itself",
    ),
    note="third-party checkpoints tagged MIT or Apache-2.0 are relabels of "
    "AGPL-licensed weights; the tag does not remove the obligation",
)

YOLO_DERM_SEG = Provenance(
    name="YOLO dermatological segmentation",
    tier=Tier.COPYLEFT,
    license_id="AGPL-3.0",
    source_url="https://www.ultralytics.com/license",
    inherited_from=(
        "Ultralytics AGPL-3.0",
        "Roboflow acne detection sets, CC BY 4.0, which require attribution",
    ),
    note="multi-instance lesion detection is genuinely YOLO-shaped work and "
    "no permissively-licensed equivalent exists, so this capability sits "
    "behind the copyleft tier by necessity rather than by preference",
)

FACE_PARSING_SEGFORMER = Provenance(
    name="SegFormer face parsing",
    tier=Tier.NONCOMMERCIAL,
    license_id="research-only",
    source_url="https://huggingface.co/jonathandinu/face-parsing",
    inherited_from=(
        "CelebAMask-HQ, which forbids reproducing, selling or trading any "
        "portion of the derived data",
    ),
    note="MIT tags on BiSeNet ports cover the code, not the weights",
)

THREEDDFA_V2 = Provenance(
    name="3DDFA_V2 dense 3D fitter",
    tier=Tier.NONCOMMERCIAL,
    license_id="MIT code, non-commercial basis",
    source_url="https://github.com/cleardusk/3DDFA_V2",
    inherited_from=(
        "Basel Face Model, internal non-commercial research use only, no "
        "redistribution",
    ),
    note="the code is genuinely MIT and runs on Apple silicon; it is the basis "
    "file that carries the restriction, so the user must supply it themselves",
)

MICA = Provenance(
    name="MICA metric face reconstruction",
    tier=Tier.NONCOMMERCIAL,
    license_id="MPI non-commercial",
    source_url="https://github.com/Zielon/MICA",
    inherited_from=("FLAME 2020", "InsightFace antelopev2 pretrained weights"),
    note="the only genuinely metric-scale face model in existence, and also the "
    "most encumbered; it predicts neutral identity shape only",
)

STAR_LOSS = Provenance(
    name="STAR loss landmark model",
    tier=Tier.UNLICENSED,
    license_id="none",
    source_url="https://github.com/ZhenglinZhou/STAR",
    note="best published landmark accuracy (WFLW NME 4.02) and no license file "
    "at all, so all rights are reserved and Faciometry will not load it",
)

CATALOGUE: tuple[Provenance, ...] = (
    YUNET,
    RETINAFACE_MOBILE,
    SPIGA,
    MEDIAPIPE_FACE_LANDMARKER,
    SIXDREPNET,
    YOLO_FACE,
    YOLO_DERM_SEG,
    FACE_PARSING_SEGFORMER,
    THREEDDFA_V2,
    MICA,
    STAR_LOSS,
)

BY_NAME: dict[str, Provenance] = {p.name: p for p in CATALOGUE}


def available_at(tier: Tier) -> tuple[Provenance, ...]:
    return tuple(p for p in CATALOGUE if p.tier <= tier)


def obligations_at(tier: Tier) -> tuple[str, ...]:
    """Every obligation the user takes on by selecting ``tier``."""
    out: list[str] = []
    for p in available_at(tier):
        if p.tier is Tier.PERMISSIVE and not p.inherited_from:
            continue
        out.append(p.describe())
    return tuple(out)
