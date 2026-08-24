"""SPIGA: 98 landmarks, and the heatmaps that say how well each one is pinned.

SPIGA is the primary landmarker because of what it emits, not because of where
it sits in the accuracy tables. Coordinate-regression landmarkers return a
point and nothing else; the entire uncertainty story in `measure.evaluate`
needs a *field* per landmark, and a heatmap network is the only kind that has
one. SPIGA is BSD-3-Clause, reports WFLW NME 4.060 and 300W NME 2.994, and runs
a forward pass in about 0.2 s on an M1 CPU.

Three things about the upstream package had to be worked around, and each is a
deliberate choice rather than an accident:

**The shipped inference wrapper is CUDA-only.** `SPIGAFramework.__init__` calls
``self.model.cuda(gpus[0])`` unconditionally and `_data2device` does the same
to every tensor, so on Apple silicon it raises before it loads anything. This
module therefore drives `spiga.models.spiga.SPIGA` directly and reuses only the
package's pre-processing transforms and its POSIT reference shape. The
crop-to-image mapping below is the same arithmetic `SPIGAFramework.postreatment`
performs, transcribed rather than called.

**The heatmaps are internal.** `MultitaskCNN.forward` returns only the visual
field and the pose core; the per-landmark attention maps are consumed inside
the loop that builds the next hourglass input and never surface. A forward hook
on the last attention head recovers them, which is why `locate` registers and
removes a hook around each forward pass rather than reading a return value.

**Coordinates come from the graph regressor, covariances from the heatmap.**
These are two different quantities from two different heads and mixing them is
intentional. The cascaded graph-attention regressor is what makes SPIGA's NME
good, so it gives the position; the attention heatmap is the only thing that
knows how *concentrated* the evidence for that position was, so it gives the
second moment. Taking the position from the heatmap argmax instead would throw
away the refinement, and taking the covariance from the regressor is not
possible because a regressor has none.

**The pose head is reported but not trusted.** SPIGA's `Pose` output is six
numbers, of which the first three are Euler angles in degrees. Which channel is
which was settled by a mirror test; the absolute sign was not, and the two pose
estimates do not share a convention: on a frontal portrait SPIGA reported
pitch +9.2 degrees where 6DRepNet reported -3.8, a thirteen-degree per-axis
disagreement on a face that is close to level. Averaging them would be
meaningless. The pipeline should read `pose_sixdrepnet`, which at least has a
published error figure on a labelled benchmark, and use this one only through
`HeadPose.disagreement` as the cross-check it exists to be.

**This backend runs on the CPU, and that is not a performance choice.** The
upstream package assumes CUDA in two places that are not reachable from a
configuration flag: `SPIGAFramework` calls ``.cuda()`` unconditionally, and,
more awkwardly, `spiga.models.gnn.pose_proj` builds its rotation matrices with
``device=None`` unless the input tensor reports ``is_cuda``, so on Metal the
forward pass dies with ``Passed CPU tensor to MPS op`` inside the POSIT
projection. Working around that would mean monkey-patching a module in
site-packages, which would break silently on the next upstream release. A
forward pass costs about 0.2 s on an M1 CPU, which is a fifth of the budget for
a whole report, so the trade is easy. The substitution is recorded on
`device_note` and belongs in the run manifest; asking for a non-CPU device
explicitly raises rather than being quietly ignored.

**The heatmap width is a training artefact, and it is calibrated.** This is the
most important caveat in the file and it was measured rather than assumed. Over
all 98 attention channels on a frontal portrait, the second-moment standard
deviation ranges from 1.52 to 1.96 feature-map pixels -- a spread of 1.29x
between the best and worst localised landmark on the whole face -- and the
anisotropy ratio averages 1.14 with a maximum of 1.38. The jaw contour, which
ought to be the clearest anisotropy in a face, comes out 1.18 against the eye
region's 1.13. That is not a measurement of localisation quality; it is the
width of the Gaussian target the network was trained to reproduce, which is the
same for every channel by construction.

Two consequences, both stated rather than hidden:

* The *shape* of the covariance is kept, because it is the only per-landmark
  signal available and it is weakly informative. It is not the strong
  anisotropy `measure.evaluate` was designed to consume, and a report built on
  this backend should not claim it is.
* The *scale* is calibrated, because the raw second moment carried through the
  crop magnification puts every landmark at roughly 13 image pixels of standard
  deviation on a 270-pixel face, which withholds the entire catalogue for a
  reason that is an artefact of the training target. :data:`WFLW_NME` converts
  the model's published accuracy into a pixel scale using the image's own
  interocular distance, and the covariances are rescaled to match it while
  keeping their relative magnitudes. That is a *provisional* calibration from a
  published aggregate, not a measured per-landmark error; arm 1 of the
  validation design (landmark NME against the published benchmark, per
  landmark) is what replaces it, and ``nme_calibration=None`` turns it off.

**Installation.** ``spiga`` imports `pkg_resources`, which setuptools removed in
version 81. The environment needs ``setuptools<81`` or `spiga.inference.
pretreatment` fails to import; the error message below says so, because the
raw `ModuleNotFoundError: No module named 'pkg_resources'` points at the wrong
package entirely.

**Landmark mapping.** The WFLW index assignments below were read off the
canonical 98-point reference shape that ships inside the package
(``spiga/data/models3D/mean_face_3D_98.txt``), not off a blog post: the shape
puts the nose tip at the origin with maximum z, the chin at minimum y, and it
separates the two sides cleanly in x, so which index is which landmark is a
matter of reading coordinates. The side assignment was then confirmed against
two independently trained models on the same photograph, where SPIGA 96,
YuNet's first keypoint and MediaPipe's iris centre 468 all landed within two
pixels of the same physical eye.

Landmarks WFLW cannot supply are absent from the map and stay absent. The
face-contour points 0 to 32 are a *silhouette*, and a silhouette is not a
skeleton: zygion is the widest point of the zygomatic arch and gonion is the
corner of the mandible, and neither is where the outline of a photographed face
happens to be widest. Reading bizygomatic width off a contour point is the
standard move in this product category and it is measuring hair and soft
tissue. Trichion needs a hairline, porion needs an ear canal, and the neck is
not in a frontal crop.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.landmarks import Landmark, PointSet
from ..core.landmarks import Landmark as L
from . import licensing, weights
from .device import Device, pin_to_cpu, to_host
from .licensing import Provenance, Tier
from .protocols import (
    FaceBox,
    Image,
    LandmarkResult,
    heatmap_moments,
    scale_covariances,
    uncertainty_from_covariances,
)

WEIGHT_KEY = "spiga_wflw"

PROVENANCE: Provenance = licensing.SPIGA

#: WFLW row index for each landmark SPIGA can actually see.
#:
#: Read from the package's own canonical shape. The eyelid points are the mid-lid
#: vertices directly above and below the pupil, which is what palpebrale
#: superius and inferius mean; the crista philtri pair are the two vertices that
#: sit *higher* than the midline upper-lip vertex between them, which is the
#: Cupid's bow and is a geometric signature rather than a guess.
WFLW_INDEX: dict[Landmark, int] = {
    L.SELLION: 51,
    L.SUPERCILIARE_R: 35,
    L.SUPERCILIARE_L: 44,
    L.EXOCANTHION_R: 60,
    L.ENDOCANTHION_R: 64,
    L.ENDOCANTHION_L: 68,
    L.EXOCANTHION_L: 72,
    L.PALPEBRALE_SUP_R: 62,
    L.PALPEBRALE_INF_R: 66,
    L.PALPEBRALE_SUP_L: 70,
    L.PALPEBRALE_INF_L: 74,
    L.PUPIL_R: 96,
    L.PUPIL_L: 97,
    L.PRONASALE: 54,
    L.ALARE_R: 55,
    L.ALARE_L: 59,
    L.SUBNASALE: 57,
    L.CRISTA_PHILTRI_R: 78,
    L.CRISTA_PHILTRI_L: 80,
    L.LABIALE_SUPERIUS: 79,
    L.LABIALE_INFERIUS: 85,
    L.CHEILION_R: 76,
    L.CHEILION_L: 82,
    # The lowest median point of the chin outline. The catalogue's frontal
    # facial-thirds specs call this landmark menton, so it is emitted under that
    # name; gnathion is deliberately *not* also emitted, because it would be the
    # same pixel under a second name and a measurement combining both would be
    # counting one observation twice.
    L.MENTON: 16,
}

#: Landmarks in the vocabulary that this backend structurally cannot supply,
#: recorded so the report can say "the model cannot see this" rather than
#: leaving a silent gap. Each entry has a reason in the module docstring.
UNSUPPORTED: tuple[Landmark, ...] = (
    L.TRICHION,
    L.GLABELLA,
    L.NASION,
    L.ORBITALE_L,
    L.ORBITALE_R,
    L.SUBALARE_L,
    L.SUBALARE_R,
    L.COLUMELLA,
    L.STOMION,
    L.SUBLABIALE,
    L.POGONION,
    L.GNATHION,
    L.GONION_L,
    L.GONION_R,
    L.ZYGION_L,
    L.ZYGION_R,
    L.TRAGION_L,
    L.TRAGION_R,
    L.PORION_L,
    L.PORION_R,
    L.CERVICALE,
)

#: The crop the model was trained on, and the resolution of its attention maps.
CROP_PX = 256
FTMAP_PX = 64

#: Half-width of the window the heatmap second moment is taken over, in feature
#: map pixels. Eleven feature-map pixels is about 44 crop pixels, which is wide
#: enough to contain the peak of a poorly localised contour point and narrow
#: enough that two neighbouring landmarks do not share mass.
HEATMAP_WINDOW = 11

#: SPIGA is trained on a crop somewhat larger than a tight detection box.
#: Feeding it a tight box moves every contour point inward.
BOX_EXPANSION = 1.2

#: SPIGA's published WFLW normalised mean error, as a fraction of the
#: interocular distance rather than the percentage the paper prints. It is a
#: mean over all 98 points including the jaw contour, so it is pessimistic for
#: the two dozen well-localised landmarks Vitruve actually maps.
WFLW_NME = 0.04060

#: Mean absolute error to per-axis standard deviation for a 2D isotropic
#: Gaussian: E|e| = sigma * sqrt(pi/2). NME is a mean L2 error, so undoing this
#: factor is what turns it into the quantity a covariance holds.
_MAE_TO_SD = 1.0 / np.sqrt(np.pi / 2.0)


def _calibrate_to_nme(
    covs: NDArray[np.float64],
    points_px: NDArray[np.float64],
    nme: float,
) -> NDArray[np.float64]:
    """Rescale covariances so their pooled standard deviation matches the NME.

    Relative magnitudes and orientations are preserved; only the overall scale
    moves. The reference length is the image's own interocular distance, because
    that is the denominator WFLW normalises by, so this converts a published
    percentage into pixels for *this* photograph rather than assuming a face
    size.

    Pooling over all 98 channels rather than over the mapped subset is
    deliberate: the published NME is an average over all 98, and normalising a
    hand-picked subset of well-localised points to a whole-model average would
    inflate exactly the landmarks that are best.
    """
    interocular = float(
        np.linalg.norm(points_px[WFLW_INDEX[L.EXOCANTHION_L]] - points_px[WFLW_INDEX[L.EXOCANTHION_R]])
    )
    if interocular <= 0:
        raise ValueError("degenerate interocular distance, so no calibration reference exists")
    target_sd = nme * interocular * _MAE_TO_SD
    pooled_var = float(np.mean(np.trace(covs, axis1=-2, axis2=-1) / 2.0))
    if pooled_var <= 0:
        raise ValueError("heatmap covariances are degenerate, so they cannot be calibrated")
    return covs * (target_sd**2 / pooled_var)


class _MissingDependency(RuntimeError):
    """Raised with an actionable message when the upstream package is unusable."""


def _import_spiga() -> tuple[Any, Any, Any]:
    try:
        from spiga.inference import pretreatment
        from spiga.inference.config import ModelConfig
        from spiga.models.spiga import SPIGA as SpigaNet
    except ModuleNotFoundError as exc:
        if exc.name == "pkg_resources":
            raise _MissingDependency(
                "spiga imports pkg_resources, which setuptools removed in version 81. "
                "Pin `setuptools<81` in this environment. The bare ImportError names "
                "pkg_resources and points at the wrong package."
            ) from exc
        raise _MissingDependency(
            "SPIGA is not installed. Install it with `uv pip install spiga 'setuptools<81'`. "
            "There is no fallback landmarker: a coordinate-regression model would "
            "produce points with no covariance, and every interval in the report "
            "descends from those covariances."
        ) from exc
    return ModelConfig, pretreatment, SpigaNet


#: What the run manifest should say when the device was substituted.
CPU_ONLY_REASON = (
    "SPIGA runs on the CPU: spiga.models.gnn.pose_proj constructs its rotation "
    "matrices on the host unless the input tensor reports is_cuda, so the POSIT "
    "projection raises 'Passed CPU tensor to MPS op' on Apple silicon. The forward "
    "pass costs about 0.2 s on CPU, so this is a bounded cost rather than a "
    "degradation."
)


class SpigaLandmarker:
    """98-point landmarks with per-point covariance from the attention maps."""

    provenance = PROVENANCE

    def __init__(
        self,
        *,
        allowed_tier: Tier = Tier.PERMISSIVE,
        device: Device | None = None,
        dataset: str = "wflw",
        weights_path: Path | None = None,
        box_expansion: float = BOX_EXPANSION,
        heatmap_window: int = HEATMAP_WINDOW,
        nme_calibration: float | None = WFLW_NME,
    ) -> None:
        # Before the 243 MB checkpoint is touched.
        licensing.require(PROVENANCE, allowed_tier)

        import torch

        ModelConfig, pretreatment, SpigaNet = _import_spiga()

        if dataset != "wflw":
            raise ValueError(
                f"only the wflw checkpoint is pinned, not {dataset!r}; the 98-point "
                "index map above is specific to it"
            )

        self._torch = torch
        self._device, self._device_note = pin_to_cpu(device, CPU_ONLY_REASON)
        self._box_expansion = float(box_expansion)
        self._heatmap_window = int(heatmap_window)
        self._nme_calibration = None if nme_calibration is None else float(nme_calibration)

        cfg = ModelConfig(dataset, load_model_url=False)
        self._cfg = cfg
        self._transforms = pretreatment.get_transformers(cfg)

        self._path = weights_path or weights.resolve(WEIGHT_KEY)
        self._sha256 = weights.spec_for(WEIGHT_KEY).sha256
        net = SpigaNet(
            num_landmarks=cfg.dataset.num_landmarks, num_edges=cfg.dataset.num_edges
        )
        # weights_only because a checkpoint is data, and this one comes off a
        # mirror; the pin guarantees the bytes but not that unpickling them is safe.
        state = torch.load(str(self._path), map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval()
        net.to(self._device.torch_device)
        self._net = net

        # The POSIT reference shape and camera intrinsics. Loaded from a text
        # file inside the package, so this is not a network access.
        params = pretreatment.AddModel3D(
            cfg.dataset.ldm_ids,
            ftmap_size=cfg.ftmap_size,
            focal_ratio=cfg.focal_ratio,
            totensor=True,
        )()
        dev = self._device.torch_device
        self._model3d = params["model3d"].unsqueeze(0).to(dev)
        self._cam_matrix = params["cam_matrix"].unsqueeze(0).to(dev)

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

    @property
    def index_map(self) -> Mapping[Landmark, int]:
        return dict(WFLW_INDEX)

    # -- inference ---------------------------------------------------------

    def _forward(self, crop: NDArray[np.float64]) -> tuple[Any, Any, Any]:
        torch = self._torch
        batch = torch.tensor(np.asarray([crop]), dtype=torch.float32).to(
            self._device.torch_device
        )
        caught: dict[str, Any] = {}

        def grab(_module: Any, _inputs: Any, output: Any) -> None:
            caught["heatmaps"] = output.detach()

        handle = self._net.visual_cnn.outs_points[-1].register_forward_hook(grab)
        try:
            with torch.no_grad():
                out = self._net([batch, self._model3d, self._cam_matrix])
        finally:
            # Removed in a finally block: a hook left registered would fire on
            # every later call and quietly hold a reference to a device tensor.
            handle.remove()
        if "heatmaps" not in caught:
            raise RuntimeError(
                "the SPIGA attention head did not fire, so no heatmap is available. "
                "The upstream module layout has changed and the hook target in "
                "this file needs revisiting; falling back to an argmax would "
                "silently drop every covariance."
            )
        return out["Landmarks"][-1], caught["heatmaps"], out["Pose"]

    def locate(self, image: Image, box: object) -> LandmarkResult:
        img = np.asarray(image)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"expected an (h, w, 3) RGB image, got shape {img.shape}")

        # SPIGA's transforms are written against cv2.imread output, so the
        # model saw BGR during training.
        bgr = np.ascontiguousarray(img[:, :, ::-1])
        target = FaceBox.from_detection(box, PROVENANCE).expanded(self._box_expansion)
        sample = self._transforms({"image": bgr, "bbox": list(target.xywh)})
        crop_bbox = np.asarray(sample["bbox"], dtype=np.float64)

        raw_points, raw_heatmaps, raw_pose = self._forward(sample["image"])

        # Crop-relative [0, 1] -> crop pixels -> box-relative -> image pixels.
        # This is SPIGA's own postreatment arithmetic; the scale factors are
        # reused below to carry the covariances through the same mapping.
        pts_crop = to_host(raw_points[0]) * CROP_PX
        sx = target.w / crop_bbox[2]
        sy = target.h / crop_bbox[3]
        pts_img = np.empty_like(pts_crop)
        pts_img[:, 0] = (pts_crop[:, 0] - crop_bbox[0]) * sx + target.x
        pts_img[:, 1] = (pts_crop[:, 1] - crop_bbox[1]) * sy + target.y

        heatmaps = to_host(raw_heatmaps[0])
        _centres, covs_ft = heatmap_moments(heatmaps, window=self._heatmap_window)
        # Feature map -> crop pixels is a fixed factor of four; crop -> image is
        # the same anisotropic scaling the points went through.
        ft_to_crop = CROP_PX / FTMAP_PX
        covs_img = scale_covariances(covs_ft, sx=ft_to_crop * sx, sy=ft_to_crop * sy)
        if self._nme_calibration is not None:
            covs_img = _calibrate_to_nme(covs_img, pts_img, self._nme_calibration)

        index = {name: i for i, name in enumerate(WFLW_INDEX)}
        coords = np.stack([pts_img[WFLW_INDEX[name]] for name in index], axis=0)
        points = PointSet(index=index, coords=coords)
        uncertainty = uncertainty_from_covariances(index, covs_img, WFLW_INDEX)

        pose = to_host(raw_pose[0])
        return LandmarkResult(
            points=points,
            uncertainty=uncertainty,
            provenance=PROVENANCE,
            # Channel identity was fixed by a mirror test: pitch is invariant
            # under a horizontal flip while the other two move. The absolute
            # sign was not resolved (see protocols.HeadPose), and the
            # cross-check in pose_sixdrepnet is the estimate the pipeline
            # should prefer.
            yaw_deg=float(pose[0]),
            pitch_deg=float(pose[1]),
            roll_deg=float(pose[2]),
            unsupported=UNSUPPORTED,
        )


def build(
    *, allowed_tier: Tier = Tier.PERMISSIVE, **kwargs: object
) -> SpigaLandmarker:
    return SpigaLandmarker(allowed_tier=allowed_tier, **kwargs)  # type: ignore[arg-type]
