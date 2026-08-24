"""Fine-tune the acne lesion detector, and write down where the data came from.

Acne is the only dermatological finding in this project with real public box
annotations, and the annotations differ sharply in what they let you do:

* **ACNE04** (Wu et al., roughly 1,457 images and 18,983 boxes, with Hayashi
  severity 0-3) is the reference academic set and is released "free for academic
  usage". That is not an OSI licence and it is not a commercial grant, so
  Vitruve will not ship weights trained on it.
* **Roboflow Universe** carries several acne detection and segmentation sets
  under **CC BY 4.0**. These are the commercially clean option. CC BY is an
  attribution licence, which means the obligation does not disappear once
  training finishes: the dataset has to be credited wherever the model is, so
  this script records it in the model card and the detector reprints it with
  every finding.

The obligation that does not come from the data is Ultralytics'. Ultralytics
asserts AGPL-3.0 over models produced by its training code, not merely over the
code. A checkpoint fine-tuned here is therefore AGPL-3.0 regardless of the
dataset licence, which is why the detector sits at :attr:`Tier.COPYLEFT` and why
this script refuses to run below that tier.

Acquiring the data
------------------

1. Pick a set on Roboflow Universe whose licence line reads exactly
   ``CC BY 4.0``. Check the licence on the *version* page, not the project
   page; they can differ.
2. Export it in ``YOLOv8`` format. The export gives a directory with
   ``data.yaml`` plus ``train/``, ``valid/`` and ``test/``.
3. Record the citation Roboflow generates. It is the attribution CC BY requires
   and it is a required argument here, not an optional one.
4. Run::

     python -m vitruve.derm.train \\
        --data /path/to/export/data.yaml \\
        --dataset-name "acne-detection-xyz" \\
        --dataset-license CC-BY-4.0 \\
        --dataset-url https://universe.roboflow.com/... \\
        --dataset-citation "..." \\
        --license-tier copyleft \\
        --out runs/acne-v1

Reproducibility
---------------

Seed, image size, epochs, the resolved dataset digest and the exact package
versions all land in the model card next to the weights. A model card that
records the metrics but not the data hash cannot support the claim that a rerun
produced the same detector, and that claim is the only reason to write one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from ..models.licensing import YOLO_DERM_SEG, LicenseViolation, Tier, require

#: Dataset licences under which a resulting checkpoint may be redistributed.
#: CC BY carries an attribution obligation rather than a use restriction, which
#: is why it is the only one here that is not also a refusal.
COMMERCIALLY_CLEAN_LICENCES: frozenset[str] = frozenset(
    {"CC-BY-4.0", "CC-BY-3.0", "MIT", "APACHE-2.0", "BSD-3-CLAUSE", "CC0-1.0", "PUBLIC-DOMAIN"}
)

#: Licences that permit training but not shipping. ACNE04's terms are here.
RESEARCH_ONLY_LICENCES: frozenset[str] = frozenset(
    {"CC-BY-NC-4.0", "CC-BY-NC-SA-4.0", "ACADEMIC-ONLY", "RESEARCH-ONLY", "NON-COMMERCIAL"}
)

TIERS: dict[str, Tier] = {t.name.lower(): t for t in Tier}


@dataclass
class TrainConfig:
    """Everything that changes the resulting weights, in one hashable place."""

    data: str
    dataset_name: str
    dataset_license: str
    dataset_url: str
    dataset_citation: str
    out: str
    base_weights: str = "yolov8n-seg.pt"
    epochs: int = 60
    imgsz: int = 640
    batch: int = 16
    seed: int = 0
    device: str = "cpu"
    lr0: float = 0.01
    patience: int = 20
    task: str = "segment"
    license_tier: str = "copyleft"
    accept_noncommercial: bool = False

    def normalised_licence(self) -> str:
        return self.dataset_license.strip().upper().replace(" ", "-")

    def digest(self) -> str:
        """Twelve hex characters over the config, so two runs can be compared."""
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m vitruve.derm.train",
        description=(
            "Fine-tune a YOLO acne lesion detector against a CC BY 4.0 dataset and "
            "write a model card that records the licence."
        ),
    )
    p.add_argument("--data", required=True, help="path to the YOLO data.yaml export")
    p.add_argument("--dataset-name", required=True, help="dataset name, for the model card")
    p.add_argument(
        "--dataset-license",
        required=True,
        help="dataset licence id, e.g. CC-BY-4.0. Refused unless it permits the "
        "intended use; there is no default because guessing it is the failure mode",
    )
    p.add_argument("--dataset-url", required=True, help="canonical dataset URL")
    p.add_argument(
        "--dataset-citation",
        required=True,
        help="attribution text. CC BY requires it and it travels with the weights",
    )
    p.add_argument("--out", required=True, help="output directory for weights and model card")
    p.add_argument("--base-weights", default="yolov8n-seg.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu, mps, or a cuda index")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--task", choices=("segment", "detect"), default="segment")
    p.add_argument(
        "--license-tier",
        choices=sorted(TIERS),
        default="copyleft",
        help="tier the caller accepts. Training with the Ultralytics trainer "
        "produces an AGPL-3.0 model, so anything below copyleft is refused",
    )
    p.add_argument(
        "--accept-noncommercial",
        action="store_true",
        help="permit a research-only dataset such as ACNE04. The resulting weights "
        "may not be redistributed and the model card says so",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve the configuration, check the licences and print the model "
        "card without training",
    )
    return p


def check_licences(cfg: TrainConfig) -> list[str]:
    """Raise on a licence conflict; return the obligations that survive it."""
    tier = TIERS[cfg.license_tier]
    require(YOLO_DERM_SEG, tier)

    licence = cfg.normalised_licence()
    obligations = [
        "the Ultralytics trainer asserts AGPL-3.0 over the model it produces, so "
        "these weights are AGPL-3.0 whatever the dataset licence says",
    ]
    if licence in COMMERCIALLY_CLEAN_LICENCES:
        if licence.startswith("CC-BY"):
            obligations.append(
                f"{licence} requires attribution wherever the model is used: "
                f"{cfg.dataset_citation}"
            )
    elif licence in RESEARCH_ONLY_LICENCES:
        if not cfg.accept_noncommercial:
            raise LicenseViolation(
                f"dataset {cfg.dataset_name} is {licence}, which does not permit "
                "shipping the resulting weights. Re-run with --accept-noncommercial "
                "to train anyway, and expect a model card that says the weights "
                "cannot be redistributed."
            )
        obligations.append(
            f"{licence}: these weights are derived from a research-only dataset and "
            "may not be redistributed or used commercially"
        )
    else:
        raise LicenseViolation(
            f"unrecognised dataset licence {licence!r}. Add it to "
            "COMMERCIALLY_CLEAN_LICENCES or RESEARCH_ONLY_LICENCES in "
            "vitruve/derm/train.py after reading the actual terms; this script will "
            "not guess what a licence permits."
        )
    return obligations


def dataset_digest(data_yaml: str) -> str:
    """Hash the data.yaml and the file listing of each split.

    Hashing every image would be better and is far too slow to do on every run,
    so this hashes the manifest: the split paths and the sorted label filenames.
    It catches a re-export, a changed split and a dropped class, which are the
    changes that actually happen between two runs someone claims are the same.
    """
    path = Path(data_yaml)
    h = hashlib.sha256()
    if not path.exists():
        return "missing"
    h.update(path.read_bytes())
    root = path.parent
    for split in ("train", "valid", "test"):
        labels = root / split / "labels"
        if not labels.is_dir():
            continue
        names = sorted(p.name for p in labels.iterdir() if p.is_file())
        h.update(split.encode())
        h.update(str(len(names)).encode())
        for name in names:
            h.update(name.encode())
    return h.hexdigest()[:16]


def model_card(
    cfg: TrainConfig,
    obligations: Sequence[str],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The record that has to exist for the weights to be usable by anyone else."""
    return {
        "model": "vitruve acne lesion detector",
        "task": cfg.task,
        "base_weights": cfg.base_weights,
        "config_digest": cfg.digest(),
        "dataset": {
            "name": cfg.dataset_name,
            "license": cfg.normalised_licence(),
            "url": cfg.dataset_url,
            "citation": cfg.dataset_citation,
            "digest": dataset_digest(cfg.data),
        },
        "license": {
            "model_license": YOLO_DERM_SEG.license_id,
            "tier": YOLO_DERM_SEG.tier.name.lower(),
            "obligations": list(obligations),
            "provenance": YOLO_DERM_SEG.describe(),
        },
        "training": {
            "epochs": cfg.epochs,
            "imgsz": cfg.imgsz,
            "batch": cfg.batch,
            "lr0": cfg.lr0,
            "seed": cfg.seed,
            "device": cfg.device,
            "patience": cfg.patience,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "grading_convention": (
            "Severity thresholds applied downstream follow Hayashi et al. (2008): "
            "inflammatory lesions counted on one half of the face, 0-5 mild, 6-20 "
            "moderate, 21-50 severe, above 50 very severe. A detector inherits its "
            "training set's definition of a countable lesion, so the grade is "
            "calibrated on that convention."
        ),
        "evaluation": metrics or {},
        "known_gaps": [
            "no stratified evaluation by skin tone is recorded unless one was run "
            "and added here; detector recall on darker skin is the documented weak "
            "point of lesion detection and an unstratified mAP hides it",
            "the dataset's labelling of what counts as a lesion is not the same as "
            "a clinician's count, so the absolute count is not comparable to a "
            "published lesion count without a calibration study",
        ],
    }


def render_card(card: dict[str, Any]) -> str:
    """Markdown next to the JSON, because the JSON is not what a human reads."""
    d, lic, tr = card["dataset"], card["license"], card["training"]
    lines = [
        f"# {card['model']}",
        "",
        f"Task: `{card['task']}`  |  base weights: `{card['base_weights']}`  |  "
        f"config digest: `{card['config_digest']}`",
        "",
        "## Data",
        "",
        f"- Dataset: **{d['name']}** ({d['license']})",
        f"- URL: {d['url']}",
        f"- Manifest digest: `{d['digest']}`",
        f"- Attribution: {d['citation']}",
        "",
        "## Licence",
        "",
        f"- Model licence: **{lic['model_license']}** (tier `{lic['tier']}`)",
        f"- Provenance: {lic['provenance']}",
        "",
    ]
    lines += [f"- {o}" for o in lic["obligations"]]
    lines += [
        "",
        "## Training",
        "",
        f"- epochs {tr['epochs']}, imgsz {tr['imgsz']}, batch {tr['batch']}, "
        f"lr0 {tr['lr0']}, seed {tr['seed']}, device {tr['device']}",
        f"- python {card['environment']['python']} on {card['environment']['platform']}",
        "",
        "## Grading convention",
        "",
        card["grading_convention"],
        "",
        "## Evaluation",
        "",
        "```json",
        json.dumps(card["evaluation"], indent=2, sort_keys=True),
        "```",
        "",
        "## Known gaps",
        "",
    ]
    lines += [f"- {g}" for g in card["known_gaps"]]
    return "\n".join(lines) + "\n"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def train(cfg: TrainConfig, obligations: Sequence[str]) -> dict[str, Any]:
    """Run the fine-tune. Imports ultralytics only after the licence check."""
    from ultralytics import YOLO

    seed_everything(cfg.seed)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(cfg.base_weights)
    results = model.train(
        data=cfg.data,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        seed=cfg.seed,
        device=cfg.device,
        lr0=cfg.lr0,
        patience=cfg.patience,
        project=str(out),
        name="train",
        exist_ok=True,
        deterministic=True,
        plots=False,
    )
    metrics = {}
    box = getattr(getattr(results, "box", None), "map50", None)
    if box is not None:
        metrics["box_map50"] = float(box)
        metrics["box_map50_95"] = float(results.box.map)
    seg = getattr(getattr(results, "seg", None), "map50", None)
    if seg is not None:
        metrics["mask_map50"] = float(seg)
        metrics["mask_map50_95"] = float(results.seg.map)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = TrainConfig(
        data=args.data,
        dataset_name=args.dataset_name,
        dataset_license=args.dataset_license,
        dataset_url=args.dataset_url,
        dataset_citation=args.dataset_citation,
        out=args.out,
        base_weights=args.base_weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        device=args.device,
        lr0=args.lr0,
        patience=args.patience,
        task=args.task,
        license_tier=args.license_tier,
        accept_noncommercial=args.accept_noncommercial,
    )
    try:
        obligations = check_licences(cfg)
    except LicenseViolation as exc:
        # A licence refusal is an expected outcome of this script, not a crash,
        # and a traceback would bury the one sentence the user has to read.
        print(f"refused: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(render_card(model_card(cfg, obligations)))
        return 0

    metrics = train(cfg, obligations)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    card = model_card(cfg, obligations, metrics)
    (out / "model_card.json").write_text(json.dumps(card, indent=2, sort_keys=True))
    (out / "MODEL_CARD.md").write_text(render_card(card))
    print(render_card(card))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
