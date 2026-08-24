"""The dermatological findings layer.

Two capabilities, at two different licence tiers, for one blunt reason: the
public data exists for one of them and does not exist for the other.

* **Acne** has real box annotations -- ACNE04, and several Roboflow Universe
  sets under CC BY 4.0 -- so it is detected, with YOLO instance segmentation
  over region crops. Ultralytics asserts AGPL-3.0 over models its trainer
  produces, so this capability sits at :attr:`Tier.COPYLEFT` and is off by
  default.
* **Erythema, periorbital hyperpigmentation and skin tone** have no usable
  annotated public data at all, so they are not detected. They are measured, as
  paired within-face contrasts in CIELAB, which needs no weights, carries no
  licence, and is more defensible than a model trained on data nobody can show
  you. This is the default path and it runs at :attr:`Tier.PERMISSIVE`.

Regions are derived geometrically from landmarks rather than from a face parser,
because every public face parser inherits CelebAMask-HQ's non-commercial terms.
Parser refinement is available to callers who accept that tier; the geometric
path works alone.

See ``docs/DERMATOLOGY.md`` for what each of those claims rests on.
"""

from __future__ import annotations

from .colorimetry import (
    PairedContrast,
    PigmentationContrast,
    RegionColour,
    SkinTone,
    WhiteBalance,
    erythema,
    grey_card_balance,
    individual_typology_angle,
    monk_tone,
    paired_contrast,
    periorbital_pigmentation,
    region_colour,
    rgb_to_lab,
    sample_regions,
    tone_from_regions,
)
from .findings import (
    DISCLAIMER,
    Finding,
    FindingKind,
    FindingSet,
    collect,
    contains_advice,
)
from .regions import (
    REFERENCE_PAIRS,
    Region,
    RegionPolygon,
    RegionSet,
    RegionUnavailable,
    build_regions,
    rasterize,
    refine_with_parser,
)

__all__ = [
    "DISCLAIMER",
    "REFERENCE_PAIRS",
    "Finding",
    "FindingKind",
    "FindingSet",
    "PairedContrast",
    "PigmentationContrast",
    "Region",
    "RegionColour",
    "RegionPolygon",
    "RegionSet",
    "RegionUnavailable",
    "SkinTone",
    "WhiteBalance",
    "build_regions",
    "collect",
    "contains_advice",
    "erythema",
    "grey_card_balance",
    "individual_typology_angle",
    "monk_tone",
    "paired_contrast",
    "periorbital_pigmentation",
    "rasterize",
    "refine_with_parser",
    "region_colour",
    "rgb_to_lab",
    "sample_regions",
    "tone_from_regions",
]
