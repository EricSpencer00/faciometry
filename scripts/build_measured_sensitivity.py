"""Regenerate the measured pose slopes from the evaluation harness output.

The a-priori sensitivities in `core/sensitivity.py` are a first-order
projection model. They are useful and they are wrong in specific, findable
ways: the sweep found 115 of 201 measurement-axis pairs where the declared
slope understated what was measured, including one that declared 0.010 per
degree of pitch against a measured 1.189.

So the model is the fallback and the measurement is the source of truth. This
script copies the sweep's output into the package so inference does not depend
on `evals/` being present.

Usage:  make evals && python scripts/build_measured_sensitivity.py
"""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

SOURCE = Path("evals/results/arm02_recommendations.csv")
DEST = Path("src/vitruve/measure/data/measured_sensitivity.json")


def _number(row: dict, key: str) -> float:
    try:
        value = float(row[key])
    except (TypeError, ValueError):
        return 0.0
    return value if value == value else 0.0


def main() -> None:
    rows = list(csv.DictReader(SOURCE.open()))
    slopes: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for row in rows:
        axis = row["axis"].strip()
        if axis in ("yaw", "pitch", "roll"):
            slopes[row["id"]][axis] = round(_number(row, "worst_measured"), 6)
    DEST.write_text(
        json.dumps(
            {
                "_comment": (
                    "Pose slopes measured by evals/arms/arm02_pose_sweep.py, in "
                    "fractional change per degree for lengths and ratios and "
                    "degrees per degree for angles. Generated, not hand-written."
                ),
                "source": str(SOURCE),
                "worst_of": (
                    "orthographic, perspective at 1.0 m and 0.5 m, and the "
                    "left/right asymmetry"
                ),
                "n_measurements": len(slopes),
                "slopes": {k: v for k, v in sorted(slopes.items())},
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {DEST} with {len(slopes)} measurements")


if __name__ == "__main__":
    main()
