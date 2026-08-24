"""Derive between-subject spreads from the NIOSH 2003 head-and-face survey.

NIOSH surveyed 3,997 US respirator users with calipers, recorded twenty facial
dimensions per subject, and released the per-subject data as a work of the US
Government -- public domain, no agreement, no attribution requirement. It is
the only openly redistributable source of direct craniofacial anthropometry at
this scale, and it measures exactly the dimensions this project needs,
including the two that photogrammetry cannot recover.

That last point is what makes it valuable here. NIOSH gives the *true*
between-person spread of bigonial and bizygomatic breadth, measured with
calipers on a living face. Faciometry compares its own photographic error against
that spread to decide whether a number is worth printing. Using a
photogrammetric study for the same purpose would be circular.

Two data quirks, both discovered by cross-checking rather than from the
documentation:

* Missing values are written ``-9,999`` -- with a thousands separator, inside a
  quoted CSV field. Parsing this naively yields facial dimensions of minus ten
  thousand millimetres.
* ``NECKCIRC`` was added part-way through collection, so its 45% missingness is
  structural rather than random.

Usage:  python scripts/build_niosh_norms.py <path-to-RespiratorUsersData-all-subjects.csv>
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

SEX = {"1": "male", "2": "female"}
RACE = {"1": "White", "2": "Black", "3": "Hispanic", "4": "Other"}
AGE = {"1": "17-29", "2": "30-44", "3": "45+"}

#: NIOSH column -> the Faciometry measurement it corresponds to. Only exact
#: correspondences are mapped; a NIOSH dimension whose landmark definition
#: differs from the catalogue's is left out rather than approximated.
COLUMN_TO_MEASUREMENT = {
    "INTPUPBR": "interpupillary_distance",
    "BIZYGOBR": "bizygomatic_width",
    "BIGONLBR": "bigonial_width",
    "NOSEBR": "nose_breadth",
    "LIPLTH": "mouth_width",
    "SUBNASAL": "nose_height",
    "MENSELL": "face_height_sellion_menton",
}

#: Ratios computed per subject before aggregating. Computing the ratio on each
#: subject and then taking its spread is not the same as dividing two spreads,
#: and it is the correct order: the numerator and denominator covary strongly
#: within a person, so dividing the marginal spreads would badly overstate how
#: much these ratios vary between people.
RATIOS = {
    "jaw_cheekbone_ratio": ("BIGONLBR", "BIZYGOBR"),
    "nose_mouth_width_ratio": ("NOSEBR", "LIPLTH"),
}


def _num(raw: str) -> float | None:
    s = raw.strip().strip('"').replace(",", "")
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return None if v <= -9000 else v


def summarise(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "sd": round(statistics.stdev(values), 4),
    }


def main(path: Path) -> None:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    rows = [r for r in rows if r.get("SUBNO", "").strip()]
    print(f"read {len(rows)} subjects from {path.name}", file=sys.stderr)

    out: dict[str, dict] = {}

    def collect(key: str, selector) -> None:
        buckets: dict[tuple[str, str], list[float]] = {}
        for r in rows:
            v = selector(r)
            if v is None:
                continue
            sex = SEX.get(r["SEX"].strip(), "unknown")
            race = RACE.get(r["RACEGRP"].strip(), "unknown")
            buckets.setdefault(("both", "pooled"), []).append(v)
            buckets.setdefault((sex, "pooled"), []).append(v)
            buckets.setdefault((sex, race), []).append(v)
        strata = {
            f"{s}|{race}": summarise(vals)
            for (s, race), vals in sorted(buckets.items())
            if len(vals) >= 30
        }
        out[key] = strata

    for col, mid in COLUMN_TO_MEASUREMENT.items():
        collect(mid, lambda r, c=col: _num(r[c]))

    for mid, (num_c, den_c) in RATIOS.items():
        def sel(r, n=num_c, d=den_c):
            a, b = _num(r[n]), _num(r[d])
            return a / b if a is not None and b else None

        collect(mid, sel)

    dest = Path("src/faciometry/norms/data/niosh2003.json")
    dest.write_text(
        json.dumps(
            {
                "source": "NIOSH head-and-face anthropometric survey of US respirator users, 2003",
                "citation": "Zhuang & Bradtmiller 2005; data released as CDC RD-10130-2020-0",
                "url": "https://www.cdc.gov/niosh/data/datasets/rd-10130-2020-0/default.html",
                "license": "US Government work, public domain",
                "method": "direct caliper anthropometry",
                "n_subjects": len(rows),
                "units": "mm, except ratios which are dimensionless",
                "caveat": (
                    "US respirator users, deliberately oversampled for racial "
                    "minorities and not reweighted here; strata are unbalanced, "
                    "so pooled figures are not population estimates"
                ),
                "strata": out,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {dest} with {len(out)} measurements", file=sys.stderr)
    for mid, strata in out.items():
        p = strata.get("both|pooled")
        if p:
            print(f"  {mid:32s} n={p['n']:5d} mean={p['mean']:9.3f} sd={p['sd']:7.3f} "
                  f"rsd={p['sd']/p['mean']:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
