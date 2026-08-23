"""Published between-subject spreads, from open-access sources only.

The numerator of every discriminability ratio comes from here. Each entry
records the spread of a measurement *between people*, so it can be compared
against how much the same measurement moves between photographs of one person.

Sourcing rules, applied strictly:

* Only openly licensed sources are transcribed. Farkas' 2005 international
  series is the canonical multi-ancestry table and it is paywalled, so it
  appears here only through CC-BY articles that reprint fragments of it.
* Creative Commons NonCommercial-NoDerivatives sources may be cited but their
  tables may not be shipped as a modified derivative, so those values are
  referenced in prose and not tabulated.
* Where sources disagree because they measured different things under the same
  name, the disagreement is recorded rather than averaged away.

That last point is not pedantry. Kramer et al. (2012) measured the facial
width-to-height ratio on the same 155 people two ways and got 2.01 from
photographs against 1.83 from 3D scans. That 0.18 gap is larger than every sex
difference reported anywhere in the fWHR literature. Pooling those numbers
would manufacture a finding out of a methods artifact.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Spread:
    """Between-subject spread for one measurement, from one population."""

    measurement_id: str
    population: str
    sex: str
    n: int
    mean: float
    sd: float
    unit: str
    source: str
    license: str
    caveat: str = ""

    @property
    def rsd(self) -> float:
        """Relative standard deviation. Meaningless for angles; use ``sd``."""
        return self.sd / self.mean if self.mean else float("nan")


CC_BY = "CC BY"

SPREADS: tuple[Spread, ...] = (
    # -- Interpupillary distance ------------------------------------------
    Spread("interpupillary_distance", "ANSUR 1988, pooled", "both", 3976, 63.36, 3.832, "mm",
           "Dodgson 2004 over ANSUR; ANSUR itself is a US Government work",
           "public domain (underlying survey)"),
    Spread("interpupillary_distance", "ANSUR 1988", "male", 1771, 64.67, 3.708, "mm",
           "Dodgson 2004 over ANSUR", "public domain (underlying survey)"),
    Spread("interpupillary_distance", "ANSUR 1988", "female", 2205, 62.31, 3.599, "mm",
           "Dodgson 2004 over ANSUR", "public domain (underlying survey)"),

    # -- Intercanthal and biocular ----------------------------------------
    Spread("intercanthal_width", "North American white (Farkas)", "male", 109, 33.3, 2.7, "mm",
           "Farkas 2005, reprinted in Al-Sebaei 2015 (PMC4369102)", CC_BY),
    Spread("intercanthal_width", "North American white (Farkas)", "female", 200, 31.8, 2.3, "mm",
           "Farkas 2005, reprinted in Al-Sebaei 2015 (PMC4369102)", CC_BY),
    Spread("intercanthal_width", "Hong Kong Chinese, 3dMD", "male", 51, 40.61, 4.91, "mm",
           "Jayaratne et al. 2013 (PMC3730197)", CC_BY),
    Spread("intercanthal_width", "Hong Kong Chinese, 3dMD", "female", 52, 38.27, 2.61, "mm",
           "Jayaratne et al. 2013 (PMC3730197)", CC_BY),
    Spread("biocular_width", "Hong Kong Chinese, 3dMD", "male", 51, 93.00, 5.56, "mm",
           "Jayaratne et al. 2013 (PMC3730197)", CC_BY),
    Spread("biocular_width", "Hong Kong Chinese, 3dMD", "female", 52, 88.39, 3.74, "mm",
           "Jayaratne et al. 2013 (PMC3730197)", CC_BY),

    # -- Periocular --------------------------------------------------------
    Spread("palpebral_fissure_width_r", "Hong Kong Chinese, 3dMD", "male", 51, 27.64, 1.67, "mm",
           "Jayaratne et al. 2013 (PMC3730197)", CC_BY),
    Spread("palpebral_fissure_height_r", "Hong Kong Chinese, 3dMD", "male", 51, 11.55, 1.05, "mm",
           "Jayaratne et al. 2013 (PMC3730197)", CC_BY),
    Spread("intercanthal_biocular_ratio", "Hong Kong Chinese, 3dMD", "female", 52, 43.29, 2.35,
           "percent", "Jayaratne et al. 2013 (PMC3730197)", CC_BY,
           "the male cell of the same table has SD 8.23 at a nearly identical "
           "mean, which is an outlier artifact rather than a population fact"),

    # -- Nose --------------------------------------------------------------
    Spread("nose_breadth", "North American white (Farkas)", "male", 109, 34.9, 2.1, "mm",
           "Farkas 2005, reprinted in Al-Sebaei 2015 (PMC4369102)", CC_BY),
    Spread("nose_breadth", "North American white (Farkas)", "female", 200, 31.4, 2.0, "mm",
           "Farkas 2005, reprinted in Al-Sebaei 2015 (PMC4369102)", CC_BY),
    Spread("nose_height", "North American white (Farkas)", "male", 109, 54.8, 3.3, "mm",
           "Farkas 2005, reprinted in Al-Sebaei 2015 (PMC4369102)", CC_BY),

    # -- Vertical thirds ---------------------------------------------------
    Spread("middle_third_height", "Saudi", "male", 75, 54.12, 4.34, "mm",
           "Al-Sebaei 2015 (PMC4369102)", CC_BY),
    Spread("lower_third_height", "Saudi", "male", 75, 65.02, 5.16, "mm",
           "Al-Sebaei 2015 (PMC4369102)", CC_BY),
    Spread("lower_third_height", "Saudi", "female", 93, 60.27, 4.62, "mm",
           "Al-Sebaei 2015 (PMC4369102)", CC_BY),

    # -- Angles ------------------------------------------------------------
    Spread("nasolabial_angle", "Turkish", "both", 96, 107.05, 8.45, "deg",
           "Celebi et al. 2013 (PMC3606791)", CC_BY,
           "the journal typeset decimal commas; 107,05 means 107.05"),
    Spread("nasolabial_angle", "Lebanese", "male", 99, 109.03, 12.05, "deg",
           "Saadeh et al. 2025 (PMC12228583)", CC_BY),
    Spread("nasofrontal_angle", "Lebanese", "male", 99, 130.89, 11.59, "deg",
           "Saadeh et al. 2025 (PMC12228583)", CC_BY),
    Spread("nasofrontal_angle", "Lebanese", "female", 106, 134.96, 9.19, "deg",
           "Saadeh et al. 2025 (PMC12228583)", CC_BY),
    Spread("gonial_angle_l", "Lebanese, Co-Go-Me on lateral cephalogram", "male", 99,
           120.98, 6.05, "deg", "Saadeh et al. 2025 (PMC12228583)", CC_BY,
           "cephalometric hard tissue, not the soft-tissue angle Vitruve measures; "
           "Ar-Go-Me, Co-Go-Me and Me-Go-Co differ by 5 to 15 degrees in the same jaw"),
    Spread("gonial_angle_l", "Nepalese, normodivergent", "both", 54, 124.06, 3.88, "deg",
           "Bajracharya et al. 2021 (PMC8673450)", CC_BY),
    Spread("canthal_tilt_l", "Caucasian, landmark-based", "female", 21, 8.50, 2.10, "deg",
           "Liu et al. 2023 (PMC10335162)", CC_BY,
           "attractiveness-selected celebrity sample, explicitly not a population norm"),
    Spread("canthal_tilt_l", "Caucasian, landmark-based", "male", 21, 6.51, 2.67, "deg",
           "Liu et al. 2023 (PMC10335162)", CC_BY,
           "attractiveness-selected celebrity sample"),

    # -- Facial width-to-height ratio -------------------------------------
    Spread("facial_width_height_ratio", "White, 2D photographs", "male", 2075, 1.94, 0.17,
           "ratio", "Kramer 2015 (PMC10430000)", "CC BY-NC",
           "cited, not shipped: NonCommercial licence"),
    Spread("facial_width_height_ratio", "White European, 2D photographs", "male", 66, 2.01, 0.16,
           "ratio", "Kramer et al. 2012 (PMC3413652)", CC_BY,
           "the same 66 men measured by 3D scan give 1.83, a gap larger than any "
           "published sex difference in this measurement"),
    Spread("facial_width_height_ratio", "White European, 3D scans", "male", 66, 1.83, 0.11,
           "ratio", "Kramer et al. 2012 (PMC3413652)", CC_BY),
)


#: Default relative spread for a craniofacial linear measurement with no
#: published table. Across every open source transcribed above, linear
#: measurements cluster tightly around a 6% relative standard deviation, so
#: this is an interpolation rather than a guess -- but it is still an
#: assumption, and measurements using it say so in the report.
DEFAULT_LINEAR_RSD = 0.06


def spreads_for(measurement_id: str) -> tuple[Spread, ...]:
    return tuple(s for s in SPREADS if s.measurement_id == measurement_id)


def representative_spread(measurement_id: str, unit: str) -> float | None:
    """A single between-subject spread to use as the discriminability numerator.

    Takes the largest-n open source, which is the least likely to be a
    small-sample artifact. Angles return an absolute spread in degrees; lengths
    and ratios return a relative one.
    """
    candidates = [s for s in spreads_for(measurement_id) if s.license.startswith("CC BY") or "public domain" in s.license]
    candidates = [s for s in candidates if not s.license.startswith("CC BY-NC")]
    if not candidates:
        return None
    best = max(candidates, key=lambda s: s.n)
    return best.sd if unit == "deg" else best.rsd
