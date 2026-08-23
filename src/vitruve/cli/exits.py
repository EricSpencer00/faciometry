"""Exit codes, so a script can tell the four outcomes apart.

A withheld measurement is a successful run. A photograph the gate rejects
outright is not, and a caller that pipes Vitruve into something else needs to
know which happened without parsing prose.
"""

from __future__ import annotations

from enum import IntEnum


class Exit(IntEnum):
    OK = 0
    """The analysis ran. Some measurements may have been withheld, which is a
    result rather than a failure."""

    ERROR = 1
    """Something unexpected. A stack trace belongs on stderr with this code."""

    BAD_INPUT = 2
    """A file that is not there, an image that will not decode, a flag value
    outside its allowed set."""

    QUALITY_GATE = 3
    """The photograph did not clear the gate, so no measurement was reported.
    Head pose, blur, exposure, occlusion or subject distance."""

    LICENSE = 4
    """A backend was requested whose license exceeds the tier the user
    permitted. Raised before any weight is loaded."""
