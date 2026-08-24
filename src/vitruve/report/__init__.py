"""Rendering: overlays, prose, and one self-contained HTML file.

Depends on ``core``, ``measure`` and ``norms``. It does not import
``vitruve.models`` or ``vitruve.pipeline``, so the renderer stays free of torch
and can be exercised from measurements built by hand in a test.

The entry point is :func:`vitruve.report.html.render`, which takes a
:class:`~vitruve.report.model.ReportInput` and returns a string.
"""

from __future__ import annotations

from .html import render, write
from .model import (
    REGIONS,
    MeasurementGroup,
    NormativeStratum,
    OverlayImage,
    QualityIssue,
    Region,
    ReportInput,
    niosh_stratum,
    region_of,
)
from .overlay import (
    PlottedLandmark,
    landmarks_from,
    overlays_for_groups,
    render_group_overlay,
    to_png,
)
from .prose import describe, report_text, summary

__all__ = [
    "REGIONS",
    "MeasurementGroup",
    "NormativeStratum",
    "OverlayImage",
    "PlottedLandmark",
    "QualityIssue",
    "Region",
    "ReportInput",
    "describe",
    "landmarks_from",
    "niosh_stratum",
    "overlays_for_groups",
    "region_of",
    "render",
    "render_group_overlay",
    "report_text",
    "summary",
    "to_png",
    "write",
]
