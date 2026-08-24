"""Annotated images, drawn from the covariances rather than from the points.

The one thing an overlay in this project has to show is the thing every other
face-analysis overlay hides: a landmark is not a point. A heatmap model returns
a probability field, and a jaw-contour point is well localised across the
contour and badly localised along it. Drawn as a dot, that landmark looks as
certain as the pupil next to it. Drawn as its 95% ellipse, it is visibly a
smear lying along the jawline, and the reader can see for themselves why the
measurement that depends on it carries the interval it carries.

So every landmark is drawn as an ellipse. The dot is the centre of the ellipse
and never appears without it.

One overlay per measurement group, not per measurement. A face carrying
forty-five annotated segments is a decoration; a face carrying the four
segments of the periocular group is a diagram.

Pillow only. Matplotlib would drag in a font cache, a backend selection and a
figure lifecycle for what is a few dozen polygons.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any, Collection, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from ..measure.evaluate import Measured
from ..measure.registry import BY_ID
from .model import MeasurementGroup, OverlayImage


def chi2_radius(p: float = 0.95) -> float:
    """Mahalanobis radius enclosing ``p`` of a two-dimensional Gaussian.

    The chi-squared quantile has a closed form at two degrees of freedom, so
    the 95% ellipse costs a logarithm rather than a scipy dependency.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {p}")
    return math.sqrt(-2.0 * math.log(1.0 - p))


#: Restrained palette. The photograph carries the tone; the annotation is an
#: instrument laid over it, so it is drawn in two greys and one accent.
INK = (24, 22, 20, 235)
CHALK = (238, 235, 228, 200)
GHOST = (238, 235, 228, 90)
ACCENT = (200, 92, 42, 255)
ACCENT_FILL = (200, 92, 42, 60)
HORIZON = (120, 196, 214, 190)


@dataclass(frozen=True)
class PlottedLandmark:
    """One landmark in image pixels, with the covariance that placed it.

    ``cov`` is the 2x2 positional covariance in square pixels, in image axes.
    ``None`` means the backend supplied no uncertainty, which is drawn as a
    hollow marker rather than as a small ellipse: an unknown uncertainty must
    not look like a confident one.
    """

    name: str
    x: float
    y: float
    cov: NDArray[np.float64] | None = None


def ellipse_polygon(
    lm: PlottedLandmark, *, confidence: float = 0.95, n: int = 72
) -> list[tuple[float, float]]:
    """The confidence ellipse of a landmark, as a polygon.

    Pillow draws axis-aligned ellipses only, and an axis-aligned ellipse would
    throw away the orientation, which is the whole point of keeping the
    covariance.
    """
    if lm.cov is None:
        raise ValueError(f"{lm.name} has no covariance to draw")
    cov = np.asarray(lm.cov, dtype=float).reshape(2, 2)
    vals, vecs = np.linalg.eigh((cov + cov.T) / 2.0)
    vals = np.clip(vals, 0.0, None)
    k = chi2_radius(confidence)
    a, b = k * math.sqrt(float(vals[1])), k * math.sqrt(float(vals[0]))
    major, minor = vecs[:, 1], vecs[:, 0]
    pts = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        dx = a * math.cos(t) * major[0] + b * math.sin(t) * minor[0]
        dy = a * math.cos(t) * major[1] + b * math.sin(t) * minor[1]
        pts.append((lm.x + dx, lm.y + dy))
    return pts


def anisotropy(lm: PlottedLandmark) -> float:
    """Ratio of the long to the short axis of a landmark's uncertainty.

    A value near 1 is a round blob. A jaw-contour point runs to 4 or more, and
    that number is worth printing next to the picture, because it says the
    error is a direction and not a magnitude.
    """
    if lm.cov is None:
        return float("nan")
    cov = np.asarray(lm.cov, dtype=float).reshape(2, 2)
    vals = np.clip(np.linalg.eigvalsh((cov + cov.T) / 2.0), 1e-12, None)
    return float(math.sqrt(vals[1] / vals[0]))


# ---------------------------------------------------------------------------
# Which points a measurement actually joins
# ---------------------------------------------------------------------------

#: Formula nodes that join two named points into something a reader would draw
#: as a line. Read off the serialised form so this stays outside the algebra's
#: internals: ``to_dict`` is part of the core contract, the node classes are
#: not.
_PAIR_OPS = {"dist": ("a", "b"), "vec": ("a", "b"), "signed_tilt": ("a", "b"),
             "proj_length": ("a", "b")}
_TRIPLE_OPS = {"angle_at": ("a", "vertex", "c")}


def _walk(node: Any, out: list[tuple[str, str]]) -> None:
    if not isinstance(node, Mapping):
        return
    op = node.get("op")
    if op in _PAIR_OPS:
        a, b = (node.get(k) for k in _PAIR_OPS[op])
        if (
            isinstance(a, Mapping)
            and isinstance(b, Mapping)
            and a.get("op") == "pt"
            and b.get("op") == "pt"
        ):
            out.append((a["name"], b["name"]))
    if op in _TRIPLE_OPS:
        a, v, c = (node.get(k) for k in _TRIPLE_OPS[op])
        names = [n.get("name") for n in (a, v, c) if isinstance(n, Mapping) and n.get("op") == "pt"]
        if len(names) == 3:
            out.append((names[0], names[1]))
            out.append((names[1], names[2]))
    for value in node.values():
        if isinstance(value, Mapping):
            _walk(value, out)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _walk(item, out)


def segments_for(spec_id: str) -> tuple[tuple[str, str], ...]:
    """The point pairs a measurement joins, for drawing.

    Returns nothing for a measurement whose formula joins synthesised points
    such as a midline, because a line drawn to a point the reader cannot see is
    worse than no line.
    """
    spec = BY_ID.get(spec_id)
    if spec is None:
        return ()
    found: list[tuple[str, str]] = []
    _walk(spec.formula.to_dict(), found)
    seen: list[tuple[str, str]] = []
    for pair in found:
        if pair not in seen and tuple(reversed(pair)) not in seen:
            seen.append(pair)
    return tuple(seen)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - Pillow below 10.1
        return ImageFont.load_default()


def _dashed(
    draw: ImageDraw.ImageDraw,
    p0: tuple[float, float],
    p1: tuple[float, float],
    colour: tuple[int, int, int, int],
    *,
    width: int = 1,
    dash: float = 9.0,
    gap: float = 7.0,
) -> None:
    (x0, y0), (x1, y1) = p0, p1
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    t = 0.0
    while t < length:
        s = min(t + dash, length)
        draw.line([(x0 + ux * t, y0 + uy * t), (x0 + ux * s, y0 + uy * s)], fill=colour, width=width)
        t = s + gap


def _horizon(
    draw: ImageDraw.ImageDraw,
    size: tuple[int, int],
    anchor: tuple[float, float],
    roll_deg: float,
) -> None:
    """A level line and the estimated roll, drawn against each other.

    The roll line alone says nothing. Drawn across the level line it shows the
    angle the report keeps warning about, and for a canthal tilt whose normal
    range is four to eight degrees, three degrees of visible camera roll is the
    entire finding.
    """
    w, h = size
    cx, cy = anchor
    reach = float(w + h)
    _dashed(draw, (cx - reach, cy), (cx + reach, cy), GHOST, width=1)
    # Image y grows downward, so a subject rolled anticlockwise in the
    # canonical frame tips downward to the right on the screen.
    a = -math.radians(roll_deg)
    dx, dy = math.cos(a) * reach, math.sin(a) * reach
    draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=HORIZON, width=2)


def _caption_bar(img: Image.Image, title: str, subtitle: str) -> Image.Image:
    """A strip under the picture carrying the title and the roll readout."""
    w, h = img.size
    bar = max(30, int(h * 0.075))
    out = Image.new("RGB", (w, h + bar), (18, 17, 15))
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    pad = max(6, w // 90)
    draw.text((pad, h + bar // 2), title, font=_font(max(11, bar // 3)),
              fill=(238, 235, 228), anchor="lm")
    if subtitle:
        draw.text((w - pad, h + bar // 2), subtitle, font=_font(max(10, bar // 3 - 1)),
                  fill=(158, 152, 142), anchor="rm")
    return out


def render_group_overlay(
    image: Image.Image,
    landmarks: Sequence[PlottedLandmark],
    *,
    highlight: Collection[str] = (),
    segments: Iterable[tuple[str, str]] = (),
    roll_deg: float = 0.0,
    title: str = "",
    confidence: float = 0.95,
    label_highlighted: bool = True,
) -> Image.Image:
    """Draw one group's landmarks, their uncertainty, and the roll reference."""
    base = image.convert("RGB")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    by_name = {lm.name: lm for lm in landmarks}
    hot = set(highlight)

    focus = [by_name[n] for n in hot if n in by_name] or list(landmarks)
    anchor = (
        float(np.mean([lm.x for lm in focus])),
        float(np.mean([lm.y for lm in focus])),
    )
    _horizon(draw, base.size, anchor, roll_deg)

    for a, b in segments:
        if a in by_name and b in by_name:
            draw.line(
                [(by_name[a].x, by_name[a].y), (by_name[b].x, by_name[b].y)],
                fill=ACCENT,
                width=2,
            )

    # The dot is kept small on purpose. It marks the centre of the ellipse and
    # it must not grow large enough to hide it, which is what happens on every
    # overlay that draws landmarks as discs.
    radius = max(1.2, min(base.size) / 460.0)
    for lm in landmarks:
        on = lm.name in hot or not hot
        if lm.cov is not None:
            poly = ellipse_polygon(lm, confidence=confidence)
            draw.polygon(
                poly,
                outline=ACCENT if on else CHALK,
                fill=ACCENT_FILL if on else None,
                width=2 if on else 1,
            )
        else:
            r = radius * 3
            draw.ellipse([lm.x - r, lm.y - r, lm.x + r, lm.y + r],
                         outline=CHALK, width=1)
        r = radius if on else radius * 0.8
        draw.ellipse([lm.x - r, lm.y - r, lm.x + r, lm.y + r],
                     fill=ACCENT if on else CHALK)

    if label_highlighted and hot:
        # Labels go in two columns with leader lines back to their points,
        # rather than beside the points. Seven periocular landmarks inside
        # forty pixels of each other cannot each carry a label in place, and
        # the version that tries produces a legible face under an illegible
        # pile of words.
        size = max(10, int(min(base.size) / 52))
        font = _font(size)
        pad = max(8.0, radius * 6)
        line_h = size * 1.3
        marked = sorted(
            (by_name[n] for n in hot if n in by_name), key=lambda p: (p.y, p.x)
        )
        left_col = max(4.0, min(p.x for p in marked) - pad * 2)
        right_col = min(base.size[0] - 4.0, max(p.x for p in marked) + pad * 2)
        taken: dict[str, list[float]] = {"l": [], "r": []}
        for lm in marked:
            side = "l" if lm.x < anchor[0] else "r"
            y = lm.y
            while any(abs(y - t) < line_h for t in taken[side]):
                y += line_h
            taken[side].append(y)
            x = left_col if side == "l" else right_col
            draw.line([(lm.x, lm.y), (x, y)], fill=GHOST, width=1)
            tip = (x - 4.0, y) if side == "l" else (x + 4.0, y)
            anchor_at = "rm" if side == "l" else "lm"
            draw.text((tip[0] + 1, tip[1] + 1), lm.name, font=font,
                      fill=(0, 0, 0, 170), anchor=anchor_at)
            draw.text(tip, lm.name, font=font, fill=CHALK, anchor=anchor_at)

    out = Image.alpha_composite(base.convert("RGBA"), layer).convert("RGB")
    roll = f"estimated roll {roll_deg:+.1f} deg, against the dashed level line"
    return _caption_bar(out, title, roll)


def to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def overlay_caption(
    group_title: str, landmarks: Sequence[PlottedLandmark], confidence: float = 0.95
) -> str:
    """The sentence under the picture, which explains the ellipses."""
    drawn = [lm for lm in landmarks if lm.cov is not None]
    lead = (
        f"{group_title}: every landmark is drawn as the ellipse containing "
        f"{confidence * 100:.0f}% of where the model believes the point lies, "
        "so an elongated ellipse is a point that is well placed across one "
        "direction and poorly placed along the other."
    )
    if not drawn:
        return (
            f"{lead} No covariance was supplied for this group, so the points "
            "are drawn hollow rather than as ellipses."
        )
    worst = max(drawn, key=anisotropy)
    return (
        f"{lead} The most directional here is {worst.name}, whose long axis is "
        f"{anisotropy(worst):.1f} times its short one."
    )


def overlays_for_groups(
    image: Image.Image,
    landmarks: Sequence[PlottedLandmark],
    groups: Sequence[MeasurementGroup],
    *,
    roll_deg: float = 0.0,
    confidence: float = 0.95,
) -> tuple[OverlayImage, ...]:
    """One annotated image per measurement group.

    Groups whose landmarks the image does not carry are skipped rather than
    drawn empty, which is what happens to the profile group when only a frontal
    photograph was supplied.
    """
    present = {lm.name for lm in landmarks}
    out: list[OverlayImage] = []
    for group in groups:
        wanted: set[str] = set()
        segments: list[tuple[str, str]] = []
        for m in group.measurements:
            wanted.update(m.landmarks_used)
            segments.extend(segments_for(m.spec_id))
        hot = wanted & present
        if not hot:
            continue
        segments = [s for s in dict.fromkeys(segments) if set(s) <= hot]
        img = render_group_overlay(
            image,
            landmarks,
            highlight=hot,
            segments=segments,
            roll_deg=roll_deg,
            title=group.region.title,
            confidence=confidence,
        )
        out.append(
            OverlayImage(
                region=group.region.key,
                title=group.region.title,
                caption=overlay_caption(
                    group.region.title,
                    [lm for lm in landmarks if lm.name in hot],
                    confidence,
                ),
                png=to_png(img),
            )
        )
    return tuple(out)


def landmarks_from(
    coords: Mapping[str, tuple[float, float]],
    covariances: Mapping[str, NDArray[np.float64]] | None = None,
) -> tuple[PlottedLandmark, ...]:
    """Build the plot input from two plain mappings.

    The pipeline holds a ``PointSet`` and a ``LandmarkUncertainty``; the report
    layer takes names and pixels, so that a test can draw an overlay without
    constructing either.
    """
    cov = covariances or {}
    return tuple(
        PlottedLandmark(name=n, x=float(xy[0]), y=float(xy[1]), cov=cov.get(n))
        for n, xy in coords.items()
    )


def used_landmarks(measurements: Sequence[Measured]) -> tuple[str, ...]:
    names: list[str] = []
    for m in measurements:
        for n in m.landmarks_used:
            if n not in names:
                names.append(n)
    return tuple(names)
