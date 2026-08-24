#!/usr/bin/env python3
"""Draw Faciometry's icon and compile it to an .icns.

The mark is the two figures Leonardo drew around the man: a circle and a
square, sharing a base line, with the median that every craniofacial
measurement in this tool is taken against. Nothing else. It is drawn rather
than exported from a design file so that the icon is reproducible from the
repository and reviewable as text.

Two decisions worth stating:

* Each size is drawn from scratch at 4x and downsampled, instead of resampling
  one 1024px master. A hairline that is right at 1024 disappears at 16, so the
  stroke is a fraction of the canvas and gets a floor of one device pixel.
* The ground is an opaque rounded rectangle on Apple's icon grid (824/1024
  content, 185/1024 corner radius) rather than a full-bleed square, because a
  full-bleed square reads as a foreign object in the Dock.

Usage: make_icon.py OUT.icns [--png OUT.png]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

# The report's palette. Ivory ground, ink mark, one warm accent used at a
# weight low enough that it reads as paper rather than as colour.
IVORY = (243, 240, 232, 255)
INK = (26, 25, 22, 255)
RULE = (150, 145, 133, 255)

SS = 4  # supersample factor

# Apple's icon grid, as fractions of the full canvas.
CONTENT = 824 / 1024
RADIUS = 185 / 1024


def draw(size: int) -> Image.Image:
    """One icon at one size, drawn at SSx and reduced."""
    n = size * SS
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    inset = n * (1 - CONTENT) / 2
    box = (inset, inset, n - inset, n - inset)
    side = box[2] - box[0]
    d.rounded_rectangle(box, radius=n * RADIUS, fill=IVORY)

    # Stroke weights. A floor of one *device* pixel (SS units) keeps the 16px
    # icon from rendering as an empty ivory tile.
    hair = max(SS, round(side * 0.0105))
    fine = max(SS, round(side * 0.0065))

    cx = box[0] + side / 2

    # The circle, and inside it the square. They share a base line and not a
    # centre, which is the offset in the original drawing; keeping the square
    # clearly smaller is what stops the pair reading as one thick ring.
    r = side * 0.345
    base = box[1] + side * 0.86
    cy = base - r
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=INK, width=hair)

    sq = side * 0.46
    d.rectangle((cx - sq / 2, base - sq, cx + sq / 2, base), outline=INK, width=hair)

    # The median. Every bilateral measurement in the catalogue is signed
    # against this line, so it is the one thing in the mark that is not
    # Leonardo's. It runs the height of the circle and no further.
    d.line([(cx, cy - r), (cx, cy + r)], fill=RULE, width=fine)

    return img.resize((size, size), Image.LANCZOS)


def build(icns: Path, png: Path | None) -> None:
    if png is not None:
        png.parent.mkdir(parents=True, exist_ok=True)
        draw(1024).save(png)

    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "Faciometry.iconset"
        iconset.mkdir()
        # iconutil insists on exactly these names.
        for base in (16, 32, 128, 256, 512):
            draw(base).save(iconset / f"icon_{base}x{base}.png")
            draw(base * 2).save(iconset / f"icon_{base}x{base}@2x.png")
        icns.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["iconutil", "--convert", "icns", "--output", str(icns), str(iconset)],
            check=True,
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("icns", type=Path)
    ap.add_argument("--png", type=Path, help="also write a 1024px PNG here")
    args = ap.parse_args(argv)
    build(args.icns, args.png)
    print(f"icon: {args.icns}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
