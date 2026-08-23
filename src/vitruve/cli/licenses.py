"""What each license tier costs, printed before anyone commits to it.

`vitruve.models.licensing` holds the per-backend provenance and enforces the
tier at load time. This module is the human-facing half: it turns that data
into a page a user can read and decide from, and it adds the one fact the
provenance records cannot carry, which is what installing an extra does to the
license of the user's own deployment.

The extras descriptions are duplicated in ``pyproject.toml`` under
``[tool.vitruve.extras]``, because that is where a packager looks, and
``tests/unit/test_cli.py`` asserts the two copies agree. An obligation that
lives only in a comment is an obligation nobody reads.
"""

from __future__ import annotations

import textwrap
from ..models.licensing import CATALOGUE as PROVENANCES
from ..models.licensing import Provenance, Tier, available_at, obligations_at

#: Mirrors ``[tool.vitruve.extras]`` in pyproject.toml, checked by a test.
EXTRAS: dict[str, str] = {
    "permissive": (
        "OpenCV (Apache-2.0) and MediaPipe (Apache-2.0), including MediaPipe's "
        "bundled .task models. No copyleft and no field-of-use restriction. "
        "This is the default stack."
    ),
    "copyleft": (
        "Ultralytics (AGPL-3.0). Installing this extra makes the deployment "
        "AGPL-3.0: Ultralytics asserts AGPL-3.0 over the models its training "
        "code produces as well as over the code, so serving Vitruve over a "
        "network with this extra installed obliges you to release your "
        "corresponding source under AGPL-3.0, or to buy an Ultralytics "
        "Enterprise License."
    ),
    "api": (
        "FastAPI and Uvicorn (both MIT), for `vitruve serve`. No effect on the "
        "license of the deployment."
    ),
    "dev": "Test and lint tooling. Not installed at runtime.",
}

#: Which extras a tier expects to have installed.
TIER_EXTRAS: dict[Tier, tuple[str, ...]] = {
    Tier.PERMISSIVE: ("permissive",),
    Tier.COPYLEFT: ("permissive", "copyleft"),
    Tier.NONCOMMERCIAL: ("permissive", "copyleft"),
    Tier.UNLICENSED: ("permissive", "copyleft"),
}

#: What the tier does to the license of the thing the user ships.
TIER_EFFECT: dict[Tier, str] = {
    Tier.PERMISSIVE: (
        "Vitruve's own code stays Apache-2.0 and so does your deployment. "
        "Nothing you load imposes a copyleft or a field-of-use term."
    ),
    Tier.COPYLEFT: (
        "Your deployment becomes AGPL-3.0. Section 13 of the AGPL treats "
        "network use as distribution, so a Vitruve instance other people can "
        "reach obliges you to offer them the corresponding source of the whole "
        "combined work. Vitruve's Apache-2.0 code is compatible with that in "
        "one direction only: Apache-2.0 code may be combined into an AGPL-3.0 "
        "work, and the result is AGPL-3.0."
    ),
    Tier.NONCOMMERCIAL: (
        "Your deployment is not commercially usable and the weights may not be "
        "redistributed. Research and personal use only, and the restriction "
        "comes from the training data or the morphable-model basis rather than "
        "from any code license, so it does not appear in a pip metadata field."
    ),
    Tier.UNLICENSED: (
        "There is no grant at all. All rights are reserved by the author, and "
        "using the weights is copyright infringement rather than a license "
        "breach. Vitruve will not load these; the tier exists so the catalogue "
        "can name them and say why."
    ),
}

#: Obligations that come from the training data rather than from a code
#: license, and that survive even at the permissive tier.
ATTRIBUTION_NOTES: tuple[str, ...] = (
    "MediaPipe's face landmarker is Apache-2.0, which requires you to keep the "
    "NOTICE file and state your changes.",
    "6DRepNet's weights descend from 300W-LP, which is derived from the Basel "
    "Face Model. The head-pose estimate is used as a cross-check only, and the "
    "BFM basis itself is never loaded or redistributed.",
)


def tier_from_string(name: str) -> Tier:
    """Parse a ``--license-tier`` value. Raises ``ValueError`` on anything else."""
    try:
        return Tier[name.strip().upper()]
    except KeyError:
        allowed = ", ".join(t.name.lower() for t in Tier)
        raise ValueError(f"unknown license tier {name!r}; choose one of {allowed}") from None


def _wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, width=88, initial_indent=indent, subsequent_indent=indent)


def _bullet(text: str, indent: str = "  ") -> str:
    return textwrap.fill(
        text, width=88, initial_indent=indent + "- ", subsequent_indent=indent + "  "
    )


def render(tier: Tier) -> str:
    """The full obligation page for one tier."""
    out: list[str] = []
    out.append(f"Vitruve license tier: {tier.name.lower()}")
    out.append("=" * 88)
    out.append("")
    out.append("Vitruve's own code is Apache-2.0. Model weights are not Vitruve's code,")
    out.append("and their obligations are the reason this command exists.")
    out.append("")
    out.append("EFFECT ON YOUR DEPLOYMENT")
    out.append(_wrap(TIER_EFFECT[tier]))
    out.append("")

    out.append("EXTRAS THIS TIER EXPECTS")
    for name in TIER_EXTRAS[tier]:
        out.append(f"  pip install 'vitruve[{name}]'")
        out.append(_wrap(EXTRAS[name], indent="    "))
    out.append("")

    permitted = available_at(tier)
    out.append(f"BACKENDS THIS TIER PERMITS ({len(permitted)})")
    for prov in permitted:
        out.append(f"  {prov.name}")
        out.append(f"    {prov.license_id} ({prov.tier.name.lower()})  {prov.source_url}")
        for inherited in prov.inherited_from:
            out.append(_wrap(f"inherits: {inherited}", indent="    "))
    out.append("")

    refused = tuple(p for p in PROVENANCES if p.tier > tier)
    if refused:
        out.append(f"BACKENDS THIS TIER REFUSES TO LOAD ({len(refused)})")
        for prov in refused:
            out.append(f"  {prov.name}: {prov.license_id} ({prov.tier.name.lower()})")
            if prov.note:
                out.append(_wrap(prov.note, indent="    "))
        out.append("")

    obligations = obligations_at(tier)
    out.append(f"OBLIGATIONS YOU TAKE ON ({len(obligations)})")
    if obligations:
        for line in obligations:
            out.append(_bullet(line))
    else:
        out.append("  None beyond keeping the Apache-2.0 notices.")
    out.append("")

    out.append("ATTRIBUTION THAT APPLIES AT EVERY TIER")
    for note in ATTRIBUTION_NOTES:
        out.append(_bullet(note))
    return "\n".join(out)


def refused_at(tier: Tier) -> tuple[Provenance, ...]:
    """Backends this tier will not load."""
    return tuple(p for p in PROVENANCES if p.tier > tier)


def run(tier: Tier) -> int:
    print(render(tier))
    return 0
