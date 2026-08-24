"""License tiers, enforced before weights load."""

from __future__ import annotations

import pytest

from faciometry.models.licensing import (
    CATALOGUE,
    MICA,
    SPIGA,
    STAR_LOSS,
    YOLO_DERM_SEG,
    YOLO_FACE,
    LicenseViolation,
    Tier,
    available_at,
    obligations_at,
    require,
)


def test_tiers_are_ordered():
    assert Tier.PERMISSIVE < Tier.COPYLEFT < Tier.NONCOMMERCIAL < Tier.UNLICENSED


def test_permissive_tier_permits_a_working_stack():
    """The default install must actually be able to run, or the permissive tier
    is decoration."""
    names = {p.name for p in available_at(Tier.PERMISSIVE)}
    assert any("YuNet" in n or "RetinaFace" in n for n in names)
    assert any("SPIGA" in n for n in names)
    assert any("MediaPipe" in n for n in names)
    assert any("6DRepNet" in n for n in names)


def test_ultralytics_lineage_is_copyleft_regardless_of_upstream_tags():
    """Ultralytics asserts AGPL-3.0 over models produced by its training code.
    Third-party checkpoints tagged MIT or Apache-2.0 are relabels that do not
    launder the obligation."""
    assert YOLO_FACE.tier is Tier.COPYLEFT
    assert any("AGPL" in s for s in YOLO_FACE.inherited_from)
    with pytest.raises(LicenseViolation, match="exceeds the permitted tier"):
        require(YOLO_FACE, Tier.PERMISSIVE)


def test_yolo_derm_is_permitted_once_copyleft_is_accepted():
    require(YOLO_DERM_SEG, Tier.COPYLEFT)


def test_unlicensed_is_refused_even_at_the_loosest_tier():
    """STAR has the best published landmark accuracy and no license file at all,
    so all rights are reserved."""
    assert STAR_LOSS.tier is Tier.UNLICENSED
    with pytest.raises(LicenseViolation):
        require(STAR_LOSS, Tier.NONCOMMERCIAL)


def test_basis_obligations_are_tracked_separately_from_code_licenses():
    """The surprises in this field come from training data and morphable-model
    bases, not from code licenses."""
    assert MICA.inherited_from
    assert any("FLAME" in s for s in MICA.inherited_from)


def test_permissive_obligations_disclose_inherited_terms():
    text = " ".join(obligations_at(Tier.PERMISSIVE))
    assert "300W-LP" in text or "Basel" in text


def test_error_message_names_the_escape_hatch():
    with pytest.raises(LicenseViolation) as exc:
        require(YOLO_FACE, Tier.PERMISSIVE)
    assert "--license-tier copyleft" in str(exc.value)


def test_every_backend_declares_a_source():
    for p in CATALOGUE:
        assert p.source_url.startswith("http"), p.name
        assert p.license_id, p.name
