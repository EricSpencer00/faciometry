"""The run manifest: does it round-trip, and does it keep the rules it enforces?

Two classes of assertion.

The mechanical one: what goes into ``run.json`` must come back out equal. A
record that cannot be read back is not a record. Rounding on the way out, a
tuple that becomes a list, an enum that becomes a string and never becomes an
enum again -- each of those quietly turns the accountability artifact into a
log file.

The load-bearing one: a withheld measurement's value is not in the file. Rule
three of the core contract says to print the reason and never the number, and a
manifest that stored the number would leave that rule one field access from
being broken by a renderer nobody reviewed.
"""

from __future__ import annotations

import json
import re

import pytest

from vitruve.core.sensitivity import discriminability
from vitruve.core.spec import Reportability, Unit, Verdict
from vitruve.measure.evaluate import Measured, Unavailable
from vitruve.measure.registry import BY_ID
from vitruve.models.licensing import SIXDREPNET, YOLO_FACE, YUNET, Tier, obligations_at
from vitruve.pipeline.manifest import (
    SCHEMA_VERSION,
    ImageRecord,
    MeasurementRecord,
    ModelRecord,
    RunManifest,
    StageTiming,
)


def a_measured(spec_id: str, *, reportability: Reportability, value: float = 32.5) -> Measured:
    spec = BY_ID[spec_id]
    return Measured(
        spec_id=spec.id,
        label=spec.label,
        unit=spec.unit,
        value=value,
        ci_low=value * 0.9,
        ci_high=value * 1.1,
        sd=value * 0.05,
        verdict=Verdict(reportability, ("because the test said so",)),
        discriminability=discriminability(
            between_subject_sd=0.06, pose_error=0.01, landmark_error=0.02
        ),
        formula_fingerprint=spec.fingerprint,
        landmarks_used=tuple(sorted(x.value for x in spec.landmarks)),
        n_samples=2048,
        n_valid=2040,
        scale_source="fused",
    )


def an_unavailable(spec_id: str) -> Unavailable:
    spec = BY_ID[spec_id]
    return Unavailable(spec.id, spec.label, ("crista_philtri_l", "crista_philtri_r"))


def a_manifest(tier: Tier = Tier.PERMISSIVE) -> RunManifest:
    measured = [
        a_measured("intercanthal_width", reportability=Reportability.REPORT),
        a_measured("canthal_tilt_r", reportability=Reportability.CAVEAT, value=4.9),
        a_measured("bizygomatic_width", reportability=Reportability.WITHHOLD, value=135.0),
    ]
    unavailable = [an_unavailable("philtrum_width")]
    return (
        RunManifest.begin(
            tier=tier,
            device="cpu",
            seed=7,
            n_samples=2048,
            declared_sex="female",
        )
        .with_(
            models=(
                ModelRecord.from_provenance("detector", YUNET, "a" * 64),
                ModelRecord.from_provenance("pose_estimator", SIXDREPNET, "b" * 64),
            ),
            images=(
                ImageRecord(
                    view="frontal",
                    sha256="c" * 64,
                    width=1600,
                    height=2000,
                    filename="front.png",
                    exif_focal_length_35mm=85.0,
                    exif_tags_present=("Make", "Model", "GPSLatitude"),
                ),
            ),
            stages=(StageTiming("frontal.ingest", 0.0123456789), StageTiming("measure", 1.5)),
            measurements=RunManifest.records_for(measured, unavailable),
            scale={"source": "fused", "mm_per_px": 0.4022, "notes": ["two cues"]},
            quality={"frontal": {"interocular_px": 156.0, "issues": []}},
        )
    )


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


def test_manifest_round_trips_through_json():
    manifest = a_manifest()
    assert RunManifest.from_json(manifest.to_json()) == manifest


def test_stage_timings_survive_serialisation():
    """Rounded at construction, not at serialisation, so a read-back compares
    equal to what was written."""
    manifest = a_manifest()
    restored = RunManifest.from_json(manifest.to_json())
    assert [s.seconds for s in restored.stages] == [s.seconds for s in manifest.stages]
    assert manifest.stages[0].seconds == pytest.approx(0.012346, abs=1e-9)


def test_manifest_writes_a_readable_run_json(tmp_path):
    path = a_manifest().write(tmp_path)
    assert path.name == "run.json"
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert RunManifest.from_dict(payload) == a_manifest().with_(
        created_utc=payload["created_utc"]
    )


def test_an_unreadable_schema_version_refuses_rather_than_guesses():
    payload = json.loads(a_manifest().to_json())
    payload["schema_version"] = SCHEMA_VERSION + 99
    with pytest.raises(ValueError, match="schema version"):
        RunManifest.from_dict(payload)


# ---------------------------------------------------------------------------
# What each record has to carry
# ---------------------------------------------------------------------------


def test_a_withheld_measurement_records_its_reasons_and_not_its_value():
    manifest = a_manifest()
    record = manifest.record("bizygomatic_width")
    assert record is not None
    assert record.outcome == "withheld"
    assert record.value is None
    assert record.ci_low is None and record.ci_high is None and record.sd is None
    assert record.reasons
    # The number must not survive anywhere in the serialised form either.
    assert "135.0" not in manifest.to_json()


def test_a_reported_measurement_keeps_its_interval():
    record = a_manifest().record("intercanthal_width")
    assert record.outcome == "measured"
    assert record.value is not None
    assert record.ci_low < record.value < record.ci_high
    assert record.sd is not None


def test_a_caveated_measurement_is_still_shown():
    record = a_manifest().record("canthal_tilt_r")
    assert record.outcome == "measured"
    assert record.reportability == Reportability.CAVEAT.value
    assert record.value is not None


def test_an_unavailable_measurement_names_what_was_missing():
    record = a_manifest().record("philtrum_width")
    assert record.outcome == "unavailable"
    assert record.value is None
    assert set(record.missing) == {"crista_philtri_l", "crista_philtri_r"}
    assert any("does not supply" in r for r in record.reasons)
    # Unavailable and withheld are different outcomes and must stay so: one
    # says the model could not see the landmark, the other says the number
    # would not mean anything.
    assert record.outcome != "withheld"


def test_every_measurement_carries_the_provenance_needed_to_audit_it():
    for record in a_manifest().measurements:
        spec = BY_ID[record.spec_id]
        assert record.formula_fingerprint == spec.fingerprint
        assert record.landmarks_used == tuple(sorted(x.value for x in spec.landmarks))
        assert record.view == spec.view.value
        assert record.evidence == spec.evidence.value
        assert Unit(record.unit) is spec.unit


def test_outcomes_counts_the_three_kinds_apart():
    assert a_manifest().outcomes() == {"measured": 2, "withheld": 1, "unavailable": 1}


# ---------------------------------------------------------------------------
# Run-level accountability
# ---------------------------------------------------------------------------


def test_the_tier_records_the_obligations_it_entails_not_just_its_name():
    permissive = a_manifest(Tier.PERMISSIVE)
    copyleft = a_manifest(Tier.COPYLEFT)
    assert permissive.license_tier == "permissive"
    assert copyleft.obligations == obligations_at(Tier.COPYLEFT)
    assert len(copyleft.obligations) > len(permissive.obligations)
    assert any(YOLO_FACE.license_id in o for o in copyleft.obligations)


def test_models_record_their_weight_hashes_and_inherited_obligations():
    manifest = a_manifest()
    by_role = {m.role: m for m in manifest.models}
    assert by_role["detector"].weights_sha256 == "a" * 64
    assert by_role["detector"].license_id == YUNET.license_id
    # 6DRepNet's obligation comes from its training data, not its code licence,
    # and that is exactly the kind of thing a manifest exists to carry.
    assert by_role["pose_estimator"].inherited_from == SIXDREPNET.inherited_from


def test_the_environment_is_pinned_enough_to_reproduce():
    manifest = a_manifest()
    assert manifest.git_sha is None or re.fullmatch(r"[0-9a-f]{40}", manifest.git_sha)
    assert manifest.python_version.count(".") >= 2
    assert manifest.platform and manifest.device
    assert manifest.seed == 7 and manifest.n_samples == 2048


def test_exif_is_recorded_as_tag_names_and_never_as_values():
    manifest = a_manifest()
    image = manifest.images[0]
    assert "GPSLatitude" in image.exif_tags_present
    assert image.exif_stripped
    # The name of the tag says what the file contained; the value would make
    # the manifest a copy of the metadata ingest exists to discard.
    payload = json.loads(manifest.to_json())
    assert payload["images"][0]["exif_tags_present"] == list(image.exif_tags_present)
    assert "exif_gps" not in json.dumps(payload).lower()


def test_declared_attributes_are_recorded_as_declared():
    manifest = a_manifest()
    assert manifest.declared_sex == "female"
    assert manifest.declared_ancestry is None


def test_the_manifest_has_no_aggregate_score_field():
    """Rule one, asserted where a score would be most tempting to add."""
    payload = json.loads(a_manifest().to_json())
    banned = ("score", "overall", "harmony", "rating", "rank", "attractiveness")
    flat = json.dumps(payload).lower()
    for word in banned:
        assert f'"{word}"' not in flat, f"a field named {word!r} appeared in the manifest"


def test_records_for_sorts_so_two_runs_diff_cleanly():
    records = a_manifest().measurements
    assert [r.spec_id for r in records] == sorted(r.spec_id for r in records)


def test_measurement_record_round_trips_on_its_own():
    for record in a_manifest().measurements:
        assert MeasurementRecord.from_dict(record.to_dict()) == record
