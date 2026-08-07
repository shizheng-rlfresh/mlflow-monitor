"""Specifications for Timeline and LKG Selection domain models."""

from dataclasses import FrozenInstanceError, fields

import pytest

import mlflow_monitor.domain as domain
from mlflow_monitor.domain import (
    ComparabilityStatus,
    LifecycleStatus,
    LKGSelection,
    Timeline,
    TimelineEntry,
)


def _entry(**overrides: object) -> TimelineEntry:
    values: dict[str, object] = {
        "monitoring_run_id": "monitoring-run-1",
        "source_run_id": "train-run-1",
        "sequence_index": 0,
        "lifecycle_status": LifecycleStatus.CLOSED,
        "comparability_status": ComparabilityStatus.PASS,
    }
    values.update(overrides)
    return TimelineEntry(**values)  # type: ignore[arg-type]


def _timeline(**overrides: object) -> Timeline:
    values: dict[str, object] = {
        "timeline_id": "timeline-1",
        "subject_id": "churn-model",
        "baseline_source_run_id": "train-run-baseline",
        "entries": (),
    }
    values.update(overrides)
    return Timeline(**values)  # type: ignore[arg-type]


def _selection(**overrides: object) -> LKGSelection:
    values: dict[str, object] = {
        "lkg_selection_id": "lkg-selection-1",
        "timeline_id": "timeline-1",
        "monitoring_run_id": "monitoring-run-1",
        "source_run_id": "train-run-1",
        "supersedes_lkg_selection_ids": (),
    }
    values.update(overrides)
    return LKGSelection(**values)  # type: ignore[arg-type]


def test_timeline_models_have_the_approved_shapes() -> None:
    assert tuple(field.name for field in fields(domain.TimelineEntry)) == (
        "monitoring_run_id",
        "source_run_id",
        "sequence_index",
        "lifecycle_status",
        "comparability_status",
    )
    assert tuple(field.name for field in fields(domain.Timeline)) == (
        "timeline_id",
        "subject_id",
        "baseline_source_run_id",
        "entries",
    )
    assert tuple(field.name for field in fields(domain.LKGSelection)) == (
        "lkg_selection_id",
        "timeline_id",
        "monitoring_run_id",
        "source_run_id",
        "supersedes_lkg_selection_ids",
    )
    assert not hasattr(domain, "LKG")


@pytest.mark.parametrize(
    ("value_factory", "field_name"),
    [
        (_entry, "monitoring_run_id"),
        (_timeline, "timeline_id"),
        (_selection, "lkg_selection_id"),
    ],
)
def test_timeline_models_are_immutable(value_factory: object, field_name: str) -> None:
    value = value_factory()  # type: ignore[operator]

    with pytest.raises(FrozenInstanceError):
        setattr(value, field_name, "changed")


@pytest.mark.parametrize("field", ["monitoring_run_id", "source_run_id"])
def test_timeline_entry_rejects_empty_identity_fields(field: str) -> None:
    with pytest.raises(ValueError):
        _entry(**{field: " "})


@pytest.mark.parametrize("sequence_index", [-1, True, 1.5, "1"])
def test_timeline_entry_requires_a_nonnegative_integer_sequence(
    sequence_index: object,
) -> None:
    with pytest.raises(ValueError):
        _entry(sequence_index=sequence_index)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lifecycle_status", "closed"),
        ("comparability_status", "pass"),
    ],
)
def test_timeline_entry_requires_typed_statuses(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _entry(**{field: value})


@pytest.mark.parametrize(
    "lifecycle_status",
    [
        LifecycleStatus.CREATED,
        LifecycleStatus.PREPARED,
        LifecycleStatus.CHECKED,
        LifecycleStatus.ANALYZED,
    ],
)
def test_timeline_entry_rejects_nonterminal_lifecycle_states(
    lifecycle_status: LifecycleStatus,
) -> None:
    with pytest.raises(ValueError):
        _entry(lifecycle_status=lifecycle_status)


def test_timeline_entry_allows_missing_comparability() -> None:
    assert _entry(comparability_status=None).comparability_status is None


@pytest.mark.parametrize("field", ["timeline_id", "subject_id"])
def test_timeline_rejects_empty_identity_fields(field: str) -> None:
    with pytest.raises(ValueError):
        _timeline(**{field: " "})


def test_timeline_accepts_an_uninitialized_baseline() -> None:
    timeline = _timeline(baseline_source_run_id=None)

    assert timeline.baseline_source_run_id is None


def test_timeline_rejects_an_empty_initialized_baseline() -> None:
    with pytest.raises(ValueError):
        _timeline(baseline_source_run_id=" ")


def test_timeline_freezes_and_orders_entries_by_sequence() -> None:
    first = _entry()
    third = _entry(
        monitoring_run_id="monitoring-run-3",
        source_run_id="train-run-3",
        sequence_index=2,
        lifecycle_status=LifecycleStatus.FAILED,
        comparability_status=None,
    )
    supplied_entries = [third, first]

    timeline = _timeline(entries=supplied_entries)
    supplied_entries.clear()

    assert timeline.entries == (first, third)


@pytest.mark.parametrize(
    "entries",
    [
        (
            _entry(),
            _entry(
                monitoring_run_id="monitoring-run-2",
                source_run_id="train-run-2",
                sequence_index=0,
            ),
        ),
        (
            _entry(),
            _entry(source_run_id="train-run-other", sequence_index=1),
        ),
    ],
)
def test_timeline_rejects_duplicate_entry_identity_or_sequence(
    entries: tuple[TimelineEntry, ...],
) -> None:
    with pytest.raises(ValueError):
        _timeline(entries=entries)


@pytest.mark.parametrize("entries", ["monitoring-run-1", (object(),)])
def test_timeline_requires_typed_entry_collections(entries: object) -> None:
    with pytest.raises(ValueError):
        _timeline(entries=entries)


@pytest.mark.parametrize(
    "field",
    ["lkg_selection_id", "timeline_id", "monitoring_run_id", "source_run_id"],
)
def test_lkg_selection_rejects_empty_identity_fields(field: str) -> None:
    with pytest.raises(ValueError):
        _selection(**{field: " "})


def test_lkg_selection_canonicalizes_superseded_identities() -> None:
    supplied_ids = ["lkg-selection-b", "lkg-selection-a", "lkg-selection-a"]

    selection = _selection(supersedes_lkg_selection_ids=supplied_ids)
    supplied_ids.clear()

    assert selection.supersedes_lkg_selection_ids == (
        "lkg-selection-a",
        "lkg-selection-b",
    )


@pytest.mark.parametrize(
    "supersedes_lkg_selection_ids",
    ["lkg-selection-1", (" ",), (object(),)],
)
def test_lkg_selection_requires_nonempty_identity_collections(
    supersedes_lkg_selection_ids: object,
) -> None:
    with pytest.raises(ValueError):
        _selection(supersedes_lkg_selection_ids=supersedes_lkg_selection_ids)


def test_timeline_entry_serializes_to_an_exact_dictionary() -> None:
    assert _entry().to_dict() == {
        "monitoring_run_id": "monitoring-run-1",
        "source_run_id": "train-run-1",
        "sequence_index": 0,
        "lifecycle_status": "closed",
        "comparability_status": "pass",
    }
    assert _entry(
        lifecycle_status=LifecycleStatus.FAILED,
        comparability_status=None,
    ).to_dict() == {
        "monitoring_run_id": "monitoring-run-1",
        "source_run_id": "train-run-1",
        "sequence_index": 0,
        "lifecycle_status": "failed",
        "comparability_status": None,
    }


def test_timeline_serializes_nullable_baseline_and_ordered_entries() -> None:
    first = _entry(lifecycle_status=LifecycleStatus.FAILED)
    third = _entry(
        monitoring_run_id="monitoring-run-3",
        source_run_id="train-run-3",
        sequence_index=2,
        lifecycle_status=LifecycleStatus.FAILED,
        comparability_status=None,
    )

    assert _timeline(
        baseline_source_run_id=None,
        entries=[third, first],
    ).to_dict() == {
        "timeline_id": "timeline-1",
        "subject_id": "churn-model",
        "baseline_source_run_id": None,
        "entries": [first.to_dict(), third.to_dict()],
    }


def test_lkg_selection_serializes_canonical_supersession_history() -> None:
    assert _selection(
        supersedes_lkg_selection_ids=(
            "lkg-selection-b",
            "lkg-selection-a",
            "lkg-selection-a",
        )
    ).to_dict() == {
        "lkg_selection_id": "lkg-selection-1",
        "timeline_id": "timeline-1",
        "monitoring_run_id": "monitoring-run-1",
        "source_run_id": "train-run-1",
        "supersedes_lkg_selection_ids": [
            "lkg-selection-a",
            "lkg-selection-b",
        ],
    }


def test_equivalent_collection_inputs_serialize_identically() -> None:
    first = _entry()
    second = _entry(
        monitoring_run_id="monitoring-run-2",
        source_run_id="train-run-2",
        sequence_index=1,
    )

    assert (
        _timeline(entries=[second, first]).to_dict() == _timeline(entries=(first, second)).to_dict()
    )
    assert (
        _selection(supersedes_lkg_selection_ids=["lkg-selection-b", "lkg-selection-a"]).to_dict()
        == _selection(supersedes_lkg_selection_ids=("lkg-selection-a", "lkg-selection-b")).to_dict()
    )


def test_timeline_with_null_baseline_cannot_accept_closed_entries() -> None:
    """Test that a Timeline with a null baseline cannot accept closed entries."""

    closed_timeline_entry = TimelineEntry(
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-1",
        sequence_index=0,
        lifecycle_status=LifecycleStatus.CLOSED,
        comparability_status=ComparabilityStatus.FAIL,
    )

    failed_timeline_entry = TimelineEntry(
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-1",
        sequence_index=1,
        lifecycle_status=LifecycleStatus.FAILED,
        comparability_status=ComparabilityStatus.FAIL,
    )

    with pytest.raises(
        ValueError, match="cannot accept closed entries without a baseline_source_run_id"
    ):
        Timeline(
            timeline_id="timeline-1",
            subject_id="churn_model",
            baseline_source_run_id=None,
            entries=(closed_timeline_entry, failed_timeline_entry),
        )
