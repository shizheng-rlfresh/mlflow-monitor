"""Specifications for Timeline and LKG Selection domain models."""

from dataclasses import FrozenInstanceError, fields

import pytest

import mlflow_monitor.domain as domain
from mlflow_monitor.domain import ComparabilityStatus, LifecycleStatus


def _entry(**overrides: object) -> domain.TimelineEntry:
    values: dict[str, object] = {
        "monitoring_run_id": "monitoring-run-1",
        "source_run_id": "train-run-1",
        "sequence_index": 0,
        "lifecycle_status": LifecycleStatus.CLOSED,
        "comparability_status": ComparabilityStatus.PASS,
    }
    values.update(overrides)
    return domain.TimelineEntry(**values)  # type: ignore[arg-type]


def _timeline(**overrides: object) -> domain.Timeline:
    values: dict[str, object] = {
        "timeline_id": "timeline-1",
        "subject_id": "churn-model",
        "baseline_source_run_id": "train-run-baseline",
        "entries": (),
    }
    values.update(overrides)
    return domain.Timeline(**values)  # type: ignore[arg-type]


def _selection(**overrides: object) -> domain.LKGSelection:
    values: dict[str, object] = {
        "lkg_selection_id": "lkg-selection-1",
        "timeline_id": "timeline-1",
        "monitoring_run_id": "monitoring-run-1",
        "source_run_id": "train-run-1",
        "supersedes_lkg_selection_ids": (),
    }
    values.update(overrides)
    return domain.LKGSelection(**values)  # type: ignore[arg-type]


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
    entries: tuple[domain.TimelineEntry, ...],
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
