"""Specifications for atomic metric Diffs and reference coverage."""

from dataclasses import FrozenInstanceError, fields
from typing import Any

import pytest

import mlflow_monitor.domain as domain
from mlflow_monitor.domain import Diff, DiffReference, DiffReferenceKind

METRIC_UNAVAILABILITY_REASONS = (
    "current_metric_missing",
    "reference_metric_missing",
    "current_metric_not_finite",
    "reference_metric_not_finite",
    "delta_not_finite",
)
_ONE_METRIC_UNAVAILABILITY = object()


def _domain_type(name: str) -> Any:
    return getattr(domain, name)


def _baseline_reference() -> DiffReference:
    return DiffReference(
        kind=DiffReferenceKind.BASELINE,
        monitoring_run_id=None,
        source_run_id="train-run-baseline",
    )


def _previous_reference() -> DiffReference:
    return DiffReference(
        kind=DiffReferenceKind.PREVIOUS,
        monitoring_run_id="monitoring-run-previous",
        source_run_id="train-run-previous",
    )


def _diff(**overrides: object) -> Diff:
    values: dict[str, object] = {
        "diff_id": "diff-accuracy-baseline",
        "monitoring_run_id": "monitoring-run-current",
        "source_run_id": "train-run-current",
        "reference": _baseline_reference(),
        "metric_name": "accuracy",
        "current_value": 0.75,
        "reference_value": 0.5,
        "delta": 0.25,
    }
    values.update(overrides)
    return Diff(**values)  # type: ignore[arg-type]


def _metric_unavailability(**overrides: object) -> Any:
    values: dict[str, object] = {
        "metric_name": "precision",
        "reason": "reference_metric_missing",
    }
    values.update(overrides)
    model_type = _domain_type("MetricComparisonUnavailable")
    return model_type(**values)


def _coverage(**overrides: object) -> Any:
    status_type = _domain_type("ReferenceComparisonStatus")
    values: dict[str, object] = {
        "reference_kind": DiffReferenceKind.BASELINE,
        "reference": _baseline_reference(),
        "status": status_type.COMPLETED,
        "diff_ids": (),
        "metric_unavailability": (),
        "reason": None,
    }
    values.update(overrides)
    model_type = _domain_type("ReferenceComparisonCoverage")
    return model_type(**values)


def test_diff_reference_kind_contains_only_metric_reference_kinds() -> None:
    assert {kind.value for kind in DiffReferenceKind} == {
        "baseline",
        "previous",
        "lkg",
        "custom",
    }


def test_diff_has_one_atomic_metric_shape() -> None:
    assert tuple(field.name for field in fields(domain.Diff)) == (
        "diff_id",
        "monitoring_run_id",
        "source_run_id",
        "reference",
        "metric_name",
        "current_value",
        "reference_value",
        "delta",
    )

    diff = _diff()

    assert diff.source_run_id == "train-run-current"
    assert diff.reference == _baseline_reference()
    assert diff.metric_name == "accuracy"
    assert diff.current_value == 0.75
    assert diff.reference_value == 0.5
    assert diff.delta == 0.25


def test_diff_is_immutable() -> None:
    diff = _diff()

    with pytest.raises(FrozenInstanceError):
        diff.delta = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "field",
    ["diff_id", "monitoring_run_id", "source_run_id", "metric_name"],
)
def test_diff_rejects_empty_identity_and_metric_fields(field: str) -> None:
    with pytest.raises(ValueError):
        _diff(**{field: " "})


def test_diff_requires_a_typed_reference() -> None:
    with pytest.raises(ValueError):
        _diff(reference=None)


@pytest.mark.parametrize("field", ["current_value", "reference_value", "delta"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_diff_rejects_non_finite_numeric_evidence(field: str, value: float) -> None:
    with pytest.raises(ValueError):
        _diff(**{field: value})


def test_diff_rejects_delta_that_does_not_equal_current_minus_reference() -> None:
    with pytest.raises(ValueError):
        _diff(delta=0.5)


def test_reference_comparison_status_vocabulary_is_fixed() -> None:
    status_type = _domain_type("ReferenceComparisonStatus")

    assert {status.value for status in status_type} == {
        "completed",
        "skipped",
        "unavailable",
    }


@pytest.mark.parametrize("reason", METRIC_UNAVAILABILITY_REASONS)
def test_metric_comparison_unavailability_accepts_only_metric_reasons(reason: str) -> None:
    unavailable = _metric_unavailability(reason=reason)

    assert tuple(field.name for field in fields(unavailable)) == ("metric_name", "reason")
    assert unavailable.metric_name == "precision"
    assert unavailable.reason == reason


@pytest.mark.parametrize(
    "overrides",
    [
        {"metric_name": " "},
        {"reason": "unknown_reason"},
    ],
)
def test_metric_comparison_unavailability_rejects_invalid_shape(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _metric_unavailability(**overrides)


def test_metric_comparison_unavailability_is_immutable() -> None:
    unavailable = _metric_unavailability()

    with pytest.raises(FrozenInstanceError):
        unavailable.reason = "current_metric_missing"


def test_completed_coverage_accepts_diff_and_metric_unavailability_outcomes() -> None:
    unavailable = _metric_unavailability()
    coverage = _coverage(
        diff_ids=("diff-accuracy-baseline",),
        metric_unavailability=(unavailable,),
    )

    assert tuple(field.name for field in fields(coverage)) == (
        "reference_kind",
        "reference",
        "status",
        "diff_ids",
        "metric_unavailability",
        "reason",
    )
    assert coverage.diff_ids == ("diff-accuracy-baseline",)
    assert coverage.metric_unavailability == (unavailable,)
    assert coverage.reason is None


def test_completed_coverage_allows_an_empty_metric_selection() -> None:
    coverage = _coverage()

    assert coverage.diff_ids == ()
    assert coverage.metric_unavailability == ()


def test_skipped_coverage_requires_a_resolved_reference_and_exact_reason() -> None:
    status_type = _domain_type("ReferenceComparisonStatus")

    coverage = _coverage(
        status=status_type.SKIPPED,
        reason="current_not_comparable",
    )

    assert coverage.reference == _baseline_reference()
    assert coverage.diff_ids == ()
    assert coverage.metric_unavailability == ()


@pytest.mark.parametrize(
    ("reference_kind", "reference", "reason"),
    [
        (DiffReferenceKind.PREVIOUS, None, "previous_reference_missing"),
        (DiffReferenceKind.LKG, None, "lkg_not_selected"),
        (DiffReferenceKind.LKG, None, "lkg_selection_inconsistent"),
        (
            DiffReferenceKind.PREVIOUS,
            _previous_reference(),
            "reference_source_run_missing",
        ),
    ],
)
def test_unavailable_coverage_accepts_each_reference_reason(
    reference_kind: DiffReferenceKind,
    reference: DiffReference | None,
    reason: str,
) -> None:
    status_type = _domain_type("ReferenceComparisonStatus")

    coverage = _coverage(
        reference_kind=reference_kind,
        reference=reference,
        status=status_type.UNAVAILABLE,
        reason=reason,
    )

    assert coverage.reason == reason
    assert coverage.diff_ids == ()
    assert coverage.metric_unavailability == ()


def test_coverage_is_immutable() -> None:
    coverage = _coverage()

    with pytest.raises(FrozenInstanceError):
        coverage.reason = "current_not_comparable"


@pytest.mark.parametrize(
    ("status_name", "overrides"),
    [
        ("COMPLETED", {"reference": None}),
        ("COMPLETED", {"reason": "current_not_comparable"}),
        ("SKIPPED", {"reference": None, "reason": "current_not_comparable"}),
        ("SKIPPED", {"reason": None}),
        ("SKIPPED", {"reason": "unknown_reason"}),
        (
            "SKIPPED",
            {"reason": "current_not_comparable", "diff_ids": ("diff-1",)},
        ),
        (
            "SKIPPED",
            {
                "reason": "current_not_comparable",
                "metric_unavailability": _ONE_METRIC_UNAVAILABILITY,
            },
        ),
        ("UNAVAILABLE", {"reference": None, "reason": None}),
        ("UNAVAILABLE", {"reference": None, "reason": "unknown_reason"}),
        (
            "UNAVAILABLE",
            {
                "reference": None,
                "reason": "previous_reference_missing",
                "diff_ids": ("diff-1",),
            },
        ),
        (
            "UNAVAILABLE",
            {
                "reference": None,
                "reason": "previous_reference_missing",
                "metric_unavailability": _ONE_METRIC_UNAVAILABILITY,
            },
        ),
    ],
)
def test_coverage_rejects_status_inconsistent_shapes(
    status_name: str,
    overrides: dict[str, object],
) -> None:
    status_type = _domain_type("ReferenceComparisonStatus")
    overrides = dict(overrides)
    if overrides.get("metric_unavailability") is _ONE_METRIC_UNAVAILABILITY:
        overrides["metric_unavailability"] = (_metric_unavailability(),)

    with pytest.raises(ValueError):
        _coverage(status=getattr(status_type, status_name), **overrides)


def test_coverage_rejects_unknown_status() -> None:
    with pytest.raises(ValueError):
        _coverage(status="partial")


@pytest.mark.parametrize(
    ("status_name", "reason"),
    [
        ("COMPLETED", None),
        ("SKIPPED", "current_not_comparable"),
        ("UNAVAILABLE", "reference_source_run_missing"),
    ],
)
def test_coverage_reference_kind_must_match_resolved_reference(
    status_name: str,
    reason: str | None,
) -> None:
    status_type = _domain_type("ReferenceComparisonStatus")

    with pytest.raises(ValueError):
        _coverage(
            reference_kind=DiffReferenceKind.PREVIOUS,
            reference=_baseline_reference(),
            status=getattr(status_type, status_name),
            reason=reason,
        )
