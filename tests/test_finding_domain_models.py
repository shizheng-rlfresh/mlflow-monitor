"""Specifications for Finding Drafts and final Findings."""

from dataclasses import FrozenInstanceError, fields
from typing import Any

import pytest

import mlflow_monitor.domain as domain


def _draft(**overrides: object) -> Any:
    values: dict[str, object] = {
        "finding_rule_id": "quality.accuracy_regression",
        "severity": domain.FindingSeverity.HIGH,
        "category": "quality",
        "summary": "Accuracy regressed against the baseline.",
        "recommendation": "Review the model and input changes.",
        "evidence_diff_ids": ("diff-v1-example",),
        "evidence_compatibility_ids": (),
    }
    values.update(overrides)
    return domain.FindingDraft(**values)  # type: ignore[arg-type]


def _finding(**overrides: object) -> domain.Finding:
    values: dict[str, object] = {
        "finding_id": "finding-v1-example",
        "monitoring_run_id": "monitoring-run-current",
        "source_run_id": "train-run-current",
        "finding_policy_id": "relative-regression",
        "finding_policy_version": "1",
        "finding_rule_id": "quality.accuracy_regression",
        "severity": domain.FindingSeverity.HIGH,
        "category": "quality",
        "summary": "Accuracy regressed against the baseline.",
        "recommendation": "Review the model and input changes.",
        "evidence_diff_ids": ("diff-v1-example",),
        "evidence_compatibility_ids": (),
    }
    values.update(overrides)
    return domain.Finding(**values)  # type: ignore[arg-type]


def test_finding_models_have_the_approved_shapes() -> None:
    finding_draft_type = domain.FindingDraft

    assert tuple(field.name for field in fields(finding_draft_type)) == (
        "finding_rule_id",
        "severity",
        "category",
        "summary",
        "recommendation",
        "evidence_diff_ids",
        "evidence_compatibility_ids",
    )
    assert tuple(field.name for field in fields(domain.Finding)) == (
        "finding_id",
        "monitoring_run_id",
        "source_run_id",
        "finding_policy_id",
        "finding_policy_version",
        "finding_rule_id",
        "severity",
        "category",
        "summary",
        "recommendation",
        "evidence_diff_ids",
        "evidence_compatibility_ids",
    )


@pytest.mark.parametrize("factory", [_draft, _finding])
def test_finding_models_are_immutable(factory: Any) -> None:
    value = factory()

    with pytest.raises(FrozenInstanceError):
        value.summary = "Changed"  # type: ignore[misc]


@pytest.mark.parametrize("factory", [_draft, _finding])
def test_finding_models_defensively_freeze_evidence_collections(factory: Any) -> None:
    diff_ids = ["diff-v1-example"]
    compatibility_ids = ["compatibility-evidence-v1-example"]

    value = factory(
        evidence_diff_ids=diff_ids,
        evidence_compatibility_ids=compatibility_ids,
    )
    diff_ids.append("diff-v1-changed")
    compatibility_ids.append("compatibility-evidence-v1-changed")

    assert value.evidence_diff_ids == ("diff-v1-example",)
    assert value.evidence_compatibility_ids == ("compatibility-evidence-v1-example",)


@pytest.mark.parametrize(
    "field",
    [
        "finding_rule_id",
        "category",
        "summary",
        "recommendation",
    ],
)
def test_finding_draft_rejects_empty_text_fields(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _draft(**{field: " "})


@pytest.mark.parametrize(
    "field",
    [
        "finding_id",
        "monitoring_run_id",
        "source_run_id",
        "finding_policy_id",
        "finding_policy_version",
        "finding_rule_id",
        "category",
        "summary",
        "recommendation",
    ],
)
def test_finding_rejects_empty_text_fields(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _finding(**{field: " "})


@pytest.mark.parametrize("factory", [_draft, _finding])
def test_finding_models_require_typed_severity(factory: Any) -> None:
    with pytest.raises(ValueError, match="severity"):
        factory(severity="high")


@pytest.mark.parametrize("factory", [_draft, _finding])
def test_finding_models_require_combined_evidence(factory: Any) -> None:
    with pytest.raises(ValueError, match="evidence"):
        factory(evidence_diff_ids=(), evidence_compatibility_ids=())


@pytest.mark.parametrize("factory", [_draft, _finding])
@pytest.mark.parametrize(
    "overrides",
    [
        {"evidence_diff_ids": (" ",)},
        {"evidence_compatibility_ids": (" ",), "evidence_diff_ids": ()},
        {"evidence_diff_ids": (1,)},
        {"evidence_compatibility_ids": (1,), "evidence_diff_ids": ()},
    ],
)
def test_finding_models_require_nonempty_string_evidence_ids(
    factory: Any,
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="evidence"):
        factory(**overrides)


@pytest.mark.parametrize(
    ("evidence_diff_ids", "evidence_compatibility_ids"),
    [
        (("diff-v1-example",), ()),
        ((), ("compatibility-evidence-v1-example",)),
        (("diff-v1-example",), ("compatibility-evidence-v1-example",)),
    ],
)
def test_finding_draft_accepts_diff_compatibility_or_combined_evidence(
    evidence_diff_ids: tuple[str, ...],
    evidence_compatibility_ids: tuple[str, ...],
) -> None:
    draft = _draft(
        evidence_diff_ids=evidence_diff_ids,
        evidence_compatibility_ids=evidence_compatibility_ids,
    )

    assert draft.evidence_diff_ids == evidence_diff_ids
    assert draft.evidence_compatibility_ids == evidence_compatibility_ids
