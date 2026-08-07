"""Specifications for Finding evidence invariants."""

from dataclasses import replace

import pytest

import mlflow_monitor.invariant as invariant
from mlflow_monitor.domain import (
    CompatibilityEvidence,
    ContractCheckReason,
    Diff,
    DiffReference,
    DiffReferenceKind,
    Finding,
    FindingSeverity,
)
from mlflow_monitor.errors import InvariantViolation
from mlflow_monitor.identity import make_compatibility_evidence_id, make_diff_id

MONITORING_RUN_ID = "monitoring-run-current"
SOURCE_RUN_ID = "train-run-current"


def _diff(
    *,
    monitoring_run_id: str = MONITORING_RUN_ID,
    source_run_id: str = SOURCE_RUN_ID,
) -> Diff:
    reference = DiffReference(
        kind=DiffReferenceKind.BASELINE,
        monitoring_run_id=None,
        source_run_id="train-run-baseline",
    )
    return Diff(
        diff_id=make_diff_id(
            monitoring_run_id=monitoring_run_id,
            source_run_id=source_run_id,
            reference=reference,
            metric_name="accuracy",
        ),
        monitoring_run_id=monitoring_run_id,
        source_run_id=source_run_id,
        reference=reference,
        metric_name="accuracy",
        current_value=0.75,
        reference_value=0.5,
        delta=0.25,
    )


def _compatibility_evidence(
    *,
    monitoring_run_id: str = MONITORING_RUN_ID,
    source_run_id: str = SOURCE_RUN_ID,
) -> CompatibilityEvidence:
    reason = ContractCheckReason(
        code="environment_mismatch",
        message="Execution environment does not match the baseline.",
        blocking=False,
    )
    return CompatibilityEvidence(
        compatibility_evidence_id=make_compatibility_evidence_id(
            monitoring_run_id=monitoring_run_id,
            source_run_id=source_run_id,
            baseline_source_run_id="train-run-baseline",
            contract_id="default_permissive",
            contract_version="v0",
            reason_code=reason.code,
        ),
        monitoring_run_id=monitoring_run_id,
        source_run_id=source_run_id,
        baseline_source_run_id="train-run-baseline",
        contract_id="default_permissive",
        contract_version="v0",
        reason=reason,
    )


def _finding(
    *,
    evidence_diff_ids: tuple[str, ...],
    evidence_compatibility_ids: tuple[str, ...],
) -> Finding:
    return Finding(
        finding_id="finding-v1-example",
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        finding_policy_id="relative-regression",
        finding_policy_version="1",
        finding_rule_id="quality.accuracy_regression",
        severity=FindingSeverity.HIGH,
        category="quality",
        summary="Accuracy regressed against the baseline.",
        recommendation="Review the model and input changes.",
        evidence_diff_ids=evidence_diff_ids,
        evidence_compatibility_ids=evidence_compatibility_ids,
    )


def test_finding_evidence_accepts_supplied_records_from_the_current_pair() -> None:
    diff = _diff()
    compatibility_evidence = _compatibility_evidence()
    finding = _finding(
        evidence_diff_ids=(diff.diff_id,),
        evidence_compatibility_ids=(compatibility_evidence.compatibility_evidence_id,),
    )

    invariant.validate_finding_evidence(
        finding,
        diffs=(diff,),
        compatibility_evidence=(compatibility_evidence,),
    )


def test_finding_evidence_rejects_an_unknown_diff_identity() -> None:
    finding = _finding(
        evidence_diff_ids=("diff-v1-unknown",),
        evidence_compatibility_ids=(),
    )

    with pytest.raises(InvariantViolation) as exc_info:
        invariant.validate_finding_evidence(
            finding,
            diffs=(),
            compatibility_evidence=(),
        )

    error = exc_info.value
    assert error.code == "finding_diff_evidence_violation"
    assert error.entity == "Finding"
    assert error.field == "evidence_diff_ids"


def test_finding_evidence_rejects_an_unknown_compatibility_identity() -> None:
    finding = _finding(
        evidence_diff_ids=(),
        evidence_compatibility_ids=("compatibility-evidence-v1-unknown",),
    )

    with pytest.raises(InvariantViolation) as exc_info:
        invariant.validate_finding_evidence(
            finding,
            diffs=(),
            compatibility_evidence=(),
        )

    error = exc_info.value
    assert error.code == "finding_compatibility_evidence_violation"
    assert error.entity == "Finding"
    assert error.field == "evidence_compatibility_ids"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("monitoring_run_id", "monitoring-run-other"),
        ("source_run_id", "train-run-other"),
    ],
)
def test_finding_evidence_rejects_a_cross_pair_diff(field: str, value: str) -> None:
    diff = _diff(**{field: value})
    finding = _finding(
        evidence_diff_ids=(diff.diff_id,),
        evidence_compatibility_ids=(),
    )

    with pytest.raises(InvariantViolation) as exc_info:
        invariant.validate_finding_evidence(
            finding,
            diffs=(diff,),
            compatibility_evidence=(),
        )

    error = exc_info.value
    assert error.code == "finding_diff_evidence_violation"
    assert error.entity == "Diff"
    assert error.field == field


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("monitoring_run_id", "monitoring-run-other"),
        ("source_run_id", "train-run-other"),
    ],
)
def test_finding_evidence_rejects_cross_pair_compatibility_evidence(
    field: str,
    value: str,
) -> None:
    compatibility_evidence = _compatibility_evidence(**{field: value})
    finding = _finding(
        evidence_diff_ids=(),
        evidence_compatibility_ids=(compatibility_evidence.compatibility_evidence_id,),
    )

    with pytest.raises(InvariantViolation) as exc_info:
        invariant.validate_finding_evidence(
            finding,
            diffs=(),
            compatibility_evidence=(compatibility_evidence,),
        )

    error = exc_info.value
    assert error.code == "finding_compatibility_evidence_violation"
    assert error.entity == "CompatibilityEvidence"
    assert error.field == field


def test_finding_evidence_validates_the_supplied_diff_identity() -> None:
    diff = replace(_diff(), diff_id="diff-v1-incorrect")
    finding = _finding(
        evidence_diff_ids=(diff.diff_id,),
        evidence_compatibility_ids=(),
    )

    with pytest.raises(InvariantViolation) as exc_info:
        invariant.validate_finding_evidence(
            finding,
            diffs=(diff,),
            compatibility_evidence=(),
        )

    assert exc_info.value.code == "diff_identity_mismatch"


def test_finding_evidence_validates_the_supplied_compatibility_identity() -> None:
    evidence = replace(
        _compatibility_evidence(),
        compatibility_evidence_id="compatibility-evidence-v1-incorrect",
    )
    finding = _finding(
        evidence_diff_ids=(),
        evidence_compatibility_ids=(evidence.compatibility_evidence_id,),
    )

    with pytest.raises(InvariantViolation) as exc_info:
        invariant.validate_finding_evidence(
            finding,
            diffs=(),
            compatibility_evidence=(evidence,),
        )

    assert exc_info.value.code == "compatibility_evidence_identity_mismatch"


def test_finding_evidence_accepts_identical_evidence_retries() -> None:
    diff = _diff()
    finding = _finding(
        evidence_diff_ids=(diff.diff_id,),
        evidence_compatibility_ids=(),
    )

    invariant.validate_finding_evidence(
        finding,
        diffs=(diff, diff),
        compatibility_evidence=(),
    )
