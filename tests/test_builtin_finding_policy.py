"""Specifications for the built-in compatibility Finding policy."""

from __future__ import annotations

import pytest

from mlflow_monitor.builtins import SYSTEM_COMPATIBILITY_FINDING_POLICY
from mlflow_monitor.domain import (
    CompatibilityEvidence,
    ContractCheckReason,
    Diff,
    DiffReference,
    DiffReferenceKind,
    FindingDraft,
    FindingSeverity,
)

MONITORING_RUN_ID = "monitoring-run-current"
SOURCE_RUN_ID = "train-run-current"
BASELINE_SOURCE_RUN_ID = "train-run-baseline"

SUPPORTED_REASON_MAPPINGS = (
    (
        "environment_mismatch",
        False,
        "compatibility.environment_mismatch",
        (
            "Review the execution-environment differences and confirm that the current "
            "evidence is comparable with the baseline before relying on metric comparisons."
        ),
    ),
    (
        "schema_mismatch",
        True,
        "compatibility.schema_mismatch",
        (
            "Review the schema changes and either restore baseline-compatible data or "
            "intentionally update the Contract for a future Monitoring Run."
        ),
    ),
    (
        "feature_mismatch",
        True,
        "compatibility.feature_mismatch",
        (
            "Review the feature-set changes and either restore baseline-compatible features or "
            "intentionally update the Contract for a future Monitoring Run."
        ),
    ),
    (
        "data_scope_mismatch",
        True,
        "compatibility.data_scope_mismatch",
        (
            "Confirm the intended data population and either restore the baseline-compatible "
            "scope or intentionally update the Contract for a future Monitoring Run."
        ),
    ),
)


def _compatibility_evidence(*, reason_code: str, blocking: bool) -> CompatibilityEvidence:
    reason = ContractCheckReason(
        code=reason_code,
        message=f"Committed {reason_code} reason message.",
        blocking=blocking,
    )
    return CompatibilityEvidence(
        compatibility_evidence_id=f"compatibility-evidence-{reason_code}",
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        baseline_source_run_id=BASELINE_SOURCE_RUN_ID,
        contract_id="default_permissive",
        contract_version="v0",
        reason=reason,
    )


def _evaluate(
    *,
    diffs: tuple[Diff, ...] = (),
    compatibility_evidence: tuple[CompatibilityEvidence, ...] = (),
) -> tuple[FindingDraft, ...]:
    policy = SYSTEM_COMPATIBILITY_FINDING_POLICY
    return policy.evaluate(
        parameters=policy.validate_parameters({}),
        diffs=diffs,
        compatibility_evidence=compatibility_evidence,
        reference_comparison_coverage=(),
    )


def test_builtin_policy_maps_each_supported_compatibility_reason_exactly() -> None:
    evidence = tuple(
        _compatibility_evidence(reason_code=reason_code, blocking=blocking)
        for reason_code, blocking, _, _ in SUPPORTED_REASON_MAPPINGS
    )

    drafts = _evaluate(compatibility_evidence=evidence)

    assert len(drafts) == len(evidence)
    drafts_by_evidence_id = {draft.evidence_compatibility_ids[0]: draft for draft in drafts}
    assert len(drafts_by_evidence_id) == len(evidence)
    for compatibility_evidence, (_, _, finding_rule_id, recommendation) in zip(
        evidence,
        SUPPORTED_REASON_MAPPINGS,
        strict=True,
    ):
        assert drafts_by_evidence_id[
            compatibility_evidence.compatibility_evidence_id
        ] == FindingDraft(
            finding_rule_id=finding_rule_id,
            severity=FindingSeverity.HIGH,
            category="compatibility",
            summary=compatibility_evidence.reason.message,
            recommendation=recommendation,
            evidence_diff_ids=(),
            evidence_compatibility_ids=(compatibility_evidence.compatibility_evidence_id,),
        )


def test_builtin_policy_ignores_metric_diffs_without_compatibility_evidence() -> None:
    reference = DiffReference(
        kind=DiffReferenceKind.BASELINE,
        monitoring_run_id=None,
        source_run_id=BASELINE_SOURCE_RUN_ID,
    )
    diff = Diff(
        diff_id="diff-current-accuracy-baseline",
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        reference=reference,
        metric_name="accuracy",
        current_value=0.75,
        reference_value=0.5,
        delta=0.25,
    )

    assert _evaluate(diffs=(diff,)) == ()


def test_builtin_policy_rejects_unknown_compatibility_reason_code() -> None:
    evidence = _compatibility_evidence(reason_code="future_mismatch", blocking=True)

    with pytest.raises(ValueError):
        _evaluate(compatibility_evidence=(evidence,))
