"""Specifications for package-owned Finding materialization."""

from dataclasses import replace

import pytest

from mlflow_monitor.domain import (
    CompatibilityEvidence,
    ContractCheckReason,
    Diff,
    DiffReference,
    DiffReferenceKind,
    FindingDraft,
    FindingSeverity,
)
from mlflow_monitor.errors import InvariantViolation
from mlflow_monitor.finding_policy import materialize_finding_drafts
from mlflow_monitor.identity import (
    make_compatibility_evidence_id,
    make_diff_id,
    make_finding_id,
)

MONITORING_RUN_ID = "monitoring-run-current"
SOURCE_RUN_ID = "train-run-current"
FINDING_POLICY_ID = "relative-regression"
FINDING_POLICY_VERSION = "1"


def _diff(
    *,
    metric_name: str = "accuracy",
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
            metric_name=metric_name,
        ),
        monitoring_run_id=monitoring_run_id,
        source_run_id=source_run_id,
        reference=reference,
        metric_name=metric_name,
        current_value=0.75,
        reference_value=0.5,
        delta=0.25,
    )


def _compatibility_evidence(*, code: str = "environment_mismatch") -> CompatibilityEvidence:
    reason_by_code = {
        "environment_mismatch": ContractCheckReason(
            code="environment_mismatch",
            message="Execution environment does not match the baseline.",
            blocking=False,
        ),
        "schema_mismatch": ContractCheckReason(
            code="schema_mismatch",
            message="Data schema does not match the baseline.",
            blocking=True,
        ),
    }
    reason = reason_by_code[code]
    return CompatibilityEvidence(
        compatibility_evidence_id=make_compatibility_evidence_id(
            monitoring_run_id=MONITORING_RUN_ID,
            source_run_id=SOURCE_RUN_ID,
            baseline_source_run_id="train-run-baseline",
            contract_id="default_permissive",
            contract_version="v0",
            reason_code=reason.code,
        ),
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        baseline_source_run_id="train-run-baseline",
        contract_id="default_permissive",
        contract_version="v0",
        reason=reason,
    )


def _draft(**overrides: object) -> FindingDraft:
    values: dict[str, object] = {
        "finding_rule_id": "quality.accuracy_regression",
        "severity": FindingSeverity.HIGH,
        "category": "quality",
        "summary": "Accuracy regressed against the baseline.",
        "recommendation": "Review the model and input changes.",
        "evidence_diff_ids": (),
        "evidence_compatibility_ids": ("compatibility-evidence-v1-placeholder",),
    }
    values.update(overrides)
    return FindingDraft(**values)  # type: ignore[arg-type]


def test_materialization_attaches_package_owned_identity() -> None:
    diff = _diff()
    draft = _draft(
        evidence_diff_ids=(diff.diff_id,),
        evidence_compatibility_ids=(),
    )

    findings = materialize_finding_drafts(
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        finding_policy_id=FINDING_POLICY_ID,
        finding_policy_version=FINDING_POLICY_VERSION,
        drafts=(draft,),
        diffs=(diff,),
        compatibility_evidence=(),
    )

    assert len(findings) == 1
    finding = findings[0]
    assert finding.finding_id == make_finding_id(
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        finding_policy_id=FINDING_POLICY_ID,
        finding_policy_version=FINDING_POLICY_VERSION,
        finding_rule_id=draft.finding_rule_id,
        evidence_diff_ids=draft.evidence_diff_ids,
        evidence_compatibility_ids=draft.evidence_compatibility_ids,
    )
    assert finding.monitoring_run_id == MONITORING_RUN_ID
    assert finding.source_run_id == SOURCE_RUN_ID
    assert finding.finding_policy_id == FINDING_POLICY_ID
    assert finding.finding_policy_version == FINDING_POLICY_VERSION
    assert finding.finding_rule_id == draft.finding_rule_id


def test_materialization_canonicalizes_each_evidence_collection() -> None:
    diffs = (_diff(metric_name="accuracy"), _diff(metric_name="precision"))
    compatibility_evidence = (
        _compatibility_evidence(code="environment_mismatch"),
        _compatibility_evidence(code="schema_mismatch"),
    )
    draft = _draft(
        evidence_diff_ids=tuple(diff.diff_id for diff in reversed(diffs)),
        evidence_compatibility_ids=tuple(
            evidence.compatibility_evidence_id for evidence in reversed(compatibility_evidence)
        ),
    )

    (finding,) = materialize_finding_drafts(
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        finding_policy_id=FINDING_POLICY_ID,
        finding_policy_version=FINDING_POLICY_VERSION,
        drafts=(draft,),
        diffs=diffs,
        compatibility_evidence=compatibility_evidence,
    )

    assert finding.evidence_diff_ids == tuple(sorted(draft.evidence_diff_ids))
    assert finding.evidence_compatibility_ids == tuple(sorted(draft.evidence_compatibility_ids))


def test_materialization_deduplicates_repeated_evidence_identities() -> None:
    diff = _diff()
    compatibility_evidence = _compatibility_evidence()
    draft = _draft(
        evidence_diff_ids=(diff.diff_id, diff.diff_id),
        evidence_compatibility_ids=(
            compatibility_evidence.compatibility_evidence_id,
            compatibility_evidence.compatibility_evidence_id,
        ),
    )

    (finding,) = materialize_finding_drafts(
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        finding_policy_id=FINDING_POLICY_ID,
        finding_policy_version=FINDING_POLICY_VERSION,
        drafts=(draft,),
        diffs=(diff,),
        compatibility_evidence=(compatibility_evidence,),
    )

    assert finding.evidence_diff_ids == (diff.diff_id,)
    assert finding.evidence_compatibility_ids == (compatibility_evidence.compatibility_evidence_id,)


def test_materialization_accepts_empty_draft_output() -> None:
    assert (
        materialize_finding_drafts(
            monitoring_run_id=MONITORING_RUN_ID,
            source_run_id=SOURCE_RUN_ID,
            finding_policy_id=FINDING_POLICY_ID,
            finding_policy_version=FINDING_POLICY_VERSION,
            drafts=(),
            diffs=(),
            compatibility_evidence=(),
        )
        == ()
    )


def test_materialization_deduplicates_identical_drafts() -> None:
    diff = _diff()
    draft = _draft(
        evidence_diff_ids=(diff.diff_id,),
        evidence_compatibility_ids=(),
    )

    findings = materialize_finding_drafts(
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        finding_policy_id=FINDING_POLICY_ID,
        finding_policy_version=FINDING_POLICY_VERSION,
        drafts=(draft, draft),
        diffs=(diff,),
        compatibility_evidence=(),
    )

    assert len(findings) == 1


def test_materialization_deduplicates_identity_equivalent_evidence_order() -> None:
    diffs = (_diff(metric_name="accuracy"), _diff(metric_name="precision"))
    draft = _draft(
        evidence_diff_ids=tuple(diff.diff_id for diff in diffs),
        evidence_compatibility_ids=(),
    )
    reordered = replace(draft, evidence_diff_ids=tuple(reversed(draft.evidence_diff_ids)))

    findings = materialize_finding_drafts(
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        finding_policy_id=FINDING_POLICY_ID,
        finding_policy_version=FINDING_POLICY_VERSION,
        drafts=(draft, reordered),
        diffs=diffs,
        compatibility_evidence=(),
    )

    assert len(findings) == 1


def test_materialization_rejects_conflicting_content_under_one_identity() -> None:
    diff = _diff()
    draft = _draft(
        evidence_diff_ids=(diff.diff_id,),
        evidence_compatibility_ids=(),
    )
    changed = replace(draft, summary="Changed summary under the same identity.")

    with pytest.raises(InvariantViolation) as exc_info:
        materialize_finding_drafts(
            monitoring_run_id=MONITORING_RUN_ID,
            source_run_id=SOURCE_RUN_ID,
            finding_policy_id=FINDING_POLICY_ID,
            finding_policy_version=FINDING_POLICY_VERSION,
            drafts=(draft, changed),
            diffs=(diff,),
            compatibility_evidence=(),
        )

    error = exc_info.value
    assert error.code == "finding_identity_content_conflict"
    assert error.entity == "Finding"
    assert error.field == "finding_id"


def test_materialization_rejects_unknown_draft_evidence() -> None:
    draft = _draft(
        evidence_diff_ids=("diff-v1-unknown",),
        evidence_compatibility_ids=(),
    )

    with pytest.raises(InvariantViolation) as exc_info:
        materialize_finding_drafts(
            monitoring_run_id=MONITORING_RUN_ID,
            source_run_id=SOURCE_RUN_ID,
            finding_policy_id=FINDING_POLICY_ID,
            finding_policy_version=FINDING_POLICY_VERSION,
            drafts=(draft,),
            diffs=(),
            compatibility_evidence=(),
        )

    assert exc_info.value.code == "finding_diff_evidence_violation"


def test_materialization_rejects_cross_pair_draft_evidence() -> None:
    diff = _diff(source_run_id="train-run-other")
    draft = _draft(
        evidence_diff_ids=(diff.diff_id,),
        evidence_compatibility_ids=(),
    )

    with pytest.raises(InvariantViolation) as exc_info:
        materialize_finding_drafts(
            monitoring_run_id=MONITORING_RUN_ID,
            source_run_id=SOURCE_RUN_ID,
            finding_policy_id=FINDING_POLICY_ID,
            finding_policy_version=FINDING_POLICY_VERSION,
            drafts=(draft,),
            diffs=(diff,),
            compatibility_evidence=(),
        )

    assert exc_info.value.code == "finding_diff_evidence_violation"
    assert exc_info.value.field == "source_run_id"


def test_materialization_rejects_a_non_draft_policy_value() -> None:
    with pytest.raises(InvariantViolation) as exc_info:
        materialize_finding_drafts(
            monitoring_run_id=MONITORING_RUN_ID,
            source_run_id=SOURCE_RUN_ID,
            finding_policy_id=FINDING_POLICY_ID,
            finding_policy_version=FINDING_POLICY_VERSION,
            drafts=(object(),),  # type: ignore[arg-type]
            diffs=(),
            compatibility_evidence=(),
        )

    error = exc_info.value
    assert error.code == "finding_draft_invalid"
    assert error.entity == "FindingDraft"


def test_materialization_orders_distinct_findings_by_identity() -> None:
    diff = _diff()
    first = _draft(
        finding_rule_id="quality.z_rule",
        evidence_diff_ids=(diff.diff_id,),
        evidence_compatibility_ids=(),
    )
    second = replace(first, finding_rule_id="quality.a_rule")

    findings = materialize_finding_drafts(
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        finding_policy_id=FINDING_POLICY_ID,
        finding_policy_version=FINDING_POLICY_VERSION,
        drafts=(first, second),
        diffs=(diff,),
        compatibility_evidence=(),
    )

    assert tuple(finding.finding_id for finding in findings) == tuple(
        sorted(finding.finding_id for finding in findings)
    )
