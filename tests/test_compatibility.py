"""Specifications for Compatibility Evidence materialization and projection."""

from workflow._support import _CONTRACT, make_prepared_context

from mlflow_monitor.compatibility import (
    compatibility_evidence_to_dict,
    materialize_compatibility_evidence,
)
from mlflow_monitor.domain import (
    ComparabilityStatus,
    CompatibilityEvidence,
    ContractCheckReason,
    ContractCheckResult,
)

ENVIRONMENT_EVIDENCE_ID = (
    "compatibility-evidence-v1-e1a5ea244a761da1a25a1a3712e9324339df45c99179ebb8f5117e1c3cfd7f36"
)
SCHEMA_EVIDENCE_ID = (
    "compatibility-evidence-v1-0a27ab75592628b1f6b95d27e0b60e9310397359565d7e789614dbf2249fc09b"
)


def _failed_check_result() -> ContractCheckResult:
    return ContractCheckResult(
        status=ComparabilityStatus.FAIL,
        reasons=(
            ContractCheckReason(
                code="environment_mismatch",
                message="Execution environment does not match the baseline.",
                blocking=False,
            ),
            ContractCheckReason(
                code="schema_mismatch",
                message="Data schema does not match the baseline.",
                blocking=True,
            ),
        ),
    )


def test_materialize_compatibility_evidence_preserves_complete_ordered_reasons() -> None:
    prepared_context = make_prepared_context(contract=_CONTRACT)
    contract_check_result = _failed_check_result()

    evidence_records = materialize_compatibility_evidence(
        prepared_context,
        contract_check_result,
    )

    assert evidence_records == (
        CompatibilityEvidence(
            compatibility_evidence_id=ENVIRONMENT_EVIDENCE_ID,
            monitoring_run_id=prepared_context.monitoring_run_id,
            source_run_id=prepared_context.source_run_id,
            baseline_source_run_id=prepared_context.baseline_source_run_id,
            contract_id=prepared_context.contract.contract_id,
            contract_version=prepared_context.contract.contract_version,
            reason=contract_check_result.reasons[0],
        ),
        CompatibilityEvidence(
            compatibility_evidence_id=SCHEMA_EVIDENCE_ID,
            monitoring_run_id=prepared_context.monitoring_run_id,
            source_run_id=prepared_context.source_run_id,
            baseline_source_run_id=prepared_context.baseline_source_run_id,
            contract_id=prepared_context.contract.contract_id,
            contract_version=prepared_context.contract.contract_version,
            reason=contract_check_result.reasons[1],
        ),
    )
    assert tuple(record.reason for record in evidence_records) == contract_check_result.reasons
    assert (
        materialize_compatibility_evidence(prepared_context, contract_check_result)
        == evidence_records
    )


def test_compatibility_evidence_to_dict_projects_complete_ordered_records() -> None:
    prepared_context = make_prepared_context(contract=_CONTRACT)
    contract_check_result = _failed_check_result()
    evidence_records = materialize_compatibility_evidence(
        prepared_context,
        contract_check_result,
    )

    assert compatibility_evidence_to_dict(prepared_context, evidence_records) == {
        "artifact_schema_version": "v0",
        "monitoring_run_id": "monitoring-run-1",
        "source_run_id": "train-run-123",
        "baseline_source_run_id": "train-run-baseline",
        "contract_id": "default_permissive",
        "contract_version": "v0",
        "evidence": [
            {
                "compatibility_evidence_id": ENVIRONMENT_EVIDENCE_ID,
                "reason": {
                    "code": "environment_mismatch",
                    "message": "Execution environment does not match the baseline.",
                    "blocking": False,
                },
            },
            {
                "compatibility_evidence_id": SCHEMA_EVIDENCE_ID,
                "reason": {
                    "code": "schema_mismatch",
                    "message": "Data schema does not match the baseline.",
                    "blocking": True,
                },
            },
        ],
    }


def test_pass_materializes_and_projects_empty_evidence() -> None:
    prepared_context = make_prepared_context(contract=_CONTRACT)
    contract_check_result = ContractCheckResult(
        status=ComparabilityStatus.PASS,
        reasons=(),
    )

    evidence_records = materialize_compatibility_evidence(
        prepared_context,
        contract_check_result,
    )

    assert evidence_records == ()
    assert compatibility_evidence_to_dict(prepared_context, evidence_records) == {
        "artifact_schema_version": "v0",
        "monitoring_run_id": "monitoring-run-1",
        "source_run_id": "train-run-123",
        "baseline_source_run_id": "train-run-baseline",
        "contract_id": "default_permissive",
        "contract_version": "v0",
        "evidence": [],
    }
