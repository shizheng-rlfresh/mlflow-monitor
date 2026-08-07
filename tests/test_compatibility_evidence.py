"""Specifications for run-scoped Compatibility Evidence."""

from dataclasses import FrozenInstanceError, fields

import pytest

from mlflow_monitor.domain import CompatibilityEvidence, ContractCheckReason


def _evidence(**overrides: object) -> CompatibilityEvidence:
    values: dict[str, object] = {
        "compatibility_evidence_id": "compatibility-evidence-v1-example",
        "monitoring_run_id": "monitoring-run-current",
        "source_run_id": "train-run-current",
        "baseline_source_run_id": "train-run-baseline",
        "contract_id": "default_permissive",
        "contract_version": "v0",
        "reason": ContractCheckReason(
            code="environment_mismatch",
            message="Execution environment does not match the baseline.",
            blocking=False,
        ),
    }
    values.update(overrides)
    return CompatibilityEvidence(**values)  # type: ignore[arg-type]


def test_compatibility_evidence_has_approved_shape() -> None:
    evidence = _evidence()

    assert tuple(field.name for field in fields(evidence)) == (
        "compatibility_evidence_id",
        "monitoring_run_id",
        "source_run_id",
        "baseline_source_run_id",
        "contract_id",
        "contract_version",
        "reason",
    )


def test_compatibility_evidence_is_immutable() -> None:
    evidence = _evidence()

    with pytest.raises(FrozenInstanceError):
        evidence.reason = ContractCheckReason(  # type: ignore[misc]
            code="schema_mismatch",
            message="Data schema does not match the baseline.",
            blocking=True,
        )


@pytest.mark.parametrize(
    "field",
    [
        "compatibility_evidence_id",
        "monitoring_run_id",
        "source_run_id",
        "baseline_source_run_id",
        "contract_id",
        "contract_version",
    ],
)
def test_compatibility_evidence_rejects_empty_identity_fields(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _evidence(**{field: " "})


def test_compatibility_evidence_requires_a_contract_check_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        _evidence(reason=None)
