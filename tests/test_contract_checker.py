"""Unit tests for the contract checker boundary."""

from dataclasses import dataclass

from mlflow_monitor.contract_checker import ContractChecker, ContractEvaluationContext
from mlflow_monitor.domain import (
    Baseline,
    ComparabilityStatus,
    Contract,
    ContractCheckReason,
    ContractCheckResult,
)

CONTRACT = Contract(
    contract_id="default",
    version="v0",
    schema_contract_ref=None,
    feature_contract_ref=None,
    metric_contract_ref=None,
    data_scope_contract_ref=None,
    execution_contract_ref=None,
)

BASELINE = Baseline(
    timeline_id="timeline-1",
    source_run_id="train-run-1",
    model_identity="model-a",
    parameter_fingerprint="params-v1",
    data_snapshot_ref="dataset-2026-03-01",
    run_config_ref="config-v1",
    metric_snapshot={"f1": 0.87},
    environment_context={"python": "3.12"},
)

CONTEXT = ContractEvaluationContext(
    subject_id="churn_model",
    source_run_id="train-run-2",
    baseline=BASELINE,
    current_metrics={"f1": 0.85},
    current_environment={"python": "3.12"},
    current_features=("age", "income"),
    current_schema={"age": "int", "income": "float"},
    current_data_scope="validation:2026-03-10",
)


@dataclass
class StubContractChecker:
    """Stub checker used to validate the workflow-facing checker protocol."""

    result: ContractCheckResult

    def check(
        self,
        contract: Contract,
        context: ContractEvaluationContext,
    ) -> ContractCheckResult:
        assert contract is CONTRACT
        assert context is CONTEXT
        return self.result


def test_contract_checker_protocol_accepts_stub_checker() -> None:
    """Checker boundary should accept a single coordinator implementation."""
    checker: ContractChecker = StubContractChecker(
        result=ContractCheckResult(
            status=ComparabilityStatus.WARN,
            reasons=(
                ContractCheckReason(
                    code="environment_mismatch",
                    message="Python version differs.",
                    blocking=False,
                ),
            ),
        ),
    )

    result = checker.check(CONTRACT, CONTEXT)

    assert result.status is ComparabilityStatus.WARN


def test_contract_evaluation_context_is_platform_agnostic() -> None:
    """Evaluation context should carry resolved evidence, not MLflow objects."""
    assert CONTEXT.subject_id == "churn_model"
    assert CONTEXT.source_run_id == "train-run-2"
    assert CONTEXT.baseline is BASELINE
    assert CONTEXT.current_metrics == {"f1": 0.85}
    assert CONTEXT.current_environment == {"python": "3.12"}
    assert CONTEXT.current_features == ("age", "income")
    assert CONTEXT.current_schema == {"age": "int", "income": "float"}
    assert CONTEXT.current_data_scope == "validation:2026-03-10"
