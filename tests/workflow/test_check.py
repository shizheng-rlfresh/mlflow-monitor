import pytest

from mlflow_monitor.contract import SYSTEM_DEFAULT_CONTRACT_ID
from mlflow_monitor.contract_checker import DefaultContractChecker
from mlflow_monitor.domain import (
    ComparabilityStatus,
    Contract,
    ContractCheckReason,
    ContractCheckResult,
)
from mlflow_monitor.errors import CheckStageError, GatewayConsistencyViolation
from mlflow_monitor.gateway import GatewayConfig, InMemoryMonitoringGateway
from mlflow_monitor.workflow import (
    contract_check_result_to_dict,
    execute_contract_check,
    hydrate_contract_check_result,
)

from ._support import _CONTRACT, make_prepared_context


class RaisingContractChecker:
    """Test double whose contract check execution raises an exception."""

    def check(self, contract: Contract, context: object) -> ContractCheckResult:
        """Raise a deterministic checker failure."""
        del contract, context
        raise RuntimeError("checker exploded")


class InvalidResultContractChecker:
    """Test double returning an invariant-invalid contract check result."""

    def check(self, contract: Contract, context: object) -> ContractCheckResult:
        """Return an invalid result shape for workflow validation tests."""
        del contract, context
        return ContractCheckResult(
            status=ComparabilityStatus.PASS,
            reasons=(
                ContractCheckReason(
                    code="environment_mismatch",
                    message="Execution environment does not match the baseline.",
                    blocking=False,
                ),
            ),
        )


class DuplicateReasonContractChecker:
    """Test double returning duplicate Contract Check reason codes."""

    def check(self, contract: Contract, context: object) -> ContractCheckResult:
        """Return a result whose reason codes violate uniqueness."""
        del contract, context
        reason = ContractCheckReason(
            code="environment_mismatch",
            message="Execution environment does not match the baseline.",
            blocking=False,
        )
        return ContractCheckResult(
            status=ComparabilityStatus.WARN,
            reasons=(reason, reason),
        )


def test_contract_check_result_to_dict_preserves_complete_ordered_output() -> None:
    """Contract Check artifacts should retain identity and exact reason order."""
    context = make_prepared_context(contract=_CONTRACT)
    result = ContractCheckResult(
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

    assert contract_check_result_to_dict(context, result) == {
        "artifact_schema_version": "v0",
        "monitoring_run_id": "monitoring-run-1",
        "source_run_id": "train-run-123",
        "contract_id": SYSTEM_DEFAULT_CONTRACT_ID,
        "contract_version": "v0",
        "status": "fail",
        "reasons": [
            {
                "code": "environment_mismatch",
                "message": "Execution environment does not match the baseline.",
                "blocking": False,
            },
            {
                "code": "schema_mismatch",
                "message": "Data schema does not match the baseline.",
                "blocking": True,
            },
        ],
    }


def test_hydrate_contract_check_result_preserves_complete_ordered_output() -> None:
    """Hydration should reconstruct exact persisted reason content and order."""
    context = make_prepared_context(contract=_CONTRACT)
    expected = ContractCheckResult(
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

    hydrated = hydrate_contract_check_result(
        contract_check_result_to_dict(context, expected),
        prepared_context=context,
        projected_comparability_status=ComparabilityStatus.FAIL,
    )

    assert hydrated == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {"artifact_schema_version": "v0"},
    ],
)
def test_hydrate_contract_check_result_rejects_missing_or_malformed_artifact(
    raw: dict[str, object] | None,
) -> None:
    """Committed Check hydration should fail closed with bounded diagnostics."""
    context = make_prepared_context(contract=_CONTRACT)

    with pytest.raises(GatewayConsistencyViolation) as exc_info:
        hydrate_contract_check_result(
            raw,
            prepared_context=context,
            projected_comparability_status=ComparabilityStatus.PASS,
        )

    assert exc_info.value.code == "monitoring_run_json_artifact_inconsistent"
    assert exc_info.value.details == (
        ("monitoring_run_id", context.monitoring_run_id),
        ("path", "outputs/contract_check.json"),
    )


def test_hydrate_contract_check_result_rejects_persisted_duplicate_reason_codes() -> None:
    """Persisted duplicate reasons are consistency failures, not Check failures."""
    context = make_prepared_context(contract=_CONTRACT)
    reason = {
        "code": "environment_mismatch",
        "message": "Execution environment does not match the baseline.",
        "blocking": False,
    }
    raw = contract_check_result_to_dict(
        context,
        ContractCheckResult(
            status=ComparabilityStatus.WARN,
            reasons=(ContractCheckReason(**reason),),
        ),
    )
    raw["reasons"] = [reason, reason]

    with pytest.raises(GatewayConsistencyViolation) as exc_info:
        hydrate_contract_check_result(
            raw,
            prepared_context=context,
            projected_comparability_status=ComparabilityStatus.WARN,
        )

    assert exc_info.value.code == "monitoring_run_json_artifact_inconsistent"


def test_hydrate_contract_check_result_rejects_projection_disagreement() -> None:
    """A nonempty contradictory comparability projection should fail closed."""
    context = make_prepared_context(contract=_CONTRACT)
    raw = contract_check_result_to_dict(
        context,
        ContractCheckResult(status=ComparabilityStatus.PASS, reasons=()),
    )

    with pytest.raises(GatewayConsistencyViolation):
        hydrate_contract_check_result(
            raw,
            prepared_context=context,
            projected_comparability_status=ComparabilityStatus.FAIL,
        )


def test_hydrate_contract_check_result_accepts_missing_projection() -> None:
    """A missing comparability projection is noncontradictory."""
    context = make_prepared_context(contract=_CONTRACT)
    expected = ContractCheckResult(status=ComparabilityStatus.PASS, reasons=())

    hydrated = hydrate_contract_check_result(
        contract_check_result_to_dict(context, expected),
        prepared_context=context,
        projected_comparability_status=None,
    )

    assert hydrated == expected


def test_execute_contract_check_returns_warn_result_for_environment_mismatch() -> None:
    """Check should return the canonical warning result for env mismatch."""
    contract = Contract(
        contract_id="env_repro",
        contract_version="v0",
        schema_contract_ref=None,
        feature_contract_ref=None,
        metric_contract_ref=None,
        data_scope_contract_ref=None,
        execution_contract_ref="builtin:env_repro",
    )
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-baseline",
        source_experiment="training/churn",
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-123",
        source_experiment="training/churn",
        metrics={"f1": 0.89, "auc": 0.94},
        artifacts=("metrics.json",),
        environment={"python": "3.11"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    result = execute_contract_check(
        prepared_context=make_prepared_context(contract=contract),
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    assert result == ContractCheckResult(
        status=ComparabilityStatus.WARN,
        reasons=(
            ContractCheckReason(
                code="environment_mismatch",
                message="Execution environment does not match the baseline.",
                blocking=False,
            ),
        ),
    )


def test_execute_contract_check_fails_when_baseline_evidence_is_missing() -> None:
    """Check should fail explicitly when baseline evidence cannot be loaded."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-123",
        source_experiment="training/churn",
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(CheckStageError) as exc_info:
        execute_contract_check(
            prepared_context=make_prepared_context(contract=_CONTRACT),
            gateway=gateway,
            contract_checker=DefaultContractChecker(),
        )

    assert exc_info.value.code == "check_missing_baseline_evidence"


def test_execute_contract_check_fails_when_current_evidence_is_missing() -> None:
    """Check should fail explicitly when current evidence cannot be loaded."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-baseline",
        source_experiment="training/churn",
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(CheckStageError) as exc_info:
        execute_contract_check(
            prepared_context=make_prepared_context(contract=_CONTRACT),
            gateway=gateway,
            contract_checker=DefaultContractChecker(),
        )

    assert exc_info.value.code == "check_missing_current_evidence"


def test_execute_contract_check_propagates_checker_failures() -> None:
    """Check should surface unexpected checker runtime failures."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-baseline",
        source_experiment="training/churn",
        metrics={"f1": 0.87, "auc": 0.93},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-123",
        source_experiment="training/churn",
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(RuntimeError, match="checker exploded"):
        execute_contract_check(
            prepared_context=make_prepared_context(contract=_CONTRACT),
            gateway=gateway,
            contract_checker=RaisingContractChecker(),
        )


def test_execute_contract_check_rejects_invalid_checker_result() -> None:
    """Check should reject results that violate contract-check invariants."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-baseline",
        source_experiment="training/churn",
        metrics={"f1": 0.87, "auc": 0.93},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-123",
        source_experiment="training/churn",
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(CheckStageError) as exc_info:
        execute_contract_check(
            prepared_context=make_prepared_context(contract=_CONTRACT),
            gateway=gateway,
            contract_checker=InvalidResultContractChecker(),
        )

    assert exc_info.value.code == "check_result_invalid"


def test_execute_contract_check_rejects_duplicate_reason_codes() -> None:
    """Check should reject checker results with duplicate reason codes."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    for source_run_id in ("train-run-baseline", "train-run-123"):
        gateway.add_source_run(
            subject_id="churn_model",
            source_run_id=source_run_id,
            source_experiment="training/churn",
            metrics={"f1": 0.91, "auc": 0.95},
            artifacts=("metrics.json",),
            environment={"python": "3.12"},
            features=("age", "income"),
            schema={"age": "int", "income": "float"},
            data_scope="validation:2026-03-01",
        )

    with pytest.raises(CheckStageError) as exc_info:
        execute_contract_check(
            prepared_context=make_prepared_context(contract=_CONTRACT),
            gateway=gateway,
            contract_checker=DuplicateReasonContractChecker(),
        )

    assert exc_info.value.code == "check_result_invalid"
