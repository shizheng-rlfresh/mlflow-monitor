"""Unit tests for the contract checker boundary."""

from dataclasses import dataclass

from mlflow_monitor.contract_checker import (
    ContractChecker,
    ContractEvaluationContext,
    ContractEvidence,
    DefaultContractChecker,
)
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

BASELINE_EVIDENCE = ContractEvidence(
    metrics={"f1": 0.87},
    environment={"python": "3.12"},
    features=("age", "income"),
    schema={"age": "int", "income": "float"},
    data_scope="validation:2026-03-01",
)

CURRENT_EVIDENCE = ContractEvidence(
    metrics={"f1": 0.85},
    environment={"python": "3.12"},
    features=("age", "income"),
    schema={"age": "int", "income": "float"},
    data_scope="validation:2026-03-10",
)

CONTEXT = ContractEvaluationContext(
    subject_id="churn_model",
    source_run_id="train-run-2",
    baseline_source_run_id="train-run-1",
    baseline_evidence=BASELINE_EVIDENCE,
    current_evidence=CURRENT_EVIDENCE,
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
    assert CONTEXT.baseline_source_run_id == "train-run-1"
    assert CONTEXT.baseline_evidence is BASELINE_EVIDENCE
    assert CONTEXT.current_evidence is CURRENT_EVIDENCE


def test_default_contract_checker_returns_pass_when_no_checks_are_enabled() -> None:
    """Concrete checker should pass when the contract enables no checks."""
    contract = Contract(
        contract_id="default_permissive",
        version="v0",
        schema_contract_ref=None,
        feature_contract_ref=None,
        metric_contract_ref=None,
        data_scope_contract_ref=None,
        execution_contract_ref=None,
    )
    context = ContractEvaluationContext(
        subject_id="churn_model",
        source_run_id="train-run-2",
        baseline_source_run_id="train-run-1",
        baseline_evidence=ContractEvidence(
            metrics={"f1": 0.87},
            environment={"python": "3.12"},
            features=("age", "income"),
            schema={"age": "int", "income": "float"},
            data_scope="validation:2026-03-01",
        ),
        current_evidence=ContractEvidence(
            metrics={"f1": 0.85},
            environment={"python": "3.12"},
            features=("age", "income"),
            schema={"age": "int", "income": "float"},
            data_scope="validation:2026-03-10",
        ),
    )

    checker = DefaultContractChecker()

    result = checker.check(contract, context)

    assert result.status is ComparabilityStatus.PASS
    assert result.reasons == ()


def test_default_contract_checker_warns_for_execution_environment_mismatch() -> None:
    """Concrete checker should warn when execution checking is enabled and env differs."""
    contract = Contract(
        contract_id="env_repro",
        version="v0",
        schema_contract_ref=None,
        feature_contract_ref=None,
        metric_contract_ref=None,
        data_scope_contract_ref=None,
        execution_contract_ref="builtin:env_repro",
    )
    context = ContractEvaluationContext(
        subject_id="churn_model",
        source_run_id="train-run-2",
        baseline_source_run_id="train-run-1",
        baseline_evidence=ContractEvidence(
            metrics={"f1": 0.87},
            environment={"python": "3.12"},
            features=("age", "income"),
            schema={"age": "int", "income": "float"},
            data_scope="validation:2026-03-01",
        ),
        current_evidence=ContractEvidence(
            metrics={"f1": 0.85},
            environment={"python": "3.11"},
            features=("age", "income"),
            schema={"age": "int", "income": "float"},
            data_scope="validation:2026-03-10",
        ),
    )

    checker = DefaultContractChecker()

    result = checker.check(contract, context)

    assert result.status is ComparabilityStatus.WARN
    assert result.reasons == (
        ContractCheckReason(
            code="environment_mismatch",
            message="Execution environment does not match the baseline.",
            blocking=False,
        ),
    )


def test_default_contract_checker_fails_for_schema_mismatch() -> None:
    """Concrete checker should fail when schema checking is enabled and schema differs."""
    contract = Contract(
        contract_id="schema_exact",
        version="v0",
        schema_contract_ref="builtin:schema_exact",
        feature_contract_ref=None,
        metric_contract_ref=None,
        data_scope_contract_ref=None,
        execution_contract_ref=None,
    )
    context = ContractEvaluationContext(
        subject_id="churn_model",
        source_run_id="train-run-2",
        baseline_source_run_id="train-run-1",
        baseline_evidence=ContractEvidence(
            metrics={"f1": 0.87},
            environment={"python": "3.12"},
            features=("age", "income"),
            schema={"age": "int", "income": "float"},
            data_scope="validation:2026-03-01",
        ),
        current_evidence=ContractEvidence(
            metrics={"f1": 0.85},
            environment={"python": "3.12"},
            features=("age", "income"),
            schema={"age": "int", "income": "double"},
            data_scope="validation:2026-03-10",
        ),
    )

    checker = DefaultContractChecker()

    result = checker.check(contract, context)

    assert result.status is ComparabilityStatus.FAIL
    assert result.reasons == (
        ContractCheckReason(
            code="schema_mismatch",
            message="Data schema does not match the baseline.",
            blocking=True,
        ),
    )


def test_default_contract_checker_fails_for_feature_mismatch() -> None:
    """Concrete checker should fail when feature checking is enabled and features differ."""
    contract = Contract(
        contract_id="feature_exact",
        version="v0",
        schema_contract_ref=None,
        feature_contract_ref="builtin:feature_exact",
        metric_contract_ref=None,
        data_scope_contract_ref=None,
        execution_contract_ref=None,
    )
    context = ContractEvaluationContext(
        subject_id="churn_model",
        source_run_id="train-run-2",
        baseline_source_run_id="train-run-1",
        baseline_evidence=ContractEvidence(
            metrics={"f1": 0.87},
            environment={"python": "3.12"},
            features=("age", "income"),
            schema={"age": "int", "income": "float"},
            data_scope="validation:2026-03-01",
        ),
        current_evidence=ContractEvidence(
            metrics={"f1": 0.85},
            environment={"python": "3.12"},
            features=("age", "income", "tenure"),
            schema={"age": "int", "income": "float"},
            data_scope="validation:2026-03-10",
        ),
    )

    checker = DefaultContractChecker()

    result = checker.check(contract, context)

    assert result.status is ComparabilityStatus.FAIL
    assert result.reasons == (
        ContractCheckReason(
            code="feature_mismatch",
            message="Feature set does not match the baseline.",
            blocking=True,
        ),
    )


def test_default_contract_checker_fails_for_data_scope_mismatch() -> None:
    """Concrete checker should fail when data-scope checking is enabled and scope differs."""
    contract = Contract(
        contract_id="data_scope_exact",
        version="v0",
        schema_contract_ref=None,
        feature_contract_ref=None,
        metric_contract_ref=None,
        data_scope_contract_ref="builtin:data_scope_exact",
        execution_contract_ref=None,
    )
    context = ContractEvaluationContext(
        subject_id="churn_model",
        source_run_id="train-run-2",
        baseline_source_run_id="train-run-1",
        baseline_evidence=ContractEvidence(
            metrics={"f1": 0.87},
            environment={"python": "3.12"},
            features=("age", "income"),
            schema={"age": "int", "income": "float"},
            data_scope="validation:2026-03-01",
        ),
        current_evidence=ContractEvidence(
            metrics={"f1": 0.85},
            environment={"python": "3.12"},
            features=("age", "income"),
            schema={"age": "int", "income": "float"},
            data_scope="validation:2026-03-10",
        ),
    )

    checker = DefaultContractChecker()

    result = checker.check(contract, context)

    assert result.status is ComparabilityStatus.FAIL
    assert result.reasons == (
        ContractCheckReason(
            code="data_scope_mismatch",
            message="Data scope does not match the baseline.",
            blocking=True,
        ),
    )


def test_default_contract_checker_fails_when_blocking_and_warning_reasons_coexist() -> None:
    """Concrete checker should fail when any blocking reason exists."""
    contract = Contract(
        contract_id="env_and_schema_exact",
        version="v0",
        schema_contract_ref="builtin:schema_exact",
        feature_contract_ref=None,
        metric_contract_ref=None,
        data_scope_contract_ref=None,
        execution_contract_ref="builtin:env_repro",
    )
    context = ContractEvaluationContext(
        subject_id="churn_model",
        source_run_id="train-run-2",
        baseline_source_run_id="train-run-1",
        baseline_evidence=ContractEvidence(
            metrics={"f1": 0.87},
            environment={"python": "3.12"},
            features=("age", "income"),
            schema={"age": "int", "income": "float"},
            data_scope="validation:2026-03-01",
        ),
        current_evidence=ContractEvidence(
            metrics={"f1": 0.85},
            environment={"python": "3.11"},
            features=("age", "income"),
            schema={"age": "int", "income": "double"},
            data_scope="validation:2026-03-10",
        ),
    )

    checker = DefaultContractChecker()

    result = checker.check(contract, context)

    assert result.status is ComparabilityStatus.FAIL
    assert result.reasons == (
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
    )


def test_default_contract_checker_ignores_metric_differences_for_deferred_metric_check() -> None:
    """Concrete checker should not emit metric mismatch reasons in D-004D."""
    contract = Contract(
        contract_id="metric_exact_deferred",
        version="v0",
        schema_contract_ref=None,
        feature_contract_ref=None,
        metric_contract_ref="builtin:metric_exact",
        data_scope_contract_ref=None,
        execution_contract_ref=None,
    )
    context = ContractEvaluationContext(
        subject_id="churn_model",
        source_run_id="train-run-2",
        baseline_source_run_id="train-run-1",
        baseline_evidence=ContractEvidence(
            metrics={"f1": 0.87},
            environment={"python": "3.12"},
            features=("age", "income"),
            schema={"age": "int", "income": "float"},
            data_scope="validation:2026-03-01",
        ),
        current_evidence=ContractEvidence(
            metrics={"f1": 0.55},
            environment={"python": "3.12"},
            features=("age", "income"),
            schema={"age": "int", "income": "float"},
            data_scope="validation:2026-03-10",
        ),
    )

    checker = DefaultContractChecker()

    result = checker.check(contract, context)

    assert result.status is ComparabilityStatus.PASS
    assert result.reasons == ()


def test_default_contract_checker_emits_reasons_in_deterministic_dimension_order() -> None:
    """Concrete checker should preserve a stable reason ordering across dimensions."""
    contract = Contract(
        contract_id="all_dimensions",
        version="v0",
        schema_contract_ref="builtin:schema_exact",
        feature_contract_ref="builtin:feature_exact",
        metric_contract_ref="builtin:metric_exact",
        data_scope_contract_ref="builtin:data_scope_exact",
        execution_contract_ref="builtin:env_repro",
    )
    context = ContractEvaluationContext(
        subject_id="churn_model",
        source_run_id="train-run-2",
        baseline_source_run_id="train-run-1",
        baseline_evidence=ContractEvidence(
            metrics={"f1": 0.87},
            environment={"python": "3.12"},
            features=("age", "income"),
            schema={"age": "int", "income": "float"},
            data_scope="validation:2026-03-01",
        ),
        current_evidence=ContractEvidence(
            metrics={"f1": 0.55},
            environment={"python": "3.11"},
            features=("age", "income", "tenure"),
            schema={"age": "int", "income": "double"},
            data_scope="validation:2026-03-10",
        ),
    )

    checker = DefaultContractChecker()

    result = checker.check(contract, context)

    assert result.status is ComparabilityStatus.FAIL
    assert result.reasons == (
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
        ContractCheckReason(
            code="feature_mismatch",
            message="Feature set does not match the baseline.",
            blocking=True,
        ),
        ContractCheckReason(
            code="data_scope_mismatch",
            message="Data scope does not match the baseline.",
            blocking=True,
        ),
    )


def test_default_contract_checker_returns_pass_when_enabled_checks_match() -> None:
    """Concrete checker should pass when enabled checks find no mismatches."""
    contract = Contract(
        contract_id="env_repro",
        version="v0",
        schema_contract_ref=None,
        feature_contract_ref=None,
        metric_contract_ref=None,
        data_scope_contract_ref=None,
        execution_contract_ref="builtin:env_repro",
    )
    context = ContractEvaluationContext(
        subject_id="churn_model",
        source_run_id="train-run-2",
        baseline_source_run_id="train-run-1",
        baseline_evidence=ContractEvidence(
            metrics={"f1": 0.87},
            environment={"python": "3.12"},
            features=("age", "income"),
            schema={"age": "int", "income": "float"},
            data_scope="validation:2026-03-01",
        ),
        current_evidence=ContractEvidence(
            metrics={"f1": 0.85},
            environment={"python": "3.12"},
            features=("age", "income"),
            schema={"age": "int", "income": "float"},
            data_scope="validation:2026-03-10",
        ),
    )

    checker = DefaultContractChecker()

    result = checker.check(contract, context)

    assert result.status is ComparabilityStatus.PASS
    assert result.reasons == ()
