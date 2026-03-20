"""Unit tests for monitor-layer orchestration helpers."""

import pytest

from mlflow_monitor.contract_checker import DefaultContractChecker
from mlflow_monitor.domain import (
    ComparabilityStatus,
    Contract,
    ContractCheckReason,
    ContractCheckResult,
    LifecycleStatus,
)
from mlflow_monitor.errors import PrepareStageError
from mlflow_monitor.gateway import GatewayConfig, InMemoryMonitoringGateway
from mlflow_monitor.monitor import _check_stage, _prepare_stage, run
from mlflow_monitor.recipe import (
    SYSTEM_DEFAULT_CONTRACT_ID,
    RecipeReferenceCatalog,
    resolve_recipe_v0_lite,
)
from mlflow_monitor.recipe_compiler import CompiledRunPlan, compile_recipe_v0_lite


def make_compiled_run_plan(
    *,
    run_selector: str = "train-run-123",
    source_experiment: str | None = "training/churn",
    required_metrics: tuple[str, ...] = ("f1", "auc"),
    required_artifacts: tuple[str, ...] = ("metrics.json",),
    custom_reference_run_id: str | None = None,
    recipe_id: str = "default",
    contract_id: str = "default",
) -> CompiledRunPlan:
    """Build a compiled run plan using the real recipe compiler."""
    raw = {
        "identity": {"recipe_id": recipe_id, "version": "v0"},
        "input_binding": {
            "run_selector": run_selector,
            "source_experiment": source_experiment,
            "required_metrics": list(required_metrics),
            "required_artifacts": list(required_artifacts),
            "custom_reference_run_id": custom_reference_run_id,
        },
        "contract_binding": {"contract_id": contract_id},
        "metrics_slices": {"metrics": ["f1", "auc"], "slices": ["region", "segment"]},
        "finding_policy": {"profile": "default_policy"},
        "output_binding": {"summary_mode": "standard"},
    }
    recipe = resolve_recipe_v0_lite(
        raw,
        references=RecipeReferenceCatalog(
            contract_ids=frozenset({"default", "env_repro", SYSTEM_DEFAULT_CONTRACT_ID}),
            finding_policy_profiles=frozenset({"default_policy"}),
            summary_modes=frozenset({"standard"}),
        ),
    )
    return compile_recipe_v0_lite(recipe)


DEFAULT_CONTRACT = Contract(
    contract_id="default",
    version="v0",
    schema_contract_ref=None,
    feature_contract_ref=None,
    metric_contract_ref=None,
    data_scope_contract_ref=None,
    execution_contract_ref=None,
)

ENV_REPRO_CONTRACT = Contract(
    contract_id="env_repro",
    version="v0",
    schema_contract_ref=None,
    feature_contract_ref=None,
    metric_contract_ref=None,
    data_scope_contract_ref=None,
    execution_contract_ref="builtin:env_repro",
)


def test_prepare_stage_returns_prepared_run_and_context() -> None:
    """Prepare orchestration should build, transition, and persist a prepared run."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.initialize_timeline("churn_model", "train-run-baseline")
    gateway.add_source_run(
        subject_id="churn_model",
        run_id="train-run-baseline",
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
        run_id="train-run-123",
        source_experiment="training/churn",
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    prepared_run, prepared_context = _prepare_stage(
        run_id="run-1",
        subject_id="churn_model",
        compiled_plan=make_compiled_run_plan(),
        resolved_contract=DEFAULT_CONTRACT,
        gateway=gateway,
    )

    assert prepared_run.lifecycle_status is LifecycleStatus.PREPARED
    assert prepared_run.comparability_status is ComparabilityStatus.PASS
    assert prepared_run.contract_check_result is None
    assert prepared_run.timeline_id == prepared_context.timeline_id
    assert prepared_run.source_run_id == prepared_context.source_run_id
    assert prepared_run.baseline_source_run_id == prepared_context.baseline_source_run_id
    assert prepared_run.contract == prepared_context.contract

    stored = gateway.get_monitoring_run("churn_model", "run-1")
    assert stored is not None
    assert stored.lifecycle_status is LifecycleStatus.PREPARED
    assert stored.sequence_index == prepared_run.sequence_index
    assert stored.contract_check_result is None


def test_prepare_stage_reserves_sequence_after_success_only() -> None:
    """Prepare orchestration should not consume sequence indices on failed prepare."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.initialize_timeline("churn_model", "train-run-baseline")
    gateway.add_source_run(
        subject_id="churn_model",
        run_id="train-run-baseline",
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
        run_id="train-run-123",
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
        run_id="train-run-456",
        source_experiment="training/churn",
        metrics={"f1": 0.92, "auc": 0.96},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-08",
    )

    first_run, _ = _prepare_stage(
        run_id="run-1",
        subject_id="churn_model",
        compiled_plan=make_compiled_run_plan(),
        resolved_contract=DEFAULT_CONTRACT,
        gateway=gateway,
    )

    with pytest.raises(PrepareStageError):
        _prepare_stage(
            run_id="run-bad",
            subject_id="churn_model",
            compiled_plan=make_compiled_run_plan(run_selector="missing-run"),
            resolved_contract=DEFAULT_CONTRACT,
            gateway=gateway,
        )

    second_run, _ = _prepare_stage(
        run_id="run-2",
        subject_id="churn_model",
        compiled_plan=make_compiled_run_plan(run_selector="train-run-456"),
        resolved_contract=DEFAULT_CONTRACT,
        gateway=gateway,
    )

    assert first_run.sequence_index == 0
    assert second_run.sequence_index == 1


def test_check_stage_updates_run_and_persists_checked_state() -> None:
    """Check orchestration should apply results, transition, and persist checked state."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.initialize_timeline("churn_model", "train-run-baseline")
    gateway.add_source_run(
        subject_id="churn_model",
        run_id="train-run-baseline",
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
        run_id="train-run-123",
        source_experiment="training/churn",
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("metrics.json",),
        environment={"python": "3.11"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    prepared_run, prepared_context = _prepare_stage(
        run_id="run-1",
        subject_id="churn_model",
        compiled_plan=make_compiled_run_plan(contract_id="env_repro"),
        resolved_contract=ENV_REPRO_CONTRACT,
        gateway=gateway,
    )

    checked_run = _check_stage(
        prepared_run=prepared_run,
        prepared_context=prepared_context,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    expected_result = ContractCheckResult(
        status=ComparabilityStatus.WARN,
        reasons=(
            ContractCheckReason(
                code="environment_mismatch",
                message="Execution environment does not match the baseline.",
                blocking=False,
            ),
        ),
    )

    assert checked_run.lifecycle_status is LifecycleStatus.CHECKED
    assert checked_run.comparability_status is ComparabilityStatus.WARN
    assert checked_run.contract_check_result == expected_result

    stored = gateway.get_monitoring_run("churn_model", "run-1")
    assert stored is not None
    assert stored.lifecycle_status is LifecycleStatus.CHECKED
    assert stored.contract_check_result == expected_result
    assert stored.comparability_status is ComparabilityStatus.WARN


def test_monitor_run_remains_unimplemented() -> None:
    """Public monitor entrypoint should remain a stub in this slice."""
    with pytest.raises(NotImplementedError):
        run()
