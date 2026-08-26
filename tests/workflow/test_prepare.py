"""Unit tests for workflow lifecycle transitions and prepare-stage behavior."""

import pytest

from mlflow_monitor.contract import SYSTEM_DEFAULT_CONTRACT_ID
from mlflow_monitor.domain import (
    ComparabilityStatus,
    ContractCheckReason,
    ContractCheckResult,
    LifecycleStatus,
)
from mlflow_monitor.errors import PrepareStageError
from mlflow_monitor.gateway import (
    GatewayConfig,
    IdempotencyKey,
    InMemoryMonitoringGateway,
    TimelineState,
)
from mlflow_monitor.recipe import SYSTEM_DEFAULT_RECIPE_ID
from mlflow_monitor.recipe_compiler import compile_recipe

from ._prepare_support import (
    _PREPARE_BASELINE_OVERRIDE_EXISTING_BASELINE,
    AliasResolvingBaselineGateway,
    RecordingBaselineClaimGateway,
    allocate_test_monitoring_run,
    make_compiled_invocation,
    make_gateway_with_timeline,
    prepare_test_context,
    reconcile_test_timeline_baseline,
)
from ._support import _CONTRACT


def test_prepare_run_context_succeeds_with_initialized_timeline() -> None:
    """Prepare should resolve references and required source-run inputs."""
    fixture = make_gateway_with_timeline()
    gateway = fixture.gateway
    gateway.set_active_lkg_monitoring_run_id("churn_model", fixture.previous_monitoring_run_id)
    compiled = make_compiled_invocation(
        custom_reference_monitoring_run_id=fixture.custom_monitoring_run_id
    )

    prepared = prepare_test_context(
        subject_id="churn_model",
        compiled_invocation=compiled,
        gateway=gateway,
    )

    assert (
        prepared.monitoring_run_id
        == gateway.idempotency_bindings("churn_model")["train-run-123|default|v0"]
    )
    assert prepared.subject_id == "churn_model"
    assert prepared.timeline_id == "timeline-churn_model"
    assert prepared.source_run_id == "train-run-123"
    assert prepared.baseline_source_run_id == "train-run-baseline"
    assert prepared.previous_monitoring_run_id == fixture.custom_monitoring_run_id
    assert prepared.active_lkg_monitoring_run_id is None
    assert prepared.reference_plan[2].unavailable_reason == "lkg_not_selected"
    assert prepared.custom_reference_monitoring_run_id == fixture.custom_monitoring_run_id
    assert prepared.contract == _CONTRACT
    assert prepared.required_metrics == ("auc", "f1")
    assert prepared.required_artifacts == ("metrics.json",)
    assert prepared.recipe_id == "default"
    assert prepared.recipe_version == "v0"
    assert prepared.contract_id == SYSTEM_DEFAULT_CONTRACT_ID


def test_prepare_run_context_bootstraps_allocated_uninitialized_timeline() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    for source_run_id in ("train-run-1", "train-run-current"):
        gateway.add_source_run(
            subject_id="churn_model",
            source_run_id=source_run_id,
            source_experiment="training/churn",
            metrics={"f1": 0.87},
            artifacts=("metrics.json",),
            environment={"python": "3.12"},
            features=("age", "income"),
            schema={"age": "int", "income": "float"},
            data_scope="validation:2026-03-01",
        )
    compiled_invocation = make_compiled_invocation(
        source_run_id="train-run-current",
        required_metrics=("f1",),
        custom_reference_monitoring_run_id=None,
    )
    compiled_recipe = compiled_invocation.compiled_recipe
    allocation = gateway.create_or_reuse_monitoring_run(
        IdempotencyKey(
            subject_id="churn_model",
            source_run_id="train-run-current",
            recipe_id=compiled_recipe.identity.recipe_id,
            recipe_version=compiled_recipe.identity.recipe_version,
        )
    )

    timeline_state = gateway.get_timeline_state("churn_model")

    assert timeline_state is not None
    assert timeline_state.timeline_id == allocation.timeline_id
    assert timeline_state.baseline_source_run_id is None

    prepared = prepare_test_context(
        subject_id="churn_model",
        compiled_invocation=compiled_invocation,
        gateway=gateway,
        baseline_source_run_id="train-run-1",
    )
    bootstrapped_timeline_state = gateway.get_timeline_state("churn_model")

    assert prepared.monitoring_run_id == allocation.monitoring_run_id
    assert prepared.timeline_id == allocation.timeline_id
    assert bootstrapped_timeline_state is not None
    assert bootstrapped_timeline_state.timeline_id == allocation.timeline_id
    assert bootstrapped_timeline_state.baseline_source_run_id == "train-run-1"


def test_prepare_run_context_succeeds_without_previous_run() -> None:
    """Prepare should tolerate a missing previous run."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    reconcile_test_timeline_baseline(gateway)
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

    prepared = prepare_test_context(
        subject_id="churn_model",
        compiled_invocation=make_compiled_invocation(custom_reference_monitoring_run_id=None),
        gateway=gateway,
    )

    assert prepared.previous_monitoring_run_id is None


def test_prepare_selects_greatest_lower_closed_monitoring_run() -> None:
    """Previous should skip ineligible states and later closed allocations."""
    fixture = make_gateway_with_timeline()
    gateway = fixture.gateway
    compiled_recipe = make_compiled_invocation().compiled_recipe

    for source_run_id, lifecycle_status in (
        ("train-run-checked", LifecycleStatus.CHECKED),
        ("train-run-failed", LifecycleStatus.FAILED),
        ("train-run-prepared", LifecycleStatus.PREPARED),
    ):
        allocation = allocate_test_monitoring_run(
            gateway,
            subject_id="churn_model",
            source_run_id=source_run_id,
            compiled_recipe=compiled_recipe,
        )
        gateway.upsert_monitoring_run(
            subject_id="churn_model",
            monitoring_run_id=allocation.monitoring_run_id,
            source_run_id=source_run_id,
            lifecycle_status=lifecycle_status,
            sequence_index=allocation.sequence_index,
        )

    closed_allocation = allocate_test_monitoring_run(
        gateway,
        subject_id="churn_model",
        source_run_id="train-run-closed-fail",
        compiled_recipe=compiled_recipe,
    )
    gateway.upsert_monitoring_run(
        subject_id="churn_model",
        monitoring_run_id=closed_allocation.monitoring_run_id,
        source_run_id="train-run-closed-fail",
        lifecycle_status=LifecycleStatus.CLOSED,
        sequence_index=closed_allocation.sequence_index,
        contract_check_result=ContractCheckResult(
            status=ComparabilityStatus.FAIL,
            reasons=(
                ContractCheckReason(
                    code="schema_mismatch",
                    message="Schema is incompatible.",
                    blocking=True,
                ),
            ),
        ),
    )
    current_allocation = allocate_test_monitoring_run(
        gateway,
        subject_id="churn_model",
        source_run_id="train-run-123",
        compiled_recipe=compiled_recipe,
    )
    later_closed_allocation = allocate_test_monitoring_run(
        gateway,
        subject_id="churn_model",
        source_run_id="train-run-later-closed",
        compiled_recipe=compiled_recipe,
    )
    gateway.upsert_monitoring_run(
        subject_id="churn_model",
        monitoring_run_id=later_closed_allocation.monitoring_run_id,
        source_run_id="train-run-later-closed",
        lifecycle_status=LifecycleStatus.CLOSED,
        sequence_index=later_closed_allocation.sequence_index,
    )

    prepared = prepare_test_context(
        subject_id="churn_model",
        compiled_invocation=make_compiled_invocation(),
        gateway=gateway,
    )

    assert prepared.sequence_index == current_allocation.sequence_index
    assert prepared.previous_monitoring_run_id == closed_allocation.monitoring_run_id


def test_prepare_run_context_allows_omitted_source_experiment_filter() -> None:
    """Prepare should resolve a raw source run when source_experiment is omitted."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    reconcile_test_timeline_baseline(gateway)
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

    prepared = prepare_test_context(
        subject_id="churn_model",
        compiled_invocation=make_compiled_invocation(
            source_experiment=None,
            custom_reference_monitoring_run_id=None,
        ),
        gateway=gateway,
    )

    assert prepared.source_run_id == "train-run-123"


def test_prepare_run_context_preserves_omitted_custom_reference() -> None:
    """Prepare should keep an omitted custom reference as None."""
    gateway = make_gateway_with_timeline().gateway

    prepared = prepare_test_context(
        subject_id="churn_model",
        compiled_invocation=make_compiled_invocation(custom_reference_monitoring_run_id=None),
        gateway=gateway,
    )

    assert prepared.custom_reference_monitoring_run_id is None


def test_prepare_run_context_fails_when_source_run_cannot_be_resolved() -> None:
    """Prepare should fail explicitly when the source run is missing."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    reconcile_test_timeline_baseline(gateway)

    with pytest.raises(PrepareStageError, match="Source training run could not be resolved"):
        prepare_test_context(
            subject_id="churn_model",
            compiled_invocation=make_compiled_invocation(custom_reference_monitoring_run_id=None),
            gateway=gateway,
        )


def test_prepare_run_context_fails_when_required_metric_is_missing() -> None:
    """Prepare should fail explicitly when a required metric is absent."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    reconcile_test_timeline_baseline(gateway)
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-123",
        source_experiment="training/churn",
        metrics={"auc": 0.95},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError, match="missing required metric"):
        prepare_test_context(
            subject_id="churn_model",
            compiled_invocation=make_compiled_invocation(
                required_metrics=("f1", "auc"),
                custom_reference_monitoring_run_id=None,
            ),
            gateway=gateway,
        )


def test_prepare_run_context_fails_when_required_artifact_is_missing() -> None:
    """Prepare should fail explicitly when a required artifact is absent."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    reconcile_test_timeline_baseline(gateway)
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-123",
        source_experiment="training/churn",
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("model.pkl",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError, match="missing required artifact"):
        prepare_test_context(
            subject_id="churn_model",
            compiled_invocation=make_compiled_invocation(
                required_artifacts=("metrics.json",),
                custom_reference_monitoring_run_id=None,
            ),
            gateway=gateway,
        )


def test_prepare_run_context_uses_invocation_owned_source_run_id() -> None:
    """Prepare should use the invocation identity rather than Recipe selection."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    reconcile_test_timeline_baseline(
        gateway,
        source_run_id="train-run-runtime",
        compiled_recipe=make_compiled_invocation(
            source_run_id="train-run-runtime",
            recipe_id=SYSTEM_DEFAULT_RECIPE_ID,
        ).compiled_recipe,
    )
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-runtime",
        source_experiment=None,
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    prepared = prepare_test_context(
        subject_id="churn_model",
        compiled_invocation=make_compiled_invocation(
            source_run_id="train-run-runtime",
            source_experiment=None,
            custom_reference_monitoring_run_id=None,
            recipe_id=SYSTEM_DEFAULT_RECIPE_ID,
            contract_id=SYSTEM_DEFAULT_CONTRACT_ID,
        ),
        gateway=gateway,
    )

    assert prepared.source_run_id == "train-run-runtime"


def test_prepare_run_context_succeeds_for_resolved_system_default_recipe() -> None:
    """Prepare should treat the built-in default recipe as a first-class runtime input."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    compiled = compile_recipe()
    reconcile_test_timeline_baseline(
        gateway,
        source_run_id="train-run-runtime",
        compiled_recipe=compiled,
    )
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-runtime",
        source_experiment=None,
        metrics={"f1": 0.91},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age",),
        schema={"age": "int"},
        data_scope="validation:2026-03-01",
    )

    prepared = prepare_test_context(
        subject_id="churn_model",
        compiled_invocation=compiled,
        gateway=gateway,
        source_run_id="train-run-runtime",
    )

    assert compiled.identity.recipe_id == SYSTEM_DEFAULT_RECIPE_ID
    assert compiled.contract.contract_id == SYSTEM_DEFAULT_CONTRACT_ID
    assert prepared.recipe_id == SYSTEM_DEFAULT_RECIPE_ID
    assert prepared.contract_id == SYSTEM_DEFAULT_CONTRACT_ID
    assert prepared.contract == compiled.contract
    assert prepared.source_run_id == "train-run-runtime"
    assert prepared.source_experiment is None
    assert prepared.required_metrics == ()
    assert prepared.required_artifacts == ()
    assert prepared.custom_reference_monitoring_run_id is None


def test_prepare_run_context_allows_system_default_recipe_without_optional_evidence() -> None:
    """Prepare should not require extra metrics or artifacts for the system default recipe."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    compiled = compile_recipe()
    reconcile_test_timeline_baseline(
        gateway,
        source_run_id="train-run-runtime",
        compiled_recipe=compiled,
    )
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-runtime",
        source_experiment=None,
        metrics={},
        artifacts=(),
        environment={"python": "3.12"},
        features=(),
        schema={},
        data_scope="validation:2026-03-01",
    )

    prepared = prepare_test_context(
        subject_id="churn_model",
        compiled_invocation=compiled,
        gateway=gateway,
        source_run_id="train-run-runtime",
    )

    assert prepared.required_metrics == ()
    assert prepared.required_artifacts == ()
    assert prepared.custom_reference_monitoring_run_id is None
    assert prepared.source_run_id == "train-run-runtime"


def test_prepare_run_context_fails_when_custom_reference_is_missing() -> None:
    """Prepare should fail when configured custom reference is absent."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    reconcile_test_timeline_baseline(gateway)
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

    with pytest.raises(
        PrepareStageError, match="Custom reference monitoring run could not be resolved"
    ):
        prepare_test_context(
            subject_id="churn_model",
            compiled_invocation=make_compiled_invocation(
                custom_reference_monitoring_run_id="monitoring-run-missing"
            ),
            gateway=gateway,
        )


def test_prepare_run_context_fails_when_custom_reference_is_on_another_subject() -> None:
    """Prepare should reject a custom reference from another subject timeline."""
    gateway = make_gateway_with_timeline().gateway
    compiled_invocation = make_compiled_invocation()
    foreign_allocation = allocate_test_monitoring_run(
        gateway,
        subject_id="fraud_model",
        source_run_id="train-run-foreign",
        compiled_recipe=compiled_invocation.compiled_recipe,
    )
    gateway.upsert_monitoring_run(
        subject_id="fraud_model",
        monitoring_run_id=foreign_allocation.monitoring_run_id,
        source_run_id="train-run-foreign",
        lifecycle_status=LifecycleStatus.CLOSED,
        sequence_index=0,
    )

    with pytest.raises(
        PrepareStageError, match="Custom reference monitoring run could not be resolved"
    ):
        prepare_test_context(
            subject_id="churn_model",
            compiled_invocation=make_compiled_invocation(
                custom_reference_monitoring_run_id=foreign_allocation.monitoring_run_id
            ),
            gateway=gateway,
        )


@pytest.mark.parametrize(
    "lifecycle_status",
    (
        LifecycleStatus.CREATED,
        LifecycleStatus.PREPARED,
        LifecycleStatus.CHECKED,
        LifecycleStatus.ANALYZED,
        LifecycleStatus.FAILED,
    ),
)
def test_prepare_run_context_fails_when_custom_reference_is_not_closed(
    lifecycle_status: LifecycleStatus,
) -> None:
    """Prepare should reject any non-closed custom Reference Monitoring Run."""
    fixture = make_gateway_with_timeline()
    gateway = fixture.gateway
    compiled_recipe = make_compiled_invocation().compiled_recipe
    checked_allocation = allocate_test_monitoring_run(
        gateway,
        subject_id="churn_model",
        source_run_id="train-run-checked",
        compiled_recipe=compiled_recipe,
    )
    gateway.upsert_monitoring_run(
        subject_id="churn_model",
        monitoring_run_id=checked_allocation.monitoring_run_id,
        source_run_id="train-run-checked",
        lifecycle_status=lifecycle_status,
        sequence_index=checked_allocation.sequence_index,
    )

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_test_context(
            subject_id="churn_model",
            compiled_invocation=make_compiled_invocation(
                custom_reference_monitoring_run_id=checked_allocation.monitoring_run_id
            ),
            gateway=gateway,
        )

    assert exc_info.value.code == "prepare_custom_reference_not_closed"


def test_prepare_run_context_fails_for_uninitialized_timeline_with_no_baseline() -> None:
    """Prepare should require a baseline when allocation has not pinned one."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-123",
        source_experiment="training/churn",
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("metrics.json",),
        environment={},
        features=(),
        schema={},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_test_context(
            subject_id="churn_model",
            compiled_invocation=make_compiled_invocation(
                source_run_id="train-run-123",
                source_experiment="training/churn",
                required_metrics=("f1", "auc"),
                required_artifacts=("metrics.json",),
                custom_reference_monitoring_run_id=None,
            ),
            gateway=gateway,
        )

    error = exc_info.value
    assert error.code == "prepare_missing_baseline_for_uninitialized_timeline"
    assert error.details == (
        ("subject_id", "churn_model"),
        ("baseline_source_run_id", None),
    )
    assert error.message == (
        "The timeline for subject_id='churn_model' has no pinned baseline "
        "and no baseline_source_run_id was provided. "
        "A valid baseline_source_run_id is required to bootstrap the timeline."
    )
    timeline_state = gateway.get_timeline_state("churn_model")
    assert timeline_state is not None
    assert timeline_state.timeline_id == "timeline-churn_model"
    assert timeline_state.baseline_source_run_id is None


def test_prepare_run_context_fails_for_uninitialized_timeline_with_empty_baseline() -> None:
    """Prepare should reject an empty baseline for an allocated Timeline."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-123",
        source_experiment="training/churn",
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("metrics.json",),
        environment={},
        features=(),
        schema={},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_test_context(
            subject_id="churn_model",
            compiled_invocation=make_compiled_invocation(
                source_run_id="train-run-123",
                source_experiment="training/churn",
                required_metrics=("f1", "auc"),
                required_artifacts=("metrics.json",),
                custom_reference_monitoring_run_id=None,
            ),
            gateway=gateway,
            baseline_source_run_id="",
        )

    error = exc_info.value
    assert error.code == "prepare_missing_baseline_for_uninitialized_timeline"
    assert error.details == (
        ("subject_id", "churn_model"),
        ("baseline_source_run_id", ""),
    )
    assert error.message == (
        "The timeline for subject_id='churn_model' has no pinned baseline "
        "and no baseline_source_run_id was provided. "
        "A valid baseline_source_run_id is required to bootstrap the timeline."
    )
    timeline_state = gateway.get_timeline_state("churn_model")
    assert timeline_state is not None
    assert timeline_state.timeline_id == "timeline-churn_model"
    assert timeline_state.baseline_source_run_id is None


def test_prepare_run_context_fails_for_uninitialized_timeline_with_missing_baseline_run() -> None:
    """Prepare should reject bootstrap baselines that do not resolve."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-1",
        source_experiment="training/churn",
        metrics={"f1": 0.87},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_test_context(
            subject_id="churn_model",
            compiled_invocation=make_compiled_invocation(
                source_run_id="train-run-1",
                source_experiment="training/churn",
                required_metrics=("f1",),
                required_artifacts=("metrics.json",),
                custom_reference_monitoring_run_id=None,
            ),
            gateway=gateway,
            baseline_source_run_id="missing-baseline",
        )

    error = exc_info.value
    assert error.code == "prepare_invalid_bootstrap_baseline"
    assert error.details == (
        ("subject_id", "churn_model"),
        ("compiled_recipe.source_requirements.source_experiment", "training/churn"),
        ("baseline_source_run_id", "missing-baseline"),
    )
    assert error.message == (
        "Baseline source run could not be resolved for subject_id='churn_model', "
        "source_experiment='training/churn', "
        "and baseline_source_run_id='missing-baseline'."
    )


def test_prepare_run_context_does_not_bootstrap_when_source_run_resolution_fails() -> None:
    """Failed Prepare should retain allocation without pinning a baseline."""
    gateway = RecordingBaselineClaimGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-1",
        source_experiment="training/churn",
        metrics={"f1": 0.87},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_test_context(
            subject_id="churn_model",
            compiled_invocation=make_compiled_invocation(
                source_run_id="missing-source",
                source_experiment="training/churn",
                required_metrics=("f1",),
                required_artifacts=("metrics.json",),
                custom_reference_monitoring_run_id=None,
            ),
            gateway=gateway,
            baseline_source_run_id="train-run-1",
        )

    assert exc_info.value.code == "prepare_source_run_not_found"
    timeline_state = gateway.get_timeline_state("churn_model")
    assert timeline_state is not None
    assert timeline_state.timeline_id == "timeline-churn_model"
    assert timeline_state.baseline_source_run_id is None
    assert gateway.baseline_claim_calls == []


def test_prepare_run_context_does_not_bootstrap_when_metric_validation_fails() -> None:
    """Metric validation failure should leave the baseline uninitialized."""
    gateway = RecordingBaselineClaimGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-1",
        source_experiment="training/churn",
        metrics={"auc": 0.95},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_test_context(
            subject_id="churn_model",
            compiled_invocation=make_compiled_invocation(
                source_run_id="train-run-1",
                source_experiment="training/churn",
                required_metrics=("f1", "auc"),
                required_artifacts=("metrics.json",),
                custom_reference_monitoring_run_id=None,
            ),
            gateway=gateway,
            baseline_source_run_id="train-run-1",
        )

    assert exc_info.value.code == "prepare_missing_required_metric"
    timeline_state = gateway.get_timeline_state("churn_model")
    assert timeline_state is not None
    assert timeline_state.timeline_id == "timeline-churn_model"
    assert timeline_state.baseline_source_run_id is None
    assert gateway.baseline_claim_calls == []


def test_prepare_run_context_does_not_bootstrap_when_artifact_validation_fails() -> None:
    """Artifact validation failure should leave the baseline uninitialized."""
    gateway = RecordingBaselineClaimGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-1",
        source_experiment="training/churn",
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("model.pkl",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_test_context(
            subject_id="churn_model",
            compiled_invocation=make_compiled_invocation(
                source_run_id="train-run-1",
                source_experiment="training/churn",
                required_metrics=("f1", "auc"),
                required_artifacts=("metrics.json",),
                custom_reference_monitoring_run_id=None,
            ),
            gateway=gateway,
            baseline_source_run_id="train-run-1",
        )

    assert exc_info.value.code == "prepare_missing_required_artifact"
    timeline_state = gateway.get_timeline_state("churn_model")
    assert timeline_state is not None
    assert timeline_state.timeline_id == "timeline-churn_model"
    assert timeline_state.baseline_source_run_id is None
    assert gateway.baseline_claim_calls == []


def test_prepare_run_context_does_not_bootstrap_when_custom_reference_is_invalid() -> None:
    """Reference validation failure should leave the baseline uninitialized."""
    gateway = RecordingBaselineClaimGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-1",
        source_experiment="training/churn",
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_test_context(
            subject_id="churn_model",
            compiled_invocation=make_compiled_invocation(
                source_run_id="train-run-1",
                source_experiment="training/churn",
                required_metrics=("f1", "auc"),
                required_artifacts=("metrics.json",),
                custom_reference_monitoring_run_id="run-missing",
            ),
            gateway=gateway,
            baseline_source_run_id="train-run-1",
        )

    assert exc_info.value.code == "prepare_custom_reference_not_found"
    timeline_state = gateway.get_timeline_state("churn_model")
    assert timeline_state is not None
    assert timeline_state.timeline_id == "timeline-churn_model"
    assert timeline_state.baseline_source_run_id is None
    assert gateway.baseline_claim_calls == []


def test_prepare_ignores_legacy_lkg_pointer_during_baseline_bootstrap() -> None:
    """Legacy LKG pointer state should not affect current Prepare integration."""
    gateway = RecordingBaselineClaimGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-1",
        source_experiment="training/churn",
        metrics={"f1": 0.91},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age",),
        schema={"age": "int"},
        data_scope="validation:2026-03-01",
    )
    gateway.set_active_lkg_monitoring_run_id(
        "churn_model",
        "monitoring-run-missing",
    )

    prepared = prepare_test_context(
        subject_id="churn_model",
        compiled_invocation=make_compiled_invocation(
            source_run_id="train-run-1",
            source_experiment="training/churn",
            required_metrics=("f1",),
            required_artifacts=("metrics.json",),
        ),
        gateway=gateway,
        baseline_source_run_id="train-run-1",
    )

    assert prepared.active_lkg_monitoring_run_id is None
    assert prepared.reference_plan[2].unavailable_reason == "lkg_not_selected"
    assert len(gateway.baseline_claim_calls) == 1
    assert gateway.get_timeline_state("churn_model") == TimelineState(
        timeline_id="timeline-churn_model",
        baseline_source_run_id="train-run-1",
    )


def test_prepare_rejects_foreign_subject_baseline_before_timeline_bootstrap() -> None:
    """Prepare should reject bootstrap baselines from another subject."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-1",
        source_experiment="training/churn",
        metrics={"f1": 0.87},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )
    gateway.add_source_run(
        subject_id="fraud_model",
        source_run_id="fraud-baseline",
        source_experiment="training/churn",
        metrics={"f1": 0.87},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_test_context(
            subject_id="churn_model",
            compiled_invocation=make_compiled_invocation(
                source_run_id="train-run-1",
                source_experiment="training/churn",
                required_metrics=("f1",),
                required_artifacts=("metrics.json",),
                custom_reference_monitoring_run_id=None,
            ),
            gateway=gateway,
            baseline_source_run_id="fraud-baseline",
        )

    error = exc_info.value
    assert error.code == "prepare_invalid_bootstrap_baseline"
    assert error.details == (
        ("subject_id", "churn_model"),
        ("compiled_recipe.source_requirements.source_experiment", "training/churn"),
        ("baseline_source_run_id", "fraud-baseline"),
    )
    assert error.message == (
        "Baseline source run could not be resolved for subject_id='churn_model', "
        "source_experiment='training/churn', "
        "and baseline_source_run_id='fraud-baseline'."
    )


def test_prepare_rejects_foreign_experiment_baseline_before_timeline_bootstrap() -> None:
    """Prepare should reject bootstrap baselines from another experiment."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-1",
        source_experiment="training/churn",
        metrics={"f1": 0.87},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="fraud-baseline",
        source_experiment="validation/fraudeval",
        metrics={"f1": 0.87},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_test_context(
            subject_id="churn_model",
            compiled_invocation=make_compiled_invocation(
                source_run_id="train-run-1",
                source_experiment="training/churn",
                required_metrics=("f1",),
                required_artifacts=("metrics.json",),
                custom_reference_monitoring_run_id=None,
            ),
            gateway=gateway,
            baseline_source_run_id="fraud-baseline",
        )

    error = exc_info.value
    assert error.code == "prepare_invalid_bootstrap_baseline"
    assert error.details == (
        ("subject_id", "churn_model"),
        ("compiled_recipe.source_requirements.source_experiment", "training/churn"),
        ("baseline_source_run_id", "fraud-baseline"),
    )
    assert error.message == (
        "Baseline source run could not be resolved for subject_id='churn_model', "
        "source_experiment='training/churn', "
        "and baseline_source_run_id='fraud-baseline'."
    )


def test_prepare_claims_existing_timeline_baseline_for_new_monitoring_run() -> None:
    """Every successful Prepare should persist its own identical baseline claim."""
    gateway = RecordingBaselineClaimGateway(GatewayConfig())
    compiled_invocation = make_compiled_invocation(
        source_run_id="train-run-current",
        source_experiment="training/churn",
        required_metrics=("f1",),
        required_artifacts=("metrics.json",),
        custom_reference_monitoring_run_id=None,
    )
    reconciled_timeline_state = reconcile_test_timeline_baseline(
        gateway,
        source_run_id="train-run-established",
        baseline_source_run_id="train-run-baseline",
        compiled_recipe=compiled_invocation.compiled_recipe,
    )
    gateway.baseline_claim_calls.clear()

    for source_run_id in ("train-run-baseline", "train-run-current"):
        gateway.add_source_run(
            subject_id="churn_model",
            source_run_id=source_run_id,
            source_experiment="training/churn",
            metrics={"f1": 0.87},
            artifacts=("metrics.json",),
            environment={"python": "3.12"},
            features=("age", "income"),
            schema={"age": "int", "income": "float"},
            data_scope="validation:2026-03-01",
        )

    prepared = prepare_test_context(
        subject_id="churn_model",
        compiled_invocation=compiled_invocation,
        gateway=gateway,
        baseline_source_run_id="train-run-baseline",
    )

    timeline_state = gateway.get_timeline_state("churn_model")

    assert reconciled_timeline_state == TimelineState(
        timeline_id="timeline-churn_model",
        baseline_source_run_id="train-run-baseline",
    )
    assert prepared.sequence_index == 1
    assert gateway.baseline_claim_calls == [
        (
            "churn_model",
            prepared.monitoring_run_id,
            "train-run-baseline",
        )
    ]
    assert timeline_state is not None
    assert timeline_state.baseline_source_run_id == "train-run-baseline"
    assert timeline_state.timeline_id == "timeline-churn_model"


def test_prepare_run_context_accepts_baseline_alias_resolving_to_pinned_baseline() -> None:
    """Prepare should compare a resolved baseline identity with the pinned identity."""
    gateway = AliasResolvingBaselineGateway(GatewayConfig())
    compiled_invocation = make_compiled_invocation(
        source_run_id="train-run-current",
        source_experiment="training/churn",
        required_metrics=("f1", "auc"),
        required_artifacts=("metrics.json",),
    )
    reconcile_test_timeline_baseline(
        gateway,
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        compiled_recipe=compiled_invocation.compiled_recipe,
    )
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-current",
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
        source_run_id="train-run-baseline",
        source_experiment="training/churn",
        metrics={"f1": 0.87, "auc": 0.93},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )
    gateway.add_source_run_alias("baseline-alias", "train-run-baseline")

    prepared = prepare_test_context(
        subject_id="churn_model",
        compiled_invocation=compiled_invocation,
        gateway=gateway,
        baseline_source_run_id="baseline-alias",
    )

    assert prepared.baseline_source_run_id == "train-run-baseline"


def test_prepare_run_context_succeeds_with_existed_timeline_and_no_baseline() -> None:
    """Baseline resolution should succeed by returning the existing pinned baseline."""
    gateway = make_gateway_with_timeline().gateway
    timeline_state = gateway.get_timeline_state("churn_model")

    prepare_test_context(
        subject_id="churn_model",
        compiled_invocation=make_compiled_invocation(
            source_run_id="train-run-123",
            source_experiment="training/churn",
            required_metrics=("f1", "auc"),
            required_artifacts=("metrics.json",),
            custom_reference_monitoring_run_id=None,
        ),
        gateway=gateway,
    )

    assert timeline_state is not None
    assert timeline_state.baseline_source_run_id == "train-run-baseline"
    assert timeline_state.timeline_id == "timeline-churn_model"


def test_prepare_run_context_rejects_empty_baseline_for_initialized_timeline() -> None:
    """An explicit empty baseline should not be treated as an omitted baseline."""
    gateway = make_gateway_with_timeline().gateway

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_test_context(
            subject_id="churn_model",
            compiled_invocation=make_compiled_invocation(
                source_run_id="train-run-123",
                source_experiment="training/churn",
                required_metrics=("f1", "auc"),
                required_artifacts=("metrics.json",),
            ),
            gateway=gateway,
            baseline_source_run_id="",
        )

    error = exc_info.value
    assert error.code == "prepare_invalid_baseline"
    assert error.details == (
        ("subject_id", "churn_model"),
        ("compiled_recipe.source_requirements.source_experiment", "training/churn"),
        ("baseline_source_run_id", ""),
    )


def test_prepare_run_context_succeeds_with_created_timeline_matching_baseline() -> None:
    """Prepare should succeed when provided baseline matches the existing timeline baseline."""
    gateway = make_gateway_with_timeline().gateway

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

    timeline_state = gateway.get_timeline_state("churn_model")

    prepare_test_context(
        subject_id="churn_model",
        compiled_invocation=make_compiled_invocation(
            source_run_id="train-run-123",
            source_experiment="training/churn",
            required_metrics=("f1", "auc"),
            required_artifacts=("metrics.json",),
            custom_reference_monitoring_run_id=None,
        ),
        gateway=gateway,
        baseline_source_run_id="train-run-baseline",
    )

    assert timeline_state is not None
    assert timeline_state.baseline_source_run_id == "train-run-baseline"
    assert timeline_state.timeline_id == "timeline-churn_model"


def test_prepare_run_context_fail_with_created_timeline_mismatch_baseline() -> None:
    """Prepare should fail when provided baseline does not match existing timeline baseline."""
    gateway = make_gateway_with_timeline().gateway

    # add source run that does not match the existing timeline baseline
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-other",
        source_experiment="training/churn",
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    timeline_state = gateway.get_timeline_state("churn_model")
    assert timeline_state is not None
    assert timeline_state.baseline_source_run_id == "train-run-baseline"
    assert timeline_state.timeline_id == "timeline-churn_model"

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_test_context(
            subject_id="churn_model",
            compiled_invocation=make_compiled_invocation(
                source_run_id="train-run-other",
                source_experiment="training/churn",
                required_metrics=("f1", "auc"),
                required_artifacts=("metrics.json",),
                custom_reference_monitoring_run_id=None,
            ),
            gateway=gateway,
            baseline_source_run_id="train-run-other",
        )

    error = exc_info.value
    assert error.code == _PREPARE_BASELINE_OVERRIDE_EXISTING_BASELINE
    assert error.details == (
        ("subject_id", "churn_model"),
        ("baseline_source_run_id", "train-run-other"),
    )
    assert error.message == (
        "Provided baseline_source_run_id='train-run-other' "
        "with resolved baseline_source_run_id='train-run-other' "
        "does not match existing timeline pinned "
        "baseline_source_run_id='train-run-baseline' for subject_id='churn_model'. "
        "Overriding an existing timeline's baseline is not allowed."
    )


def test_prepare_run_context_fails_for_uninitialized_timeline_and_invalid_baseline() -> None:
    """Prepare should reject an unresolved baseline before bootstrapping."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-current",
        source_experiment="training/churn",
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("metrics.json",),
        environment={},
        features=(),
        schema={},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_test_context(
            subject_id="churn_model",
            compiled_invocation=make_compiled_invocation(
                source_run_id="train-run-current",
                source_experiment="training/churn",
                required_metrics=("f1", "auc"),
                required_artifacts=("metrics.json",),
                custom_reference_monitoring_run_id=None,
            ),
            gateway=gateway,
            baseline_source_run_id="train-run-baseline",
        )

    error = exc_info.value
    assert error.code == "prepare_invalid_bootstrap_baseline"
    assert error.details == (
        ("subject_id", "churn_model"),
        ("compiled_recipe.source_requirements.source_experiment", "training/churn"),
        ("baseline_source_run_id", "train-run-baseline"),
    )
    assert error.message == (
        "Baseline source run could not be resolved for subject_id='churn_model', "
        "source_experiment='training/churn', "
        "and baseline_source_run_id='train-run-baseline'."
    )
