"""Unit tests for workflow lifecycle transitions and prepare-stage behavior."""

import pytest

from mlflow_monitor.domain import (
    Baseline,
    ComparabilityStatus,
    Contract,
    ContractCheckReason,
    ContractCheckResult,
    LifecycleStatus,
    Run,
)
from mlflow_monitor.errors import CheckStageError, InvalidRunTransition, PrepareStageError
from mlflow_monitor.gateway import (
    GatewayConfig,
    InMemoryMonitoringGateway,
    TimelineInitializationResult,
)
from mlflow_monitor.recipe import (
    SYSTEM_DEFAULT_CONTRACT_ID,
    SYSTEM_DEFAULT_RECIPE_ID,
    SYSTEM_DEFAULT_RUN_SELECTOR_TOKEN,
    RecipeReferenceCatalog,
    resolve_recipe_v0_lite,
)
from mlflow_monitor.recipe_compiler import CompiledRunPlan, compile_recipe_v0_lite
from mlflow_monitor.workflow import (
    PreparedContext,
    execute_contract_check,
    prepare_run_context,
    transition_run,
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


def make_compiled_run_plan(
    *,
    run_selector: str = "train-run-123",
    source_experiment: str | None = "training/churn",
    required_metrics: tuple[str, ...] = ("f1", "auc"),
    required_artifacts: tuple[str, ...] = ("metrics.json",),
    custom_reference_run_id: str | None = "run-custom-1",
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


def make_gateway_with_timeline() -> InMemoryMonitoringGateway:
    """Build a gateway fixture with timeline and source-run state."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.initialize_timeline("churn_model", "train-run-baseline")
    gateway.upsert_monitoring_run(
        subject_id="churn_model",
        run_id="run-prev",
        lifecycle_status=LifecycleStatus.CLOSED,
        sequence_index=0,
    )
    gateway.upsert_monitoring_run(
        subject_id="churn_model",
        run_id="run-custom-1",
        lifecycle_status=LifecycleStatus.CLOSED,
        sequence_index=1,
    )
    gateway.add_source_run(
        subject_id="churn_model",
        run_id="train-run-123",
        source_experiment="training/churn",
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("metrics.json", "model.pkl"),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )
    return gateway


class BrokenInitializeTimelineGateway(InMemoryMonitoringGateway):
    """Test double whose timeline initialization does not persist state."""

    def initialize_timeline(
        self, subject_id: str, baseline_source_run_id: str
    ) -> TimelineInitializationResult:
        """Pretend to initialize the timeline without storing timeline state."""
        return TimelineInitializationResult(
            timeline_id=f"timeline-{subject_id}",
            created=True,
        )


class RaceWinningInitializeTimelineGateway(InMemoryMonitoringGateway):
    """Test double that simulates another writer winning timeline bootstrap."""

    def __init__(
        self,
        config: GatewayConfig,
        *,
        competing_baseline_source_run_id: str,
    ) -> None:
        """Initialize with the baseline the competing writer materializes."""
        super().__init__(config)
        self._competing_baseline_source_run_id = competing_baseline_source_run_id

    def initialize_timeline(
        self, subject_id: str, baseline_source_run_id: str
    ) -> TimelineInitializationResult:
        """Materialize timeline state as if another writer initialized first."""
        if self.get_timeline_state(subject_id) is None:
            self._timeline_by_subject[subject_id] = self._timeline_by_subject.get(
                subject_id,
                None,
            ) or self._build_competing_timeline_state(subject_id)
        return TimelineInitializationResult(
            timeline_id=f"timeline-{subject_id}",
            created=False,
        )

    def _build_competing_timeline_state(self, subject_id: str):
        """Return timeline state pinned to the competing baseline."""
        from mlflow_monitor.gateway import TimelineState

        return TimelineState(
            timeline_id=f"timeline-{subject_id}",
            baseline_source_run_id=self._competing_baseline_source_run_id,
        )


class AliasResolvingBaselineGateway(InMemoryMonitoringGateway):
    """Test double that resolves baseline aliases to canonical source run ids."""

    def __init__(self, config: GatewayConfig) -> None:
        """Initialize alias mapping for source-run resolution."""
        super().__init__(config)
        self._aliases: dict[str, str] = {}

    def add_source_run_alias(self, alias: str, run_id: str) -> None:
        """Register an alias that resolves to a canonical source run id."""
        self._aliases[alias] = run_id

    def resolve_source_run_id(
        self,
        subject_id: str,
        source_experiment: str | None,
        run_selector: str,
        runtime_source_run_id: str | None = None,
    ) -> str | None:
        """Resolve aliases first, then delegate to the base gateway behavior."""
        candidate = self._aliases.get(run_selector, run_selector)
        return super().resolve_source_run_id(
            subject_id=subject_id,
            source_experiment=source_experiment,
            run_selector=candidate,
            runtime_source_run_id=runtime_source_run_id,
        )


def make_run(
    *,
    lifecycle_status: LifecycleStatus = LifecycleStatus.CREATED,
    comparability_status: ComparabilityStatus = ComparabilityStatus.PASS,
    contract_check_result: ContractCheckResult | None = None,
) -> Run:
    """Build a canonical run for workflow transition tests."""
    return Run(
        run_id="run-1",
        timeline_id="timeline-1",
        sequence_index=0,
        subject_id="churn_model",
        source_run_id="train-run-2",
        baseline_source_run_id=BASELINE.source_run_id,
        contract=CONTRACT,
        lifecycle_status=lifecycle_status,
        comparability_status=comparability_status,
        contract_check_result=contract_check_result,
        diff_ids=(),
        finding_ids=(),
    )


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


def make_prepared_context(
    *,
    contract: Contract,
    source_run_id: str = "train-run-123",
    baseline_source_run_id: str = "train-run-baseline",
) -> PreparedContext:
    """Build a prepared context aligned with the common workflow test subject."""
    return PreparedContext(
        run_id="run-1",
        subject_id="churn_model",
        recipe_id="default",
        recipe_version="v0",
        contract_id=contract.contract_id,
        run_selector=source_run_id,
        source_experiment="training/churn",
        timeline_id="timeline-churn_model",
        baseline_source_run_id=baseline_source_run_id,
        previous_run_id=None,
        active_lkg_run_id=None,
        custom_reference_run_id=None,
        source_run_id=source_run_id,
        contract=contract,
        required_metrics=("f1", "auc"),
        required_artifacts=("metrics.json",),
    )


def test_transition_run_advances_through_happy_path() -> None:
    """Run should move through the allowed lifecycle sequence to closed."""
    run = make_run()

    run = transition_run(run, LifecycleStatus.PREPARED)
    run = transition_run(run, LifecycleStatus.CHECKED)
    run = transition_run(run, LifecycleStatus.ANALYZED)
    run = transition_run(run, LifecycleStatus.CLOSED)

    assert run.lifecycle_status is LifecycleStatus.CLOSED


@pytest.mark.parametrize(
    "from_status",
    (
        LifecycleStatus.CREATED,
        LifecycleStatus.PREPARED,
        LifecycleStatus.CHECKED,
        LifecycleStatus.ANALYZED,
    ),
)
def test_transition_run_allows_failure_from_active_states(
    from_status: LifecycleStatus,
) -> None:
    """Run should be able to fail from any active non-terminal state."""
    run = make_run(lifecycle_status=from_status)

    failed_run = transition_run(run, LifecycleStatus.FAILED)

    assert failed_run.lifecycle_status is LifecycleStatus.FAILED


@pytest.mark.parametrize(
    ("from_status", "to_status"),
    (
        (LifecycleStatus.CREATED, LifecycleStatus.CHECKED),
        (LifecycleStatus.CHECKED, LifecycleStatus.PREPARED),
        (LifecycleStatus.CLOSED, LifecycleStatus.FAILED),
        (LifecycleStatus.FAILED, LifecycleStatus.CLOSED),
    ),
)
def test_transition_run_rejects_illegal_transitions(
    from_status: LifecycleStatus,
    to_status: LifecycleStatus,
) -> None:
    """Run should reject skipped, backward, and terminal-state transitions."""
    run = make_run(lifecycle_status=from_status)

    with pytest.raises(InvalidRunTransition) as exc_info:
        transition_run(run, to_status)

    assert exc_info.value.from_status is from_status
    assert exc_info.value.to_status is to_status


def test_transition_run_preserves_comparability_fields() -> None:
    """Lifecycle transitions should not alter comparability-related fields."""
    contract_check_result = ContractCheckResult(
        status=ComparabilityStatus.WARN,
        reasons=(
            ContractCheckReason(
                code="environment_mismatch",
                message="Python version differs.",
                blocking=False,
            ),
        ),
    )
    run = make_run(
        comparability_status=ComparabilityStatus.WARN,
        contract_check_result=contract_check_result,
    )

    prepared_run = transition_run(run, LifecycleStatus.PREPARED)

    assert prepared_run.comparability_status is ComparabilityStatus.WARN
    assert prepared_run.contract_check_result == contract_check_result


def test_prepare_run_context_succeeds_with_initialized_timeline() -> None:
    """Prepare should resolve references and required source-run inputs."""
    gateway = make_gateway_with_timeline()
    gateway.set_active_lkg_run_id("churn_model", "run-lkg")
    compiled = make_compiled_run_plan()

    prepared = prepare_run_context(
        run_id="run-1",
        subject_id="churn_model",
        compiled_plan=compiled,
        resolved_contract=CONTRACT,
        gateway=gateway,
    )

    assert prepared.run_id == "run-1"
    assert prepared.subject_id == "churn_model"
    assert prepared.timeline_id == "timeline-churn_model"
    assert prepared.source_run_id == "train-run-123"
    assert prepared.baseline_source_run_id == "train-run-baseline"
    assert prepared.previous_run_id == "run-custom-1"
    assert prepared.active_lkg_run_id == "run-lkg"
    assert prepared.custom_reference_run_id == "run-custom-1"
    assert prepared.contract is CONTRACT
    assert prepared.required_metrics == ("f1", "auc")
    assert prepared.required_artifacts == ("metrics.json",)
    assert prepared.recipe_id == "default"
    assert prepared.recipe_version == "v0"
    assert prepared.contract_id == "default"


def test_execute_contract_check_returns_warn_result_for_environment_mismatch() -> None:
    """Check should return the canonical warning result for env mismatch."""
    contract = Contract(
        contract_id="env_repro",
        version="v0",
        schema_contract_ref=None,
        feature_contract_ref=None,
        metric_contract_ref=None,
        data_scope_contract_ref=None,
        execution_contract_ref="builtin:env_repro",
    )
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        run_id="train-run-baseline",
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
        run_id="train-run-123",
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
        run_id="train-run-123",
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
            prepared_context=make_prepared_context(contract=CONTRACT),
            gateway=gateway,
        )

    assert exc_info.value.code == "check_missing_baseline_evidence"


def test_execute_contract_check_fails_when_current_evidence_is_missing() -> None:
    """Check should fail explicitly when current evidence cannot be loaded."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        run_id="train-run-baseline",
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
            prepared_context=make_prepared_context(contract=CONTRACT),
            gateway=gateway,
        )

    assert exc_info.value.code == "check_missing_current_evidence"


def test_execute_contract_check_wraps_checker_failures() -> None:
    """Check should wrap checker execution failures in a workflow error."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
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

    with pytest.raises(CheckStageError) as exc_info:
        execute_contract_check(
            prepared_context=make_prepared_context(contract=CONTRACT),
            gateway=gateway,
            contract_checker=RaisingContractChecker(),
        )

    assert exc_info.value.code == "check_checker_execution_failed"


def test_execute_contract_check_rejects_invalid_checker_result() -> None:
    """Check should reject results that violate contract-check invariants."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
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

    with pytest.raises(CheckStageError) as exc_info:
        execute_contract_check(
            prepared_context=make_prepared_context(contract=CONTRACT),
            gateway=gateway,
            contract_checker=InvalidResultContractChecker(),
        )

    assert exc_info.value.code == "check_result_invalid"


def test_prepare_run_context_succeeds_without_previous_run() -> None:
    """Prepare should tolerate a missing previous run."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.initialize_timeline("churn_model", "train-run-baseline")
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

    prepared = prepare_run_context(
        run_id="run-1",
        subject_id="churn_model",
        compiled_plan=make_compiled_run_plan(custom_reference_run_id=None),
        resolved_contract=CONTRACT,
        gateway=gateway,
    )

    assert prepared.previous_run_id is None


def test_prepare_run_context_succeeds_without_active_lkg() -> None:
    """Prepare should tolerate a missing active LKG."""
    gateway = make_gateway_with_timeline()

    prepared = prepare_run_context(
        run_id="run-1",
        subject_id="churn_model",
        compiled_plan=make_compiled_run_plan(),
        resolved_contract=CONTRACT,
        gateway=gateway,
    )

    assert prepared.active_lkg_run_id is None


def test_prepare_run_context_allows_omitted_source_experiment_filter() -> None:
    """Prepare should resolve a raw source run when source_experiment is omitted."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.initialize_timeline("churn_model", "train-run-baseline")
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

    prepared = prepare_run_context(
        run_id="run-1",
        subject_id="churn_model",
        compiled_plan=make_compiled_run_plan(
            source_experiment=None,
            custom_reference_run_id=None,
        ),
        resolved_contract=CONTRACT,
        gateway=gateway,
    )

    assert prepared.source_run_id == "train-run-123"


def test_prepare_run_context_preserves_omitted_custom_reference() -> None:
    """Prepare should keep an omitted custom reference as None."""
    gateway = make_gateway_with_timeline()

    prepared = prepare_run_context(
        run_id="run-1",
        subject_id="churn_model",
        compiled_plan=make_compiled_run_plan(custom_reference_run_id=None),
        resolved_contract=CONTRACT,
        gateway=gateway,
    )

    assert prepared.custom_reference_run_id is None


def test_prepare_run_context_fails_when_source_run_cannot_be_resolved() -> None:
    """Prepare should fail explicitly when the source run is missing."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.initialize_timeline("churn_model", "train-run-baseline")

    with pytest.raises(PrepareStageError, match="Source training run could not be resolved"):
        prepare_run_context(
            run_id="run-1",
            subject_id="churn_model",
            compiled_plan=make_compiled_run_plan(custom_reference_run_id=None),
            resolved_contract=CONTRACT,
            gateway=gateway,
        )


def test_prepare_run_context_fails_when_required_metric_is_missing() -> None:
    """Prepare should fail explicitly when a required metric is absent."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.initialize_timeline("churn_model", "train-run-baseline")
    gateway.add_source_run(
        subject_id="churn_model",
        run_id="train-run-123",
        source_experiment="training/churn",
        metrics={"auc": 0.95},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError, match="missing required metric"):
        prepare_run_context(
            run_id="run-1",
            subject_id="churn_model",
            compiled_plan=make_compiled_run_plan(
                required_metrics=("f1", "auc"),
                custom_reference_run_id=None,
            ),
            resolved_contract=CONTRACT,
            gateway=gateway,
        )


def test_prepare_run_context_fails_when_required_artifact_is_missing() -> None:
    """Prepare should fail explicitly when a required artifact is absent."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.initialize_timeline("churn_model", "train-run-baseline")
    gateway.add_source_run(
        subject_id="churn_model",
        run_id="train-run-123",
        source_experiment="training/churn",
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("model.pkl",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError, match="missing required artifact"):
        prepare_run_context(
            run_id="run-1",
            subject_id="churn_model",
            compiled_plan=make_compiled_run_plan(
                required_artifacts=("metrics.json",),
                custom_reference_run_id=None,
            ),
            resolved_contract=CONTRACT,
            gateway=gateway,
        )


def test_prepare_run_context_uses_runtime_source_run_id_for_reserved_selector() -> None:
    """Prepare should honor the reserved runtime selector token."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.initialize_timeline("churn_model", "train-run-baseline")
    gateway.add_source_run(
        subject_id="churn_model",
        run_id="train-run-runtime",
        source_experiment=None,
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    prepared = prepare_run_context(
        run_id="run-1",
        subject_id="churn_model",
        compiled_plan=make_compiled_run_plan(
            run_selector=SYSTEM_DEFAULT_RUN_SELECTOR_TOKEN,
            source_experiment=None,
            custom_reference_run_id=None,
            recipe_id=SYSTEM_DEFAULT_RECIPE_ID,
            contract_id=SYSTEM_DEFAULT_CONTRACT_ID,
        ),
        resolved_contract=Contract(
            contract_id=SYSTEM_DEFAULT_CONTRACT_ID,
            version="v0",
            schema_contract_ref=None,
            feature_contract_ref=None,
            metric_contract_ref=None,
            data_scope_contract_ref=None,
            execution_contract_ref=None,
        ),
        gateway=gateway,
        runtime_source_run_id="train-run-runtime",
    )

    assert prepared.source_run_id == "train-run-runtime"


def test_prepare_run_context_fails_when_custom_reference_is_missing() -> None:
    """Prepare should fail when configured custom reference is absent."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.initialize_timeline("churn_model", "train-run-baseline")
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

    with pytest.raises(PrepareStageError, match="Custom reference run could not be resolved"):
        prepare_run_context(
            run_id="run-1",
            subject_id="churn_model",
            compiled_plan=make_compiled_run_plan(custom_reference_run_id="run-missing"),
            resolved_contract=CONTRACT,
            gateway=gateway,
        )


def test_prepare_run_context_fails_when_custom_reference_is_on_another_subject() -> None:
    """Prepare should reject a custom reference from another subject timeline."""
    gateway = make_gateway_with_timeline()
    gateway.upsert_monitoring_run(
        subject_id="fraud_model",
        run_id="run-foreign",
        lifecycle_status=LifecycleStatus.CLOSED,
        sequence_index=0,
    )

    with pytest.raises(PrepareStageError, match="Custom reference run could not be resolved"):
        prepare_run_context(
            run_id="run-1",
            subject_id="churn_model",
            compiled_plan=make_compiled_run_plan(custom_reference_run_id="run-foreign"),
            resolved_contract=CONTRACT,
            gateway=gateway,
        )


def test_prepare_run_context_fails_when_resolved_contract_mismatches_compiled_plan() -> None:
    """Prepare should reject contradictory resolved contract inputs."""
    gateway = make_gateway_with_timeline()
    mismatched_contract = Contract(
        contract_id="other-contract",
        version="v0",
        schema_contract_ref=None,
        feature_contract_ref=None,
        metric_contract_ref=None,
        data_scope_contract_ref=None,
        execution_contract_ref=None,
    )

    with pytest.raises(PrepareStageError, match="Resolved contract does not match compiled plan"):
        prepare_run_context(
            run_id="run-1",
            subject_id="churn_model",
            compiled_plan=make_compiled_run_plan(custom_reference_run_id=None),
            resolved_contract=mismatched_contract,
            gateway=gateway,
        )


def test_prepare_run_context_succeeds_for_first_run_with_baseline_passed_in() -> None:
    """Prepare should resolve references and required source-run inputs."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        run_id=BASELINE.source_run_id,
        source_experiment="training/churn",
        metrics=BASELINE.metric_snapshot,
        artifacts=("metrics.json",),
        environment=BASELINE.environment_context,
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    prepare_run_context(
        run_id="run-1",
        subject_id="churn_model",
        compiled_plan=make_compiled_run_plan(
            run_selector=BASELINE.source_run_id,
            source_experiment="training/churn",
            required_metrics=tuple(BASELINE.metric_snapshot.keys()),
            required_artifacts=("metrics.json",),
            custom_reference_run_id=None,
        ),
        resolved_contract=CONTRACT,
        gateway=gateway,
        baseline_source_run_id=BASELINE.source_run_id,
    )

    timeline_state = gateway.get_timeline_state("churn_model")

    assert timeline_state is not None
    assert timeline_state.baseline_source_run_id == "train-run-1"
    assert timeline_state.timeline_id == "timeline-churn_model"


def test_prepare_run_context_fails_for_first_run_with_no_baseline() -> None:
    """Prepare should fail without existing timeline and provided baseline."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_run_context(
            run_id="run-1",
            subject_id="churn_model",
            compiled_plan=make_compiled_run_plan(
                run_selector="train-run-123",
                source_experiment="training/churn",
                required_metrics=("f1", "auc"),
                required_artifacts=("metrics.json",),
                custom_reference_run_id=None,
            ),
            resolved_contract=CONTRACT,
            gateway=gateway,
        )

    error = exc_info.value
    assert error.code == "prepare_missing_baseline_no_timeline"
    assert error.details == (
        ("subject_id", "churn_model"),
        ("baseline_source_run_id", None),
    )
    assert error.message == (
        "No timeline exists for subject_id=churn_model "
        "and no baseline_source_run_id was provided. "
        "A valid baseline_source_run_id is required to bootstrap a new timeline."
    )


def test_prepare_run_context_fails_for_first_run_with_empty_baseline() -> None:
    """Prepare should fail without existing timeline and provided baseline."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_run_context(
            run_id="run-1",
            subject_id="churn_model",
            compiled_plan=make_compiled_run_plan(
                run_selector="train-run-123",
                source_experiment="training/churn",
                required_metrics=("f1", "auc"),
                required_artifacts=("metrics.json",),
                custom_reference_run_id=None,
            ),
            resolved_contract=CONTRACT,
            gateway=gateway,
            baseline_source_run_id="",
        )

    error = exc_info.value
    assert error.code == "prepare_missing_baseline_no_timeline"
    assert error.details == (
        ("subject_id", "churn_model"),
        ("baseline_source_run_id", ""),
    )
    assert error.message == (
        "No timeline exists for subject_id=churn_model "
        "and no baseline_source_run_id was provided. "
        "A valid baseline_source_run_id is required to bootstrap a new timeline."
    )


def test_prepare_run_context_fails_for_first_run_with_missing_baseline_run() -> None:
    """Prepare should reject bootstrap baselines that do not resolve."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        run_id=BASELINE.source_run_id,
        source_experiment="training/churn",
        metrics=BASELINE.metric_snapshot,
        artifacts=("metrics.json",),
        environment=BASELINE.environment_context,
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_run_context(
            run_id="run-1",
            subject_id="churn_model",
            compiled_plan=make_compiled_run_plan(
                run_selector=BASELINE.source_run_id,
                source_experiment="training/churn",
                required_metrics=tuple(BASELINE.metric_snapshot.keys()),
                required_artifacts=("metrics.json",),
                custom_reference_run_id=None,
            ),
            resolved_contract=CONTRACT,
            gateway=gateway,
            baseline_source_run_id="missing-baseline",
        )

    error = exc_info.value
    assert error.code == "prepare_invalid_bootstrap_baseline"
    assert error.details == (
        ("subject_id", "churn_model"),
        ("compiled_plan.input.source_experiment", "training/churn"),
        ("baseline_source_run_id", "missing-baseline"),
    )
    assert error.message == (
        "Baseline source run could not be resolved for subject_id=churn_model, "
        "compiled_plan.input.source_experiment='training/churn', "
        "and baseline_source_run_id='missing-baseline'."
    )


def test_prepare_run_context_does_not_persist_timeline_when_source_run_resolution_fails() -> None:
    """Failed first prepare should not leave timeline state behind."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        run_id=BASELINE.source_run_id,
        source_experiment="training/churn",
        metrics=BASELINE.metric_snapshot,
        artifacts=("metrics.json",),
        environment=BASELINE.environment_context,
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_run_context(
            run_id="run-1",
            subject_id="churn_model",
            compiled_plan=make_compiled_run_plan(
                run_selector="missing-source",
                source_experiment="training/churn",
                required_metrics=("f1",),
                required_artifacts=("metrics.json",),
                custom_reference_run_id=None,
            ),
            resolved_contract=CONTRACT,
            gateway=gateway,
            baseline_source_run_id=BASELINE.source_run_id,
        )

    assert exc_info.value.code == "prepare_source_run_not_found"
    assert gateway.get_timeline_state("churn_model") is None


def test_prepare_run_context_does_not_persist_timeline_when_metric_validation_fails() -> None:
    """Failed first prepare should not persist bootstrap state on metric errors."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        run_id=BASELINE.source_run_id,
        source_experiment="training/churn",
        metrics={"auc": 0.95},
        artifacts=("metrics.json",),
        environment=BASELINE.environment_context,
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_run_context(
            run_id="run-1",
            subject_id="churn_model",
            compiled_plan=make_compiled_run_plan(
                run_selector=BASELINE.source_run_id,
                source_experiment="training/churn",
                required_metrics=("f1", "auc"),
                required_artifacts=("metrics.json",),
                custom_reference_run_id=None,
            ),
            resolved_contract=CONTRACT,
            gateway=gateway,
            baseline_source_run_id=BASELINE.source_run_id,
        )

    assert exc_info.value.code == "prepare_missing_required_metric"
    assert gateway.get_timeline_state("churn_model") is None


def test_prepare_run_context_does_not_persist_timeline_when_artifact_validation_fails() -> None:
    """Failed first prepare should not persist bootstrap state on artifact errors."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        run_id=BASELINE.source_run_id,
        source_experiment="training/churn",
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("model.pkl",),
        environment=BASELINE.environment_context,
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_run_context(
            run_id="run-1",
            subject_id="churn_model",
            compiled_plan=make_compiled_run_plan(
                run_selector=BASELINE.source_run_id,
                source_experiment="training/churn",
                required_metrics=("f1", "auc"),
                required_artifacts=("metrics.json",),
                custom_reference_run_id=None,
            ),
            resolved_contract=CONTRACT,
            gateway=gateway,
            baseline_source_run_id=BASELINE.source_run_id,
        )

    assert exc_info.value.code == "prepare_missing_required_artifact"
    assert gateway.get_timeline_state("churn_model") is None


def test_prepare_run_context_does_not_persist_timeline_when_custom_reference_is_invalid() -> None:
    """Failed first prepare should not persist bootstrap state on reference errors."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        run_id=BASELINE.source_run_id,
        source_experiment="training/churn",
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("metrics.json",),
        environment=BASELINE.environment_context,
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_run_context(
            run_id="run-1",
            subject_id="churn_model",
            compiled_plan=make_compiled_run_plan(
                run_selector=BASELINE.source_run_id,
                source_experiment="training/churn",
                required_metrics=("f1", "auc"),
                required_artifacts=("metrics.json",),
                custom_reference_run_id="run-missing",
            ),
            resolved_contract=CONTRACT,
            gateway=gateway,
            baseline_source_run_id=BASELINE.source_run_id,
        )

    assert exc_info.value.code == "prepare_custom_reference_not_found"
    assert gateway.get_timeline_state("churn_model") is None


def test_prepare_run_context_fails_for_first_run_with_foreign_subject_baseline() -> None:
    """Prepare should reject bootstrap baselines from another subject."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        run_id=BASELINE.source_run_id,
        source_experiment="training/churn",
        metrics=BASELINE.metric_snapshot,
        artifacts=("metrics.json",),
        environment=BASELINE.environment_context,
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )
    gateway.add_source_run(
        subject_id="fraud_model",
        run_id="fraud-baseline",
        source_experiment="training/churn",
        metrics=BASELINE.metric_snapshot,
        artifacts=("metrics.json",),
        environment=BASELINE.environment_context,
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_run_context(
            run_id="run-1",
            subject_id="churn_model",
            compiled_plan=make_compiled_run_plan(
                run_selector=BASELINE.source_run_id,
                source_experiment="training/churn",
                required_metrics=tuple(BASELINE.metric_snapshot.keys()),
                required_artifacts=("metrics.json",),
                custom_reference_run_id=None,
            ),
            resolved_contract=CONTRACT,
            gateway=gateway,
            baseline_source_run_id="fraud-baseline",
        )

    error = exc_info.value
    assert error.code == "prepare_invalid_bootstrap_baseline"
    assert error.details == (
        ("subject_id", "churn_model"),
        ("compiled_plan.input.source_experiment", "training/churn"),
        ("baseline_source_run_id", "fraud-baseline"),
    )
    assert error.message == (
        "Baseline source run could not be resolved for subject_id=churn_model, "
        "compiled_plan.input.source_experiment='training/churn', "
        "and baseline_source_run_id='fraud-baseline'."
    )


def test_prepare_run_context_fails_for_first_run_with_foreign_experiment_baseline() -> None:
    """Prepare should reject bootstrap baselines from another experiment."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        run_id=BASELINE.source_run_id,
        source_experiment="training/churn",
        metrics=BASELINE.metric_snapshot,
        artifacts=("metrics.json",),
        environment=BASELINE.environment_context,
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )
    gateway.add_source_run(
        subject_id="churn_model",
        run_id="fraud-baseline",
        source_experiment="validation/fraudeval",
        metrics=BASELINE.metric_snapshot,
        artifacts=("metrics.json",),
        environment=BASELINE.environment_context,
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_run_context(
            run_id="run-1",
            subject_id="churn_model",
            compiled_plan=make_compiled_run_plan(
                run_selector=BASELINE.source_run_id,
                source_experiment="training/churn",
                required_metrics=tuple(BASELINE.metric_snapshot.keys()),
                required_artifacts=("metrics.json",),
                custom_reference_run_id=None,
            ),
            resolved_contract=CONTRACT,
            gateway=gateway,
            baseline_source_run_id="fraud-baseline",
        )

    error = exc_info.value
    assert error.code == "prepare_invalid_bootstrap_baseline"
    assert error.details == (
        ("subject_id", "churn_model"),
        ("compiled_plan.input.source_experiment", "training/churn"),
        ("baseline_source_run_id", "fraud-baseline"),
    )
    assert error.message == (
        "Baseline source run could not be resolved for subject_id=churn_model, "
        "compiled_plan.input.source_experiment='training/churn', "
        "and baseline_source_run_id='fraud-baseline'."
    )


def test_prepare_run_context_fails_when_timeline_init_does_not_materialize_state() -> None:
    """Prepare should fail deterministically if bootstrap does not create timeline state."""
    gateway = BrokenInitializeTimelineGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        run_id=BASELINE.source_run_id,
        source_experiment="training/churn",
        metrics=BASELINE.metric_snapshot,
        artifacts=("metrics.json",),
        environment=BASELINE.environment_context,
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_run_context(
            run_id="run-1",
            subject_id="churn_model",
            compiled_plan=make_compiled_run_plan(
                run_selector=BASELINE.source_run_id,
                source_experiment="training/churn",
                required_metrics=tuple(BASELINE.metric_snapshot.keys()),
                required_artifacts=("metrics.json",),
                custom_reference_run_id=None,
            ),
            resolved_contract=CONTRACT,
            gateway=gateway,
            baseline_source_run_id=BASELINE.source_run_id,
        )

    error = exc_info.value
    assert error.code == "prepare_timeline_initialization_failed"
    assert error.details == (("subject_id", "churn_model"),)
    assert error.message == (
        "Timeline initialization did not materialize state for subject_id=churn_model."
    )


def test_prepare_run_context_succeeds_when_competing_bootstrap_pins_same_baseline() -> None:
    """Prepare should succeed if another writer created the same pinned baseline."""
    gateway = RaceWinningInitializeTimelineGateway(
        GatewayConfig(),
        competing_baseline_source_run_id=BASELINE.source_run_id,
    )
    gateway.add_source_run(
        subject_id="churn_model",
        run_id=BASELINE.source_run_id,
        source_experiment="training/churn",
        metrics=BASELINE.metric_snapshot,
        artifacts=("metrics.json",),
        environment=BASELINE.environment_context,
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    prepared = prepare_run_context(
        run_id="run-1",
        subject_id="churn_model",
        compiled_plan=make_compiled_run_plan(
            run_selector=BASELINE.source_run_id,
            source_experiment="training/churn",
            required_metrics=tuple(BASELINE.metric_snapshot.keys()),
            required_artifacts=("metrics.json",),
            custom_reference_run_id=None,
        ),
        resolved_contract=CONTRACT,
        gateway=gateway,
        baseline_source_run_id=BASELINE.source_run_id,
    )

    assert prepared.timeline_id == "timeline-churn_model"
    assert prepared.baseline_source_run_id == BASELINE.source_run_id


def test_prepare_run_context_fails_when_competing_bootstrap_pins_different_baseline() -> None:
    """Prepare should reject a competing timeline pinned to another baseline."""
    gateway = RaceWinningInitializeTimelineGateway(
        GatewayConfig(),
        competing_baseline_source_run_id="train-run-other",
    )
    gateway.add_source_run(
        subject_id="churn_model",
        run_id=BASELINE.source_run_id,
        source_experiment="training/churn",
        metrics=BASELINE.metric_snapshot,
        artifacts=("metrics.json",),
        environment=BASELINE.environment_context,
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )
    gateway.add_source_run(
        subject_id="churn_model",
        run_id="train-run-other",
        source_experiment="training/churn",
        metrics=BASELINE.metric_snapshot,
        artifacts=("metrics.json",),
        environment=BASELINE.environment_context,
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_run_context(
            run_id="run-1",
            subject_id="churn_model",
            compiled_plan=make_compiled_run_plan(
                run_selector=BASELINE.source_run_id,
                source_experiment="training/churn",
                required_metrics=tuple(BASELINE.metric_snapshot.keys()),
                required_artifacts=("metrics.json",),
                custom_reference_run_id=None,
            ),
            resolved_contract=CONTRACT,
            gateway=gateway,
            baseline_source_run_id=BASELINE.source_run_id,
        )

    error = exc_info.value
    assert error.code == "prepare_baseline_override_existing_timeline"
    assert error.details == (
        ("subject_id", "churn_model"),
        ("baseline_source_run_id", BASELINE.source_run_id),
    )


def test_prepare_run_context_succeeds_existing_timeline_with_correct_baseline_passed_in() -> None:
    """Prepare should resolve references and required source-run inputs."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    timeline_initialization_result = gateway.initialize_timeline(
        "churn_model", BASELINE.source_run_id
    )

    gateway.add_source_run(
        subject_id="churn_model",
        run_id=BASELINE.source_run_id,
        source_experiment="training/churn",
        metrics=BASELINE.metric_snapshot,
        artifacts=("metrics.json",),
        environment=BASELINE.environment_context,
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    prepare_run_context(
        run_id="run-1",
        subject_id="churn_model",
        compiled_plan=make_compiled_run_plan(
            run_selector=BASELINE.source_run_id,
            source_experiment="training/churn",
            required_metrics=tuple(BASELINE.metric_snapshot.keys()),
            required_artifacts=("metrics.json",),
            custom_reference_run_id=None,
        ),
        resolved_contract=CONTRACT,
        gateway=gateway,
        baseline_source_run_id=BASELINE.source_run_id,
    )

    timeline_state = gateway.get_timeline_state("churn_model")

    assert timeline_initialization_result.created is True
    assert timeline_initialization_result.timeline_id == "timeline-churn_model"
    assert timeline_state is not None
    assert timeline_state.baseline_source_run_id == BASELINE.source_run_id
    assert timeline_state.timeline_id == "timeline-churn_model"


def test_prepare_run_context_succeeds_with_existed_timeline_and_no_baseline() -> None:
    """Baseline resolution should succeed by returning the existing pinned baseline."""
    gateway = make_gateway_with_timeline()
    timeline_state = gateway.get_timeline_state("churn_model")

    prepare_run_context(
        run_id="run-1",
        subject_id="churn_model",
        compiled_plan=make_compiled_run_plan(
            run_selector="train-run-123",
            source_experiment="training/churn",
            required_metrics=("f1", "auc"),
            required_artifacts=("metrics.json",),
            custom_reference_run_id=None,
        ),
        resolved_contract=CONTRACT,
        gateway=gateway,
    )

    assert timeline_state is not None
    assert timeline_state.baseline_source_run_id == "train-run-baseline"
    assert timeline_state.timeline_id == "timeline-churn_model"


def test_prepare_run_context_succeeds_with_created_timeline_matching_baseline() -> None:
    """Prepare should succeed when provided baseline matches the existing timeline baseline."""
    gateway = make_gateway_with_timeline()

    gateway.add_source_run(
        subject_id="churn_model",
        run_id="train-run-baseline",
        source_experiment="training/churn",
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    timeline_state = gateway.get_timeline_state("churn_model")

    prepare_run_context(
        run_id="run-1",
        subject_id="churn_model",
        compiled_plan=make_compiled_run_plan(
            run_selector="train-run-123",
            source_experiment="training/churn",
            required_metrics=("f1", "auc"),
            required_artifacts=("metrics.json",),
            custom_reference_run_id=None,
        ),
        resolved_contract=CONTRACT,
        gateway=gateway,
        baseline_source_run_id="train-run-baseline",
    )

    assert timeline_state is not None
    assert timeline_state.baseline_source_run_id == "train-run-baseline"
    assert timeline_state.timeline_id == "timeline-churn_model"


def test_prepare_run_context_fail_with_created_timeline_mismatch_baseline() -> None:
    """Prepare should fail when provided baseline does not match existing timeline baseline."""
    gateway = make_gateway_with_timeline()

    # add source run that does not match the existing timeline baseline
    gateway.add_source_run(
        subject_id="churn_model",
        run_id="train-run-other",
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
        prepare_run_context(
            run_id="run-1",
            subject_id="churn_model",
            compiled_plan=make_compiled_run_plan(
                run_selector="train-run-other",
                source_experiment="training/churn",
                required_metrics=("f1", "auc"),
                required_artifacts=("metrics.json",),
                custom_reference_run_id=None,
            ),
            resolved_contract=CONTRACT,
            gateway=gateway,
            baseline_source_run_id="train-run-other",
        )

    error = exc_info.value
    assert error.code == "prepare_baseline_override_existing_timeline"
    assert error.details == (
        ("subject_id", "churn_model"),
        ("baseline_source_run_id", "train-run-other"),
    )
    assert error.message == (
        "Provided baseline_source_run_id='train-run-other' "
        "with resolved_baseline_source_run_id='train-run-other' "
        "does not match existing timeline "
        "baseline_source_run_id='train-run-baseline' for subject_id=churn_model. "
        "Overriding an existing timeline's baseline is not allowed."
    )


def test_prepare_run_context_fail_with_no_timeline_and_invalid_baseline() -> None:
    """Prepare should fail with no timeline and invalid baseline that cannot be resolved."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())

    with pytest.raises(PrepareStageError) as exc_info:
        prepare_run_context(
            run_id="run-1",
            subject_id="churn_model",
            compiled_plan=make_compiled_run_plan(
                run_selector="train-run-baseline",
                source_experiment="training/churn",
                required_metrics=("f1", "auc"),
                required_artifacts=("metrics.json",),
                custom_reference_run_id=None,
            ),
            resolved_contract=CONTRACT,
            gateway=gateway,
            baseline_source_run_id="train-run-baseline",
        )

    error = exc_info.value
    assert error.code == "prepare_invalid_bootstrap_baseline"
    assert error.details == (
        ("subject_id", "churn_model"),
        ("compiled_plan.input.source_experiment", "training/churn"),
        ("baseline_source_run_id", "train-run-baseline"),
    )
    assert error.message == (
        "Baseline source run could not be resolved for subject_id=churn_model, "
        "compiled_plan.input.source_experiment='training/churn', "
        "and baseline_source_run_id='train-run-baseline'."
    )
