from dataclasses import dataclass

from mlflow_monitor.contract import SYSTEM_DEFAULT_CONTRACT_ID
from mlflow_monitor.domain import LifecycleStatus
from mlflow_monitor.gateway import (
    CreateOrReuseMonitoringRunResult,
    GatewayConfig,
    IdempotencyKey,
    InMemoryMonitoringGateway,
    TimelineState,
)
from mlflow_monitor.recipe_compiler import CompiledRecipe, compile_recipe
from mlflow_monitor.workflow import (
    PreparedContext,
)
from mlflow_monitor.workflow import (
    prepare_run_context as _prepare_run_context,
)

_PREPARE_BASELINE_OVERRIDE_EXISTING_BASELINE = "prepare_baseline_override_existing_baseline"


@dataclass(frozen=True, slots=True)
class CompiledInvocation:
    """Compiled Recipe paired with invocation-owned Prepare inputs."""

    compiled_recipe: CompiledRecipe
    source_run_id: str
    custom_reference_monitoring_run_id: str | None


@dataclass(frozen=True, slots=True)
class InitializedTimelineFixture:
    """Gateway plus opaque Monitoring Run identities created by its fixture."""

    gateway: InMemoryMonitoringGateway
    previous_monitoring_run_id: str
    custom_monitoring_run_id: str


def make_compiled_invocation(
    *,
    source_run_id: str = "train-run-123",
    source_experiment: str | None = "training/churn",
    required_metrics: tuple[str, ...] = ("f1", "auc"),
    required_artifacts: tuple[str, ...] = ("metrics.json",),
    custom_reference_monitoring_run_id: str | None = None,
    recipe_id: str = "default",
    contract_id: str = SYSTEM_DEFAULT_CONTRACT_ID,
) -> CompiledInvocation:
    """Build a Compiled Recipe and distinct invocation-owned identities."""
    raw = {
        "recipe_schema_version": "v0",
        "identity": {"recipe_id": recipe_id, "recipe_version": "v0"},
        "source_requirements": {
            "required_metric_names": list(required_metrics),
            "required_artifact_paths": list(required_artifacts),
        },
        "contract": {"contract_id": contract_id, "contract_version": "v0"},
        "analysis": {"metric_names": ["f1", "auc"], "finding_policy_bindings": []},
    }
    source_requirements = raw["source_requirements"]
    assert isinstance(source_requirements, dict)
    if source_experiment is not None:
        source_requirements["source_experiment"] = source_experiment
    return CompiledInvocation(
        compiled_recipe=compile_recipe(raw),
        source_run_id=source_run_id,
        custom_reference_monitoring_run_id=custom_reference_monitoring_run_id,
    )


def prepare_test_context(
    *,
    subject_id: str,
    compiled_invocation: CompiledInvocation | CompiledRecipe,
    gateway: InMemoryMonitoringGateway,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
) -> PreparedContext:
    """Call Prepare while keeping Recipe and invocation identities separate."""
    if isinstance(compiled_invocation, CompiledInvocation):
        compiled_recipe = compiled_invocation.compiled_recipe
        effective_source_run_id = compiled_invocation.source_run_id
        custom_reference_monitoring_run_id = compiled_invocation.custom_reference_monitoring_run_id
    else:
        compiled_recipe = compiled_invocation
        if source_run_id is None:
            raise AssertionError("A Source Training Run is required for a bare CompiledRecipe.")
        effective_source_run_id = source_run_id
        custom_reference_monitoring_run_id = None
    allocation = allocate_test_monitoring_run(
        gateway,
        subject_id=subject_id,
        source_run_id=effective_source_run_id,
        compiled_recipe=compiled_recipe,
    )
    return _prepare_run_context(
        monitoring_run_id=allocation.monitoring_run_id,
        subject_id=subject_id,
        compiled_recipe=compiled_recipe,
        gateway=gateway,
        source_run_id=effective_source_run_id,
        sequence_index=allocation.sequence_index,
        baseline_source_run_id=baseline_source_run_id,
        custom_reference_monitoring_run_id=custom_reference_monitoring_run_id,
    )


def allocate_test_monitoring_run(
    gateway: InMemoryMonitoringGateway,
    *,
    subject_id: str,
    source_run_id: str,
    compiled_recipe: CompiledRecipe,
) -> CreateOrReuseMonitoringRunResult:
    """Allocate a Monitoring Run through the same public API as production."""
    return gateway.create_or_reuse_monitoring_run(
        IdempotencyKey(
            subject_id=subject_id,
            source_run_id=source_run_id,
            recipe_id=compiled_recipe.identity.recipe_id,
            recipe_version=compiled_recipe.identity.recipe_version,
        )
    )


def reconcile_test_timeline_baseline(
    gateway: InMemoryMonitoringGateway,
    *,
    subject_id: str = "churn_model",
    source_run_id: str = "train-run-123",
    baseline_source_run_id: str = "train-run-baseline",
    compiled_recipe: CompiledRecipe | None = None,
) -> TimelineState:
    """Allocate a Monitoring Run before bootstrapping its Timeline baseline."""
    effective_compiled_recipe = (
        compiled_recipe or make_compiled_invocation(source_run_id=source_run_id).compiled_recipe
    )
    allocation = allocate_test_monitoring_run(
        gateway,
        subject_id=subject_id,
        source_run_id=source_run_id,
        compiled_recipe=effective_compiled_recipe,
    )
    return gateway.reconcile_timeline_baseline(
        subject_id,
        allocation.monitoring_run_id,
        baseline_source_run_id,
    )


def make_gateway_with_timeline() -> InitializedTimelineFixture:
    """Build an initialized Timeline using public allocation operations."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    compiled_recipe = make_compiled_invocation().compiled_recipe
    previous_allocation = allocate_test_monitoring_run(
        gateway,
        subject_id="churn_model",
        source_run_id="train-run-prev",
        compiled_recipe=compiled_recipe,
    )
    gateway.reconcile_timeline_baseline(
        "churn_model",
        previous_allocation.monitoring_run_id,
        "train-run-baseline",
    )
    gateway.upsert_monitoring_run(
        subject_id="churn_model",
        monitoring_run_id=previous_allocation.monitoring_run_id,
        source_run_id="train-run-prev",
        lifecycle_status=LifecycleStatus.CLOSED,
        sequence_index=0,
    )
    custom_allocation = allocate_test_monitoring_run(
        gateway,
        subject_id="churn_model",
        source_run_id="train-run-custom-1",
        compiled_recipe=compiled_recipe,
    )
    gateway.upsert_monitoring_run(
        subject_id="churn_model",
        monitoring_run_id=custom_allocation.monitoring_run_id,
        source_run_id="train-run-custom-1",
        lifecycle_status=LifecycleStatus.CLOSED,
        sequence_index=1,
    )
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-123",
        source_experiment="training/churn",
        metrics={"f1": 0.91, "auc": 0.95},
        artifacts=("metrics.json", "model.pkl"),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )
    return InitializedTimelineFixture(
        gateway=gateway,
        previous_monitoring_run_id=previous_allocation.monitoring_run_id,
        custom_monitoring_run_id=custom_allocation.monitoring_run_id,
    )


class AliasResolvingBaselineGateway(InMemoryMonitoringGateway):
    """Test double that resolves baseline aliases to canonical source run ids."""

    def __init__(self, config: GatewayConfig) -> None:
        """Initialize alias mapping for source-run resolution."""
        super().__init__(config)
        self._aliases: dict[str, str] = {}

    def add_source_run_alias(self, alias: str, source_run_id: str) -> None:
        """Register an alias that resolves to a canonical source run id."""
        self._aliases[alias] = source_run_id

    def resolve_source_run_id(
        self,
        subject_id: str,
        source_experiment: str | None,
        source_run_id: str,
    ) -> str | None:
        """Resolve aliases first, then delegate to the base gateway behavior."""
        candidate = self._aliases.get(source_run_id, source_run_id)
        return super().resolve_source_run_id(
            subject_id=subject_id,
            source_experiment=source_experiment,
            source_run_id=candidate,
        )


class RecordingBaselineClaimGateway(InMemoryMonitoringGateway):
    """Record baseline reconciliation attempts made by Prepare."""

    def __init__(self, config: GatewayConfig) -> None:
        super().__init__(config)
        self.baseline_claim_calls: list[tuple[str, str, str]] = []

    def reconcile_timeline_baseline(
        self,
        subject_id: str,
        monitoring_run_id: str,
        baseline_source_run_id: str,
    ) -> TimelineState:
        """Record the claim before delegating to the in-memory gateway."""
        self.baseline_claim_calls.append((subject_id, monitoring_run_id, baseline_source_run_id))
        return super().reconcile_timeline_baseline(
            subject_id,
            monitoring_run_id,
            baseline_source_run_id,
        )
