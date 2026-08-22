"""Unit tests for workflow lifecycle transitions and prepare-stage behavior."""

from dataclasses import dataclass

import pytest

from mlflow_monitor.contract import SYSTEM_DEFAULT_CONTRACT_ID, resolve_contract_v0
from mlflow_monitor.contract_checker import DefaultContractChecker
from mlflow_monitor.domain import (
    ComparabilityStatus,
    Contract,
    ContractCheckReason,
    ContractCheckResult,
    DiffReferenceKind,
    LifecycleStatus,
    MonitoringRunReference,
)
from mlflow_monitor.errors import (
    CheckStageError,
    GatewayConsistencyViolation,
    PrepareStageError,
)
from mlflow_monitor.gateway import InMemoryMonitoringGateway
from mlflow_monitor.gateway_models import (
    CreateOrReuseMonitoringRunResult,
    GatewayConfig,
    IdempotencyKey,
    TimelineState,
)
from mlflow_monitor.recipe import SYSTEM_DEFAULT_RECIPE_ID
from mlflow_monitor.recipe_compiler import CompiledRecipe, compile_recipe
from mlflow_monitor.workflow import (
    PreparedContext,
    PreparedReferencePlanEntry,
    execute_contract_check,
    hydrate_prepared_context,
    prepared_context_to_dict,
)
from mlflow_monitor.workflow import (
    prepare_run_context as _prepare_run_context,
)

_CONTRACT = resolve_contract_v0(SYSTEM_DEFAULT_CONTRACT_ID)
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


def make_prepared_context(
    *,
    contract: Contract,
    source_run_id: str = "train-run-123",
    baseline_source_run_id: str = "train-run-baseline",
) -> PreparedContext:
    """Build a prepared context aligned with the common workflow test subject."""
    compiled_recipe = compile_recipe()
    return PreparedContext(
        monitoring_run_id="monitoring-run-1",
        source_run_id=source_run_id,
        subject_id="churn_model",
        timeline_id="timeline-churn_model",
        sequence_index=0,
        baseline_source_run_id=baseline_source_run_id,
        effective_recipe=compiled_recipe.effective_plan,
        contract=contract,
        reference_plan=(
            PreparedReferencePlanEntry(
                kind=DiffReferenceKind.BASELINE,
                reference=MonitoringRunReference(
                    kind=DiffReferenceKind.BASELINE,
                    monitoring_run_id=None,
                    source_run_id=baseline_source_run_id,
                ),
                unavailable_reason=None,
            ),
            PreparedReferencePlanEntry(
                kind=DiffReferenceKind.PREVIOUS,
                reference=None,
                unavailable_reason="previous_reference_missing",
            ),
            PreparedReferencePlanEntry(
                kind=DiffReferenceKind.LKG,
                reference=None,
                unavailable_reason="lkg_not_selected",
            ),
        ),
    )


def test_prepared_context_exposes_resolved_references_from_fixed_plan() -> None:
    """Prepared context should retain unavailable planned reference groups."""
    compiled_recipe = compile_recipe()
    baseline = MonitoringRunReference(
        kind=DiffReferenceKind.BASELINE,
        monitoring_run_id=None,
        source_run_id="train-run-baseline",
    )
    reference_plan = (
        PreparedReferencePlanEntry(
            kind=DiffReferenceKind.BASELINE,
            reference=baseline,
            unavailable_reason=None,
        ),
        PreparedReferencePlanEntry(
            kind=DiffReferenceKind.PREVIOUS,
            reference=None,
            unavailable_reason="previous_reference_missing",
        ),
        PreparedReferencePlanEntry(
            kind=DiffReferenceKind.LKG,
            reference=None,
            unavailable_reason="lkg_not_selected",
        ),
    )
    context = PreparedContext(
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-current",
        subject_id="churn_model",
        timeline_id="timeline-churn_model",
        sequence_index=0,
        baseline_source_run_id="train-run-baseline",
        effective_recipe=compiled_recipe.effective_plan,
        contract=compiled_recipe.contract,
        reference_plan=reference_plan,
    )

    assert context.references == (baseline,)
    assert prepared_context_to_dict(context)["reference_plan"] == [
        {
            "kind": "baseline",
            "monitoring_run_id": None,
            "source_run_id": "train-run-baseline",
            "unavailable_reason": None,
        },
        {
            "kind": "previous",
            "monitoring_run_id": None,
            "source_run_id": None,
            "unavailable_reason": "previous_reference_missing",
        },
        {
            "kind": "lkg",
            "monitoring_run_id": None,
            "source_run_id": None,
            "unavailable_reason": "lkg_not_selected",
        },
    ]


def test_hydrate_prepared_context_accepts_inconsistent_lkg_plan_shape() -> None:
    """Hydration should retain nonfatal LKG inconsistency for V0-030 replay."""
    compiled_recipe = compile_recipe()
    context = PreparedContext(
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-current",
        subject_id="churn_model",
        timeline_id="timeline-churn_model",
        sequence_index=3,
        baseline_source_run_id="train-run-baseline",
        effective_recipe=compiled_recipe.effective_plan,
        contract=compiled_recipe.contract,
        reference_plan=(
            PreparedReferencePlanEntry(
                kind=DiffReferenceKind.BASELINE,
                reference=MonitoringRunReference(
                    kind=DiffReferenceKind.BASELINE,
                    monitoring_run_id=None,
                    source_run_id="train-run-baseline",
                ),
                unavailable_reason=None,
            ),
            PreparedReferencePlanEntry(
                kind=DiffReferenceKind.PREVIOUS,
                reference=None,
                unavailable_reason="previous_reference_missing",
            ),
            PreparedReferencePlanEntry(
                kind=DiffReferenceKind.LKG,
                reference=None,
                unavailable_reason="lkg_selection_inconsistent",
            ),
        ),
    )

    hydrated = hydrate_prepared_context(
        prepared_context_to_dict(context),
        compiled_recipe=compiled_recipe,
        monitoring_run_id=context.monitoring_run_id,
        source_run_id=context.source_run_id,
        subject_id=context.subject_id,
        timeline_id=context.timeline_id,
        sequence_index=context.sequence_index,
    )

    assert hydrated == context


def test_hydrate_prepared_context_rejects_unpaired_unavailable_monitoring_run_id() -> None:
    """Unavailable plan entries should not retain an orphan Monitoring Run ID."""
    compiled_recipe = compile_recipe()
    context = make_prepared_context(contract=compiled_recipe.contract)
    raw = prepared_context_to_dict(context)
    references = raw["reference_plan"]
    assert isinstance(references, list)
    previous_reference = references[1]
    assert isinstance(previous_reference, dict)
    previous_reference["monitoring_run_id"] = "monitoring-run-orphan"

    with pytest.raises(GatewayConsistencyViolation) as exc_info:
        hydrate_prepared_context(
            raw,
            compiled_recipe=compiled_recipe,
            monitoring_run_id=context.monitoring_run_id,
            source_run_id=context.source_run_id,
            subject_id=context.subject_id,
            timeline_id=context.timeline_id,
            sequence_index=context.sequence_index,
        )

    assert exc_info.value.code == "prepared_context_inconsistent"


def test_hydrate_prepared_context_reports_reference_plan_for_baseline_mismatch() -> None:
    """Baseline mismatches should identify the persisted reference-plan field."""
    compiled_recipe = compile_recipe()
    context = make_prepared_context(contract=compiled_recipe.contract)
    raw = prepared_context_to_dict(context)
    reference_plan = raw["reference_plan"]
    assert isinstance(reference_plan, list)
    baseline_reference = reference_plan[0]
    assert isinstance(baseline_reference, dict)
    baseline_reference["source_run_id"] = "train-run-other"

    with pytest.raises(GatewayConsistencyViolation) as exc_info:
        hydrate_prepared_context(
            raw,
            compiled_recipe=compiled_recipe,
            monitoring_run_id=context.monitoring_run_id,
            source_run_id=context.source_run_id,
            subject_id=context.subject_id,
            timeline_id=context.timeline_id,
            sequence_index=context.sequence_index,
        )

    assert exc_info.value.details == (
        ("reason", "baseline_reference_mismatch"),
        ("field", "reference_plan"),
    )


def test_hydrate_prepared_context_rejects_obsolete_prepared_context_schema() -> None:
    """Resolved-only prepared artifacts should fail closed without migration."""
    compiled_recipe = compile_recipe()
    context = make_prepared_context(contract=compiled_recipe.contract)
    raw = prepared_context_to_dict(context)
    del raw["reference_plan"]
    raw["references"] = [reference.to_dict() for reference in context.references]

    with pytest.raises(GatewayConsistencyViolation) as exc_info:
        hydrate_prepared_context(
            raw,
            compiled_recipe=compiled_recipe,
            monitoring_run_id=context.monitoring_run_id,
            source_run_id=context.source_run_id,
            subject_id=context.subject_id,
            timeline_id=context.timeline_id,
            sequence_index=context.sequence_index,
        )

    assert exc_info.value.code == "prepared_context_inconsistent"


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
