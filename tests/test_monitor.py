"""Unit tests for W-003 orchestration and public monitor entrypoint."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Protocol, cast

import pytest

from mlflow_monitor import monitor
from mlflow_monitor.builtins import SYSTEM_DEFAULT_RECIPE_ID
from mlflow_monitor.contract_checker import DefaultContractChecker
from mlflow_monitor.domain import (
    ComparabilityStatus,
    ContractCheckReason,
    ContractCheckResult,
    DiffReferenceKind,
    LifecycleStatus,
    MonitoringRunReference,
)
from mlflow_monitor.errors import (
    PREPARE_BASELINE_OVERRIDE_EXISTING_BASELINE,
    GatewayConsistencyViolation,
)
from mlflow_monitor.gateway import (
    CreateOrReuseMonitoringRunResult,
    GatewayConfig,
    IdempotencyKey,
    InMemoryMonitoringGateway,
    TimelineState,
)
from mlflow_monitor.orchestration import run_orchestration
from mlflow_monitor.recipe import SYSTEM_DEFAULT_RECIPE_VERSION, build_system_default_recipe
from mlflow_monitor.recipe_compiler import CompiledRecipe, ComponentRegistry, compile_recipe
from mlflow_monitor.result_contract import MonitorRunResult
from mlflow_monitor.utils import canonical_json
from mlflow_monitor.workflow import PreparedReferencePlanEntry

_PREPARED_CONTEXT_PATH = "state/prepared_context.json"
_CONTRACT_CHECK_PATH = "outputs/contract_check.json"
_USE_STORED_PREPARED_CONTEXT = object()
_PREPARED_CONTEXT_INCONSISTENT_CODE_FOR_TESTS = "prepared_context_inconsistent"


class _ReadablePreparedContextGateway(Protocol):
    """Test-facing Gateway contract for prepared-context hydration."""

    def read_monitoring_run_json_artifact(
        self,
        monitoring_run_id: str,
        path: str,
    ) -> dict[str, object] | None:
        """Read one exact-path Monitoring Run JSON artifact."""
        ...


def read_prepared_context(
    gateway: InMemoryMonitoringGateway,
    monitoring_run_id: str,
) -> dict[str, object] | None:
    """Read the persisted prepared context through the planned Gateway API."""
    return cast(_ReadablePreparedContextGateway, gateway).read_monitoring_run_json_artifact(
        monitoring_run_id,
        _PREPARED_CONTEXT_PATH,
    )


def read_contract_check(
    gateway: InMemoryMonitoringGateway,
    monitoring_run_id: str,
) -> dict[str, object] | None:
    """Read persisted Contract Check output through the Gateway boundary."""
    return cast(_ReadablePreparedContextGateway, gateway).read_monitoring_run_json_artifact(
        monitoring_run_id,
        _CONTRACT_CHECK_PATH,
    )


def add_default_source_runs(gateway: InMemoryMonitoringGateway) -> None:
    """Seed a gateway with the default baseline and current Source Training Runs."""
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-baseline",
        source_experiment=None,
        metrics={"f1": 0.87},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age",),
        schema={"age": "int"},
        data_scope="validation:2026-03-01",
    )
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-current",
        source_experiment=None,
        metrics={"f1": 0.91},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age",),
        schema={"age": "int"},
        data_scope="validation:2026-03-01",
    )


def make_gateway() -> InMemoryMonitoringGateway:
    """Build a gateway with baseline and current source runs."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    add_default_source_runs(gateway)
    return gateway


def make_compiled_recipe(
    *,
    recipe_id: str = "churn-monitoring",
    recipe_version: str = "3",
    metric_names: list[str] | None = None,
) -> CompiledRecipe:
    """Build a compiled Recipe with configurable identity and analysis."""
    raw = build_system_default_recipe()
    raw["identity"] = {
        "recipe_id": recipe_id,
        "recipe_version": recipe_version,
    }
    if metric_names is not None:
        raw["analysis"] = {"metric_names": metric_names}
    return compile_recipe(raw)


def expected_prepared_context_payload(
    *,
    monitoring_run_id: str,
    source_run_id: str,
    sequence_index: int,
    compiled_recipe: CompiledRecipe,
    reference_plan: tuple[PreparedReferencePlanEntry, ...],
) -> dict[str, object]:
    """Build the complete prepared-context artifact contract."""
    contract = compiled_recipe.contract
    return {
        "artifact_schema_version": "v0",
        "monitoring_run_id": monitoring_run_id,
        "source_run_id": source_run_id,
        "subject_id": "churn_model",
        "timeline_id": "timeline-churn_model",
        "sequence_index": sequence_index,
        "baseline_source_run_id": "train-run-baseline",
        "effective_recipe": compiled_recipe.effective_plan.to_dict(),
        "contract": {
            "contract_id": contract.contract_id,
            "contract_version": contract.contract_version,
            "schema_contract_ref": contract.schema_contract_ref,
            "feature_contract_ref": contract.feature_contract_ref,
            "metric_contract_ref": contract.metric_contract_ref,
            "data_scope_contract_ref": contract.data_scope_contract_ref,
            "execution_contract_ref": contract.execution_contract_ref,
        },
        "reference_plan": [entry.to_dict() for entry in reference_plan],
    }


class InvalidResultContractChecker:
    """Return a contract-check result that violates invariants."""

    def check(self, contract, context) -> ContractCheckResult:  # type: ignore[no-untyped-def]
        return ContractCheckResult(
            status=ComparabilityStatus.PASS,
            reasons=(
                ContractCheckReason(
                    code="schema_mismatch",
                    message="Data schema does not match the baseline.",
                    blocking=True,
                ),
            ),
        )


class RaisingContractChecker:
    """Raise an unexpected runtime error from the checker."""

    def check(self, contract, context) -> ContractCheckResult:  # type: ignore[no-untyped-def]
        raise RuntimeError("checker exploded")


class BrokenUpsertGateway(InMemoryMonitoringGateway):
    """Raise a gateway consistency error during monitoring-run persistence."""

    def upsert_monitoring_run(
        self,
        subject_id: str,
        monitoring_run_id: str,
        source_run_id: str,
        lifecycle_status: LifecycleStatus,
        sequence_index: int,
        contract_check_result: ContractCheckResult | None = None,
        references: tuple[MonitoringRunReference, ...] | None = None,
    ) -> None:
        if lifecycle_status is LifecycleStatus.PREPARED:
            raise GatewayConsistencyViolation.monitoring_run_upsert_field_override(
                fields=(("lifecycle_status", lifecycle_status.value),)
            )

        super().upsert_monitoring_run(
            subject_id=subject_id,
            monitoring_run_id=monitoring_run_id,
            source_run_id=source_run_id,
            lifecycle_status=lifecycle_status,
            sequence_index=sequence_index,
            contract_check_result=contract_check_result,
            references=references,
        )


class ReplaySensitiveGateway(InMemoryMonitoringGateway):
    """Fail if checked reruns re-enter prepare-sensitive source-run reads."""

    def __init__(self, config: GatewayConfig) -> None:
        super().__init__(config)
        self.block_timeline_reads = False
        self.block_source_resolution = False
        self.block_baseline_evidence = False
        self.block_current_evidence = False
        self._allocation_in_progress = False

    def create_or_reuse_monitoring_run(
        self, key: IdempotencyKey
    ) -> CreateOrReuseMonitoringRunResult:
        self._allocation_in_progress = True
        try:
            return super().create_or_reuse_monitoring_run(key)
        finally:
            self._allocation_in_progress = False

    def get_timeline_state(self, subject_id: str) -> TimelineState | None:
        if self.block_timeline_reads and not self._allocation_in_progress:
            raise AssertionError("checked replay must not reload Timeline state")
        return super().get_timeline_state(subject_id)

    def resolve_source_run_id(
        self,
        subject_id: str,
        source_experiment: str | None,
        source_run_id: str,
    ) -> str | None:
        if self.block_source_resolution:
            raise AssertionError("checked replay must not resolve Source Training Runs")
        return super().resolve_source_run_id(
            subject_id=subject_id,
            source_experiment=source_experiment,
            source_run_id=source_run_id,
        )

    def get_source_run_contract_evidence(self, source_run_id: str):
        if source_run_id == "train-run-baseline" and self.block_baseline_evidence:
            raise AssertionError("checked replay must not reload baseline Contract evidence")
        if source_run_id == "train-run-current" and self.block_current_evidence:
            raise AssertionError("checked replay must not reload current Contract evidence")
        return super().get_source_run_contract_evidence(source_run_id)


class AllocationOnlyReplayGateway(InMemoryMonitoringGateway):
    """Replay a prior allocation before any monitoring-run record is persisted."""

    def __init__(self, config: GatewayConfig) -> None:
        super().__init__(config)
        self._replayed = False

    def create_or_reuse_monitoring_run(
        self, key: IdempotencyKey
    ) -> CreateOrReuseMonitoringRunResult:
        result = super().create_or_reuse_monitoring_run(key)
        if self._replayed or not result.allocated:
            return result
        replayed = super().create_or_reuse_monitoring_run(key)
        self._replayed = True
        return CreateOrReuseMonitoringRunResult(
            monitoring_run_id=replayed.monitoring_run_id,
            source_run_id=replayed.source_run_id,
            timeline_id=replayed.timeline_id,
            sequence_index=replayed.sequence_index,
            existing_monitoring_run=None,
            allocated=False,
        )


class UnreadableAllocatedTimelineGateway(InMemoryMonitoringGateway):
    """Hide Timeline reads after returning a successful allocation."""

    def __init__(self, config: GatewayConfig) -> None:
        super().__init__(config)
        self._timeline_reads_blocked = False
        self.allocation_timeline_id: str | None = None

    def create_or_reuse_monitoring_run(
        self, key: IdempotencyKey
    ) -> CreateOrReuseMonitoringRunResult:
        result = super().create_or_reuse_monitoring_run(key)
        self.allocation_timeline_id = result.timeline_id
        self._timeline_reads_blocked = True
        return result

    def get_timeline_state(self, subject_id: str) -> TimelineState | None:
        if self._timeline_reads_blocked:
            return None
        return super().get_timeline_state(subject_id)


class FinalizingGateway(InMemoryMonitoringGateway):
    """Capture terminal result finalization calls from orchestration."""

    def __init__(self, config: GatewayConfig) -> None:
        super().__init__(config)
        self.finalized_results: list[MonitorRunResult] = []

    def finalize_monitoring_run_result(
        self,
        *,
        monitoring_run_id: str,
        result: MonitorRunResult,
    ) -> None:
        assert monitoring_run_id == result.monitoring_run_id
        self.finalized_results.append(result)


class RecordingAllocationGateway(InMemoryMonitoringGateway):
    """Record every idempotency key presented to Monitoring Run allocation."""

    def __init__(self, config: GatewayConfig) -> None:
        super().__init__(config)
        self.allocation_keys: list[IdempotencyKey] = []

    def create_or_reuse_monitoring_run(
        self, key: IdempotencyKey
    ) -> CreateOrReuseMonitoringRunResult:
        self.allocation_keys.append(key)
        return super().create_or_reuse_monitoring_run(key)


class RecordingPreparedCommitGateway(InMemoryMonitoringGateway):
    """Record prepared artifact and lifecycle writes in call order."""

    def __init__(self, config: GatewayConfig) -> None:
        super().__init__(config)
        self.persistence_events: list[tuple[str, object]] = []

    def reconcile_timeline_baseline(
        self,
        subject_id: str,
        monitoring_run_id: str,
        baseline_source_run_id: str,
    ) -> TimelineState:
        """Record a successfully persisted baseline claim."""
        timeline_state = super().reconcile_timeline_baseline(
            subject_id,
            monitoring_run_id,
            baseline_source_run_id,
        )
        self.persistence_events.append(
            (
                "baseline_claim",
                (monitoring_run_id, baseline_source_run_id),
            )
        )
        return timeline_state

    def write_monitoring_run_json_artifact(
        self,
        monitoring_run_id: str,
        data: dict[str, object],
        path: str,
    ) -> None:
        super().write_monitoring_run_json_artifact(monitoring_run_id, data, path)
        self.persistence_events.append(("artifact", path))

    def upsert_monitoring_run(
        self,
        subject_id: str,
        monitoring_run_id: str,
        source_run_id: str,
        lifecycle_status: LifecycleStatus,
        sequence_index: int,
        contract_check_result: ContractCheckResult | None = None,
        references: tuple[MonitoringRunReference, ...] | None = None,
    ) -> None:
        super().upsert_monitoring_run(
            subject_id=subject_id,
            monitoring_run_id=monitoring_run_id,
            source_run_id=source_run_id,
            lifecycle_status=lifecycle_status,
            sequence_index=sequence_index,
            contract_check_result=contract_check_result,
            references=references,
        )
        self.persistence_events.append(("lifecycle", lifecycle_status))


class InterruptPreparedCommitGateway(InMemoryMonitoringGateway):
    """Interrupt once after Prepare should have written its artifact."""

    def __init__(self, config: GatewayConfig) -> None:
        super().__init__(config)
        self.last_allocation: CreateOrReuseMonitoringRunResult | None = None
        self._interrupt_prepared_commit = True

    def create_or_reuse_monitoring_run(
        self, key: IdempotencyKey
    ) -> CreateOrReuseMonitoringRunResult:
        result = super().create_or_reuse_monitoring_run(key)
        self.last_allocation = result
        return result

    def upsert_monitoring_run(
        self,
        subject_id: str,
        monitoring_run_id: str,
        source_run_id: str,
        lifecycle_status: LifecycleStatus,
        sequence_index: int,
        contract_check_result: ContractCheckResult | None = None,
        references: tuple[MonitoringRunReference, ...] | None = None,
    ) -> None:
        if lifecycle_status is LifecycleStatus.PREPARED and self._interrupt_prepared_commit:
            self._interrupt_prepared_commit = False
            raise RuntimeError("simulated interruption before prepared commit")
        super().upsert_monitoring_run(
            subject_id=subject_id,
            monitoring_run_id=monitoring_run_id,
            source_run_id=source_run_id,
            lifecycle_status=lifecycle_status,
            sequence_index=sequence_index,
            contract_check_result=contract_check_result,
            references=references,
        )

    def replace_prepared_context_for_test(self, data: dict[str, object]) -> None:
        """Replace the partial artifact to simulate contradictory persisted content."""
        assert self.last_allocation is not None
        self._monitoring_runs_artifact_store[
            (self.last_allocation.monitoring_run_id, _PREPARED_CONTEXT_PATH)
        ] = canonical_json(data)


class InterruptCheckCommitGateway(InMemoryMonitoringGateway):
    """Interrupt once after Check output is written but before checked commits."""

    def __init__(self, config: GatewayConfig) -> None:
        super().__init__(config)
        self.last_allocation: CreateOrReuseMonitoringRunResult | None = None
        self._interrupt_checked_commit = True
        self._lifecycle_advance_after_allocation: LifecycleStatus | None = None
        self.finalized_results: list[MonitorRunResult] = []
        self.baseline_reconciliations: list[tuple[str, str]] = []
        self.persistence_events: list[tuple[str, object]] = []

    def create_or_reuse_monitoring_run(
        self, key: IdempotencyKey
    ) -> CreateOrReuseMonitoringRunResult:
        result = super().create_or_reuse_monitoring_run(key)
        self.last_allocation = result
        if self._lifecycle_advance_after_allocation is not None:
            assert result.existing_monitoring_run is not None
            self.upsert_monitoring_run(
                subject_id=key.subject_id,
                monitoring_run_id=result.monitoring_run_id,
                source_run_id=result.source_run_id,
                lifecycle_status=self._lifecycle_advance_after_allocation,
                sequence_index=result.sequence_index,
            )
            self._lifecycle_advance_after_allocation = None
        return result

    def advance_lifecycle_after_allocation_for_test(
        self,
        lifecycle_status: LifecycleStatus,
    ) -> None:
        """Advance persisted lifecycle after the next allocation snapshot."""
        self._lifecycle_advance_after_allocation = lifecycle_status

    def upsert_monitoring_run(
        self,
        subject_id: str,
        monitoring_run_id: str,
        source_run_id: str,
        lifecycle_status: LifecycleStatus,
        sequence_index: int,
        contract_check_result: ContractCheckResult | None = None,
        references: tuple[MonitoringRunReference, ...] | None = None,
    ) -> None:
        if lifecycle_status is LifecycleStatus.CHECKED and self._interrupt_checked_commit:
            self._interrupt_checked_commit = False
            raise RuntimeError("simulated interruption before checked commit")
        super().upsert_monitoring_run(
            subject_id=subject_id,
            monitoring_run_id=monitoring_run_id,
            source_run_id=source_run_id,
            lifecycle_status=lifecycle_status,
            sequence_index=sequence_index,
            contract_check_result=contract_check_result,
            references=references,
        )
        if lifecycle_status in {LifecycleStatus.CHECKED, LifecycleStatus.FAILED}:
            self.persistence_events.append(("lifecycle_status", lifecycle_status))

    def finalize_monitoring_run_result(
        self,
        *,
        monitoring_run_id: str,
        result: MonitorRunResult,
    ) -> None:
        assert monitoring_run_id == result.monitoring_run_id
        self.finalized_results.append(result)

    def reconcile_timeline_baseline(
        self,
        subject_id: str,
        monitoring_run_id: str,
        baseline_source_run_id: str,
    ) -> TimelineState:
        """Record successful baseline reconciliation during recovery."""
        timeline_state = super().reconcile_timeline_baseline(
            subject_id,
            monitoring_run_id,
            baseline_source_run_id,
        )
        self.baseline_reconciliations.append((monitoring_run_id, baseline_source_run_id))
        self.persistence_events.append(("baseline_reconciliation", monitoring_run_id))
        return timeline_state

    def remove_baseline_claim_for_test(self) -> None:
        """Remove the current claim to require prepared-replay reconciliation."""
        assert self.last_allocation is not None
        self._baseline_claim_by_monitoring_run_id.pop(
            self.last_allocation.monitoring_run_id,
        )
        self.baseline_reconciliations.clear()
        self.persistence_events.clear()

    def has_baseline_claim_for_test(self) -> bool:
        """Return whether the interrupted Monitoring Run still owns its claim."""
        assert self.last_allocation is not None
        return self.last_allocation.monitoring_run_id in self._baseline_claim_by_monitoring_run_id

    def replace_contract_check_for_test(self, data: Mapping[str, object]) -> None:
        """Replace partial Check output to simulate persisted contradiction."""
        assert self.last_allocation is not None
        self._monitoring_runs_artifact_store[
            (self.last_allocation.monitoring_run_id, _CONTRACT_CHECK_PATH)
        ] = canonical_json(dict(data))

    def write_monitoring_run_json_artifact(
        self,
        monitoring_run_id: str,
        data: dict[str, object],
        path: str,
    ) -> None:
        """Record successful Check artifact validation or persistence."""
        super().write_monitoring_run_json_artifact(monitoring_run_id, data, path)
        if path == _CONTRACT_CHECK_PATH:
            self.persistence_events.append(("contract_check_artifact", monitoring_run_id))

    def delete_contract_check_for_test(self) -> None:
        """Remove persisted Check output to model a legacy checked record."""
        assert self.last_allocation is not None
        self._monitoring_runs_artifact_store.pop(
            (self.last_allocation.monitoring_run_id, _CONTRACT_CHECK_PATH),
            None,
        )

    def replace_comparability_projection_for_test(
        self,
        status: ComparabilityStatus,
    ) -> None:
        """Replace the in-memory comparability projection for consistency tests."""
        assert self.last_allocation is not None
        monitoring_run_id = self.last_allocation.monitoring_run_id
        stored = self._monitoring_runs_by_subject["churn_model"][monitoring_run_id]
        self._monitoring_runs_by_subject["churn_model"][monitoring_run_id] = replace(
            stored,
            comparability_status=status,
        )


class PreparedReplayGateway(InMemoryMonitoringGateway):
    """Control prepared-context reads and reject live Prepare resolution on replay."""

    def __init__(self, config: GatewayConfig) -> None:
        super().__init__(config)
        self.block_prepare_resolution = False
        self.prepared_context_override: object = _USE_STORED_PREPARED_CONTEXT
        self.corrupt_timeline_id = False
        self.baseline_reconciliations: list[tuple[str, str]] = []
        self.last_allocation: CreateOrReuseMonitoringRunResult | None = None
        self._allocation_in_progress = False

    def create_or_reuse_monitoring_run(
        self, key: IdempotencyKey
    ) -> CreateOrReuseMonitoringRunResult:
        self._allocation_in_progress = True
        try:
            result = super().create_or_reuse_monitoring_run(key)
        finally:
            self._allocation_in_progress = False
        self.last_allocation = result
        return result

    def reconcile_timeline_baseline(
        self,
        subject_id: str,
        monitoring_run_id: str,
        baseline_source_run_id: str,
    ) -> TimelineState:
        timeline_state = super().reconcile_timeline_baseline(
            subject_id,
            monitoring_run_id,
            baseline_source_run_id,
        )
        self.baseline_reconciliations.append((monitoring_run_id, baseline_source_run_id))
        return timeline_state

    def simulate_projection_only_prepared_run(self, monitoring_run_id: str) -> None:
        """Remove the durable claim to model state written before V0-013."""
        self._baseline_claim_by_monitoring_run_id.pop(monitoring_run_id)
        self.baseline_reconciliations.clear()

    def get_timeline_state(self, subject_id: str) -> TimelineState | None:
        if self.block_prepare_resolution and not self._allocation_in_progress:
            raise AssertionError("prepared replay must not reload Timeline state")
        return super().get_timeline_state(subject_id)

    def read_monitoring_run_json_artifact(
        self,
        monitoring_run_id: str,
        path: str,
    ) -> dict[str, object] | None:
        if (
            path == _PREPARED_CONTEXT_PATH
            and self.prepared_context_override is not _USE_STORED_PREPARED_CONTEXT
        ):
            if self.prepared_context_override is None:
                return None
            return cast(dict[str, object], self.prepared_context_override)

        reader = cast(_ReadablePreparedContextGateway, super())
        artifact = reader.read_monitoring_run_json_artifact(monitoring_run_id, path)
        if artifact is None or path != _PREPARED_CONTEXT_PATH or not self.corrupt_timeline_id:
            return artifact
        corrupted = dict(artifact)
        corrupted["timeline_id"] = "timeline-other"
        return corrupted


def complete_interrupted_check(gateway: InterruptCheckCommitGateway) -> str:
    """Complete one interrupted Check and return its Monitoring Run identifier."""
    add_default_source_runs(gateway)
    with pytest.raises(RuntimeError, match="simulated interruption before checked commit"):
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id="train-run-baseline",
            gateway=gateway,
            contract_checker=DefaultContractChecker(),
        )
    replay = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    assert replay.lifecycle_status is LifecycleStatus.CHECKED
    gateway.finalized_results.clear()
    return replay.monitoring_run_id


def leave_monitoring_run_prepared(
    gateway: PreparedReplayGateway,
    *,
    recipe: CompiledRecipe | None = None,
) -> str:
    """Interrupt Check after Prepare has committed and return its Monitoring Run ID."""
    with pytest.raises(RuntimeError, match="checker exploded"):
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id="train-run-baseline",
            recipe=recipe,
            gateway=gateway,
            contract_checker=RaisingContractChecker(),
        )

    assert gateway.last_allocation is not None
    monitoring_run_id = gateway.last_allocation.monitoring_run_id
    stored = gateway.get_monitoring_run("churn_model", monitoring_run_id)
    assert stored is not None
    assert stored.lifecycle_status is LifecycleStatus.PREPARED
    return monitoring_run_id


def test_run_orchestration_first_run_persists_checked_state() -> None:
    gateway = make_gateway()

    result = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    stored = gateway.get_monitoring_run("churn_model", result.monitoring_run_id)

    assert result.lifecycle_status is LifecycleStatus.CHECKED
    assert result.comparability_status is ComparabilityStatus.PASS
    assert result.timeline_id == "timeline-churn_model"
    assert result.references == (
        MonitoringRunReference(
            kind=DiffReferenceKind.BASELINE,
            monitoring_run_id=None,
            source_run_id="train-run-baseline",
        ),
    )
    assert result.finding_ids == ()
    assert result.diff_ids == ()
    assert result.summary is None
    assert result.error is None
    assert stored is not None
    assert stored.source_run_id == "train-run-current"
    assert stored.sequence_index == 0
    assert stored.lifecycle_status is LifecycleStatus.CHECKED
    assert stored.comparability_status is ComparabilityStatus.PASS
    assert stored.contract_check_result is not None
    assert stored.contract_check_result.status is ComparabilityStatus.PASS


def test_run_orchestration_persists_complete_prepared_context() -> None:
    gateway = make_gateway()
    compiled_recipe = make_compiled_recipe(metric_names=["f1"])
    reference_allocation = gateway.create_or_reuse_monitoring_run(
        IdempotencyKey(
            subject_id="churn_model",
            source_run_id="train-run-reference",
            recipe_id=compiled_recipe.identity.recipe_id,
            recipe_version=compiled_recipe.identity.recipe_version,
        )
    )
    gateway.upsert_monitoring_run(
        subject_id="churn_model",
        monitoring_run_id=reference_allocation.monitoring_run_id,
        source_run_id="train-run-reference",
        lifecycle_status=LifecycleStatus.CLOSED,
        sequence_index=reference_allocation.sequence_index,
    )
    first = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        recipe=compiled_recipe,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    gateway.set_active_lkg_monitoring_run_id("churn_model", first.monitoring_run_id)
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-next",
        source_experiment=None,
        metrics={"f1": 0.92},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age",),
        schema={"age": "int"},
        data_scope="validation:2026-03-02",
    )

    second = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-next",
        baseline_source_run_id=None,
        custom_reference_monitoring_run_id=reference_allocation.monitoring_run_id,
        recipe=compiled_recipe,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    expected_reference_plan = (
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
            reference=MonitoringRunReference(
                kind=DiffReferenceKind.PREVIOUS,
                monitoring_run_id=reference_allocation.monitoring_run_id,
                source_run_id="train-run-reference",
            ),
            unavailable_reason=None,
        ),
        PreparedReferencePlanEntry(
            kind=DiffReferenceKind.LKG,
            reference=None,
            unavailable_reason="lkg_not_selected",
        ),
        PreparedReferencePlanEntry(
            kind=DiffReferenceKind.CUSTOM,
            reference=MonitoringRunReference(
                kind=DiffReferenceKind.CUSTOM,
                monitoring_run_id=reference_allocation.monitoring_run_id,
                source_run_id="train-run-reference",
            ),
            unavailable_reason=None,
        ),
    )
    assert read_prepared_context(gateway, second.monitoring_run_id) == (
        expected_prepared_context_payload(
            monitoring_run_id=second.monitoring_run_id,
            source_run_id="train-run-next",
            sequence_index=2,
            compiled_recipe=compiled_recipe,
            reference_plan=expected_reference_plan,
        )
    )


def test_prepare_writes_context_before_prepared_lifecycle_marker() -> None:
    gateway = RecordingPreparedCommitGateway(GatewayConfig())
    add_default_source_runs(gateway)

    run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    monitoring_run_id = gateway.idempotency_bindings("churn_model")[
        "train-run-current|system_default|v0"
    ]
    claim_index = gateway.persistence_events.index(
        (
            "baseline_claim",
            (monitoring_run_id, "train-run-baseline"),
        )
    )
    artifact_index = gateway.persistence_events.index(("artifact", _PREPARED_CONTEXT_PATH))
    prepared_index = gateway.persistence_events.index(("lifecycle", LifecycleStatus.PREPARED))
    assert claim_index < artifact_index < prepared_index


def test_check_writes_complete_output_before_checked_lifecycle_marker() -> None:
    gateway = RecordingPreparedCommitGateway(GatewayConfig())
    add_default_source_runs(gateway)

    run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    artifact_index = gateway.persistence_events.index(("artifact", _CONTRACT_CHECK_PATH))
    checked_index = gateway.persistence_events.index(("lifecycle", LifecycleStatus.CHECKED))
    assert artifact_index < checked_index


def test_created_replay_reuses_identical_partial_prepared_context() -> None:
    gateway = InterruptPreparedCommitGateway(GatewayConfig())
    add_default_source_runs(gateway)

    with pytest.raises(RuntimeError, match="simulated interruption before prepared commit"):
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id="train-run-baseline",
            gateway=gateway,
            contract_checker=DefaultContractChecker(),
        )

    assert gateway.last_allocation is not None
    monitoring_run_id = gateway.last_allocation.monitoring_run_id
    stored = gateway.get_monitoring_run("churn_model", monitoring_run_id)
    assert stored is not None
    assert stored.lifecycle_status is LifecycleStatus.CREATED
    assert read_prepared_context(gateway, monitoring_run_id) is not None

    replay = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    assert replay.monitoring_run_id == monitoring_run_id
    assert replay.lifecycle_status is LifecycleStatus.CHECKED


def test_created_replay_rejects_conflicting_partial_prepared_context() -> None:
    gateway = InterruptPreparedCommitGateway(GatewayConfig())
    add_default_source_runs(gateway)

    with pytest.raises(RuntimeError, match="simulated interruption before prepared commit"):
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id="train-run-baseline",
            gateway=gateway,
            contract_checker=DefaultContractChecker(),
        )

    gateway.replace_prepared_context_for_test(
        {
            "artifact_schema_version": "v0",
            "monitoring_run_id": "contradictory-monitoring-run",
            "source_run_id": "train-run-current",
        }
    )

    with pytest.raises(GatewayConsistencyViolation):
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id=None,
            gateway=gateway,
            contract_checker=DefaultContractChecker(),
        )

    assert gateway.last_allocation is not None
    stored = gateway.get_monitoring_run(
        "churn_model",
        gateway.last_allocation.monitoring_run_id,
    )
    assert stored is not None
    assert stored.lifecycle_status is LifecycleStatus.CREATED


def test_prepared_replay_reuses_identical_partial_contract_check_output() -> None:
    gateway = InterruptCheckCommitGateway(GatewayConfig())
    add_default_source_runs(gateway)

    with pytest.raises(RuntimeError, match="simulated interruption before checked commit"):
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id="train-run-baseline",
            gateway=gateway,
            contract_checker=DefaultContractChecker(),
        )

    assert gateway.last_allocation is not None
    monitoring_run_id = gateway.last_allocation.monitoring_run_id
    stored = gateway.get_monitoring_run("churn_model", monitoring_run_id)
    assert stored is not None
    assert stored.lifecycle_status is LifecycleStatus.PREPARED
    assert read_contract_check(gateway, monitoring_run_id) is not None
    gateway.remove_baseline_claim_for_test()

    replay = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    assert replay.lifecycle_status is LifecycleStatus.CHECKED
    assert len(gateway.finalized_results) == 1
    assert gateway.has_baseline_claim_for_test() is True
    assert gateway.persistence_events == [
        ("contract_check_artifact", monitoring_run_id),
        ("baseline_reconciliation", monitoring_run_id),
        ("lifecycle_status", LifecycleStatus.CHECKED),
    ]


def test_prepared_replay_keeps_invalid_fresh_check_as_owned_failure() -> None:
    gateway = InterruptCheckCommitGateway(GatewayConfig())
    add_default_source_runs(gateway)

    with pytest.raises(RuntimeError, match="simulated interruption before checked commit"):
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id="train-run-baseline",
            gateway=gateway,
            contract_checker=DefaultContractChecker(),
        )

    assert gateway.last_allocation is not None
    monitoring_run_id = gateway.last_allocation.monitoring_run_id
    persisted_before = read_contract_check(gateway, monitoring_run_id)
    assert persisted_before is not None
    gateway.remove_baseline_claim_for_test()

    replay = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=InvalidResultContractChecker(),
    )

    stored = gateway.get_monitoring_run("churn_model", monitoring_run_id)
    assert stored is not None
    assert stored.lifecycle_status is LifecycleStatus.FAILED
    assert replay.lifecycle_status is LifecycleStatus.FAILED
    assert replay.error is not None
    assert replay.error.code == "check_result_invalid"
    assert read_contract_check(gateway, monitoring_run_id) == persisted_before
    assert gateway.finalized_results == [replay]
    assert gateway.has_baseline_claim_for_test() is False
    assert gateway.baseline_reconciliations == []
    assert gateway.persistence_events == [
        ("lifecycle_status", LifecycleStatus.FAILED),
    ]


def test_prepared_replay_rejects_conflicting_partial_contract_check_output() -> None:
    gateway = InterruptCheckCommitGateway(GatewayConfig())
    add_default_source_runs(gateway)
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-current",
        source_experiment=None,
        metrics={"f1": 0.91},
        artifacts=("metrics.json",),
        environment={"python": "3.11"},
        features=("age",),
        schema={"age": "int"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(RuntimeError, match="simulated interruption before checked commit"):
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id="train-run-baseline",
            gateway=gateway,
            contract_checker=DefaultContractChecker(),
        )

    assert gateway.last_allocation is not None
    monitoring_run_id = gateway.last_allocation.monitoring_run_id
    persisted_before = read_contract_check(gateway, monitoring_run_id)
    assert persisted_before is not None
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-current",
        source_experiment=None,
        metrics={"f1": 0.91},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age",),
        schema={"age": "int"},
        data_scope="validation:2026-03-01",
    )
    gateway.remove_baseline_claim_for_test()
    assert gateway.has_baseline_claim_for_test() is False

    with pytest.raises(GatewayConsistencyViolation):
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id=None,
            gateway=gateway,
            contract_checker=DefaultContractChecker(),
        )

    stored = gateway.get_monitoring_run("churn_model", monitoring_run_id)
    assert stored is not None
    assert stored.lifecycle_status is LifecycleStatus.PREPARED
    assert stored.contract_check_result is None
    assert read_contract_check(gateway, monitoring_run_id) == persisted_before
    assert gateway.finalized_results == []
    assert gateway.has_baseline_claim_for_test() is False
    assert gateway.baseline_reconciliations == []


def test_prepared_replay_rejects_duplicate_reasons_in_partial_contract_check() -> None:
    gateway = InterruptCheckCommitGateway(GatewayConfig())
    add_default_source_runs(gateway)
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-current",
        source_experiment=None,
        metrics={"f1": 0.91},
        artifacts=("metrics.json",),
        environment={"python": "3.11"},
        features=("age",),
        schema={"age": "int"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(RuntimeError, match="simulated interruption before checked commit"):
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id="train-run-baseline",
            gateway=gateway,
            contract_checker=DefaultContractChecker(),
        )

    assert gateway.last_allocation is not None
    monitoring_run_id = gateway.last_allocation.monitoring_run_id
    artifact = read_contract_check(gateway, monitoring_run_id)
    assert artifact is not None
    reasons = artifact["reasons"]
    assert isinstance(reasons, list)
    reasons.append(dict(reasons[0]))
    gateway.replace_contract_check_for_test(artifact)
    gateway.remove_baseline_claim_for_test()
    assert gateway.has_baseline_claim_for_test() is False

    with pytest.raises(GatewayConsistencyViolation):
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id=None,
            gateway=gateway,
            contract_checker=RaisingContractChecker(),
        )

    stored = gateway.get_monitoring_run("churn_model", monitoring_run_id)
    assert stored is not None
    assert stored.lifecycle_status is LifecycleStatus.PREPARED
    assert gateway.finalized_results == []
    assert gateway.has_baseline_claim_for_test() is False
    assert gateway.baseline_reconciliations == []


def test_checked_replay_rejects_legacy_missing_contract_check_without_mutation() -> None:
    gateway = InterruptCheckCommitGateway(GatewayConfig())
    monitoring_run_id = complete_interrupted_check(gateway)
    gateway.delete_contract_check_for_test()
    stored_before = gateway.get_monitoring_run("churn_model", monitoring_run_id)

    with pytest.raises(GatewayConsistencyViolation):
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id=None,
            gateway=gateway,
            contract_checker=RaisingContractChecker(),
        )

    assert gateway.get_monitoring_run("churn_model", monitoring_run_id) == stored_before
    assert read_contract_check(gateway, monitoring_run_id) is None
    assert gateway.finalized_results == []


def test_checked_replay_rejects_malformed_contract_check_without_mutation() -> None:
    gateway = InterruptCheckCommitGateway(GatewayConfig())
    monitoring_run_id = complete_interrupted_check(gateway)
    malformed = {"artifact_schema_version": "v0"}
    gateway.replace_contract_check_for_test(malformed)
    stored_before = gateway.get_monitoring_run("churn_model", monitoring_run_id)

    with pytest.raises(GatewayConsistencyViolation):
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id=None,
            gateway=gateway,
            contract_checker=RaisingContractChecker(),
        )

    assert gateway.get_monitoring_run("churn_model", monitoring_run_id) == stored_before
    assert read_contract_check(gateway, monitoring_run_id) == malformed
    assert gateway.finalized_results == []


def test_checked_replay_rejects_projection_disagreement_without_mutation() -> None:
    gateway = InterruptCheckCommitGateway(GatewayConfig())
    monitoring_run_id = complete_interrupted_check(gateway)
    artifact_before = read_contract_check(gateway, monitoring_run_id)
    gateway.replace_comparability_projection_for_test(ComparabilityStatus.FAIL)
    stored_before = gateway.get_monitoring_run("churn_model", monitoring_run_id)

    with pytest.raises(GatewayConsistencyViolation):
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id=None,
            gateway=gateway,
            contract_checker=RaisingContractChecker(),
        )

    assert gateway.get_monitoring_run("churn_model", monitoring_run_id) == stored_before
    assert read_contract_check(gateway, monitoring_run_id) == artifact_before
    assert gateway.finalized_results == []


@pytest.mark.parametrize(
    "lifecycle_status",
    [LifecycleStatus.ANALYZED, LifecycleStatus.CLOSED],
)
def test_post_check_replay_fails_closed_without_lifecycle_regression(
    lifecycle_status: LifecycleStatus,
) -> None:
    gateway = InterruptCheckCommitGateway(GatewayConfig())
    monitoring_run_id = complete_interrupted_check(gateway)
    stored_checked = gateway.get_monitoring_run("churn_model", monitoring_run_id)
    assert stored_checked is not None
    gateway.upsert_monitoring_run(
        subject_id="churn_model",
        monitoring_run_id=monitoring_run_id,
        source_run_id=stored_checked.source_run_id,
        lifecycle_status=lifecycle_status,
        sequence_index=stored_checked.sequence_index,
    )
    stored_before = gateway.get_monitoring_run("churn_model", monitoring_run_id)
    artifact_before = read_contract_check(gateway, monitoring_run_id)

    with pytest.raises(GatewayConsistencyViolation) as exc_info:
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id=None,
            gateway=gateway,
            contract_checker=RaisingContractChecker(),
        )

    assert exc_info.value.code == "monitoring_run_upsert_field_override"
    assert exc_info.value.details == (
        ("lifecycle_status", "checked"),
        ("persisted_lifecycle_status", lifecycle_status.value),
    )
    assert gateway.get_monitoring_run("churn_model", monitoring_run_id) == stored_before
    assert read_contract_check(gateway, monitoring_run_id) == artifact_before
    assert gateway.finalized_results == []


@pytest.mark.parametrize(
    "lifecycle_status",
    [LifecycleStatus.ANALYZED, LifecycleStatus.CLOSED],
)
def test_checked_replay_rechecks_post_check_lifecycle_before_check_execution(
    lifecycle_status: LifecycleStatus,
) -> None:
    gateway = InterruptCheckCommitGateway(GatewayConfig())
    monitoring_run_id = complete_interrupted_check(gateway)
    artifact_before = read_contract_check(gateway, monitoring_run_id)
    baseline_reconciliations_before = list(gateway.baseline_reconciliations)
    persistence_events_before = list(gateway.persistence_events)
    gateway.advance_lifecycle_after_allocation_for_test(lifecycle_status)

    with pytest.raises(GatewayConsistencyViolation) as exc_info:
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id=None,
            gateway=gateway,
            contract_checker=RaisingContractChecker(),
        )

    assert exc_info.value.code == "monitoring_run_upsert_field_override"
    assert exc_info.value.details == (
        ("lifecycle_status", "checked"),
        ("persisted_lifecycle_status", lifecycle_status.value),
    )
    stored = gateway.get_monitoring_run("churn_model", monitoring_run_id)
    assert stored is not None
    assert stored.lifecycle_status is lifecycle_status
    assert read_contract_check(gateway, monitoring_run_id) == artifact_before
    assert gateway.finalized_results == []
    assert gateway.baseline_reconciliations == baseline_reconciliations_before
    assert gateway.persistence_events == persistence_events_before


def test_checked_replay_rechecks_failed_lifecycle_before_check_execution() -> None:
    gateway = InterruptCheckCommitGateway(GatewayConfig())
    monitoring_run_id = complete_interrupted_check(gateway)
    artifact_before = read_contract_check(gateway, monitoring_run_id)
    baseline_reconciliations_before = list(gateway.baseline_reconciliations)
    gateway.advance_lifecycle_after_allocation_for_test(LifecycleStatus.FAILED)

    replay = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=RaisingContractChecker(),
    )

    stored = gateway.get_monitoring_run("churn_model", monitoring_run_id)
    assert stored is not None
    assert stored.lifecycle_status is LifecycleStatus.FAILED
    assert replay.lifecycle_status is LifecycleStatus.FAILED
    assert read_contract_check(gateway, monitoring_run_id) == artifact_before
    assert gateway.finalized_results == [replay]
    assert gateway.baseline_reconciliations == baseline_reconciliations_before


def test_prepared_replay_hydrates_context_without_prepare_resolution() -> None:
    gateway = PreparedReplayGateway(GatewayConfig())
    add_default_source_runs(gateway)
    monitoring_run_id = leave_monitoring_run_prepared(gateway)
    gateway.block_prepare_resolution = True

    replay = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    assert replay.monitoring_run_id == monitoring_run_id
    assert replay.lifecycle_status is LifecycleStatus.CHECKED


def test_prepared_replay_materializes_missing_baseline_claim() -> None:
    gateway = PreparedReplayGateway(GatewayConfig())
    add_default_source_runs(gateway)
    monitoring_run_id = leave_monitoring_run_prepared(gateway)
    gateway.simulate_projection_only_prepared_run(monitoring_run_id)
    gateway.block_prepare_resolution = True

    replay = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    assert replay.lifecycle_status is LifecycleStatus.CHECKED
    assert gateway.baseline_reconciliations == [(monitoring_run_id, "train-run-baseline")]


def test_prepared_replay_repairs_baseline_before_owned_check_failure() -> None:
    gateway = InterruptCheckCommitGateway(GatewayConfig())
    add_default_source_runs(gateway)

    with pytest.raises(RuntimeError, match="simulated interruption before checked commit"):
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id="train-run-baseline",
            gateway=gateway,
            contract_checker=DefaultContractChecker(),
        )

    assert gateway.last_allocation is not None
    monitoring_run_id = gateway.last_allocation.monitoring_run_id
    gateway.delete_contract_check_for_test()
    gateway.remove_baseline_claim_for_test()

    replay = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=InvalidResultContractChecker(),
    )

    stored = gateway.get_monitoring_run("churn_model", monitoring_run_id)
    assert stored is not None
    assert stored.lifecycle_status is LifecycleStatus.FAILED
    assert replay.lifecycle_status is LifecycleStatus.FAILED
    assert replay.error is not None
    assert replay.error.code == "check_result_invalid"
    assert read_contract_check(gateway, monitoring_run_id) is None
    assert gateway.baseline_reconciliations == [(monitoring_run_id, "train-run-baseline")]
    assert gateway.finalized_results == [replay]
    assert gateway.persistence_events == [
        ("baseline_reconciliation", monitoring_run_id),
        ("lifecycle_status", LifecycleStatus.FAILED),
    ]


def test_prepared_replay_rejects_missing_prepared_context() -> None:
    gateway = PreparedReplayGateway(GatewayConfig())
    add_default_source_runs(gateway)
    leave_monitoring_run_prepared(gateway)
    gateway.prepared_context_override = None

    with pytest.raises(GatewayConsistencyViolation):
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id=None,
            gateway=gateway,
            contract_checker=DefaultContractChecker(),
        )


def test_prepared_replay_rejects_malformed_prepared_context() -> None:
    gateway = PreparedReplayGateway(GatewayConfig())
    add_default_source_runs(gateway)
    leave_monitoring_run_prepared(gateway)
    gateway.prepared_context_override = {"artifact_schema_version": "v0"}

    with pytest.raises(GatewayConsistencyViolation) as exc_info:
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id=None,
            gateway=gateway,
            contract_checker=DefaultContractChecker(),
        )

    assert exc_info.value.code == _PREPARED_CONTEXT_INCONSISTENT_CODE_FOR_TESTS


def test_prepared_replay_rejects_conflicting_persisted_identity() -> None:
    gateway = PreparedReplayGateway(GatewayConfig())
    add_default_source_runs(gateway)
    leave_monitoring_run_prepared(gateway)
    gateway.corrupt_timeline_id = True

    with pytest.raises(GatewayConsistencyViolation) as exc_info:
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id=None,
            gateway=gateway,
            contract_checker=DefaultContractChecker(),
        )

    assert exc_info.value.code == _PREPARED_CONTEXT_INCONSISTENT_CODE_FOR_TESTS


def test_prepared_replay_rejects_changed_contract_definition_for_same_identity() -> None:
    gateway = PreparedReplayGateway(GatewayConfig())
    add_default_source_runs(gateway)
    monitoring_run_id = leave_monitoring_run_prepared(gateway)

    prepared_context = read_prepared_context(gateway, monitoring_run_id)
    assert prepared_context is not None
    corrupted_context = dict(prepared_context)
    persisted_contract = corrupted_context["contract"]
    assert isinstance(persisted_contract, dict)
    corrupted_contract = dict(persisted_contract)
    corrupted_contract["execution_contract_ref"] = None
    corrupted_context["contract"] = corrupted_contract
    gateway.prepared_context_override = corrupted_context

    with pytest.raises(GatewayConsistencyViolation) as exc_info:
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id=None,
            gateway=gateway,
            contract_checker=DefaultContractChecker(),
        )

    assert exc_info.value.code == _PREPARED_CONTEXT_INCONSISTENT_CODE_FOR_TESTS


def test_prepared_replay_rejects_different_effective_plan_for_same_recipe_identity() -> None:
    gateway = PreparedReplayGateway(GatewayConfig())
    add_default_source_runs(gateway)
    first_recipe = make_compiled_recipe(metric_names=[])
    second_recipe = make_compiled_recipe(metric_names=["f1"])
    leave_monitoring_run_prepared(gateway, recipe=first_recipe)

    with pytest.raises(GatewayConsistencyViolation) as exc_info:
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id=None,
            recipe=second_recipe,
            gateway=gateway,
            contract_checker=DefaultContractChecker(),
        )

    assert exc_info.value.code == _PREPARED_CONTEXT_INCONSISTENT_CODE_FOR_TESTS


def test_run_orchestration_upserts_created_when_allocation_exists_without_run_record() -> None:
    gateway = AllocationOnlyReplayGateway(make_gateway().config)
    for source_run_id, metrics in (
        ("train-run-baseline", {"f1": 0.87}),
        ("train-run-current", {"f1": 0.91}),
    ):
        gateway.add_source_run(
            subject_id="churn_model",
            source_run_id=source_run_id,
            source_experiment=None,
            metrics=metrics,
            artifacts=("metrics.json",),
            environment={"python": "3.12"},
            features=("age",),
            schema={"age": "int"},
            data_scope="validation:2026-03-01",
        )

    result = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    stored = gateway.get_monitoring_run("churn_model", result.monitoring_run_id)

    assert result.lifecycle_status is LifecycleStatus.CHECKED
    assert stored is not None
    assert stored.sequence_index == 0
    assert stored.lifecycle_status is LifecycleStatus.CHECKED


def test_run_orchestration_later_run_can_omit_baseline_source_run_id() -> None:
    gateway = make_gateway()
    first = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-next",
        source_experiment=None,
        metrics={"f1": 0.92},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age",),
        schema={"age": "int"},
        data_scope="validation:2026-03-02",
    )

    second = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-next",
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    assert first.lifecycle_status is LifecycleStatus.CHECKED
    assert second.lifecycle_status is LifecycleStatus.CHECKED
    assert second.references == (
        MonitoringRunReference(
            kind=DiffReferenceKind.BASELINE,
            monitoring_run_id=None,
            source_run_id="train-run-baseline",
        ),
    )


def test_run_orchestration_finalizes_newly_checked_run_once() -> None:
    gateway = FinalizingGateway(GatewayConfig())
    for source_run_id, metrics in (
        ("train-run-baseline", {"f1": 0.87}),
        ("train-run-current", {"f1": 0.91}),
    ):
        gateway.add_source_run(
            subject_id="churn_model",
            source_run_id=source_run_id,
            source_experiment=None,
            metrics=metrics,
            artifacts=("metrics.json",),
            environment={"python": "3.12"},
            features=("age",),
            schema={"age": "int"},
            data_scope="validation:2026-03-01",
        )

    result = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    assert [finalized.monitoring_run_id for finalized in gateway.finalized_results] == [
        result.monitoring_run_id
    ]
    assert gateway.finalized_results[0].lifecycle_status is LifecycleStatus.CHECKED


def test_run_orchestration_checked_rerun_finalizes_for_repair() -> None:
    gateway = FinalizingGateway(GatewayConfig())
    for source_run_id, metrics in (
        ("train-run-baseline", {"f1": 0.87}),
        ("train-run-current", {"f1": 0.91}),
    ):
        gateway.add_source_run(
            subject_id="churn_model",
            source_run_id=source_run_id,
            source_experiment=None,
            metrics=metrics,
            artifacts=("metrics.json",),
            environment={"python": "3.12"},
            features=("age",),
            schema={"age": "int"},
            data_scope="validation:2026-03-01",
        )

    first = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    second = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    assert second.monitoring_run_id == first.monitoring_run_id
    assert [finalized.monitoring_run_id for finalized in gateway.finalized_results] == [
        first.monitoring_run_id,
        first.monitoring_run_id,
    ]
    assert gateway.finalized_results[1].lifecycle_status is LifecycleStatus.CHECKED


def test_run_orchestration_finalizes_owned_failure_once() -> None:
    gateway = FinalizingGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-baseline",
        source_experiment=None,
        metrics={"f1": 0.87},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age",),
        schema={"age": "int"},
        data_scope="validation:2026-03-01",
    )

    result = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-missing",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    assert result.lifecycle_status is LifecycleStatus.FAILED
    assert [finalized.monitoring_run_id for finalized in gateway.finalized_results] == [
        result.monitoring_run_id
    ]
    assert gateway.finalized_results[0].lifecycle_status is LifecycleStatus.FAILED


def test_run_orchestration_failed_rerun_finalizes_for_repair() -> None:
    gateway = FinalizingGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-baseline",
        source_experiment=None,
        metrics={"f1": 0.87},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age",),
        schema={"age": "int"},
        data_scope="validation:2026-03-01",
    )

    first = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-missing",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    second = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-missing",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    assert second.monitoring_run_id == first.monitoring_run_id
    assert second.lifecycle_status is LifecycleStatus.FAILED
    assert second.error is not None
    assert second.error.code == "idempotent_run_retry_failed_terminal"
    assert [finalized.monitoring_run_id for finalized in gateway.finalized_results] == [
        first.monitoring_run_id,
        first.monitoring_run_id,
    ]
    assert gateway.finalized_results[1].lifecycle_status is LifecycleStatus.FAILED


def test_run_orchestration_non_comparable_check_still_returns_checked_result() -> None:
    gateway = make_gateway()
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-current",
        source_experiment=None,
        metrics={"f1": 0.91},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age",),
        schema={"age": "string"},
        data_scope="validation:2026-03-01",
    )

    result = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    stored = gateway.get_monitoring_run("churn_model", result.monitoring_run_id)

    assert result.lifecycle_status is LifecycleStatus.CHECKED
    assert result.comparability_status is ComparabilityStatus.FAIL
    assert result.error is None
    assert stored is not None
    assert stored.lifecycle_status is LifecycleStatus.CHECKED
    assert stored.comparability_status is ComparabilityStatus.FAIL


def test_run_orchestration_prepare_error_persists_failed_and_returns_runtime_error() -> None:
    gateway = make_gateway()

    result = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-missing",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    stored = gateway.get_monitoring_run("churn_model", result.monitoring_run_id)

    assert result.lifecycle_status is LifecycleStatus.FAILED
    assert result.comparability_status is None
    assert result.error is not None
    assert result.error.stage == "prepare"
    assert result.error.code == "prepare_source_run_not_found"
    assert stored is not None
    assert stored.lifecycle_status is LifecycleStatus.FAILED
    assert stored.contract_check_result is None


def test_run_orchestration_failed_prepare_rerun_short_circuits_terminal_state() -> None:
    gateway = make_gateway()
    first = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-missing",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    second = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-missing",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    stored = gateway.get_monitoring_run("churn_model", first.monitoring_run_id)

    assert first.lifecycle_status is LifecycleStatus.FAILED
    assert second.monitoring_run_id == first.monitoring_run_id
    assert second.lifecycle_status is LifecycleStatus.FAILED
    assert second.comparability_status is None
    assert second.error is not None
    assert second.error.stage == "prepare"
    assert second.error.code == "idempotent_run_retry_failed_terminal"
    assert stored is not None
    assert stored.sequence_index == 0
    assert stored.lifecycle_status is LifecycleStatus.FAILED
    assert stored.contract_check_result is None


def test_run_orchestration_prebootstrap_failure_preserves_allocation_sequence() -> None:
    gateway = make_gateway()
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-second",
        source_experiment=None,
        metrics={"f1": 0.92},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age",),
        schema={"age": "int"},
        data_scope="validation:2026-03-01",
    )

    failed_result = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    failed_monitoring_run = gateway.get_monitoring_run(
        "churn_model", failed_result.monitoring_run_id
    )
    uninitialized_timeline_state = gateway.get_timeline_state("churn_model")

    successful_result = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-second",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    successful_monitoring_run = gateway.get_monitoring_run(
        "churn_model", successful_result.monitoring_run_id
    )

    assert failed_result.lifecycle_status is LifecycleStatus.FAILED
    assert failed_result.timeline_id == "timeline-churn_model"
    assert failed_result.error is not None
    assert failed_result.error.stage == "prepare"
    assert failed_monitoring_run is not None
    assert failed_monitoring_run.sequence_index == 0
    assert failed_monitoring_run.lifecycle_status is LifecycleStatus.FAILED
    assert uninitialized_timeline_state is not None
    assert uninitialized_timeline_state.timeline_id == failed_result.timeline_id
    assert uninitialized_timeline_state.baseline_source_run_id is None
    assert successful_result.lifecycle_status is LifecycleStatus.CHECKED
    assert successful_result.timeline_id == failed_result.timeline_id
    assert successful_monitoring_run is not None
    assert successful_monitoring_run.sequence_index == 1


def test_run_orchestration_preserves_allocation_timeline_id_when_timeline_read_fails() -> None:
    """A failed allocated result should use identity returned by allocation."""
    gateway = UnreadableAllocatedTimelineGateway(GatewayConfig())

    result = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    assert result.lifecycle_status is LifecycleStatus.FAILED
    assert result.timeline_id == gateway.allocation_timeline_id == "timeline-churn_model"
    assert result.error is not None
    assert result.error.code == "prepare_missing_timeline"


def test_run_orchestration_check_error_persists_failed_and_returns_runtime_error() -> None:
    gateway = make_gateway()

    result = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=InvalidResultContractChecker(),
    )

    stored = gateway.get_monitoring_run("churn_model", result.monitoring_run_id)

    assert result.lifecycle_status is LifecycleStatus.FAILED
    assert result.comparability_status is None
    assert result.error is not None
    assert result.error.stage == "check"
    assert result.error.code == "check_result_invalid"
    assert stored is not None
    assert stored.lifecycle_status is LifecycleStatus.FAILED
    assert stored.contract_check_result is None


def test_run_orchestration_failed_check_rerun_short_circuits_terminal_state() -> None:
    gateway = make_gateway()
    first = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=InvalidResultContractChecker(),
    )
    second = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    stored = gateway.get_monitoring_run("churn_model", first.monitoring_run_id)

    assert first.lifecycle_status is LifecycleStatus.FAILED
    assert second.monitoring_run_id == first.monitoring_run_id
    assert second.lifecycle_status is LifecycleStatus.FAILED
    assert second.comparability_status is None
    assert second.error is not None
    assert second.error.stage == "prepare"
    assert second.error.code == "idempotent_run_retry_failed_terminal"
    assert stored is not None
    assert stored.sequence_index == 0
    assert stored.lifecycle_status is LifecycleStatus.FAILED
    assert stored.contract_check_result is None


def test_run_orchestration_raises_unexpected_checker_errors() -> None:
    gateway = make_gateway()

    with pytest.raises(RuntimeError, match="checker exploded"):
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id="train-run-baseline",
            gateway=gateway,
            contract_checker=RaisingContractChecker(),
        )


def test_run_orchestration_raises_internal_gateway_errors_instead_of_normalizing() -> None:
    gateway = BrokenUpsertGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-baseline",
        source_experiment=None,
        metrics={"f1": 0.87},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age",),
        schema={"age": "int"},
        data_scope="validation:2026-03-01",
    )
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-current",
        source_experiment=None,
        metrics={"f1": 0.91},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age",),
        schema={"age": "int"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(
        GatewayConsistencyViolation,
        match="Attempted to override immutable Monitoring Run field",
    ):
        run_orchestration(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id="train-run-baseline",
            gateway=gateway,
            contract_checker=DefaultContractChecker(),
        )


def test_run_orchestration_reuses_idempotent_run_without_overwriting_check_output() -> None:
    gateway = make_gateway()
    first = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    second = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    stored = gateway.get_monitoring_run("churn_model", first.monitoring_run_id)

    assert second.monitoring_run_id == first.monitoring_run_id
    assert stored is not None
    assert stored.sequence_index == 0
    assert stored.lifecycle_status is LifecycleStatus.CHECKED
    assert stored.contract_check_result is not None
    assert second.lifecycle_status is LifecycleStatus.CHECKED
    assert second.error is None


def test_run_orchestration_checked_rerun_omitting_baseline_replays_result() -> None:
    gateway = make_gateway()
    first = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    second = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    assert second.monitoring_run_id == first.monitoring_run_id
    assert second.lifecycle_status is LifecycleStatus.CHECKED
    assert second.comparability_status is ComparabilityStatus.PASS
    assert second.references == (
        MonitoringRunReference(
            kind=DiffReferenceKind.BASELINE,
            monitoring_run_id=None,
            source_run_id="train-run-baseline",
        ),
    )
    assert second.error is None


def test_checked_replay_uses_committed_context_and_output_without_live_reads() -> None:
    gateway = ReplaySensitiveGateway(GatewayConfig())
    add_default_source_runs(gateway)
    first = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    gateway.block_timeline_reads = True
    gateway.block_source_resolution = True
    gateway.block_baseline_evidence = True
    gateway.block_current_evidence = True

    replay = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=RaisingContractChecker(),
    )

    assert replay.monitoring_run_id == first.monitoring_run_id
    assert replay.lifecycle_status is LifecycleStatus.CHECKED
    assert replay.comparability_status is ComparabilityStatus.PASS


def test_run_orchestration_checked_rerun_replays_when_source_run_no_longer_resolves() -> None:
    gateway = ReplaySensitiveGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-baseline",
        source_experiment=None,
        metrics={"f1": 0.87},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age",),
        schema={"age": "int"},
        data_scope="validation:2026-03-01",
    )
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-current",
        source_experiment=None,
        metrics={"f1": 0.91},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age",),
        schema={"age": "int"},
        data_scope="validation:2026-03-01",
    )
    first = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    gateway.block_source_resolution = True

    second = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    assert second.monitoring_run_id == first.monitoring_run_id
    assert second.lifecycle_status is LifecycleStatus.CHECKED
    assert second.comparability_status is ComparabilityStatus.PASS
    assert second.error is None


def test_run_orchestration_checked_rerun_replays_when_baseline_evidence_no_longer_resolves() -> (
    None
):
    gateway = ReplaySensitiveGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-baseline",
        source_experiment=None,
        metrics={"f1": 0.87},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age",),
        schema={"age": "int"},
        data_scope="validation:2026-03-01",
    )
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-current",
        source_experiment=None,
        metrics={"f1": 0.91},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age",),
        schema={"age": "int"},
        data_scope="validation:2026-03-01",
    )
    first = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    gateway.block_baseline_evidence = True
    gateway.block_current_evidence = True

    second = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    assert second.monitoring_run_id == first.monitoring_run_id
    assert second.lifecycle_status is LifecycleStatus.CHECKED
    assert second.comparability_status is ComparabilityStatus.PASS
    assert second.error is None


def test_run_orchestration_rejects_baseline_override_on_checked_idempotent_rerun() -> None:
    gateway = make_gateway()
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-other",
        source_experiment=None,
        metrics={"f1": 0.88},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age",),
        schema={"age": "int"},
        data_scope="validation:2026-03-01",
    )

    first = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    second = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-other",
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    stored = gateway.get_monitoring_run("churn_model", first.monitoring_run_id)

    assert second.monitoring_run_id == first.monitoring_run_id
    assert second.lifecycle_status is LifecycleStatus.FAILED
    assert second.comparability_status is None
    assert second.error is not None
    assert second.error.stage == "prepare"
    assert second.error.code == PREPARE_BASELINE_OVERRIDE_EXISTING_BASELINE
    assert stored is not None
    assert stored.lifecycle_status is LifecycleStatus.CHECKED
    assert stored.contract_check_result is not None
    assert stored.contract_check_result.status is ComparabilityStatus.PASS


def test_public_run_is_a_thin_facade(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = object()
    captured: dict[str, object] = {}
    gateway = InMemoryMonitoringGateway(GatewayConfig())

    def fakerun_orchestrationing(**kwargs: object) -> object:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(monitor, "run_orchestration", fakerun_orchestrationing)

    result = monitor.run(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        gateway=gateway,
    )

    assert result is expected
    assert captured["subject_id"] == "churn_model"
    assert captured["source_run_id"] == "train-run-current"
    assert captured["baseline_source_run_id"] == "train-run-baseline"
    assert captured["gateway"] is gateway
    assert "monitoring_run_id_factory" not in captured


def test_public_run_forwards_compiled_recipe_and_custom_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = object()
    captured: dict[str, object] = {}
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    compiled_recipe = make_compiled_recipe()

    def fake_run_orchestration(**kwargs: object) -> object:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(monitor, "run_orchestration", fake_run_orchestration)

    result = monitor.run(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        custom_reference_monitoring_run_id="monitoring-run-reference",
        recipe=compiled_recipe,
        gateway=gateway,
    )

    assert result is expected
    assert captured["recipe"] is compiled_recipe
    assert captured["custom_reference_monitoring_run_id"] == "monitoring-run-reference"


def test_public_run_explicit_none_uses_system_default_recipe_identity() -> None:
    gateway = RecordingAllocationGateway(GatewayConfig())
    add_default_source_runs(gateway)

    result = monitor.run(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        recipe=None,
        gateway=gateway,
    )

    assert result.lifecycle_status is LifecycleStatus.CHECKED
    assert gateway.allocation_keys == [
        IdempotencyKey(
            subject_id="churn_model",
            source_run_id="train-run-current",
            recipe_id=SYSTEM_DEFAULT_RECIPE_ID,
            recipe_version=SYSTEM_DEFAULT_RECIPE_VERSION,
        )
    ]


def test_public_run_uses_supplied_compiled_recipe_identity_for_idempotency() -> None:
    gateway = RecordingAllocationGateway(GatewayConfig())
    add_default_source_runs(gateway)
    compiled_recipe = make_compiled_recipe(recipe_id="custom-churn", recipe_version="7")

    result = monitor.run(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        recipe=compiled_recipe,
        gateway=gateway,
    )

    assert result.lifecycle_status is LifecycleStatus.CHECKED
    assert gateway.allocation_keys == [
        IdempotencyKey(
            subject_id="churn_model",
            source_run_id="train-run-current",
            recipe_id="custom-churn",
            recipe_version="7",
        )
    ]


@pytest.mark.parametrize(
    "recipe_input",
    [
        pytest.param(build_system_default_recipe(), id="raw-mapping"),
        pytest.param(Path("recipe.json"), id="path"),
        pytest.param("recipe.json", id="string-path"),
        pytest.param(ComponentRegistry(), id="component-registry"),
    ],
)
def test_public_run_rejects_uncompiled_recipe_without_allocating(recipe_input: object) -> None:
    gateway = RecordingAllocationGateway(GatewayConfig())

    with pytest.raises(TypeError) as exc_info:
        monitor.run(
            subject_id="churn_model",
            source_run_id="train-run-current",
            baseline_source_run_id="train-run-baseline",
            # delibrately passing wrong type for test
            recipe=recipe_input,  # pyright: ignore[reportArgumentType]
            gateway=gateway,
        )

    assert "unexpected keyword argument" not in str(exc_info.value)
    assert gateway.allocation_keys == []


def test_public_run_adds_invocation_owned_custom_reference() -> None:
    gateway = make_gateway()
    compiled_recipe = make_compiled_recipe()
    reference_allocation = gateway.create_or_reuse_monitoring_run(
        IdempotencyKey(
            subject_id="churn_model",
            source_run_id="train-run-reference",
            recipe_id=compiled_recipe.identity.recipe_id,
            recipe_version=compiled_recipe.identity.recipe_version,
        )
    )
    gateway.upsert_monitoring_run(
        subject_id="churn_model",
        monitoring_run_id=reference_allocation.monitoring_run_id,
        source_run_id="train-run-reference",
        lifecycle_status=LifecycleStatus.CLOSED,
        sequence_index=reference_allocation.sequence_index,
    )

    result = monitor.run(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        custom_reference_monitoring_run_id=reference_allocation.monitoring_run_id,
        recipe=compiled_recipe,
        gateway=gateway,
    )

    assert result.lifecycle_status is LifecycleStatus.CHECKED
    assert result.references == (
        MonitoringRunReference(
            kind=DiffReferenceKind.BASELINE,
            monitoring_run_id=None,
            source_run_id="train-run-baseline",
        ),
        MonitoringRunReference(
            kind=DiffReferenceKind.PREVIOUS,
            monitoring_run_id=reference_allocation.monitoring_run_id,
            source_run_id="train-run-reference",
        ),
        MonitoringRunReference(
            kind=DiffReferenceKind.CUSTOM,
            monitoring_run_id=reference_allocation.monitoring_run_id,
            source_run_id="train-run-reference",
        ),
    )


def test_public_run_does_not_globally_compare_effective_plans_for_same_recipe_identity() -> None:
    gateway = make_gateway()
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-next",
        source_experiment=None,
        metrics={"f1": 0.92},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age",),
        schema={"age": "int"},
        data_scope="validation:2026-03-02",
    )
    select_no_metrics = make_compiled_recipe(metric_names=[])
    select_f1 = make_compiled_recipe(metric_names=["f1"])

    first = monitor.run(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        recipe=select_no_metrics,
        gateway=gateway,
    )
    second = monitor.run(
        subject_id="churn_model",
        source_run_id="train-run-next",
        recipe=select_f1,
        gateway=gateway,
    )

    assert first.lifecycle_status is LifecycleStatus.CHECKED
    assert second.lifecycle_status is LifecycleStatus.CHECKED
    assert second.monitoring_run_id != first.monitoring_run_id


def test_public_run_defaults_to_mlflow_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = object()
    captured: dict[str, object] = {}
    constructed_gateways: list[object] = []

    def fake_mlflow_gateway(config: GatewayConfig) -> object:
        constructed_gateways.append(config)
        return expected

    def fake_run_orchestration(**kwargs: object) -> object:
        captured.update(kwargs)
        return "result"

    monkeypatch.setattr(monitor, "MLflowMonitoringGateway", fake_mlflow_gateway)
    monkeypatch.setattr(monitor, "run_orchestration", fake_run_orchestration)

    result = monitor.run(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
    )

    assert result == "result"
    assert len(constructed_gateways) == 1
    assert isinstance(constructed_gateways[0], GatewayConfig)
    assert captured["gateway"] is expected


def test_public_run_constructs_fresh_mlflow_gateway_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration_gateways: list[object] = []

    def fake_mlflow_gateway(config: GatewayConfig) -> object:
        assert isinstance(config, GatewayConfig)
        return object()

    def fake_run_orchestration(**kwargs: object) -> object:
        orchestration_gateways.append(kwargs["gateway"])
        return object()

    monkeypatch.setattr(monitor, "MLflowMonitoringGateway", fake_mlflow_gateway)
    monkeypatch.setattr(monitor, "run_orchestration", fake_run_orchestration)

    monitor.run(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
    )
    monitor.run(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
    )

    assert len(orchestration_gateways) == 2
    assert orchestration_gateways[0] is not orchestration_gateways[1]
