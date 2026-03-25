"""Unit tests for W-003 orchestration and public monitor entrypoint."""

from __future__ import annotations

import pytest

from mlflow_monitor import monitor
from mlflow_monitor.contract_checker import DefaultContractChecker
from mlflow_monitor.domain import (
    ComparabilityStatus,
    ContractCheckReason,
    ContractCheckResult,
    LifecycleStatus,
    MonitoringRunReference,
)
from mlflow_monitor.errors import GatewayConsistencyViolation
from mlflow_monitor.gateway import (
    CreateOrReuseMonitoringRunResult,
    GatewayConfig,
    IdempotencyKey,
    InMemoryMonitoringGateway,
)
from mlflow_monitor.orchestration import run_orchestration
from mlflow_monitor.result_contract import MonitorRunResult


def make_gateway() -> InMemoryMonitoringGateway:
    """Build a gateway with baseline and current source runs."""
    gateway = InMemoryMonitoringGateway(GatewayConfig())
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
    return gateway


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
        lifecycle_status: LifecycleStatus,
        sequence_index: int,
        contract_check_result: ContractCheckResult | None = None,
        references: tuple[MonitoringRunReference, ...] | None = None,
    ) -> None:
        if lifecycle_status is LifecycleStatus.PREPARED:
            raise GatewayConsistencyViolation(
                code="monitoring_run_upsert_field_override",
                message="prepared upsert broke gateway consistency",
            )
        super().upsert_monitoring_run(
            subject_id=subject_id,
            monitoring_run_id=monitoring_run_id,
            lifecycle_status=lifecycle_status,
            sequence_index=sequence_index,
            contract_check_result=contract_check_result,
            references=references,
        )


class ReplaySensitiveGateway(InMemoryMonitoringGateway):
    """Fail if checked reruns re-enter prepare-sensitive source-run reads."""

    def __init__(self, config: GatewayConfig) -> None:
        super().__init__(config)
        self.block_source_resolution = False
        self.block_baseline_evidence = False
        self.block_current_evidence = False

    def resolve_source_run_id(
        self,
        subject_id: str,
        source_experiment: str | None,
        run_selector: str,
        runtime_source_run_id: str | None = None,
    ) -> str | None:
        if self.block_source_resolution:
            return None
        return super().resolve_source_run_id(
            subject_id=subject_id,
            source_experiment=source_experiment,
            run_selector=run_selector,
            runtime_source_run_id=runtime_source_run_id,
        )

    def get_source_run_contract_evidence(self, source_run_id: str):
        if source_run_id == "train-run-baseline" and self.block_baseline_evidence:
            return None
        if source_run_id == "train-run-current" and self.block_current_evidence:
            return None
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
            sequence_index=replayed.sequence_index,
            existing_monitoring_run=None,
            allocated=False,
        )


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
        MonitoringRunReference(kind="baseline", reference_run_id="train-run-baseline"),
    )
    assert result.finding_ids == ()
    assert result.diff_ids == ()
    assert result.summary is None
    assert result.error is None
    assert stored is not None
    assert stored.sequence_index == 0
    assert stored.lifecycle_status is LifecycleStatus.CHECKED
    assert stored.comparability_status is ComparabilityStatus.PASS
    assert stored.contract_check_result is not None
    assert stored.contract_check_result.status is ComparabilityStatus.PASS


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
            kind="baseline", reference_run_id="train-run-baseline"
        ),
        MonitoringRunReference(
            kind="previous", reference_run_id=first.monitoring_run_id
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


def test_run_orchestration_checked_rerun_does_not_finalize_again() -> None:
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
        first.monitoring_run_id
    ]


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


def test_run_orchestration_bootstrap_failure_returns_failed_result_without_timeline() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
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

    result = run_orchestration(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    stored = gateway.get_monitoring_run("churn_model", result.monitoring_run_id)

    assert result.lifecycle_status is LifecycleStatus.FAILED
    assert result.timeline_id is None
    assert result.error is not None
    assert result.error.stage == "prepare"
    assert stored is not None
    assert stored.lifecycle_status is LifecycleStatus.FAILED


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
        match="prepared upsert broke gateway consistency",
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
        MonitoringRunReference(kind="baseline", reference_run_id="train-run-baseline"),
    )
    assert second.error is None


def test_run_orchestration_checked_rerun_preserves_references() -> None:
    gateway = make_gateway()
    gateway.set_active_lkg_monitoring_run_id("churn_model", "monitoring-run-lkg")

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

    assert first.references == (
        MonitoringRunReference(kind="baseline", reference_run_id="train-run-baseline"),
        MonitoringRunReference(kind="lkg", reference_run_id="monitoring-run-lkg"),
    )
    assert second.monitoring_run_id == first.monitoring_run_id
    assert second.references == first.references
    assert second.error is None


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
    assert second.error.code == "prepare_baseline_override_existing_timeline"
    assert stored is not None
    assert stored.lifecycle_status is LifecycleStatus.CHECKED
    assert stored.contract_check_result is not None
    assert stored.contract_check_result.status is ComparabilityStatus.PASS


def test_public_run_is_a_thin_facade(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = object()
    captured: dict[str, object] = {}

    def fakerun_orchestrationing(**kwargs: object) -> object:
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(monitor, "run_orchestration", fakerun_orchestrationing)

    result = monitor.run(
        subject_id="churn_model",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
    )

    assert result is expected
    assert captured["subject_id"] == "churn_model"
    assert captured["source_run_id"] == "train-run-current"
    assert captured["baseline_source_run_id"] == "train-run-baseline"
    assert "monitoring_run_id_factory" not in captured
