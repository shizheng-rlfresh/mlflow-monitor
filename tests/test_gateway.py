"""Unit tests for gateway module behavior in MLflow-Monitor v0."""

import pytest

from mlflow_monitor.domain import LifecycleStatus
from mlflow_monitor.errors import GatewayNamespaceViolation, TrainingRunMutationViolation
from mlflow_monitor.gateway import GatewayConfig, IdempotencyKey, InMemoryMonitoringGateway


def test_reserve_sequence_index_is_monotonic_per_subject() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())

    assert gateway.reserve_sequence_index("churn_model") == 0
    assert gateway.reserve_sequence_index("churn_model") == 1
    assert gateway.reserve_sequence_index("fraud_model") == 0
    assert gateway.reserve_sequence_index("churn_model") == 2


def test_get_or_create_idempotent_run_id_creates_then_reuses() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    key = IdempotencyKey(
        subject_id="churn_model",
        source_run_id="train-run-1",
        recipe_id="default",
        recipe_version="v0",
    )
    call_count = {"factory": 0}

    def factory() -> str:
        call_count["factory"] += 1
        return f"run-{call_count['factory']}"

    first = gateway.get_or_create_idempotent_run_id(key, factory)
    second = gateway.get_or_create_idempotent_run_id(key, factory)

    assert first == "run-1"
    assert second == first
    assert call_count["factory"] == 1


def test_get_or_create_idempotent_run_id_diff_recipe_version_creates_new() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    key_v0 = IdempotencyKey(
        subject_id="churn_model",
        source_run_id="train-run-1",
        recipe_id="default",
        recipe_version="v0",
    )
    key_v1 = IdempotencyKey(
        subject_id="churn_model",
        source_run_id="train-run-1",
        recipe_id="default",
        recipe_version="v1",
    )
    call_count = {"factory": 0}

    def factory() -> str:
        call_count["factory"] += 1
        return f"run-{call_count['factory']}"

    run_v0 = gateway.get_or_create_idempotent_run_id(key_v0, factory)
    run_v1 = gateway.get_or_create_idempotent_run_id(key_v1, factory)

    assert run_v0 == "run-1"
    assert run_v1 == "run-2"
    assert call_count["factory"] == 2


def test_get_or_create_idempotent_run_id_uses_dataclass_value_equality() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    first_key = IdempotencyKey(
        subject_id="churn_model",
        source_run_id="train-run-1",
        recipe_id="default",
        recipe_version="v0",
    )
    equivalent_key = IdempotencyKey(
        subject_id="churn_model",
        source_run_id="train-run-1",
        recipe_id="default",
        recipe_version="v0",
    )
    call_count = {"factory": 0}

    def factory() -> str:
        call_count["factory"] += 1
        return f"run-{call_count['factory']}"

    first = gateway.get_or_create_idempotent_run_id(first_key, factory)
    second = gateway.get_or_create_idempotent_run_id(equivalent_key, factory)

    assert first == "run-1"
    assert second == first
    assert call_count["factory"] == 1


def test_initialize_timeline_is_deterministic_and_stores_baseline_reference() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())

    first_timeline_id = gateway.initialize_timeline("churn_model", "train-run-1")
    second_timeline_id = gateway.initialize_timeline("churn_model", "train-run-2")
    timeline_state = gateway.get_timeline_state("churn_model")

    assert first_timeline_id == "timeline-churn_model"
    assert second_timeline_id == first_timeline_id
    assert timeline_state is not None
    assert timeline_state.baseline_source_run_id == "train-run-1"


def test_initialize_timeline_rejects_empty_baseline_source_run_id() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())

    with pytest.raises(
        GatewayNamespaceViolation,
        match="baseline_source_run_id must be non-empty.",
    ):
        gateway.initialize_timeline("churn_model", "")


def test_set_and_resolve_active_lkg_run_id() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())

    assert gateway.resolve_active_lkg_run_id("churn_model") is None
    gateway.set_active_lkg_run_id("churn_model", "run-2")
    assert gateway.resolve_active_lkg_run_id("churn_model") == "run-2"
    gateway.set_active_lkg_run_id("churn_model", None)
    assert gateway.resolve_active_lkg_run_id("churn_model") is None


def test_list_timeline_runs_includes_failed_by_default() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    subject_id = "churn_model"
    gateway.upsert_monitoring_run(
        subject_id=subject_id,
        run_id="run-created",
        lifecycle_status=LifecycleStatus.CREATED,
        sequence_index=0,
    )
    gateway.upsert_monitoring_run(
        subject_id=subject_id,
        run_id="run-failed",
        lifecycle_status=LifecycleStatus.FAILED,
        sequence_index=1,
    )
    gateway.upsert_monitoring_run(
        subject_id=subject_id,
        run_id="run-closed",
        lifecycle_status=LifecycleStatus.CLOSED,
        sequence_index=2,
    )

    run_ids = tuple(run.run_id for run in gateway.list_timeline_runs(subject_id))

    assert run_ids == (
        "run-failed",
        "run-closed",
    )


def test_list_timeline_runs_excludes_failed_when_requested() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    subject_id = "churn_model"
    gateway.upsert_monitoring_run(
        subject_id=subject_id,
        run_id="run-1",
        lifecycle_status=LifecycleStatus.CLOSED,
        sequence_index=0,
    )
    gateway.upsert_monitoring_run(
        subject_id=subject_id,
        run_id="run-2",
        lifecycle_status=LifecycleStatus.FAILED,
        sequence_index=1,
    )

    run_ids = tuple(
        run.run_id
        for run in gateway.list_timeline_runs(
            subject_id=subject_id,
            exclude_failed=True,
        )
    )

    assert run_ids == ("run-1",)


def test_namespace_violation_raises_on_invalid_monitoring_write() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())

    with pytest.raises(
        GatewayNamespaceViolation,
        match="subject_id must be non-empty and must not contain '/'",
    ):
        gateway.upsert_monitoring_run(
            subject_id="team/churn_model",
            run_id="run-1",
            lifecycle_status=LifecycleStatus.CREATED,
            sequence_index=0,
        )


def test_training_mutation_raises_violation() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())

    with pytest.raises(
        TrainingRunMutationViolation,
        match="Training runs are read-only in MLflow-Monitor",
    ):
        gateway.mutate_training_run("train-run-1", {"status": "modified"})
