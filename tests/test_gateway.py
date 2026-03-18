"""Unit tests for gateway module behavior in MLflow-Monitor v0."""

import pytest

from mlflow_monitor.domain import LifecycleStatus
from mlflow_monitor.errors import GatewayNamespaceViolation, TrainingRunMutationViolation
from mlflow_monitor.gateway import (
    GatewayConfig,
    IdempotencyKey,
    InMemoryMonitoringGateway,
)
from mlflow_monitor.recipe import SYSTEM_DEFAULT_RUN_SELECTOR_TOKEN


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

    first_timeline_result = gateway.initialize_timeline("churn_model", "train-run-1")
    second_timeline_result = gateway.initialize_timeline("churn_model", "train-run-2")
    timeline_state = gateway.get_timeline_state("churn_model")

    assert first_timeline_result.timeline_id == "timeline-churn_model"
    assert second_timeline_result.timeline_id == first_timeline_result.timeline_id
    assert first_timeline_result.created is True
    assert second_timeline_result.created is False
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


def test_resolve_source_run_id_returns_matching_raw_run_id() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        run_id="train-run-1",
        source_experiment="training/churn",
        metrics={"f1": 0.91},
        artifacts=("metrics.json",),
    )

    resolved = gateway.resolve_source_run_id(
        subject_id="churn_model",
        source_experiment="training/churn",
        run_selector="train-run-1",
    )

    assert resolved == "train-run-1"


def test_resolve_source_run_id_uses_runtime_source_run_id_for_reserved_token() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        run_id="train-run-2",
        source_experiment=None,
        metrics={"f1": 0.88},
        artifacts=("metrics.json",),
    )

    resolved = gateway.resolve_source_run_id(
        subject_id="churn_model",
        source_experiment=None,
        run_selector=SYSTEM_DEFAULT_RUN_SELECTOR_TOKEN,
        runtime_source_run_id="train-run-2",
    )

    assert resolved == "train-run-2"


def test_resolve_source_run_id_allows_omitted_source_experiment_filter() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        run_id="train-run-3",
        source_experiment="training/churn",
        metrics={"f1": 0.89},
        artifacts=("metrics.json",),
    )

    resolved = gateway.resolve_source_run_id(
        subject_id="churn_model",
        source_experiment=None,
        run_selector="train-run-3",
    )

    assert resolved == "train-run-3"


def test_resolve_source_run_id_returns_none_for_missing_or_mismatched_run() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        run_id="train-run-1",
        source_experiment="training/churn",
        metrics={"f1": 0.91},
        artifacts=("metrics.json",),
    )

    assert (
        gateway.resolve_source_run_id(
            subject_id="churn_model",
            source_experiment="training/other",
            run_selector="train-run-1",
        )
        is None
    )
    assert (
        gateway.resolve_source_run_id(
            subject_id="fraud_model",
            source_experiment="training/churn",
            run_selector="train-run-1",
        )
        is None
    )


def test_missing_required_metrics_returns_missing_names_in_request_order() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        run_id="train-run-1",
        source_experiment="training/churn",
        metrics={"auc": 0.95},
        artifacts=("metrics.json",),
    )

    missing = gateway.get_missing_source_run_metrics(
        run_id="train-run-1",
        required_metrics=("f1", "auc", "precision"),
    )

    assert missing == ("f1", "precision")


def test_missing_required_artifacts_returns_missing_names_in_request_order() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        run_id="train-run-1",
        source_experiment="training/churn",
        metrics={"auc": 0.95},
        artifacts=("metrics.json", "model.pkl"),
    )

    missing = gateway.get_missing_source_run_artifacts(
        run_id="train-run-1",
        required_artifacts=("schema.json", "metrics.json"),
    )

    assert missing == ("schema.json",)


def test_resolve_timeline_run_id_requires_same_subject_timeline() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.upsert_monitoring_run(
        subject_id="churn_model",
        run_id="run-1",
        lifecycle_status=LifecycleStatus.CLOSED,
        sequence_index=0,
    )
    gateway.upsert_monitoring_run(
        subject_id="fraud_model",
        run_id="run-foreign",
        lifecycle_status=LifecycleStatus.CLOSED,
        sequence_index=0,
    )

    assert gateway.resolve_timeline_run_id("churn_model", "run-1") == "run-1"
    assert gateway.resolve_timeline_run_id("churn_model", "run-foreign") is None
