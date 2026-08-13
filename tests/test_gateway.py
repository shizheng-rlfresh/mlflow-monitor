"""Unit tests for gateway module behavior in MLflow-Monitor v0."""

import pytest

from mlflow_monitor.contract_checker import ContractEvidence
from mlflow_monitor.domain import (
    ComparabilityStatus,
    ContractCheckReason,
    ContractCheckResult,
    LifecycleStatus,
    MonitoringRunReference,
)
from mlflow_monitor.errors import (
    GatewayConsistencyViolation,
    GatewayNamespaceViolation,
    TrainingRunMutationViolation,
)
from mlflow_monitor.gateway import (
    GatewayConfig,
    IdempotencyKey,
    InMemoryMonitoringGateway,
)


def test_create_or_reuse_monitoring_run_creates_then_reuses() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    key = IdempotencyKey(
        subject_id="churn_model",
        source_run_id="train-run-1",
        recipe_id="default",
        recipe_version="v0",
    )

    assert gateway.get_timeline_state("churn_model") is None

    first = gateway.create_or_reuse_monitoring_run(key)
    timeline_state = gateway.get_timeline_state("churn_model")
    second = gateway.create_or_reuse_monitoring_run(key)

    assert first.monitoring_run_id.startswith("monitoring-run-")
    assert first.source_run_id == "train-run-1"
    assert first.timeline_id == "timeline-churn_model"
    assert first.sequence_index == 0
    assert first.existing_monitoring_run is None
    assert first.allocated is True
    assert timeline_state is not None
    assert timeline_state.timeline_id == first.timeline_id
    assert (
        timeline_state.baseline_source_run_id is None
    )  # no bootstrap for create_or_reuse_monitoring_run
    assert second.monitoring_run_id == first.monitoring_run_id
    assert second.source_run_id == first.source_run_id
    assert second.timeline_id == first.timeline_id
    assert second.sequence_index == first.sequence_index
    assert second.existing_monitoring_run is None
    assert second.allocated is False


def test_create_or_reuse_monitoring_run_diff_recipe_version_creates_new() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    monitoring_key_v0 = IdempotencyKey(
        subject_id="churn_model",
        source_run_id="train-run-1",
        recipe_id="default",
        recipe_version="v0",
    )
    monitoring_key_v1 = IdempotencyKey(
        subject_id="churn_model",
        source_run_id="train-run-1",
        recipe_id="default",
        recipe_version="v1",
    )
    monitoring_run_v0 = gateway.create_or_reuse_monitoring_run(monitoring_key_v0)
    monitoring_run_v1 = gateway.create_or_reuse_monitoring_run(monitoring_key_v1)

    assert monitoring_run_v0.monitoring_run_id.startswith("monitoring-run-")
    assert monitoring_run_v0.sequence_index == 0
    assert monitoring_run_v1.monitoring_run_id.startswith("monitoring-run-")
    assert monitoring_run_v1.monitoring_run_id != monitoring_run_v0.monitoring_run_id
    assert monitoring_run_v1.sequence_index == 1


def test_create_or_reuse_monitoring_run_uses_dataclass_value_equality() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    monitoring_first_key = IdempotencyKey(
        subject_id="churn_model",
        source_run_id="train-run-1",
        recipe_id="default",
        recipe_version="v0",
    )
    monitoring_equivalent_key = IdempotencyKey(
        subject_id="churn_model",
        source_run_id="train-run-1",
        recipe_id="default",
        recipe_version="v0",
    )
    monitoring_first = gateway.create_or_reuse_monitoring_run(monitoring_first_key)
    monitoring_second = gateway.create_or_reuse_monitoring_run(monitoring_equivalent_key)

    assert monitoring_first.monitoring_run_id.startswith("monitoring-run-")
    assert monitoring_second.monitoring_run_id == monitoring_first.monitoring_run_id
    assert monitoring_second.sequence_index == monitoring_first.sequence_index


def test_create_or_reuse_monitoring_run_can_replay_allocation_without_run_record() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    key = IdempotencyKey(
        subject_id="churn_model",
        source_run_id="train-run-1",
        recipe_id="default",
        recipe_version="v0",
    )

    first = gateway.create_or_reuse_monitoring_run(key)
    second = gateway.create_or_reuse_monitoring_run(key)

    assert first.allocated is True
    assert first.existing_monitoring_run is None
    assert second.allocated is False
    assert second.monitoring_run_id == first.monitoring_run_id
    assert second.sequence_index == first.sequence_index
    assert second.existing_monitoring_run is None


def test_create_or_reuse_monitoring_run_is_monotonic_per_subject() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())

    churn_first = gateway.create_or_reuse_monitoring_run(
        IdempotencyKey(
            subject_id="churn_model",
            source_run_id="train-run-1",
            recipe_id="default",
            recipe_version="v0",
        )
    )
    churn_second = gateway.create_or_reuse_monitoring_run(
        IdempotencyKey(
            subject_id="churn_model",
            source_run_id="train-run-2",
            recipe_id="default",
            recipe_version="v0",
        )
    )
    fraud_first = gateway.create_or_reuse_monitoring_run(
        IdempotencyKey(
            subject_id="fraud_model",
            source_run_id="train-run-1",
            recipe_id="default",
            recipe_version="v0",
        )
    )

    assert churn_first.sequence_index == 0
    assert churn_second.sequence_index == 1
    assert fraud_first.sequence_index == 0


def test_pin_timeline_baseline_is_deterministic_and_stores_baseline_reference() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    first_allocation = gateway.create_or_reuse_monitoring_run(
        IdempotencyKey(
            subject_id="churn_model",
            source_run_id="train-run-1",
            recipe_id="default",
            recipe_version="v0",
        )
    )

    first_timeline_result = gateway.pin_timeline_baseline("churn_model", "train-run-1")
    second_timeline_result = gateway.pin_timeline_baseline("churn_model", "train-run-2")
    second_allocation = gateway.create_or_reuse_monitoring_run(
        IdempotencyKey(
            subject_id="churn_model",
            source_run_id="train-run-2",
            recipe_id="default",
            recipe_version="v0",
        )
    )
    timeline_state = gateway.get_timeline_state("churn_model")

    assert first_allocation.timeline_id == "timeline-churn_model"
    assert first_allocation.sequence_index == 0
    assert first_timeline_result.timeline_id == first_allocation.timeline_id
    assert second_timeline_result.timeline_id == first_timeline_result.timeline_id
    assert first_timeline_result.baseline_pinned is True
    assert second_timeline_result.baseline_pinned is False
    assert second_allocation.timeline_id == first_allocation.timeline_id
    assert second_allocation.sequence_index == 1
    assert timeline_state is not None
    assert timeline_state.baseline_source_run_id == "train-run-1"


def test_pin_timeline_baseline_requires_an_existing_allocation() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())

    with pytest.raises(
        GatewayConsistencyViolation,
        match="Timeline state for subject_id 'churn_model' not found.",
    ):
        gateway.pin_timeline_baseline("churn_model", "train-run-1")

    assert gateway.get_timeline_state("churn_model") is None


def test_pin_timeline_baseline_rejects_empty_baseline_source_run_id() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())

    with pytest.raises(
        GatewayNamespaceViolation,
        match="baseline_source_run_id must be non-empty.",
    ):
        gateway.pin_timeline_baseline("churn_model", "")


def test_set_and_resolve_active_lkg_monitoring_run_id() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())

    assert gateway.resolve_active_lkg_monitoring_run_id("churn_model") is None
    gateway.set_active_lkg_monitoring_run_id("churn_model", "monitoring-run-2")
    assert gateway.resolve_active_lkg_monitoring_run_id("churn_model") == "monitoring-run-2"
    gateway.set_active_lkg_monitoring_run_id("churn_model", None)
    assert gateway.resolve_active_lkg_monitoring_run_id("churn_model") is None


def test_list_timeline_runs_includes_failed_by_default() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    subject_id = "churn_model"
    gateway.upsert_monitoring_run(
        subject_id=subject_id,
        monitoring_run_id="monitoring-run-created",
        source_run_id="train-run-created",
        lifecycle_status=LifecycleStatus.CREATED,
        sequence_index=0,
    )
    gateway.upsert_monitoring_run(
        subject_id=subject_id,
        monitoring_run_id="monitoring-run-failed",
        source_run_id="train-run-failed",
        lifecycle_status=LifecycleStatus.FAILED,
        sequence_index=1,
    )
    gateway.upsert_monitoring_run(
        subject_id=subject_id,
        monitoring_run_id="monitoring-run-closed",
        source_run_id="train-run-closed",
        lifecycle_status=LifecycleStatus.CLOSED,
        sequence_index=2,
    )

    monitoring_run_ids = tuple(
        run.monitoring_run_id for run in gateway.list_timeline_monitoring_runs(subject_id)
    )

    assert monitoring_run_ids == (
        "monitoring-run-failed",
        "monitoring-run-closed",
    )


def test_list_timeline_runs_excludes_failed_when_requested() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    subject_id = "churn_model"
    gateway.upsert_monitoring_run(
        subject_id=subject_id,
        monitoring_run_id="monitoring-run-checked",
        source_run_id="train-run-checked",
        lifecycle_status=LifecycleStatus.CHECKED,
        sequence_index=0,
    )
    gateway.upsert_monitoring_run(
        subject_id=subject_id,
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-1",
        lifecycle_status=LifecycleStatus.CLOSED,
        sequence_index=1,
    )
    gateway.upsert_monitoring_run(
        subject_id=subject_id,
        monitoring_run_id="monitoring-run-2",
        source_run_id="train-run-2",
        lifecycle_status=LifecycleStatus.FAILED,
        sequence_index=2,
    )

    monitoring_run_ids = tuple(
        run.monitoring_run_id
        for run in gateway.list_timeline_monitoring_runs(
            subject_id=subject_id,
            exclude_failed=True,
        )
    )

    assert monitoring_run_ids == ("monitoring-run-checked", "monitoring-run-1")


def test_namespace_violation_raises_on_invalid_monitoring_write() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())

    with pytest.raises(
        GatewayNamespaceViolation,
        match="subject_id must be non-empty and must not contain '/'",
    ):
        gateway.upsert_monitoring_run(
            subject_id="team/churn_model",
            monitoring_run_id="monitoring-run-1",
            source_run_id="train-run-1",
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


def test_write_monitoring_run_json_artifact_rejects_source_training_run_id() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-1",
        source_experiment="training/churn",
        metrics={"f1": 0.91},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    with pytest.raises(GatewayNamespaceViolation, match="not an allocated monitoring run"):
        gateway.write_monitoring_run_json_artifact(
            monitoring_run_id="train-run-1",
            data={"value": 1},
            path="state/prepared_context.json",
        )


def test_resolve_source_run_id_returns_matching_raw_run_id() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-1",
        source_experiment="training/churn",
        metrics={"f1": 0.91},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    resolved = gateway.resolve_source_run_id(
        subject_id="churn_model",
        source_experiment="training/churn",
        source_run_id="train-run-1",
    )

    assert resolved == "train-run-1"


def test_resolve_source_run_id_uses_invocation_owned_source_run_id() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-2",
        source_experiment=None,
        metrics={"f1": 0.88},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    resolved = gateway.resolve_source_run_id(
        subject_id="churn_model",
        source_experiment=None,
        source_run_id="train-run-2",
    )

    assert resolved == "train-run-2"


def test_resolve_source_run_id_allows_omitted_source_experiment_filter() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-3",
        source_experiment="training/churn",
        metrics={"f1": 0.89},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    resolved = gateway.resolve_source_run_id(
        subject_id="churn_model",
        source_experiment=None,
        source_run_id="train-run-3",
    )

    assert resolved == "train-run-3"


def test_resolve_source_run_id_returns_none_for_missing_or_mismatched_run() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-1",
        source_experiment="training/churn",
        metrics={"f1": 0.91},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    assert (
        gateway.resolve_source_run_id(
            subject_id="churn_model",
            source_experiment="training/other",
            source_run_id="train-run-1",
        )
        is None
    )
    assert (
        gateway.resolve_source_run_id(
            subject_id="fraud_model",
            source_experiment="training/churn",
            source_run_id="train-run-1",
        )
        is None
    )


def test_missing_required_metrics_returns_missing_names_in_request_order() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
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

    missing = gateway.get_missing_source_run_metrics(
        source_run_id="train-run-1",
        required_metrics=("f1", "auc", "precision"),
    )

    assert missing == ("f1", "precision")


def test_missing_required_artifacts_returns_missing_names_in_request_order() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-1",
        source_experiment="training/churn",
        metrics={"auc": 0.95},
        artifacts=("metrics.json", "model.pkl"),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    missing = gateway.get_missing_source_run_artifacts(
        source_run_id="train-run-1",
        required_artifacts=("schema.json", "metrics.json"),
    )

    assert missing == ("schema.json",)


def test_resolve_timeline_monitoring_run_id_requires_same_subject_timeline() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.upsert_monitoring_run(
        subject_id="churn_model",
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-1",
        lifecycle_status=LifecycleStatus.CLOSED,
        sequence_index=0,
    )
    gateway.upsert_monitoring_run(
        subject_id="fraud_model",
        monitoring_run_id="monitoring-run-foreign",
        source_run_id="train-run-foreign",
        lifecycle_status=LifecycleStatus.CLOSED,
        sequence_index=0,
    )

    assert (
        gateway.resolve_timeline_monitoring_run_id("churn_model", "monitoring-run-1")
        == "monitoring-run-1"
    )
    assert (
        gateway.resolve_timeline_monitoring_run_id("churn_model", "monitoring-run-foreign") is None
    )


def test_get_source_run_contract_evidence_returns_expected_snapshot() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.add_source_run(
        subject_id="churn_model",
        source_run_id="train-run-1",
        source_experiment="training/churn",
        metrics={"f1": 0.91},
        artifacts=("metrics.json",),
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )

    evidence = gateway.get_source_run_contract_evidence("train-run-1")

    assert evidence == ContractEvidence(
        metrics={"f1": 0.91},
        environment={"python": "3.12"},
        features=("age", "income"),
        schema={"age": "int", "income": "float"},
        data_scope="validation:2026-03-01",
    )


def test_upsert_monitoring_run_comparability_status_is_derived_from_contract_check_result() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    result = ContractCheckResult(
        status=ComparabilityStatus.WARN,
        reasons=(
            ContractCheckReason(
                code="environment_mismatch",
                message="Execution environment does not match the baseline.",
                blocking=False,
            ),
        ),
    )

    gateway.upsert_monitoring_run(
        subject_id="churn_model",
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-1",
        lifecycle_status=LifecycleStatus.CHECKED,
        sequence_index=0,
        contract_check_result=result,
    )

    stored = gateway.get_monitoring_run("churn_model", "monitoring-run-1")

    assert stored is not None
    assert stored.comparability_status is ComparabilityStatus.WARN


def test_upsert_monitoring_run_stores_contract_check_outputs() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    result = ContractCheckResult(
        status=ComparabilityStatus.WARN,
        reasons=(
            ContractCheckReason(
                code="environment_mismatch",
                message="Execution environment does not match the baseline.",
                blocking=False,
            ),
        ),
    )

    gateway.upsert_monitoring_run(
        subject_id="churn_model",
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-1",
        lifecycle_status=LifecycleStatus.CHECKED,
        sequence_index=0,
        contract_check_result=result,
    )

    stored = gateway.get_monitoring_run(
        "churn_model",
        "monitoring-run-1",
    )

    assert stored is not None
    assert stored.comparability_status is ComparabilityStatus.WARN
    assert stored.contract_check_result == result


def test_upsert_monitoring_run_stores_references() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    result = ContractCheckResult(
        status=ComparabilityStatus.WARN,
        reasons=(
            ContractCheckReason(
                code="environment_mismatch",
                message="Execution environment does not match the baseline.",
                blocking=False,
            ),
        ),
    )

    gateway.upsert_monitoring_run(
        subject_id="churn_model",
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-1",
        lifecycle_status=LifecycleStatus.CHECKED,
        sequence_index=0,
        contract_check_result=result,
        references=(
            MonitoringRunReference(
                kind="baseline",
                monitoring_run_id=None,
                source_run_id="train-run-baseline",
            ),
            MonitoringRunReference(
                kind="lkg",
                monitoring_run_id="monitoring-run-lkg",
                source_run_id="train-run-lkg",
            ),
        ),
    )

    stored = gateway.get_monitoring_run("churn_model", "monitoring-run-1")

    assert stored is not None
    assert stored.source_run_id == "train-run-1"
    assert stored.references == (
        MonitoringRunReference(
            kind="baseline",
            monitoring_run_id=None,
            source_run_id="train-run-baseline",
        ),
        MonitoringRunReference(
            kind="lkg",
            monitoring_run_id="monitoring-run-lkg",
            source_run_id="train-run-lkg",
        ),
    )


def test_upsert_monitoring_run_rejects_source_identity_conflict() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    allocation = gateway.create_or_reuse_monitoring_run(
        IdempotencyKey(
            subject_id="churn_model",
            source_run_id="train-run-1",
            recipe_id="default",
            recipe_version="v0",
        )
    )

    with pytest.raises(GatewayConsistencyViolation) as exc_info:
        gateway.upsert_monitoring_run(
            subject_id="churn_model",
            monitoring_run_id=allocation.monitoring_run_id,
            source_run_id="train-run-2",
            lifecycle_status=LifecycleStatus.CREATED,
            sequence_index=allocation.sequence_index,
        )

    assert exc_info.value.code == "monitoring_run_upsert_field_override"
    assert exc_info.value.details == (("source_run_id", "train-run-2"),)


def test_upsert_monitoring_run_requires_nonempty_source_run_id() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())

    with pytest.raises(ValueError, match="MonitoringRunRecord.source_run_id must be non-empty"):
        gateway.upsert_monitoring_run(
            subject_id="churn_model",
            monitoring_run_id="monitoring-run-1",
            source_run_id="",
            lifecycle_status=LifecycleStatus.CREATED,
            sequence_index=0,
        )


def test_upsert_monitoring_run_rejects_conflicting_reference_pair() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    gateway.upsert_monitoring_run(
        subject_id="churn_model",
        monitoring_run_id="monitoring-run-previous",
        source_run_id="train-run-persisted-reference",
        lifecycle_status=LifecycleStatus.CLOSED,
        sequence_index=0,
    )

    with pytest.raises(GatewayConsistencyViolation) as exc_info:
        gateway.upsert_monitoring_run(
            subject_id="churn_model",
            monitoring_run_id="monitoring-run-current",
            source_run_id="train-run-current",
            lifecycle_status=LifecycleStatus.CHECKED,
            sequence_index=1,
            references=(
                MonitoringRunReference(
                    kind="previous",
                    monitoring_run_id="monitoring-run-previous",
                    source_run_id="train-run-claimed-reference",
                ),
            ),
        )

    assert exc_info.value.code == "monitoring_run_upsert_field_override"
    assert dict(exc_info.value.details) == {
        "monitoring_run_id": "monitoring-run-previous",
        "source_run_id": "train-run-claimed-reference",
        "persisted_source_run_id": "train-run-persisted-reference",
    }


def test_upsert_monitoring_run_preserves_check_outputs_when_only_lifecycle_status_changes() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    result = ContractCheckResult(
        status=ComparabilityStatus.FAIL,
        reasons=(
            ContractCheckReason(
                code="schema_mismatch",
                message="Data schema does not match the baseline.",
                blocking=True,
            ),
        ),
    )

    gateway.upsert_monitoring_run(
        subject_id="churn_model",
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-1",
        lifecycle_status=LifecycleStatus.CHECKED,
        sequence_index=0,
        contract_check_result=result,
        references=(
            MonitoringRunReference(
                kind="baseline",
                monitoring_run_id=None,
                source_run_id="train-run-baseline",
            ),
            MonitoringRunReference(
                kind="lkg",
                monitoring_run_id="monitoring-run-lkg",
                source_run_id="train-run-lkg",
            ),
        ),
    )

    gateway.upsert_monitoring_run(
        subject_id="churn_model",
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-1",
        lifecycle_status=LifecycleStatus.CLOSED,
        sequence_index=0,
    )

    stored = gateway.get_monitoring_run("churn_model", "monitoring-run-1")

    assert stored is not None
    assert stored.lifecycle_status is LifecycleStatus.CLOSED
    assert stored.comparability_status is ComparabilityStatus.FAIL
    assert stored.contract_check_result == result
    assert stored.references == (
        MonitoringRunReference(
            kind="baseline",
            monitoring_run_id=None,
            source_run_id="train-run-baseline",
        ),
        MonitoringRunReference(
            kind="lkg",
            monitoring_run_id="monitoring-run-lkg",
            source_run_id="train-run-lkg",
        ),
    )


def test_upsert_monitoring_run_rejects_changed_sequence_index() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())

    gateway.upsert_monitoring_run(
        subject_id="churn_model",
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-1",
        lifecycle_status=LifecycleStatus.CREATED,
        sequence_index=0,
    )

    with pytest.raises(GatewayConsistencyViolation) as exc:
        gateway.upsert_monitoring_run(
            subject_id="churn_model",
            monitoring_run_id="monitoring-run-1",
            source_run_id="train-run-1",
            lifecycle_status=LifecycleStatus.CREATED,
            sequence_index=1,
        )

    error = exc.value
    assert error.code == "monitoring_run_upsert_field_override"
    assert error.details == (("sequence_index", 1),)


def test_upsert_monitoring_run_reports_all_immutable_field_overrides() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    original_result = ContractCheckResult(
        status=ComparabilityStatus.FAIL,
        reasons=(
            ContractCheckReason(
                code="schema_mismatch",
                message="Data schema does not match the baseline.",
                blocking=True,
            ),
        ),
    )
    replacement_result = ContractCheckResult(
        status=ComparabilityStatus.WARN,
        reasons=(
            ContractCheckReason(
                code="feature_mismatch",
                message="Feature set does not match the baseline.",
                blocking=True,
            ),
        ),
    )

    gateway.upsert_monitoring_run(
        subject_id="churn_model",
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-1",
        lifecycle_status=LifecycleStatus.CHECKED,
        sequence_index=0,
        contract_check_result=original_result,
    )

    with pytest.raises(GatewayConsistencyViolation) as exc:
        gateway.upsert_monitoring_run(
            subject_id="churn_model",
            monitoring_run_id="monitoring-run-1",
            source_run_id="train-run-1",
            lifecycle_status=LifecycleStatus.CLOSED,
            sequence_index=1,
            contract_check_result=replacement_result,
        )

    error = exc.value
    assert error.code == "monitoring_run_upsert_field_override"
    assert error.details == (
        ("sequence_index", 1),
        ("contract_check_result", str(replacement_result)),
    )


def test_upsert_monitoring_run_rejects_changed_contract_check_result_after_initial_write() -> None:
    gateway = InMemoryMonitoringGateway(GatewayConfig())
    original_result = ContractCheckResult(
        status=ComparabilityStatus.FAIL,
        reasons=(
            ContractCheckReason(
                code="schema_mismatch",
                message="Data schema does not match the baseline.",
                blocking=True,
            ),
        ),
    )
    replacement_result = ContractCheckResult(
        status=ComparabilityStatus.FAIL,
        reasons=(
            ContractCheckReason(
                code="feature_mismatch",
                message="Feature set does not match the baseline.",
                blocking=True,
            ),
        ),
    )

    gateway.upsert_monitoring_run(
        subject_id="churn_model",
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-1",
        lifecycle_status=LifecycleStatus.CHECKED,
        sequence_index=0,
        contract_check_result=original_result,
    )

    with pytest.raises(GatewayConsistencyViolation) as exc:
        gateway.upsert_monitoring_run(
            subject_id="churn_model",
            monitoring_run_id="monitoring-run-1",
            source_run_id="train-run-1",
            lifecycle_status=LifecycleStatus.CHECKED,
            sequence_index=0,
            contract_check_result=replacement_result,
        )

    error = exc.value
    assert error.code == "monitoring_run_upsert_field_override"
    assert error.details == (("contract_check_result", str(replacement_result)),)
