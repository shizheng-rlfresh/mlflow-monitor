"""Unit tests for the workflow lifecycle transitions."""

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
from mlflow_monitor.errors import InvalidRunTransition
from mlflow_monitor.workflow import transition_run

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
