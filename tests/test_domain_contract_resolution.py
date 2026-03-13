"""Tests for resolved contract semantics in domain entities."""

from mlflow_monitor.domain import Baseline, Contract, Timeline


def test_timeline_active_contract_is_resolved_contract() -> None:
    """Timeline should store the effective contract, not a recipe-facing profile."""
    contract = Contract(
        contract_id="default",
        version="v0",
        schema_contract_ref=None,
        feature_contract_ref=None,
        metric_contract_ref=None,
        data_scope_contract_ref=None,
        execution_contract_ref=None,
    )
    baseline = Baseline(
        timeline_id="timeline-1",
        source_run_id="train-run-1",
        model_identity="model-a",
        parameter_fingerprint="params-v1",
        data_snapshot_ref="dataset-2026-03-01",
        run_config_ref="config-v1",
        metric_snapshot={"f1": 0.87},
        environment_context={"python": "3.12"},
    )

    timeline = Timeline(
        timeline_id="timeline-1",
        subject_id="churn_model",
        monitoring_namespace="mlflow_monitor/churn_model",
        baseline=baseline,
        run_ids=["run-1"],
        active_lkg_run_id=None,
        active_contract=contract,
    )

    assert timeline.active_contract is contract
