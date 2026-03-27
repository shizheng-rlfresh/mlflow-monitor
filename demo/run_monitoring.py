"""Run the monitoring half of the fraud demo."""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient

try:
    from demo.setup import (
        DEMO_EXPERIMENT_NAME,
        DEMO_SCENARIO_PARAM,
        DEMO_SUBJECT_ID,
        SCENARIO_RUN_NAMES,
        _artifact_location_for_tracking_uri,
        resolve_effective_tracking_uri,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from setup import (  # type: ignore[no-redef]
        DEMO_EXPERIMENT_NAME,
        DEMO_SCENARIO_PARAM,
        DEMO_SUBJECT_ID,
        SCENARIO_RUN_NAMES,
        _artifact_location_for_tracking_uri,
        resolve_effective_tracking_uri,
    )
from mlflow_monitor import monitor
from mlflow_monitor.gateway import GatewayConfig
from mlflow_monitor.mlflow_gateway import MLflowMonitoringGateway

MONITORING_EXPERIMENT_NAME = f"mlflow_monitor/{DEMO_SUBJECT_ID}"


@dataclass(frozen=True, slots=True)
class DemoMonitoringResult:
    """One monitoring result produced by the demo runner."""

    scenario_name: str
    source_run_name: str
    source_run_id: str
    monitoring_run_id: str
    lifecycle_status: str
    comparability_status: str | None


def _load_seeded_run_ids(tracking_uri: str | None = None) -> dict[str, str]:
    """Resolve the seeded training runs by their expected run names."""
    effective_tracking_uri = resolve_effective_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=effective_tracking_uri)
    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError("Demo training experiment not found. Run `uv run demo/setup.py` first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attribute.start_time DESC"],
    )
    run_id_by_scenario: dict[str, str] = {}
    for run in runs:
        if run.info.status != "FINISHED":
            continue
        scenario_name = run.data.params.get(DEMO_SCENARIO_PARAM, "")
        if scenario_name in SCENARIO_RUN_NAMES and scenario_name not in run_id_by_scenario:
            run_id_by_scenario[scenario_name] = run.info.run_id

    resolved: dict[str, str] = {}
    for scenario_name, run_name in SCENARIO_RUN_NAMES.items():
        run_id = run_id_by_scenario.get(scenario_name)
        if run_id is None:
            raise RuntimeError(
                f"Expected seeded run {run_name!r} was not found. Run `uv run demo/setup.py` again."
            )
        resolved[scenario_name] = run_id
    return resolved


def run_demo_monitoring(tracking_uri: str | None = None) -> tuple[DemoMonitoringResult, ...]:
    """Execute the three monitoring scenarios for the fraud demo."""
    effective_tracking_uri = resolve_effective_tracking_uri(tracking_uri)
    previous_tracking_uri = mlflow.get_tracking_uri()

    try:
        if effective_tracking_uri is not None:
            mlflow.set_tracking_uri(effective_tracking_uri)

        run_ids = _load_seeded_run_ids(tracking_uri=effective_tracking_uri)
        gateway = MLflowMonitoringGateway(
            GatewayConfig(),
            tracking_uri=effective_tracking_uri,
            artifact_location=_artifact_location_for_tracking_uri(effective_tracking_uri),
        )
        pass_result = monitor.run(
            subject_id=DEMO_SUBJECT_ID,
            source_run_id=run_ids["comparable_candidate"],
            baseline_source_run_id=run_ids["baseline"],
            gateway=gateway,
        )
        warn_result = monitor.run(
            subject_id=DEMO_SUBJECT_ID,
            source_run_id=run_ids["warning_candidate"],
            gateway=gateway,
        )
        fail_result = monitor.run(
            subject_id=DEMO_SUBJECT_ID,
            source_run_id=run_ids["non_comparable_candidate"],
            gateway=gateway,
        )

        return (
            DemoMonitoringResult(
                scenario_name="comparable_candidate",
                source_run_name=SCENARIO_RUN_NAMES["comparable_candidate"],
                source_run_id=run_ids["comparable_candidate"],
                monitoring_run_id=pass_result.monitoring_run_id,
                lifecycle_status=pass_result.lifecycle_status.value,
                comparability_status=pass_result.comparability_status.value
                if pass_result.comparability_status is not None
                else None,
            ),
            DemoMonitoringResult(
                scenario_name="warning_candidate",
                source_run_name=SCENARIO_RUN_NAMES["warning_candidate"],
                source_run_id=run_ids["warning_candidate"],
                monitoring_run_id=warn_result.monitoring_run_id,
                lifecycle_status=warn_result.lifecycle_status.value,
                comparability_status=warn_result.comparability_status.value
                if warn_result.comparability_status is not None
                else None,
            ),
            DemoMonitoringResult(
                scenario_name="non_comparable_candidate",
                source_run_name=SCENARIO_RUN_NAMES["non_comparable_candidate"],
                source_run_id=run_ids["non_comparable_candidate"],
                monitoring_run_id=fail_result.monitoring_run_id,
                lifecycle_status=fail_result.lifecycle_status.value,
                comparability_status=fail_result.comparability_status.value
                if fail_result.comparability_status is not None
                else None,
            ),
        )
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)


def _print_result(result: DemoMonitoringResult) -> None:
    """Print one human-readable monitoring result summary."""
    print("=== Monitoring Result ===")
    print("Scenario:", result.scenario_name)
    print("Source Run:", result.source_run_name)
    print("Source Run ID:", result.source_run_id)
    print("Monitoring Run ID:", result.monitoring_run_id)
    print("Lifecycle:", result.lifecycle_status)
    print("Comparability:", result.comparability_status)
    print()


def main() -> None:
    """Run the seeded monitoring demo and print result summaries."""
    results = run_demo_monitoring()
    print("Monitoring experiment:", MONITORING_EXPERIMENT_NAME)
    print()
    for result in results:
        _print_result(result)


if __name__ == "__main__":
    main()
