"""Shared JSON artifact contract tests for monitoring Gateways."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import pytest

from mlflow_monitor.errors import GatewayConsistencyViolation
from mlflow_monitor.gateway import (
    GatewayConfig,
    IdempotencyKey,
    InMemoryMonitoringGateway,
    MLflowMonitoringGateway,
)


class _JsonArtifactGateway(Protocol):
    """Test-facing form of the V0-009 Gateway artifact contract."""

    def write_monitoring_run_json_artifact(
        self,
        *,
        monitoring_run_id: str,
        data: dict[str, object],
        path: str,
    ) -> None:
        """Write missing content or validate an identical existing artifact."""
        ...

    def read_monitoring_run_json_artifact(
        self,
        monitoring_run_id: str,
        path: str,
    ) -> dict[str, object] | None:
        """Read exact-path content or return None when it is missing."""
        ...


@dataclass(frozen=True, slots=True)
class _GatewayHarness:
    """One Gateway implementation and its Monitoring Run allocator."""

    gateway: _JsonArtifactGateway
    allocate: Callable[[str], str]


@pytest.fixture(params=("in_memory", "mlflow"))
def json_artifact_gateway(
    request: pytest.FixtureRequest,
    tracking_uri: str,
    artifact_root_uri: str,
) -> _GatewayHarness:
    """Return each v0 Gateway behind the same JSON artifact contract."""
    if request.param == "in_memory":
        concrete_gateway: InMemoryMonitoringGateway | MLflowMonitoringGateway = (
            InMemoryMonitoringGateway(GatewayConfig())
        )
    else:
        concrete_gateway = MLflowMonitoringGateway(
            GatewayConfig(),
            tracking_uri=tracking_uri,
            artifact_location=artifact_root_uri,
        )

    def allocate(source_run_id: str) -> str:
        return concrete_gateway.create_or_reuse_monitoring_run(
            IdempotencyKey(
                subject_id="churn_model",
                source_run_id=source_run_id,
                recipe_id="system_default",
                recipe_version="v0",
            )
        ).monitoring_run_id

    return _GatewayHarness(
        gateway=cast(_JsonArtifactGateway, concrete_gateway),
        allocate=allocate,
    )


def test_json_artifact_contract_writes_missing_accepts_identical_and_rejects_different(
    json_artifact_gateway: _GatewayHarness,
) -> None:
    monitoring_run_id = json_artifact_gateway.allocate("source-run-1")
    path = "state/prepared_context.json"
    original = {"z": 1, "a": {"z": 2, "a": 3}}
    identical = {"a": {"a": 3, "z": 2}, "z": 1}
    conflicting = {"a": {"a": 3, "z": 2}, "z": 4}

    json_artifact_gateway.gateway.write_monitoring_run_json_artifact(
        monitoring_run_id=monitoring_run_id,
        data=original,
        path=path,
    )
    json_artifact_gateway.gateway.write_monitoring_run_json_artifact(
        monitoring_run_id=monitoring_run_id,
        data=identical,
        path=path,
    )

    with pytest.raises(GatewayConsistencyViolation):
        json_artifact_gateway.gateway.write_monitoring_run_json_artifact(
            monitoring_run_id=monitoring_run_id,
            data=conflicting,
            path=path,
        )

    json_artifact_gateway.gateway.write_monitoring_run_json_artifact(
        monitoring_run_id=monitoring_run_id,
        data=original,
        path=path,
    )


def test_json_artifact_contract_reads_exact_path_and_distinguishes_missing(
    json_artifact_gateway: _GatewayHarness,
) -> None:
    monitoring_run_id = json_artifact_gateway.allocate("source-run-1")
    path = "state/prepared_context.json"
    payload = {"z": 1, "a": {"value": "prepared"}}

    assert (
        json_artifact_gateway.gateway.read_monitoring_run_json_artifact(
            monitoring_run_id,
            path,
        )
        is None
    )

    json_artifact_gateway.gateway.write_monitoring_run_json_artifact(
        monitoring_run_id=monitoring_run_id,
        data=payload,
        path=path,
    )

    assert json_artifact_gateway.gateway.read_monitoring_run_json_artifact(
        monitoring_run_id,
        path,
    ) == {"a": {"value": "prepared"}, "z": 1}
    assert (
        json_artifact_gateway.gateway.read_monitoring_run_json_artifact(
            monitoring_run_id,
            "outputs/prepared_context.json",
        )
        is None
    )


@pytest.mark.parametrize(
    ("original", "conflicting"),
    [
        ({"value": True}, {"value": 1}),
        ({"value": 1}, {"value": 1.0}),
    ],
)
def test_json_artifact_contract_preserves_json_scalar_type_distinctions(
    json_artifact_gateway: _GatewayHarness,
    original: dict[str, object],
    conflicting: dict[str, object],
) -> None:
    monitoring_run_id = json_artifact_gateway.allocate("source-run-1")
    path = "state/prepared_context.json"
    json_artifact_gateway.gateway.write_monitoring_run_json_artifact(
        monitoring_run_id=monitoring_run_id,
        data=original,
        path=path,
    )

    with pytest.raises(GatewayConsistencyViolation):
        json_artifact_gateway.gateway.write_monitoring_run_json_artifact(
            monitoring_run_id=monitoring_run_id,
            data=conflicting,
            path=path,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_json_artifact_contract_rejects_non_finite_content_without_writing(
    json_artifact_gateway: _GatewayHarness,
    value: float,
) -> None:
    monitoring_run_id = json_artifact_gateway.allocate("source-run-1")
    path = "state/prepared_context.json"

    with pytest.raises(ValueError):
        json_artifact_gateway.gateway.write_monitoring_run_json_artifact(
            monitoring_run_id=monitoring_run_id,
            data={"nested": {"value": value}},
            path=path,
        )

    json_artifact_gateway.gateway.write_monitoring_run_json_artifact(
        monitoring_run_id=monitoring_run_id,
        data={"nested": {"value": 0.0}},
        path=path,
    )
    with pytest.raises(GatewayConsistencyViolation):
        json_artifact_gateway.gateway.write_monitoring_run_json_artifact(
            monitoring_run_id=monitoring_run_id,
            data={"nested": {"value": 1.0}},
            path=path,
        )


def test_json_artifact_contract_isolates_monitoring_runs_and_exact_paths(
    json_artifact_gateway: _GatewayHarness,
) -> None:
    first_monitoring_run_id = json_artifact_gateway.allocate("source-run-1")
    second_monitoring_run_id = json_artifact_gateway.allocate("source-run-2")
    path = "state/prepared_context.json"
    json_artifact_gateway.gateway.write_monitoring_run_json_artifact(
        monitoring_run_id=first_monitoring_run_id,
        data={"value": "first"},
        path=path,
    )
    json_artifact_gateway.gateway.write_monitoring_run_json_artifact(
        monitoring_run_id=second_monitoring_run_id,
        data={"value": "second"},
        path=path,
    )
    json_artifact_gateway.gateway.write_monitoring_run_json_artifact(
        monitoring_run_id=first_monitoring_run_id,
        data={"value": "second"},
        path="outputs/prepared_context.json",
    )
    with pytest.raises(GatewayConsistencyViolation):
        json_artifact_gateway.gateway.write_monitoring_run_json_artifact(
            monitoring_run_id=first_monitoring_run_id,
            data={"value": "second"},
            path=path,
        )
