"""Unit tests for the repo-local CLI wrapper."""

from __future__ import annotations

import json

import pytest

from mlflow_monitor.domain import ComparabilityStatus, LifecycleStatus
from mlflow_monitor.result_contract import MonitorRunError, MonitorRunResult


def _success_result() -> MonitorRunResult:
    return MonitorRunResult(
        monitoring_run_id="monitoring-run-1",
        subject_id="fraud_model",
        timeline_id="mlflow_monitor/fraud_model",
        lifecycle_status=LifecycleStatus.CHECKED,
        comparability_status=ComparabilityStatus.PASS,
        summary={"status": "ok"},
        finding_ids=(),
        diff_ids=(),
        references=(),
    )


def _failed_result() -> MonitorRunResult:
    return MonitorRunResult(
        monitoring_run_id="monitoring-run-2",
        subject_id="fraud_model",
        timeline_id="mlflow_monitor/fraud_model",
        lifecycle_status=LifecycleStatus.FAILED,
        comparability_status=None,
        summary=None,
        finding_ids=(),
        diff_ids=(),
        references=(),
        error=MonitorRunError(
            code="prepare_error",
            message="prepare step failed",
            stage="prepare",
        ),
    )


def test_main_run_command_prints_canonical_json_and_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mlflow_monitor import cli

    expected = _success_result()

    def fake_run(**kwargs: str | None) -> MonitorRunResult:
        assert kwargs == {
            "subject_id": "fraud_model",
            "source_run_id": "source-run-123",
            "baseline_source_run_id": None,
        }
        return expected

    monkeypatch.setattr(cli.monitor, "run", fake_run)

    exit_code = cli.main(["run", "--subject", "fraud_model", "--source-run", "source-run-123"])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out) == expected.to_dict()
    assert captured.err == ""


def test_main_run_command_forwards_optional_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow_monitor import cli

    observed: dict[str, str | None] = {}

    def fake_run(**kwargs: str | None) -> MonitorRunResult:
        observed.update(kwargs)
        return _success_result()

    monkeypatch.setattr(cli.monitor, "run", fake_run)

    exit_code = cli.main(
        [
            "run",
            "--subject",
            "fraud_model",
            "--source-run",
            "source-run-123",
            "--baseline",
            "baseline-run-456",
        ]
    )

    assert exit_code == 0
    assert observed == {
        "subject_id": "fraud_model",
        "source_run_id": "source-run-123",
        "baseline_source_run_id": "baseline-run-456",
    }


def test_main_run_command_returns_non_zero_for_failed_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mlflow_monitor import cli

    expected = _failed_result()
    monkeypatch.setattr(cli.monitor, "run", lambda **_: expected)

    exit_code = cli.main(["run", "--subject", "fraud_model", "--source-run", "source-run-123"])

    captured = capsys.readouterr()

    assert exit_code == 1
    assert json.loads(captured.out) == expected.to_dict()
    assert captured.err == ""


def test_main_uses_argparse_for_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    from mlflow_monitor import cli

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["run", "--subject", "fraud_model"])

    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert captured.out == ""
    assert "usage:" in captured.err


def test_main_dispatches_run_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlflow_monitor import cli

    called = {"count": 0}

    def fake_execute(args: object) -> int:
        called["count"] += 1
        return 0

    monkeypatch.setattr(cli, "_execute_run_command", fake_execute)

    exit_code = cli.main(["run", "--subject", "fraud_model", "--source-run", "source-run-123"])

    assert exit_code == 0
    assert called["count"] == 1
