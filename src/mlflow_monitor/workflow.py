"""Workflow lifecycle helpers for MLflow-Monitor v0."""

from __future__ import annotations

from dataclasses import replace

from mlflow_monitor.domain import LifecycleStatus, Run
from mlflow_monitor.errors import InvalidRunTransition

_ALLOWED_TRANSITIONS = {
    LifecycleStatus.CREATED: {
        LifecycleStatus.PREPARED,
        LifecycleStatus.FAILED,
    },
    LifecycleStatus.PREPARED: {
        LifecycleStatus.CHECKED,
        LifecycleStatus.FAILED,
    },
    LifecycleStatus.CHECKED: {
        LifecycleStatus.ANALYZED,
        LifecycleStatus.FAILED,
    },
    LifecycleStatus.ANALYZED: {
        LifecycleStatus.CLOSED,
        LifecycleStatus.FAILED,
    },
    LifecycleStatus.CLOSED: set(),
    LifecycleStatus.FAILED: set(),
}


def transition_run(run: Run, to_status: LifecycleStatus) -> Run:
    """Return a new run with an updated lifecycle status if the move is legal.

    Args:
        run: The run whose lifecycle should advance.
        to_status: The target lifecycle status.

    Raises:
        InvalidRunTransition: If the requested transition is not allowed in v0.

    Returns:
        A new run value with the updated lifecycle status.
    """
    from_status = run.lifecycle_status

    if to_status not in _ALLOWED_TRANSITIONS[from_status]:
        raise InvalidRunTransition(
            from_status=from_status,
            to_status=to_status,
            message=f"Cannot transition run from {from_status} to {to_status}.",
        )

    return replace(run, lifecycle_status=to_status)
