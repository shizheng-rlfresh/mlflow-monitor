"""Lifecycle domain models for mlflow-monitor v0."""

from __future__ import annotations

from enum import StrEnum


class LifecycleStatus(StrEnum):
    """Lifecycle states for a monitoring run.

    Promotion is intentionally excluded because it is a post-close action, not a
    lifecycle transition.
    """

    CREATED = "created"
    PREPARED = "prepared"
    CHECKED = "checked"
    ANALYZED = "analyzed"
    CLOSED = "closed"
    FAILED = "failed"
