"""Canonical v0 domain models for MLflow-Monitor.

This module defines the platform-agnostic entities described in CAST v0.
These types capture shape and vocabulary for monitoring state only; workflow
rules and invariant enforcement are layered on later tickets.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


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


class ComparabilityStatus(StrEnum):
    """Comparability outcomes produced by contract evaluation."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class DiffReferenceKind(StrEnum):
    """Reference kinds supported by v0 diff records."""

    BASELINE = "baseline"
    PREVIOUS = "previous"
    LKG = "lkg"
    CUSTOM = "custom"
    STRUCTURAL = "structural"


class FindingSeverity(StrEnum):
    """Priority levels for interpreted findings."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class ContractCheckReason:
    """Machine-readable reason emitted by a contract check."""

    code: str
    message: str
    blocking: bool


@dataclass(frozen=True, slots=True)
class ContractCheckResult:
    """Comparability verdict and the reasons that produced it."""

    status: ComparabilityStatus
    reasons: tuple[ContractCheckReason, ...]


@dataclass(frozen=True, slots=True)
class Contract:
    """Versioned comparability contract bound to a timeline."""

    contract_id: str
    version: str
    schema_contract_ref: str | None
    feature_contract_ref: str | None
    metric_contract_ref: str | None
    data_scope_contract_ref: str | None
    execution_contract_ref: str | None


@dataclass(frozen=True, slots=True)
class Baseline:
    """Pinned immutable baseline snapshot for a timeline."""

    timeline_id: str
    source_run_id: str
    model_identity: str
    parameter_fingerprint: str
    data_snapshot_ref: str
    run_config_ref: str
    metric_snapshot: Mapping[str, float]
    environment_context: Mapping[str, str]

    def __post_init__(self) -> None:
        """Freeze nested baseline mappings after defensive copies."""

        object.__setattr__(
            self,
            "metric_snapshot",
            MappingProxyType(dict(self.metric_snapshot)),
        )
        object.__setattr__(
            self,
            "environment_context",
            MappingProxyType(dict(self.environment_context)),
        )


@dataclass(frozen=True, slots=True)
class Diff:
    """Objective change record between a run and one reference point."""

    diff_id: str
    run_id: str
    reference_run_id: str | None
    reference_kind: DiffReferenceKind
    metric_deltas: dict[str, float]
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class Finding:
    """Interpreted issue derived from one or more supporting diffs."""

    finding_id: str
    run_id: str
    severity: FindingSeverity
    category: str
    summary: str
    evidence_diff_ids: tuple[str, ...]
    recommendation: str


@dataclass(frozen=True, slots=True)
class Run:
    """Canonical monitoring run record owned by exactly one timeline."""

    run_id: str
    timeline_id: str
    sequence_index: int
    subject_id: str
    source_run_id: str
    baseline_source_run_id: str
    contract: Contract
    lifecycle_status: LifecycleStatus
    comparability_status: ComparabilityStatus
    contract_check_result: ContractCheckResult | None
    diff_ids: tuple[str, ...]
    finding_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LKG:
    """Pointer to the active last-known-good run on a timeline."""

    timeline_id: str
    run_id: str


@dataclass(frozen=True, slots=True)
class Timeline:
    """Ordered run history for one monitored subject."""

    timeline_id: str
    subject_id: str
    monitoring_namespace: str
    baseline: Baseline
    run_ids: list[str]
    active_lkg_run_id: str | None
    active_contract: Contract
