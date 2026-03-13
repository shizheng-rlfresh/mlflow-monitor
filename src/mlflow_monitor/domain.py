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
    """Machine-readable reason emitted by a contract check.

    Attributes:
        code: A short string code categorizing the reason.
        message: A human-readable message describing the reason.
        blocking: Whether this reason should block promotion if comparability fails.
    """

    code: str
    message: str
    blocking: bool


@dataclass(frozen=True, slots=True)
class ContractCheckResult:
    """Comparability verdict and the reasons that produced it.

    Attributes:
        status: The overall comparability status (pass/warn/fail).
        reasons: A tuple of ContractCheckReason instances explaining the verdict.
    """

    status: ComparabilityStatus
    reasons: tuple[ContractCheckReason, ...]


@dataclass(frozen=True, slots=True)
class Contract:
    """Resolved versioned comparability contract bound to a timeline.

    This is the effective contract attached to a run or timeline after any
    recipe-layer selection has been resolved. It is not the recipe-facing
    profile/binding mechanism itself.

    Attributes:
        contract_id: Unique identifier for the contract.
        version: Version string for the contract schema.
        schema_contract_ref: Optional reference to a schema contract defining expected data shapes.
        feature_contract_ref: Optional reference to a feature contract defining expected features.
        metric_contract_ref: Optional reference to a metric contract defining expected metrics.
        data_scope_contract_ref: Optional reference to a data scope contract defining expected data.
        execution_contract_ref: Optional reference to an execution contract defining expected runtime.
    """  # noqa: E501

    contract_id: str
    version: str
    schema_contract_ref: str | None
    feature_contract_ref: str | None
    metric_contract_ref: str | None
    data_scope_contract_ref: str | None
    execution_contract_ref: str | None


@dataclass(frozen=True, slots=True)
class Baseline:
    """Pinned immutable baseline snapshot for a timeline.

    Attributes:
        timeline_id: The ID of the timeline this baseline belongs to.
        source_run_id: The run ID from which this baseline was derived.
        model_identity: A string representing the model identity (e.g., name and version).
        parameter_fingerprint: A string fingerprint representing the model parameters.
        data_snapshot_ref: A reference to the data snapshot used for this baseline.
        run_config_ref: A reference to the run configuration used for this baseline.
        metric_snapshot: A mapping of metric names to their values at the time of baseline creation.
        environment_context: A mapping of environment context keys to their values at the time of baseline creation.
    """  # noqa: E501

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
    """Objective change record between a run and one reference point.

    Attributes:
        diff_id: Unique identifier for the diff record.
        run_id: The ID of the run this diff is associated with.
        reference_run_id: The ID of the reference run this diff is comparing against, if applicable.
        reference_kind: The kind of reference used for this diff (e.g., baseline, previous, lkg).
        metric_deltas: A mapping of metric names to their delta values compared to the reference.
        metadata: A mapping of additional metadata keys to values providing context for the diff.
    """

    diff_id: str
    run_id: str
    reference_run_id: str | None
    reference_kind: DiffReferenceKind
    metric_deltas: dict[str, float]
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class Finding:
    """Interpreted issue derived from one or more supporting diffs.

    Attributes:
        finding_id: Unique identifier for the finding record.
        run_id: The ID of the run this finding is associated with.
        severity: The severity level of the finding (e.g., low, medium, high, critical).
        category: A string categorizing the type of issue this finding represents.
        summary: A human-readable summary describing the finding.
        evidence_diff_ids: A tuple of diff IDs that provide evidence supporting this finding.
        recommendation: A human-readable rec for addressing the issue described by this finding.
    """

    finding_id: str
    run_id: str
    severity: FindingSeverity
    category: str
    summary: str
    evidence_diff_ids: tuple[str, ...]
    recommendation: str


@dataclass(frozen=True, slots=True)
class Run:
    """Canonical monitoring run record owned by exactly one timeline.

    Attributes:
        run_id: Unique identifier for the run.
        timeline_id: The ID of the timeline this run belongs to.
        sequence_index: The sequential index of this run within its timeline, starting at 0 for the first run.
        subject_id: The ID of the monitored subject this run is associated with.
        source_run_id: The original run ID from the training system that produced this run, if applicable.
        baseline_source_run_id: The source run ID of the baseline this run is compared against, if applicable.
        contract: The comparability contract in effect for this run.
        lifecycle_status: The current lifecycle status of this run (e.g., created, prepared, checked, analyzed, closed, failed).
        comparability_status: The latest comparability status produced by contract evaluation for this run (e.g., pass, warn, fail).
        contract_check_result: The detailed result of the latest contract check performed for this run, if applicable.
        diff_ids: A tuple of diff IDs associated with this run.
        finding_ids: A tuple of finding IDs associated with this run.
    """  # noqa: E501

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
    """Pointer to the active last-known-good run on a timeline.

    Attributes:
        timeline_id: The ID of the timeline this LKG belongs to.
        run_id: The ID of the run currently considered last-known-good for its timeline.
    """

    timeline_id: str
    run_id: str


@dataclass(frozen=True, slots=True)
class Timeline:
    """Ordered run history for one monitored subject.

    Attributes:
        timeline_id: Unique identifier for the timeline.
        subject_id: The ID of the monitored subject this timeline is associated with.
        monitoring_namespace: The namespace this timeline belongs to.
        baseline: The pinned baseline snapshot for this timeline.
        run_ids: An ordered list of run IDs belonging to this timeline.
        active_lkg_run_id: The run ID of the currently active last-known-good run for this timeline.
        active_contract: The currently active comparability contract for this timeline
    """

    timeline_id: str
    subject_id: str
    monitoring_namespace: str
    baseline: Baseline
    run_ids: list[str]
    active_lkg_run_id: str | None
    active_contract: Contract
