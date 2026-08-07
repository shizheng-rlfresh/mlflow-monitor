"""Canonical domain models for MLflow-Monitor.

This module defines the platform-agnostic entities used by the monitoring
workflow. These types capture shape and vocabulary for monitoring state only;
workflow rules and invariant enforcement live in the higher-level runtime.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

RELATIVE_DELTA_TOLERANCE = 1e-6
ABSOLUTE_DELTA_TOLERANCE = 1e-6


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
    """Reference kinds supported by diff records."""

    BASELINE = "baseline"
    PREVIOUS = "previous"
    LKG = "lkg"
    CUSTOM = "custom"


class FindingSeverity(StrEnum):
    """Priority levels for interpreted findings."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContractCheckReasonCode(StrEnum):
    """Canonical reason codes for contract check results."""

    ENV_MISMATCH = "environment_mismatch"
    SCHEMA_MISMATCH = "schema_mismatch"
    FEAT_MISMATCH = "feature_mismatch"
    DATA_SCOPE_MISMATCH = "data_scope_mismatch"


class ReferenceComparisonStatus(StrEnum):
    """Status outcomes for metrics comparison in diff."""

    COMPLETED = "completed"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"


CONTRACT_CHECK_REASON_CODE_BLOCKING = MappingProxyType(
    {
        ContractCheckReasonCode.ENV_MISMATCH: False,
        ContractCheckReasonCode.SCHEMA_MISMATCH: True,
        ContractCheckReasonCode.FEAT_MISMATCH: True,
        ContractCheckReasonCode.DATA_SCOPE_MISMATCH: True,
    }
)

_MONITORING_RUN_REFERENCE_KINDS = frozenset(
    (
        DiffReferenceKind.BASELINE,
        DiffReferenceKind.PREVIOUS,
        DiffReferenceKind.LKG,
        DiffReferenceKind.CUSTOM,
    )
)

_METRIC_UNAVAILABILITY_REASONS = frozenset(
    (
        "current_metric_missing",
        "reference_metric_missing",
        "current_metric_not_finite",
        "reference_metric_not_finite",
        "delta_not_finite",
    )
)

_SKIPPED_REFERENCE_COMPARISON_REASONS = frozenset(
    {
        "current_not_comparable",
    }
)

_UNAVAILABILITY_REFERENCE_COMPARISON_REASONS = frozenset(
    {
        "previous_reference_missing",
        "lkg_not_selected",
        "lkg_selection_inconsistent",
        "reference_source_run_missing",
    }
)


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
class CompatibilityEvidence:
    """Run-scoped compatibility observation materialized from a Check reason.

    Attributes:
        compatibility_evidence_id: Deterministic identifier for this evidence.
        monitoring_run_id: Monitoring Run that owns this evidence.
        source_run_id: Source Training Run evaluated by the Monitoring Run.
        baseline_source_run_id: Baseline Source Run used by the Contract check.
        contract_id: Identifier of the resolved Contract.
        contract_version: Version of the resolved Contract.
        reason: Complete Contract Check reason represented by this evidence.
    """

    compatibility_evidence_id: str
    monitoring_run_id: str
    source_run_id: str
    baseline_source_run_id: str
    contract_id: str
    contract_version: str
    reason: ContractCheckReason

    def __post_init__(self) -> None:
        """Validate the immutable Compatibility Evidence shape."""
        for field_name in (
            "compatibility_evidence_id",
            "monitoring_run_id",
            "source_run_id",
            "baseline_source_run_id",
            "contract_id",
            "contract_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"CompatibilityEvidence requires a non-empty string for field {field_name!r}."
                )

        if not isinstance(self.reason, ContractCheckReason):
            raise ValueError("CompatibilityEvidence requires a ContractCheckReason for 'reason'.")


@dataclass(frozen=True, slots=True)
class Contract:
    """Resolved versioned comparability contract bound to a Monitoring Run.

    This is the effective contract attached to a Monitoring Run after any
    recipe-layer selection has been resolved. It is not the recipe-facing profile
    or binding mechanism itself.

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
class MonitoringRunReference:
    """Typed run-level reference captured for one monitoring run.

    Attributes:
        kind: Reference kind for the monitoring run lineage.
        monitoring_run_id: Referenced monitoring run identifier, or None for the
            source-only baseline.
        source_run_id: Immutable source run identifier for the reference.
    """

    kind: DiffReferenceKind
    monitoring_run_id: str | None
    source_run_id: str

    def __post_init__(self) -> None:
        """Validate run-level reference shape."""
        try:
            kind = DiffReferenceKind(self.kind)
        except ValueError as exc:
            raise ValueError(f"Unsupported monitoring run reference kind {self.kind!r}.") from exc
        if kind not in _MONITORING_RUN_REFERENCE_KINDS:
            raise ValueError(f"Unsupported monitoring run reference kind {kind.value!r}.")
        object.__setattr__(self, "kind", kind)
        _validate_reference_identity(
            entity="MonitoringRunReference",
            kind=kind,
            monitoring_run_id=self.monitoring_run_id,
            source_run_id=self.source_run_id,
        )

    def to_dict(self) -> dict[str, str | None]:
        """Serialize this run-level reference into a deterministic dictionary."""
        return {
            "kind": self.kind.value,
            "monitoring_run_id": self.monitoring_run_id,
            "source_run_id": self.source_run_id,
        }


@dataclass(frozen=True, slots=True)
class DiffReference:
    """Reference descriptor for one diff comparison target.

    Attributes:
        kind: The reference kind for this diff (e.g., baseline, previous, lkg).
        monitoring_run_id: Referenced monitoring run identifier, or None for the
            source-only baseline.
        source_run_id: Immutable source run identifier for the reference. This is
            temporarily optional only for the legacy structural kind removed by
            V0-003.
    """

    kind: DiffReferenceKind
    monitoring_run_id: str | None
    source_run_id: str | None

    def __post_init__(self) -> None:
        """Validate that reference identity presence matches the reference kind."""
        _validate_reference_identity(
            entity="DiffReference",
            kind=self.kind,
            monitoring_run_id=self.monitoring_run_id,
            source_run_id=self.source_run_id,
        )


def _validate_reference_identity(
    *,
    entity: str,
    kind: DiffReferenceKind,
    monitoring_run_id: str | None,
    source_run_id: str | None,
) -> None:
    """Validate source-only baseline and paired monitoring reference identity."""
    if source_run_id is None or not source_run_id.strip():
        raise ValueError(f"{entity} with kind={kind.value!r} requires a non-empty source_run_id.")
    if kind is DiffReferenceKind.BASELINE:
        if monitoring_run_id is not None:
            raise ValueError(f"{entity} with kind='baseline' must not set monitoring_run_id.")
        return
    if monitoring_run_id is None or not monitoring_run_id.strip():
        raise ValueError(
            f"{entity} with kind={kind.value!r} requires a non-empty monitoring_run_id."
        )


@dataclass(frozen=True, slots=True)
class Diff:
    """Objective change record between a run and one reference point.

    Attributes:
        diff_id: Unique identifier for the diff record.
        monitoring_run_id: The ID of the monitoring run this diff is associated with.
        source_run_id: The immutable source training run ID of the monitoring run.
        reference: Reference descriptor containing both reference kind and reference id.
        metric_name: The name of the metric being compared.
        current_value: The value of the metric for the current run.
        reference_value: The value of the metric for the reference run.
        delta: current_value - reference_value.
    """

    diff_id: str
    monitoring_run_id: str
    source_run_id: str
    reference: DiffReference
    metric_name: str
    current_value: float
    reference_value: float
    delta: float

    def __post_init__(self) -> None:
        """Validate Diff for atomic shape."""
        for field_name in (
            "diff_id",
            "monitoring_run_id",
            "source_run_id",
            "metric_name",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Diff requires a non-empty string for field {field_name!r}.")

        if not isinstance(self.reference, DiffReference):
            raise ValueError("Diff requires a valid DiffReference for the 'reference' field.")

        for field_name in ("current_value", "reference_value", "delta"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Diff requires a finite float for field {field_name!r}.")

            if isinstance(value, int):
                object.__setattr__(self, field_name, float(value))

        expected_delta = self.current_value - self.reference_value

        if not math.isfinite(expected_delta):
            raise ValueError("Diff computed delta must be a finite float.")

        if not math.isclose(
            self.delta,
            expected_delta,
            rel_tol=RELATIVE_DELTA_TOLERANCE,
            abs_tol=ABSOLUTE_DELTA_TOLERANCE,
        ):
            raise ValueError(
                "Diff delta must equal current_value - reference_value within "
                f"rel_tol={RELATIVE_DELTA_TOLERANCE}, abs_tol={ABSOLUTE_DELTA_TOLERANCE}."
            )


@dataclass(frozen=True, slots=True)
class FindingDraft:
    """Transient policy conclusion awaiting package-owned materialization.

    Attributes:
        finding_rule_id: Stable rule identity within the Finding policy.
        severity: Priority assigned by the Finding policy.
        category: Category assigned by the Finding policy.
        summary: Human-readable conclusion produced by the Finding policy.
        recommendation: Human-readable recommended response.
        evidence_diff_ids: Diff identities supporting the conclusion.
        evidence_compatibility_ids: Compatibility Evidence identities supporting
            the conclusion.
    """

    finding_rule_id: str
    severity: FindingSeverity
    category: str
    summary: str
    recommendation: str
    evidence_diff_ids: tuple[str, ...]
    evidence_compatibility_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate and freeze the transient Finding conclusion."""
        _validate_finding_text_fields(
            entity="FindingDraft",
            value=self,
            field_names=(
                "finding_rule_id",
                "category",
                "summary",
                "recommendation",
            ),
        )
        _validate_finding_severity(entity="FindingDraft", severity=self.severity)
        _freeze_finding_evidence(self)


@dataclass(frozen=True, slots=True)
class Finding:
    """Immutable policy conclusion for one Monitoring Run.

    Attributes:
        finding_id: Deterministic identity for the Finding.
        monitoring_run_id: Monitoring Run that owns the Finding.
        source_run_id: Source Training Run evaluated by the Monitoring Run.
        finding_policy_id: Identifier of the Finding policy that emitted the draft.
        finding_policy_version: Version of the Finding policy that emitted the draft.
        finding_rule_id: Stable rule identity within the Finding policy.
        severity: Priority assigned by the Finding policy.
        category: Category assigned by the Finding policy.
        summary: Human-readable conclusion produced by the Finding policy.
        recommendation: Human-readable recommended response.
        evidence_diff_ids: Diff identities supporting the conclusion.
        evidence_compatibility_ids: Compatibility Evidence identities supporting
            the conclusion.
    """

    finding_id: str
    monitoring_run_id: str
    source_run_id: str
    finding_policy_id: str
    finding_policy_version: str
    finding_rule_id: str
    severity: FindingSeverity
    category: str
    summary: str
    recommendation: str
    evidence_diff_ids: tuple[str, ...]
    evidence_compatibility_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate and freeze the materialized Finding conclusion."""
        _validate_finding_text_fields(
            entity="Finding",
            value=self,
            field_names=(
                "finding_id",
                "monitoring_run_id",
                "source_run_id",
                "finding_policy_id",
                "finding_policy_version",
                "finding_rule_id",
                "category",
                "summary",
                "recommendation",
            ),
        )
        _validate_finding_severity(entity="Finding", severity=self.severity)
        _freeze_finding_evidence(self)


def _validate_finding_text_fields(
    *,
    entity: str,
    value: object,
    field_names: tuple[str, ...],
) -> None:
    """Validate required Finding text and identity fields."""
    for field_name in field_names:
        field_value = getattr(value, field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"{entity} requires a non-empty string for field {field_name!r}.")


def _validate_finding_severity(*, entity: str, severity: object) -> None:
    """Validate that Finding severity uses the canonical enum."""
    if not isinstance(severity, FindingSeverity):
        raise ValueError(f"{entity} requires a FindingSeverity for field 'severity'.")


def _freeze_finding_evidence(value: FindingDraft | Finding) -> None:
    """Defensively freeze and validate both Finding evidence collections."""
    entity = type(value).__name__
    evidence_collections: dict[str, tuple[str, ...]] = {}
    for field_name in ("evidence_diff_ids", "evidence_compatibility_ids"):
        supplied_ids = getattr(value, field_name)
        if isinstance(supplied_ids, str):
            raise ValueError(f"{entity} requires a collection for field {field_name!r}.")
        try:
            evidence_ids = tuple(supplied_ids)
        except TypeError as exc:
            raise ValueError(f"{entity} requires a collection for field {field_name!r}.") from exc
        if any(
            not isinstance(evidence_id, str) or not evidence_id.strip()
            for evidence_id in evidence_ids
        ):
            raise ValueError(
                f"{entity} requires non-empty string identities in field {field_name!r}."
            )
        evidence_collections[field_name] = evidence_ids
        object.__setattr__(value, field_name, evidence_ids)

    if not any(evidence_collections.values()):
        raise ValueError(f"{entity} requires at least one evidence identity.")


@dataclass(frozen=True, slots=True)
class Run:
    """Canonical monitoring run record owned by exactly one timeline.

    Attributes:
        monitoring_run_id: Unique identifier for the monitoring run.
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

    monitoring_run_id: str
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
class TimelineEntry:
    """Compact summary of one Monitoring Run in a Timeline.

    Attributes:
        monitoring_run_id: Monitoring Run summarized by this entry.
        source_run_id: Source Training Run evaluated by the Monitoring Run.
        sequence_index: Stable allocation order within the Timeline.
        lifecycle_status: Terminal lifecycle state of the Monitoring Run.
        comparability_status: Contract-check outcome when available.
    """

    monitoring_run_id: str
    source_run_id: str
    sequence_index: int
    lifecycle_status: LifecycleStatus
    comparability_status: ComparabilityStatus | None

    def __post_init__(self) -> None:
        """Validate the immutable Timeline Entry shape."""
        for field_name in ("monitoring_run_id", "source_run_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"TimelineEntry requires a non-empty string for field {field_name!r}."
                )
        if (
            not isinstance(self.sequence_index, int)
            or isinstance(self.sequence_index, bool)
            or self.sequence_index < 0
        ):
            raise ValueError("TimelineEntry sequence_index must be a nonnegative integer.")
        if not isinstance(self.lifecycle_status, LifecycleStatus):
            raise ValueError("TimelineEntry lifecycle_status must be a LifecycleStatus.")
        if self.lifecycle_status not in (LifecycleStatus.CLOSED, LifecycleStatus.FAILED):
            raise ValueError("TimelineEntry lifecycle_status must be closed or failed.")
        if self.comparability_status is not None and not isinstance(
            self.comparability_status, ComparabilityStatus
        ):
            raise ValueError(
                "TimelineEntry comparability_status must be a ComparabilityStatus or None."
            )

    def to_dict(self) -> dict[str, object]:
        """Serialize this Timeline Entry into a deterministic dictionary.

        Returns:
            JSON-compatible Timeline Entry content.
        """
        return {
            "monitoring_run_id": self.monitoring_run_id,
            "source_run_id": self.source_run_id,
            "sequence_index": self.sequence_index,
            "lifecycle_status": self.lifecycle_status.value,
            "comparability_status": (
                self.comparability_status.value if self.comparability_status is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class Timeline:
    """Ordered monitoring history for one Subject.

    Attributes:
        timeline_id: Stable Timeline identifier.
        subject_id: Subject whose monitoring history this Timeline describes.
        baseline_source_run_id: Pinned Baseline Source Run, or None before
            bootstrap succeeds.
        entries: Canonically ordered Monitoring Run summaries.
    """

    timeline_id: str
    subject_id: str
    baseline_source_run_id: str | None
    entries: tuple[TimelineEntry, ...]

    def __post_init__(self) -> None:
        """Validate identity and defensively freeze ordered entries."""
        for field_name in ("timeline_id", "subject_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Timeline requires a non-empty string for field {field_name!r}.")
        if self.baseline_source_run_id is not None and (
            not isinstance(self.baseline_source_run_id, str)
            or not self.baseline_source_run_id.strip()
        ):
            raise ValueError("Timeline baseline_source_run_id must be a non-empty string or None.")
        if isinstance(self.entries, str):
            raise ValueError("Timeline entries must be a collection of TimelineEntry values.")
        try:
            supplied_entries = tuple(self.entries)
        except TypeError as exc:
            raise ValueError(
                "Timeline entries must be a collection of TimelineEntry values."
            ) from exc
        if any(not isinstance(entry, TimelineEntry) for entry in supplied_entries):
            raise ValueError("Timeline entries must contain only TimelineEntry values.")

        entries = tuple(sorted(supplied_entries, key=lambda entry: entry.sequence_index))

        if self.baseline_source_run_id is None and any(
            entry.lifecycle_status == LifecycleStatus.CLOSED for entry in entries
        ):
            raise ValueError(
                "Timeline cannot accept closed entries without a baseline_source_run_id."
            )

        sequence_indexes = tuple(entry.sequence_index for entry in entries)
        if len(sequence_indexes) != len(set(sequence_indexes)):
            raise ValueError("Timeline entries must have unique sequence_index values.")
        monitoring_run_ids = tuple(entry.monitoring_run_id for entry in entries)
        if len(monitoring_run_ids) != len(set(monitoring_run_ids)):
            raise ValueError("Timeline entries must have unique monitoring_run_id values.")
        object.__setattr__(self, "entries", entries)

    def to_dict(self) -> dict[str, object]:
        """Serialize this Timeline into a deterministic dictionary.

        Returns:
            JSON-compatible Timeline content with ordered entries.
        """
        return {
            "timeline_id": self.timeline_id,
            "subject_id": self.subject_id,
            "baseline_source_run_id": self.baseline_source_run_id,
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True, slots=True)
class LKGSelection:
    """Immutable user selection of one trusted Monitoring Run.

    Attributes:
        lkg_selection_id: Stable identity of this trust-selection event.
        timeline_id: Timeline whose trust state this selection updates.
        monitoring_run_id: Selected Monitoring Run.
        source_run_id: Source Training Run evaluated by the selected Monitoring Run.
        supersedes_lkg_selection_ids: Prior selection identities superseded by this
            selection.
    """

    lkg_selection_id: str
    timeline_id: str
    monitoring_run_id: str
    source_run_id: str
    supersedes_lkg_selection_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate identity and freeze supersession history canonically."""
        for field_name in (
            "lkg_selection_id",
            "timeline_id",
            "monitoring_run_id",
            "source_run_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"LKGSelection requires a non-empty string for field {field_name!r}."
                )

        supplied_ids = self.supersedes_lkg_selection_ids
        if isinstance(supplied_ids, str):
            raise ValueError("LKGSelection supersedes_lkg_selection_ids must be a collection.")
        try:
            supersedes_ids = tuple(supplied_ids)
        except TypeError as exc:
            raise ValueError(
                "LKGSelection supersedes_lkg_selection_ids must be a collection."
            ) from exc
        if any(
            not isinstance(selection_id, str) or not selection_id.strip()
            for selection_id in supersedes_ids
        ):
            raise ValueError(
                "LKGSelection supersedes_lkg_selection_ids must contain non-empty strings."
            )

        if self.lkg_selection_id in supersedes_ids:
            raise ValueError("LKGSelection cannot supersede itself.")

        object.__setattr__(
            self,
            "supersedes_lkg_selection_ids",
            tuple(sorted(set(supersedes_ids))),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize this LKG Selection into a deterministic dictionary.

        Returns:
            JSON-compatible LKG Selection content.
        """
        return {
            "lkg_selection_id": self.lkg_selection_id,
            "timeline_id": self.timeline_id,
            "monitoring_run_id": self.monitoring_run_id,
            "source_run_id": self.source_run_id,
            "supersedes_lkg_selection_ids": list(self.supersedes_lkg_selection_ids),
        }


@dataclass(frozen=True, slots=True)
class MetricComparisonUnavailable:
    """Information about a metric that could not be compared in a diff.

    Attributes:
        metric_name: The name of the metric that is unavailable for comparison.
        reason: The reason why the metric comparison is unavailable.
    """

    metric_name: str
    reason: str

    def __post_init__(self) -> None:
        """Validate MetricComparisonUnavailable for atomic shape."""
        for field_name in ("metric_name", "reason"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"MetricComparisonUnavailable requires a non-empty string for "
                    f"field {field_name!r}."
                )
        metric_level_reason = self.reason
        if metric_level_reason not in _METRIC_UNAVAILABILITY_REASONS:
            raise ValueError(
                f"MetricComparisonUnavailable 'reason' must be one of "
                f"{_METRIC_UNAVAILABILITY_REASONS}, got {metric_level_reason!r}."
            )


@dataclass(frozen=True, slots=True)
class ReferenceComparisonCoverage:
    """Coverage information for a specific reference kind in a diff comparison.

    Attributes:
        reference_kind: The kind of reference (e.g., baseline, previous, lkg, custom).
        reference: The specific reference instance being compared, if applicable.
        status: The status of the reference comparison (e.g., completed, skipped, unavailable).
        diff_ids: A tuple of diff IDs associated with this reference comparison.
        metric_unavailability: A tuple of MetricComparisonUnavailable instances indicating metrics that could not be compared.
        reason: An optional reason code constrained by the reference comparison status.
    """  # noqa: E501

    reference_kind: DiffReferenceKind
    reference: DiffReference | None
    status: ReferenceComparisonStatus
    diff_ids: tuple[str, ...]
    metric_unavailability: tuple[MetricComparisonUnavailable, ...]
    reason: str | None

    def __post_init__(self) -> None:
        """Validate ReferenceComparisonCoverage for atomic shape."""
        # defensive conversion to tuples for immutability
        diff_ids_tuple = tuple(self.diff_ids)
        metric_unavailability_tuple = tuple(self.metric_unavailability)

        object.__setattr__(self, "diff_ids", diff_ids_tuple)
        object.__setattr__(self, "metric_unavailability", metric_unavailability_tuple)

        if self.reference_kind not in DiffReferenceKind:
            raise ValueError(
                "ReferenceComparisonCoverage has an unrecognized "
                f"reference_kind: {self.reference_kind!r}."
            )

        if self.reference is not None:
            if not isinstance(self.reference, DiffReference):
                raise ValueError("Coverage reference must be a DiffReference.")
            if self.reference_kind != self.reference.kind:
                raise ValueError(
                    "ReferenceComparisonCoverage 'reference_kind' must match the kind of "
                    "the provided 'reference'."
                )

        if self.status == ReferenceComparisonStatus.COMPLETED:
            self._validate_completed_coverage()

        elif self.status == ReferenceComparisonStatus.SKIPPED:
            self._validate_skipped_coverage()

        elif self.status == ReferenceComparisonStatus.UNAVAILABLE:
            self._validate_unavailable_coverage()
        else:
            raise ValueError(
                f"ReferenceComparisonCoverage has an unrecognized status: {self.status!r}."
            )

    def _validate_completed_coverage(self) -> None:
        if self.reference is None:
            raise ValueError(
                "ReferenceComparisonCoverage with status COMPLETED must have a valid reference."
            )
        if self.reason is not None:
            raise ValueError(
                "ReferenceComparisonCoverage with status COMPLETED must not have a reason code."
            )

    def _validate_skipped_coverage(self) -> None:
        if self.reference is None:
            raise ValueError(
                "ReferenceComparisonCoverage with status SKIPPED must have a valid reference."
            )
        if self.diff_ids:
            raise ValueError(
                "ReferenceComparisonCoverage with status SKIPPED must not have any diff IDs."
            )

        if self.metric_unavailability:
            raise ValueError(
                "ReferenceComparisonCoverage with status SKIPPED must not have any "
                "metric unavailability entries."
            )

        if self.reason is None or self.reason not in _SKIPPED_REFERENCE_COMPARISON_REASONS:
            raise ValueError(
                "ReferenceComparisonCoverage with status SKIPPED must have a reason code "
                f"from {_SKIPPED_REFERENCE_COMPARISON_REASONS}."
            )

    def _validate_unavailable_coverage(self) -> None:
        """Validate an unavailable reference-comparison group."""
        if self.diff_ids or self.metric_unavailability:
            raise ValueError("Unavailable coverage cannot contain metric results.")

        if self.reason is None or self.reason not in _UNAVAILABILITY_REFERENCE_COMPARISON_REASONS:
            raise ValueError(
                "ReferenceComparisonCoverage with status UNAVAILABLE must have a reason code "
                f"from {_UNAVAILABILITY_REFERENCE_COMPARISON_REASONS}."
            )

        elif self.reason == "previous_reference_missing":
            if self.reference_kind != DiffReferenceKind.PREVIOUS:
                raise ValueError("previous_reference_missing requires reference_kind='previous'.")
            if self.reference is not None:
                raise ValueError("previous_reference_missing requires reference=None.")

        elif self.reason == "lkg_not_selected":
            if self.reference_kind != DiffReferenceKind.LKG:
                raise ValueError("lkg_not_selected requires reference_kind='lkg'.")
            if self.reference is not None:
                raise ValueError("lkg_not_selected requires reference=None.")

        elif self.reason == "lkg_selection_inconsistent":
            if self.reference_kind != DiffReferenceKind.LKG:
                raise ValueError("lkg_selection_inconsistent requires reference_kind='lkg'.")
            if self.reference is not None:
                raise ValueError("lkg_selection_inconsistent requires reference=None.")

        else:
            if self.reference is None:
                raise ValueError("reference_source_run_missing requires a retained reference.")
