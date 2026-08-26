"""Canonical domain models for MLflow-Monitor v0.

This module defines the platform-agnostic entities used by the monitoring
workflow. These types capture shape and vocabulary for monitoring state only;
workflow rules and invariant enforcement live in the higher-level runtime.
"""

from .contract import (
    CONTRACT_CHECK_REASON_CODE_BLOCKING,
    ComparabilityStatus,
    CompatibilityEvidence,
    Contract,
    ContractCheckReason,
    ContractCheckReasonCode,
    ContractCheckResult,
)
from .diff import (
    ABSOLUTE_DELTA_TOLERANCE,
    REFERENCE_COMPARISON_STATUS_TO_REASON,
    RELATIVE_DELTA_TOLERANCE,
    Diff,
    MetricComparisonUnavailable,
    MetricComparisonUnavailableReason,
    ReferenceComparisonCoverage,
    ReferenceComparisonSkippedReason,
    ReferenceComparisonStatus,
    ReferenceComparisonUnavailableReason,
)
from .finding import Finding, FindingDraft, FindingSeverity
from .lifecycle import LifecycleStatus
from .reference import DiffReference, DiffReferenceKind, MonitoringRunReference
from .timeline import LKGSelection, Timeline, TimelineEntry

__all__ = [
    "DiffReferenceKind",
    "DiffReference",
    "MonitoringRunReference",
    "Diff",
    "MetricComparisonUnavailable",
    "ReferenceComparisonCoverage",
    "ReferenceComparisonSkippedReason",
    "ReferenceComparisonStatus",
    "ReferenceComparisonUnavailableReason",
    "MetricComparisonUnavailableReason",
    "REFERENCE_COMPARISON_STATUS_TO_REASON",
    "ABSOLUTE_DELTA_TOLERANCE",
    "RELATIVE_DELTA_TOLERANCE",
    "Finding",
    "FindingDraft",
    "FindingSeverity",
    "LKGSelection",
    "Timeline",
    "TimelineEntry",
    "LifecycleStatus",
    "CONTRACT_CHECK_REASON_CODE_BLOCKING",
    "ComparabilityStatus",
    "CompatibilityEvidence",
    "Contract",
    "ContractCheckReason",
    "ContractCheckReasonCode",
    "ContractCheckResult",
]
