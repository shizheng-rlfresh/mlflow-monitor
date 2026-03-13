from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class LifecycleStatus(StrEnum):
    CREATED = "created"
    PREPARED = "prepared"
    CHECKED = "checked"
    ANALYZED = "analyzed"
    CLOSED = "closed"
    FAILED = "failed"
    PROMOTED = "promoted"


class ComparabilityStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class DiffReferenceKind(StrEnum):
    BASELINE = "baseline"
    PREVIOUS = "previous"
    LKG = "lkg"
    CUSTOM = "custom"
    STRUCTURAL = "structural"


class FindingSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class ContractCheckReason:
    code: str
    message: str
    blocking: bool


@dataclass(frozen=True, slots=True)
class ContractCheckResult:
    status: ComparabilityStatus
    reasons: tuple[ContractCheckReason, ...]


@dataclass(frozen=True, slots=True)
class Contract:
    contract_id: str
    version: str
    schema_contract_ref: str | None
    feature_contract_ref: str | None
    metric_contract_ref: str | None
    data_scope_contract_ref: str | None
    execution_contract_ref: str | None


@dataclass(frozen=True, slots=True)
class Baseline:
    timeline_id: str
    source_run_id: str
    model_identity: str
    parameter_fingerprint: str
    data_snapshot_ref: str
    run_config_ref: str


@dataclass(frozen=True, slots=True)
class Diff:
    diff_id: str
    run_id: str
    reference_run_id: str | None
    reference_kind: DiffReferenceKind
    metric_deltas: dict[str, float]
    metadata: dict[str, str]


@dataclass(frozen=True, slots=True)
class Finding:
    finding_id: str
    run_id: str
    severity: FindingSeverity
    category: str
    summary: str
    evidence_diff_ids: tuple[str, ...]
    recommendation: str


@dataclass(frozen=True, slots=True)
class Run:
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
    timeline_id: str
    run_id: str


@dataclass(frozen=True, slots=True)
class Timeline:
    timeline_id: str
    subject_id: str
    monitoring_namespace: str
    baseline: Baseline
    run_ids: list[str]
    active_lkg_run_id: str | None
    active_contract: Contract
