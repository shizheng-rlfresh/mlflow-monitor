"""Deterministic identity helpers for monitoring records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from mlflow_monitor.domain import DiffReference

IDENTITY_SCHEME_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class _DiffIdentity:
    """Deterministic identity for one atomic metric Diff.

    Attributes:
        monitoring_run_id: Monitoring Run that owns the Diff.
        source_run_id: Source Training Run evaluated by the Monitoring Run.
        reference: Complete comparison reference.
        metric_name: Metric compared by the Diff.
    """

    monitoring_run_id: str
    source_run_id: str
    reference: DiffReference
    metric_name: str

    def to_dict(self) -> dict[str, object]:
        """Convert the identity to a dictionary representation."""
        return {
            "monitoring_run_id": self.monitoring_run_id,
            "source_run_id": self.source_run_id,
            "reference": {
                "kind": self.reference.kind.value,
                "monitoring_run_id": self.reference.monitoring_run_id,
                "source_run_id": self.reference.source_run_id,
            },
            "metric_name": self.metric_name,
        }


@dataclass(frozen=True, slots=True)
class _CompatibilityEvidenceIdentity:
    """Deterministic identity for one Compatibility Evidence record.

    Attributes:
        monitoring_run_id: Monitoring Run that owns the evidence.
        source_run_id: Source Training Run evaluated by the Monitoring Run.
        baseline_source_run_id: Baseline Source Run used by the Contract check.
        contract_id: Identifier of the resolved Contract.
        contract_version: Version of the resolved Contract.
        reason_code: Machine-readable Contract Check reason code.
    """

    monitoring_run_id: str
    source_run_id: str
    baseline_source_run_id: str
    contract_id: str
    contract_version: str
    reason_code: str

    def to_dict(self) -> dict[str, object]:
        """Convert the identity to a dictionary representation."""
        return {
            "monitoring_run_id": self.monitoring_run_id,
            "source_run_id": self.source_run_id,
            "baseline_source_run_id": self.baseline_source_run_id,
            "contract_id": self.contract_id,
            "contract_version": self.contract_version,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True)
class _FindingIdentity:
    """Deterministic identity for one policy Finding.

    Attributes:
        monitoring_run_id: Monitoring Run that owns the Finding.
        source_run_id: Source Training Run evaluated by the Monitoring Run.
        finding_policy_id: Identifier of the Finding policy.
        finding_policy_version: Version of the Finding policy.
        finding_rule_id: Stable rule identity within the Finding policy.
        evidence_diff_ids: Diff identities supporting the Finding.
        evidence_compatibility_ids: Compatibility Evidence identities supporting
            the Finding.
    """

    monitoring_run_id: str
    source_run_id: str
    finding_policy_id: str
    finding_policy_version: str
    finding_rule_id: str
    evidence_diff_ids: tuple[str, ...]
    evidence_compatibility_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Convert the identity to a canonical dictionary representation."""
        return {
            "monitoring_run_id": self.monitoring_run_id,
            "source_run_id": self.source_run_id,
            "finding_policy_id": self.finding_policy_id,
            "finding_policy_version": self.finding_policy_version,
            "finding_rule_id": self.finding_rule_id,
            "evidence_diff_ids": sorted(set(self.evidence_diff_ids)),
            "evidence_compatibility_ids": sorted(set(self.evidence_compatibility_ids)),
        }


def make_diff_id(
    *,
    monitoring_run_id: str,
    source_run_id: str,
    reference: DiffReference,
    metric_name: str,
) -> str:
    """Build the stable identity for one atomic metric Diff.

    Args:
        monitoring_run_id: Monitoring Run that owns the Diff.
        source_run_id: Source Training Run evaluated by the Monitoring Run.
        reference: Complete comparison reference.
        metric_name: Metric compared by the Diff.

    Returns:
        A versioned SHA-256 Diff identifier.
    """
    return _make_identity(
        entity_type="diff",
        prefix="diff",
        payload=_DiffIdentity(
            monitoring_run_id=monitoring_run_id,
            source_run_id=source_run_id,
            reference=reference,
            metric_name=metric_name,
        ),
    )


def make_compatibility_evidence_id(
    *,
    monitoring_run_id: str,
    source_run_id: str,
    baseline_source_run_id: str,
    contract_id: str,
    contract_version: str,
    reason_code: str,
) -> str:
    """Build the stable identity for one Compatibility Evidence record.

    Args:
        monitoring_run_id: Monitoring Run that owns the evidence.
        source_run_id: Source Training Run evaluated by the Monitoring Run.
        baseline_source_run_id: Baseline Source Run used by the Contract check.
        contract_id: Identifier of the resolved Contract.
        contract_version: Version of the resolved Contract.
        reason_code: Machine-readable Contract Check reason code.

    Returns:
        A versioned SHA-256 Compatibility Evidence identifier.
    """
    return _make_identity(
        entity_type="compatibility_evidence",
        prefix="compatibility-evidence",
        payload=_CompatibilityEvidenceIdentity(
            monitoring_run_id=monitoring_run_id,
            source_run_id=source_run_id,
            baseline_source_run_id=baseline_source_run_id,
            contract_id=contract_id,
            contract_version=contract_version,
            reason_code=reason_code,
        ),
    )


def make_finding_id(
    *,
    monitoring_run_id: str,
    source_run_id: str,
    finding_policy_id: str,
    finding_policy_version: str,
    finding_rule_id: str,
    evidence_diff_ids: tuple[str, ...],
    evidence_compatibility_ids: tuple[str, ...],
) -> str:
    """Build the stable identity for one policy Finding.

    Args:
        monitoring_run_id: Monitoring Run that owns the Finding.
        source_run_id: Source Training Run evaluated by the Monitoring Run.
        finding_policy_id: Identifier of the Finding policy.
        finding_policy_version: Version of the Finding policy.
        finding_rule_id: Stable rule identity within the Finding policy.
        evidence_diff_ids: Diff identities supporting the Finding.
        evidence_compatibility_ids: Compatibility Evidence identities supporting
            the Finding.

    Returns:
        A versioned SHA-256 Finding identifier.
    """
    return _make_identity(
        entity_type="finding",
        prefix="finding",
        payload=_FindingIdentity(
            monitoring_run_id=monitoring_run_id,
            source_run_id=source_run_id,
            finding_policy_id=finding_policy_id,
            finding_policy_version=finding_policy_version,
            finding_rule_id=finding_rule_id,
            evidence_diff_ids=evidence_diff_ids,
            evidence_compatibility_ids=evidence_compatibility_ids,
        ),
    )


def _make_identity(
    *,
    entity_type: str,
    prefix: str,
    payload: _DiffIdentity | _CompatibilityEvidenceIdentity | _FindingIdentity,
) -> str:
    """Hash one canonical versioned identity payload."""
    canonical_payload = {
        "entity_type": entity_type,
        "identity_scheme_version": IDENTITY_SCHEME_VERSION,
        "payload": payload.to_dict(),
    }
    encoded = json.dumps(
        canonical_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"{prefix}-{IDENTITY_SCHEME_VERSION}-{digest}"
