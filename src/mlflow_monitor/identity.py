"""Deterministic identity helpers for objective monitoring evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from mlflow_monitor.domain import DiffReference

IDENTITY_SCHEME_VERSION = "v1"


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
        payload={
            "metric_name": metric_name,
            "monitoring_run_id": monitoring_run_id,
            "reference": {
                "kind": reference.kind.value,
                "monitoring_run_id": reference.monitoring_run_id,
                "source_run_id": reference.source_run_id,
            },
            "source_run_id": source_run_id,
        },
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
        payload={
            "baseline_source_run_id": baseline_source_run_id,
            "contract_id": contract_id,
            "contract_version": contract_version,
            "monitoring_run_id": monitoring_run_id,
            "reason_code": reason_code,
            "source_run_id": source_run_id,
        },
    )


def _make_identity(
    *,
    entity_type: str,
    prefix: str,
    payload: Mapping[str, object],
) -> str:
    """Hash one canonical versioned identity payload."""
    canonical_payload = {
        "entity_type": entity_type,
        "identity_scheme_version": IDENTITY_SCHEME_VERSION,
        **payload,
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
