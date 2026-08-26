"""Contract domain models for mlflow-monitor v0."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType


class ComparabilityStatus(StrEnum):
    """Comparability outcomes produced by contract evaluation."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class ContractCheckReasonCode(StrEnum):
    """Canonical reason codes for contract check results."""

    ENV_MISMATCH = "environment_mismatch"
    SCHEMA_MISMATCH = "schema_mismatch"
    FEAT_MISMATCH = "feature_mismatch"
    DATA_SCOPE_MISMATCH = "data_scope_mismatch"


CONTRACT_CHECK_REASON_CODE_BLOCKING = MappingProxyType(
    {
        ContractCheckReasonCode.ENV_MISMATCH: False,
        ContractCheckReasonCode.SCHEMA_MISMATCH: True,
        ContractCheckReasonCode.FEAT_MISMATCH: True,
        ContractCheckReasonCode.DATA_SCOPE_MISMATCH: True,
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
        contract_version: Version string for the contract schema.
        schema_contract_ref: Optional reference to a schema contract defining expected data shapes.
        feature_contract_ref: Optional reference to a feature contract defining expected features.
        metric_contract_ref: Optional reference to a metric contract defining expected metrics.
        data_scope_contract_ref: Optional reference to a data scope contract defining expected data.
        execution_contract_ref: Optional reference to an execution contract defining expected runtime.
    """  # noqa: E501

    contract_id: str
    contract_version: str
    schema_contract_ref: str | None
    feature_contract_ref: str | None
    metric_contract_ref: str | None
    data_scope_contract_ref: str | None
    execution_contract_ref: str | None
