"""Workflow-facing contract checker boundary for MLflow-Monitor v0."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from mlflow_monitor.domain import (
    CONTRACT_CHECK_REASON_CODE_BLOCKING,
    ComparabilityStatus,
    Contract,
    ContractCheckReason,
    ContractCheckReasonCode,
    ContractCheckResult,
)

CONTRACT_CHECK_REASON_MESSAGE = MappingProxyType(
    {
        ContractCheckReasonCode.ENV_MISMATCH: "Execution environment does not match the baseline.",
        ContractCheckReasonCode.SCHEMA_MISMATCH: "Data schema does not match the baseline.",
        ContractCheckReasonCode.FEAT_MISMATCH: "Feature set does not match the baseline.",
        ContractCheckReasonCode.DATA_SCOPE_MISMATCH: "Data scope does not match the baseline.",
    },
)


@dataclass(frozen=True, slots=True)
class ContractEvidence:
    """Resolved evidence snapshot required for contract evaluation."""

    metrics: Mapping[str, float]
    environment: Mapping[str, str]
    features: tuple[str, ...]
    schema: Mapping[str, str]
    data_scope: str | None

    def __post_init__(self) -> None:
        """Freeze nested mutable structures to ensure overall immutability."""
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))
        object.__setattr__(self, "schema", MappingProxyType(dict(self.schema)))
        object.__setattr__(self, "features", tuple(self.features))


@dataclass(frozen=True, slots=True)
class ContractEvaluationContext:
    """Prepared evidence required to evaluate one resolved contract.

    This context is platform-agnostic. Workflow code should pass resolved
    evidence only, never MLflow client objects or recipe-layer configuration.

    Attributes:
        subject_id: Stable monitored subject identifier for the run.
        source_run_id: Source training run ID being evaluated.
        baseline_source_run_id: Source run ID from which the baseline was derived.
        baseline_evidence: Resolved evidence snapshot for the baseline.
        current_evidence: Resolved evidence snapshot for the current run.
    """

    subject_id: str
    source_run_id: str
    baseline_source_run_id: str
    baseline_evidence: ContractEvidence
    current_evidence: ContractEvidence


class ContractChecker(Protocol):
    """Single workflow-facing interface for contract evaluation."""

    def check(
        self,
        contract: Contract,
        context: ContractEvaluationContext,
    ) -> ContractCheckResult:
        """Evaluate one resolved contract against prepared run evidence.

        Args:
            contract: The resolved comparability contract for the run.
            context: Prepared platform-agnostic evidence for evaluation.

        Returns:
            The aggregate comparability verdict and machine-readable reasons.
        """
        ...


class DefaultContractChecker:
    """Default contract checker implementation using simple rule-based logic.

    This is a reference implementation and not intended for production use. Real
    implementations should be registered via entry points and can leverage any
    internal or external logic, including MLflow client APIs, to evaluate the
    contract.
    """

    def check(
        self,
        contract: Contract,
        context: ContractEvaluationContext,
    ) -> ContractCheckResult:
        """Evaluate the contract using simple rule-based checks on the evidence.

        This implementation performs basic equality checks between the baseline and
        current evidence for each contract aspect. It categorizes any mismatches

        Args:
            contract: The resolved comparability contract for the run.
            context: Prepared platform-agnostic evidence for evaluation.

        Returns:
            The aggregate comparability verdict and machine-readable reasons.
        """
        baseline_evidence = context.baseline_evidence
        current_evidence = context.current_evidence

        reasons: list[ContractCheckReason] = []

        if contract.execution_contract_ref is not None:
            if baseline_evidence.environment != current_evidence.environment:
                reasons.append(
                    ContractCheckReason(
                        code=ContractCheckReasonCode.ENV_MISMATCH,
                        message=CONTRACT_CHECK_REASON_MESSAGE[ContractCheckReasonCode.ENV_MISMATCH],
                        blocking=CONTRACT_CHECK_REASON_CODE_BLOCKING[
                            ContractCheckReasonCode.ENV_MISMATCH
                        ],
                    ),
                )

        if contract.schema_contract_ref is not None:
            if baseline_evidence.schema != current_evidence.schema:
                reasons.append(
                    ContractCheckReason(
                        code=ContractCheckReasonCode.SCHEMA_MISMATCH,
                        message=CONTRACT_CHECK_REASON_MESSAGE[
                            ContractCheckReasonCode.SCHEMA_MISMATCH
                        ],
                        blocking=CONTRACT_CHECK_REASON_CODE_BLOCKING[
                            ContractCheckReasonCode.SCHEMA_MISMATCH
                        ],
                    ),
                )

        if contract.feature_contract_ref is not None:
            if baseline_evidence.features != current_evidence.features:
                reasons.append(
                    ContractCheckReason(
                        code=ContractCheckReasonCode.FEAT_MISMATCH,
                        message=CONTRACT_CHECK_REASON_MESSAGE[
                            ContractCheckReasonCode.FEAT_MISMATCH
                        ],
                        blocking=CONTRACT_CHECK_REASON_CODE_BLOCKING[
                            ContractCheckReasonCode.FEAT_MISMATCH
                        ],
                    ),
                )

        if contract.data_scope_contract_ref is not None:
            if baseline_evidence.data_scope != current_evidence.data_scope:
                reasons.append(
                    ContractCheckReason(
                        code=ContractCheckReasonCode.DATA_SCOPE_MISMATCH,
                        message=CONTRACT_CHECK_REASON_MESSAGE[
                            ContractCheckReasonCode.DATA_SCOPE_MISMATCH
                        ],
                        blocking=CONTRACT_CHECK_REASON_CODE_BLOCKING[
                            ContractCheckReasonCode.DATA_SCOPE_MISMATCH
                        ],
                    ),
                )

        if contract.metric_contract_ref is not None:
            pass  # Metric checks are not implemented in this default checker

        has_reasons = bool(reasons)
        has_blocking_reason = any(reason.blocking for reason in reasons)

        if not reasons:
            return ContractCheckResult(
                status=ComparabilityStatus.PASS,
                reasons=(),
            )

        if has_blocking_reason:
            return ContractCheckResult(
                status=ComparabilityStatus.FAIL,
                reasons=tuple(reasons),
            )

        if has_reasons:
            return ContractCheckResult(
                status=ComparabilityStatus.WARN,
                reasons=tuple(reasons),
            )

        return ContractCheckResult(
            status=ComparabilityStatus.PASS,
            reasons=tuple(reasons),
        )


def _make_contract_evaluation_context(
    *,
    subject_id: str,
    source_run_id: str,
    baseline_source_run_id: str,
    baseline_context: ContractEvidence,
    current_context: ContractEvidence,
) -> ContractEvaluationContext:
    """Build a normalized contract evaluation context from resolved evidence.

    Args:
        subject_id: Stable monitored subject identifier for the run.
        source_run_id: Source training run ID being evaluated.
        baseline_source_run_id: Source run ID from which the baseline was derived.
        baseline_context: Resolved evidence snapshot for the baseline.
        current_context: Resolved evidence snapshot for the current run.

    Returns:
        A normalized, immutable contract evaluation context.
    """
    return ContractEvaluationContext(
        subject_id=subject_id,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        baseline_evidence=baseline_context,
        current_evidence=current_context,
    )
