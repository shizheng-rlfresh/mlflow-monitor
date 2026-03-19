"""Workflow-facing contract checker boundary for MLflow-Monitor v0."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from mlflow_monitor.domain import (
    ComparabilityStatus,
    Contract,
    ContractCheckReason,
    ContractCheckResult,
)


@dataclass(frozen=True, slots=True)
class ContractEvidence:
    metrics: Mapping[str, float]
    environment: Mapping[str, str]
    features: tuple[str, ...]
    schema: Mapping[str, str]
    data_scope: str | None


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

        reasons: tuple[ContractCheckReason, ...] = ()

        if contract.schema_contract_ref:
            pass

        if contract.feature_contract_ref:
            pass

        if contract.metric_contract_ref:
            pass

        if contract.data_scope_contract_ref:
            pass

        if contract.execution_contract_ref:
            pass

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
                reasons=reasons,
            )

        if has_reasons:
            return ContractCheckResult(
                status=ComparabilityStatus.WARN,
                reasons=reasons,
            )

        return ContractCheckResult(
            status=ComparabilityStatus.PASS,
            reasons=reasons,
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
