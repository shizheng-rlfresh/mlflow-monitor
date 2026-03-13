"""Workflow-facing contract checker boundary for MLflow-Monitor v0."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from mlflow_monitor.domain import Baseline, Contract, ContractCheckResult


@dataclass(frozen=True, slots=True)
class ContractEvaluationContext:
    """Prepared evidence required to evaluate one resolved contract.

    This context is platform-agnostic. Workflow code should pass resolved
    evidence only, never MLflow client objects or recipe-layer configuration.

    Attributes:
        subject_id: Stable monitored subject identifier for the run.
        source_run_id: Source training run ID being evaluated.
        baseline: Pinned baseline snapshot for the timeline.
        current_metrics: Metrics observed on the source run being checked.
        current_environment: Execution environment fingerprint for the source run.
        current_features: Feature names resolved for the source run.
        current_schema: Resolved schema view for the source run.
        current_data_scope: Human-readable scope descriptor for the source run data.
    """

    subject_id: str
    source_run_id: str
    baseline: Baseline
    current_metrics: Mapping[str, float]
    current_environment: Mapping[str, str]
    current_features: tuple[str, ...]
    current_schema: Mapping[str, str]
    current_data_scope: str | None

    def __post_init__(self) -> None:
        """Freeze mapping and sequence inputs after defensive copies."""
        object.__setattr__(
            self,
            "current_metrics",
            MappingProxyType(dict(self.current_metrics)),
        )
        object.__setattr__(
            self,
            "current_environment",
            MappingProxyType(dict(self.current_environment)),
        )
        object.__setattr__(
            self,
            "current_features",
            tuple(self.current_features),
        )
        object.__setattr__(
            self,
            "current_schema",
            MappingProxyType(dict(self.current_schema)),
        )


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


def make_contract_evaluation_context(
    *,
    subject_id: str,
    source_run_id: str,
    baseline: Baseline,
    current_metrics: Mapping[str, float],
    current_environment: Mapping[str, str],
    current_features: Sequence[str],
    current_schema: Mapping[str, str],
    current_data_scope: str | None,
) -> ContractEvaluationContext:
    """Build a normalized contract evaluation context from resolved evidence.

    Args:
        subject_id: Stable monitored subject identifier for the run.
        source_run_id: Source training run ID being evaluated.
        baseline: Pinned baseline snapshot for the timeline.
        current_metrics: Metrics observed on the source run being checked.
        current_environment: Execution environment fingerprint for the source run.
        current_features: Feature names resolved for the source run.
        current_schema: Resolved schema view for the source run.
        current_data_scope: Human-readable scope descriptor for the source run data.

    Returns:
        A normalized, immutable contract evaluation context.
    """
    return ContractEvaluationContext(
        subject_id=subject_id,
        source_run_id=source_run_id,
        baseline=baseline,
        current_metrics=current_metrics,
        current_environment=current_environment,
        current_features=tuple(current_features),
        current_schema=current_schema,
        current_data_scope=current_data_scope,
    )
