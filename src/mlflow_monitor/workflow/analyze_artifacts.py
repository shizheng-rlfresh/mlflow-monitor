"""Typed output and canonical artifacts for the internal Analyze stage."""

from dataclasses import dataclass

from mlflow_monitor.domain import CompatibilityEvidence, Diff, Finding, ReferenceComparisonCoverage


@dataclass(frozen=True, slots=True)
class AnalyzeOutput:
    """Immutable Analyze output without independent identity or persistence.

    Attributes:
        compatibility_evidence: Observations copied from committed Check reasons.
        diffs: Atomic metric comparisons in reference/metric order.
        reference_comparison_coverage: One group per planned reference.
        findings: Materialized conclusions in deterministic identity order.
    """

    compatibility_evidence: tuple[CompatibilityEvidence, ...]
    diffs: tuple[Diff, ...]
    reference_comparison_coverage: tuple[ReferenceComparisonCoverage, ...]
    findings: tuple[Finding, ...]

    def __post_init__(self) -> None:
        """Defensively freeze the output collections."""
        for field in (
            "compatibility_evidence",
            "diffs",
            "reference_comparison_coverage",
            "findings",
        ):
            object.__setattr__(self, field, tuple(getattr(self, field)))
