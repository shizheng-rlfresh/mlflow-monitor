"""System Finding-policy component required by the default Recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from mlflow_monitor.domain import (
    CompatibilityEvidence,
    Diff,
    FindingDraft,
    ReferenceComparisonCoverage,
)
from mlflow_monitor.finding_policy import (
    FrozenFindingPolicyParameters,
    JSONValue,
)

SYSTEM_COMPATIBILITY_FINDING_POLICY_ID = "system-compatibility-findings"
SYSTEM_COMPATIBILITY_FINDING_POLICY_VERSION = "v0"


@dataclass(frozen=True, slots=True)
class SystemCompatibilityFindingPolicy:
    """System policy registration used by the default Recipe.

    Attributes:
        finding_policy_id: Stable system policy identifier.
        finding_policy_version: Exact system policy version.
    """

    finding_policy_id: str = SYSTEM_COMPATIBILITY_FINDING_POLICY_ID
    finding_policy_version: str = SYSTEM_COMPATIBILITY_FINDING_POLICY_VERSION

    def validate_parameters(
        self,
        parameters: Mapping[str, JSONValue],
    ) -> FrozenFindingPolicyParameters:
        """Accept only the empty parameter mapping fixed by the v0 schema.

        Args:
            parameters: Structurally valid Recipe parameters.

        Returns:
            An immutable empty parameter mapping.

        Raises:
            ValueError: If any parameter is supplied.
        """
        if parameters:
            raise ValueError("The system compatibility Finding policy accepts no parameters.")
        return MappingProxyType({})

    def evaluate(
        self,
        *,
        parameters: FrozenFindingPolicyParameters,
        diffs: tuple[Diff, ...],
        compatibility_evidence: tuple[CompatibilityEvidence, ...],
        reference_comparison_coverage: tuple[ReferenceComparisonCoverage, ...],
    ) -> tuple[FindingDraft, ...]:
        """Fail closed until the compatibility interpretation is implemented.

        Args:
            parameters: Validated effective parameters.
            diffs: Current committed Metric Diffs.
            compatibility_evidence: Current committed Compatibility Evidence.
            reference_comparison_coverage: Current reference coverage.

        Raises:
            RuntimeError: Always, because Analyze integration is not yet available.
        """
        raise RuntimeError("System compatibility Finding policy evaluation is unavailable.")


SYSTEM_COMPATIBILITY_FINDING_POLICY = SystemCompatibilityFindingPolicy()
