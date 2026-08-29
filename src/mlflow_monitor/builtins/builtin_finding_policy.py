"""System Finding-policy component required by the default Recipe."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from mlflow_monitor.domain import (
    CompatibilityEvidence,
    ContractCheckReasonCode,
    Diff,
    FindingDraft,
    FindingSeverity,
    ReferenceComparisonCoverage,
)
from mlflow_monitor.finding_policy import (
    FrozenFindingPolicyParameters,
    JSONValue,
)

SYSTEM_COMPATIBILITY_FINDING_POLICY_ID = "system-compatibility-findings"
SYSTEM_COMPATIBILITY_FINDING_POLICY_VERSION = "v0"


_SYSTEM_COMPATIBILITY_FINDING_POLICY_CATEGORY = "compatibility"

_SYSTEM_COMPATIBILITY_FINDING_POLICY_FINDING_RULE_IDS = MappingProxyType(
    {
        ContractCheckReasonCode.ENV_MISMATCH: "compatibility.environment_mismatch",
        ContractCheckReasonCode.SCHEMA_MISMATCH: "compatibility.schema_mismatch",
        ContractCheckReasonCode.FEAT_MISMATCH: "compatibility.feature_mismatch",
        ContractCheckReasonCode.DATA_SCOPE_MISMATCH: "compatibility.data_scope_mismatch",
    }
)

_SYSTEM_COMPATIBILITY_FINDING_POLICY_RECOMMENDATIONS = MappingProxyType(
    {
        ContractCheckReasonCode.ENV_MISMATCH: (
            "Review the execution-environment differences and confirm that the current "
            "evidence is comparable with the baseline before relying on metric comparisons."
        ),
        ContractCheckReasonCode.SCHEMA_MISMATCH: (
            "Review the schema changes and either restore baseline-compatible data or "
            "intentionally update the Contract for a future Monitoring Run."
        ),
        ContractCheckReasonCode.FEAT_MISMATCH: (
            "Review the feature-set changes and either restore baseline-compatible features or "
            "intentionally update the Contract for a future Monitoring Run."
        ),
        ContractCheckReasonCode.DATA_SCOPE_MISMATCH: (
            "Confirm the intended data population and either restore the baseline-compatible "
            "scope or intentionally update the Contract for a future Monitoring Run."
        ),
    }
)


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
        """Map supported Compatibility Evidence to Finding drafts.

        Args:
            parameters: Validated empty parameters.
            diffs: Current committed Metric Diffs, which this policy ignores.
            compatibility_evidence: Current committed Compatibility Evidence.
            reference_comparison_coverage: Current reference coverage, which this
                policy ignores.

        Returns:
            One high-severity compatibility draft per evidence record.

        Raises:
            ValueError: If an evidence record has an unsupported reason code.
        """
        drafts = []

        for evidence in compatibility_evidence:
            if evidence.reason.code not in _SYSTEM_COMPATIBILITY_FINDING_POLICY_FINDING_RULE_IDS:
                raise ValueError(f"Unsupported compatibility reason code={evidence.reason.code!r}")

            drafts.append(
                FindingDraft(
                    finding_rule_id=_SYSTEM_COMPATIBILITY_FINDING_POLICY_FINDING_RULE_IDS[
                        cast(ContractCheckReasonCode, evidence.reason.code)
                    ],
                    severity=FindingSeverity.HIGH,
                    category=_SYSTEM_COMPATIBILITY_FINDING_POLICY_CATEGORY,
                    summary=evidence.reason.message,
                    recommendation=_SYSTEM_COMPATIBILITY_FINDING_POLICY_RECOMMENDATIONS[
                        cast(ContractCheckReasonCode, evidence.reason.code)
                    ],
                    evidence_diff_ids=(),
                    evidence_compatibility_ids=(evidence.compatibility_evidence_id,),
                )
            )

        return tuple(drafts)


SYSTEM_COMPATIBILITY_FINDING_POLICY = SystemCompatibilityFindingPolicy()
