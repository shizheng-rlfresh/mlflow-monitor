"""Specifications for the Finding policy extension boundary."""

from __future__ import annotations

from collections.abc import Mapping
from inspect import signature
from types import MappingProxyType

import pytest

from mlflow_monitor.domain import (
    CompatibilityEvidence,
    Diff,
    FindingDraft,
    FindingSeverity,
    ReferenceComparisonCoverage,
)
from mlflow_monitor.finding_policy import (
    FindingPolicy,
    FrozenFindingPolicyParameters,
    JSONValue,
)


class _ExamplePolicy:
    finding_policy_id = "relative-regression"
    finding_policy_version = "1"

    def validate_parameters(
        self,
        parameters: Mapping[str, JSONValue],
    ) -> FrozenFindingPolicyParameters:
        threshold = parameters.get("threshold", 0.05)
        return MappingProxyType({"threshold": threshold})

    def evaluate(
        self,
        *,
        parameters: FrozenFindingPolicyParameters,
        diffs: tuple[Diff, ...],
        compatibility_evidence: tuple[CompatibilityEvidence, ...],
        reference_comparison_coverage: tuple[ReferenceComparisonCoverage, ...],
    ) -> tuple[FindingDraft, ...]:
        assert parameters["threshold"] == 0.1
        assert diffs == ()
        assert compatibility_evidence == ()
        assert reference_comparison_coverage == ()
        return (
            FindingDraft(
                finding_rule_id="quality.accuracy_regression",
                severity=FindingSeverity.HIGH,
                category="quality",
                summary="Accuracy regressed against the baseline.",
                recommendation="Review the model and input changes.",
                evidence_diff_ids=("diff-v1-example",),
                evidence_compatibility_ids=(),
            ),
        )


def _accept_policy(policy: FindingPolicy) -> FindingPolicy:
    return policy


def test_finding_policy_exposes_identity_and_parameter_validation() -> None:
    policy = _accept_policy(_ExamplePolicy())

    parameters = policy.validate_parameters({"threshold": 0.1})

    assert policy.finding_policy_id == "relative-regression"
    assert policy.finding_policy_version == "1"
    assert parameters == {"threshold": 0.1}
    with pytest.raises(TypeError):
        parameters["threshold"] = 0.2  # type: ignore[index]


def test_finding_policy_evaluates_frozen_inputs_into_drafts() -> None:
    policy = _accept_policy(_ExamplePolicy())
    parameters = policy.validate_parameters({"threshold": 0.1})

    drafts = policy.evaluate(
        parameters=parameters,
        diffs=(),
        compatibility_evidence=(),
        reference_comparison_coverage=(),
    )

    assert isinstance(parameters, Mapping)
    assert isinstance(drafts, tuple)
    assert len(drafts) == 1
    assert isinstance(drafts[0], FindingDraft)


def test_finding_policy_evaluate_signature_names_every_immutable_input() -> None:
    assert tuple(signature(FindingPolicy.evaluate).parameters) == (
        "self",
        "parameters",
        "diffs",
        "compatibility_evidence",
        "reference_comparison_coverage",
    )
