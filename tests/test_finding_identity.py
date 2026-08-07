"""Specifications for deterministic Finding identities."""

import os
import subprocess
import sys
from dataclasses import replace

import pytest

import mlflow_monitor.identity as identity
import mlflow_monitor.invariant as invariant
from mlflow_monitor.domain import Finding, FindingSeverity
from mlflow_monitor.errors import InvariantViolation

FINDING_ID_FIXTURE = "finding-v1-aff16cb86b7d6bd262be4d52f1c9efaff92360afeed5220b161cd77d1589f84f"


def _identity_values() -> dict[str, object]:
    return {
        "monitoring_run_id": "monitoring-run-current",
        "source_run_id": "train-run-current",
        "finding_policy_id": "relative-regression",
        "finding_policy_version": "1",
        "finding_rule_id": "quality.accuracy_regression",
        "evidence_diff_ids": ("diff-v1-b", "diff-v1-a"),
        "evidence_compatibility_ids": (
            "compatibility-evidence-v1-b",
            "compatibility-evidence-v1-a",
        ),
    }


def _finding(**overrides: object) -> Finding:
    values = {
        **_identity_values(),
        "finding_id": FINDING_ID_FIXTURE,
        "severity": FindingSeverity.HIGH,
        "category": "quality",
        "summary": "Accuracy regressed against the baseline.",
        "recommendation": "Review the model and input changes.",
    }
    values.update(overrides)
    return Finding(**values)  # type: ignore[arg-type]


def test_finding_identity_container_serializes_sorted_semantic_components() -> None:
    finding_identity = identity._FindingIdentity(**_identity_values())  # type: ignore[arg-type]

    assert finding_identity.to_dict() == {
        "monitoring_run_id": "monitoring-run-current",
        "source_run_id": "train-run-current",
        "finding_policy_id": "relative-regression",
        "finding_policy_version": "1",
        "finding_rule_id": "quality.accuracy_regression",
        "evidence_diff_ids": ["diff-v1-a", "diff-v1-b"],
        "evidence_compatibility_ids": [
            "compatibility-evidence-v1-a",
            "compatibility-evidence-v1-b",
        ],
    }


def test_finding_id_matches_fixed_fixture() -> None:
    assert identity.make_finding_id(**_identity_values()) == FINDING_ID_FIXTURE  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("monitoring_run_id", "monitoring-run-other"),
        ("source_run_id", "train-run-other"),
        ("finding_policy_id", "absolute-threshold"),
        ("finding_policy_version", "2"),
        ("finding_rule_id", "quality.precision_regression"),
        ("evidence_diff_ids", ("diff-v1-c",)),
        (
            "evidence_compatibility_ids",
            ("compatibility-evidence-v1-c",),
        ),
    ],
)
def test_finding_id_changes_with_every_semantic_component(
    field: str,
    replacement: object,
) -> None:
    values = _identity_values()
    original_id = identity.make_finding_id(**values)  # type: ignore[arg-type]
    values[field] = replacement

    assert identity.make_finding_id(**values) != original_id  # type: ignore[arg-type]


def test_finding_id_is_independent_of_evidence_order() -> None:
    values = _identity_values()
    reordered = {
        **values,
        "evidence_diff_ids": tuple(reversed(values["evidence_diff_ids"])),  # type: ignore[arg-type]
        "evidence_compatibility_ids": tuple(
            reversed(values["evidence_compatibility_ids"])  # type: ignore[arg-type]
        ),
    }

    assert identity.make_finding_id(**values) == identity.make_finding_id(  # type: ignore[arg-type]
        **reordered  # type: ignore[arg-type]
    )


def test_finding_id_is_independent_of_repeated_evidence_identities() -> None:
    values = _identity_values()
    repeated = {
        **values,
        "evidence_diff_ids": (*values["evidence_diff_ids"], "diff-v1-a"),  # type: ignore[misc]
        "evidence_compatibility_ids": (
            *values["evidence_compatibility_ids"],  # type: ignore[misc]
            "compatibility-evidence-v1-a",
        ),
    }

    assert identity.make_finding_id(**values) == identity.make_finding_id(  # type: ignore[arg-type]
        **repeated  # type: ignore[arg-type]
    )


def test_finding_id_keeps_diff_and_compatibility_evidence_separate() -> None:
    first = identity.make_finding_id(
        monitoring_run_id="monitoring-run-current",
        source_run_id="train-run-current",
        finding_policy_id="policy",
        finding_policy_version="1",
        finding_rule_id="rule",
        evidence_diff_ids=("shared-id",),
        evidence_compatibility_ids=(),
    )
    second = identity.make_finding_id(
        monitoring_run_id="monitoring-run-current",
        source_run_id="train-run-current",
        finding_policy_id="policy",
        finding_policy_version="1",
        finding_rule_id="rule",
        evidence_diff_ids=(),
        evidence_compatibility_ids=("shared-id",),
    )

    assert first != second


def test_finding_id_encoding_is_delimiter_safe() -> None:
    first = identity.make_finding_id(
        monitoring_run_id="monitoring|source",
        source_run_id="training",
        finding_policy_id="policy",
        finding_policy_version="1",
        finding_rule_id="rule",
        evidence_diff_ids=("diff",),
        evidence_compatibility_ids=(),
    )
    second = identity.make_finding_id(
        monitoring_run_id="monitoring",
        source_run_id="source|training",
        finding_policy_id="policy",
        finding_policy_version="1",
        finding_rule_id="rule",
        evidence_diff_ids=("diff",),
        evidence_compatibility_ids=(),
    )

    assert first != second


@pytest.mark.parametrize("hash_seed", ["1", "8675309"])
def test_finding_id_fixture_is_stable_across_processes(hash_seed: str) -> None:
    script = """
from mlflow_monitor.identity import make_finding_id

print(make_finding_id(
    monitoring_run_id="monitoring-run-current",
    source_run_id="train-run-current",
    finding_policy_id="relative-regression",
    finding_policy_version="1",
    finding_rule_id="quality.accuracy_regression",
    evidence_diff_ids=("diff-v1-b", "diff-v1-a"),
    evidence_compatibility_ids=(
        "compatibility-evidence-v1-b",
        "compatibility-evidence-v1-a",
    ),
))
"""
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = hash_seed

    output = subprocess.check_output(  # noqa: S603
        [sys.executable, "-c", script],
        env=environment,
        text=True,
    )

    assert output.strip() == FINDING_ID_FIXTURE


def test_finding_identity_validation_accepts_the_derived_id() -> None:
    invariant.validate_finding_identity(_finding())


def test_finding_identity_validation_rejects_an_incorrect_id() -> None:
    finding = replace(_finding(), finding_id="finding-v1-incorrect")

    with pytest.raises(InvariantViolation) as exc_info:
        invariant.validate_finding_identity(finding)

    error = exc_info.value
    assert error.code == "finding_identity_mismatch"
    assert error.entity == "Finding"
    assert error.field == "finding_id"


def test_finding_identity_excludes_rendered_content() -> None:
    original = _finding()
    changed = replace(
        original,
        severity=FindingSeverity.CRITICAL,
        category="performance",
        summary="Changed summary under the same semantic identity.",
        recommendation="Changed recommendation.",
    )

    assert _make_finding_id_for(original) == _make_finding_id_for(changed)


def test_finding_identity_consistency_accepts_identical_retries() -> None:
    finding = _finding()

    invariant.validate_finding_identity_consistency((finding, finding))


def test_finding_identity_consistency_rejects_changed_rendered_content() -> None:
    original = _finding()
    changed = replace(
        original,
        summary="Changed summary under the same semantic identity.",
    )

    with pytest.raises(InvariantViolation) as exc_info:
        invariant.validate_finding_identity_consistency((original, changed))

    error = exc_info.value
    assert error.code == "finding_identity_content_conflict"
    assert error.entity == "Finding"
    assert error.field == "finding_id"


def _make_finding_id_for(finding: Finding) -> str:
    return identity.make_finding_id(
        monitoring_run_id=finding.monitoring_run_id,
        source_run_id=finding.source_run_id,
        finding_policy_id=finding.finding_policy_id,
        finding_policy_version=finding.finding_policy_version,
        finding_rule_id=finding.finding_rule_id,
        evidence_diff_ids=finding.evidence_diff_ids,
        evidence_compatibility_ids=finding.evidence_compatibility_ids,
    )
