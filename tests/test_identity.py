"""Specifications for deterministic evidence identities."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace

import pytest

from mlflow_monitor.domain import (
    CompatibilityEvidence,
    ContractCheckReason,
    Diff,
    DiffReference,
    DiffReferenceKind,
)
from mlflow_monitor.errors import InvariantViolation
from mlflow_monitor.identity import make_compatibility_evidence_id, make_diff_id
from mlflow_monitor.invariant import (
    validate_compatibility_evidence_identity,
    validate_compatibility_evidence_identity_consistency,
    validate_diff_identity,
    validate_diff_identity_consistency,
)

DIFF_ID_FIXTURE = "diff-v1-ea16f63f7055927b29f5a1b14fc5fefb3b80326d682e327df49a0164d4f19db6"
COMPATIBILITY_EVIDENCE_ID_FIXTURE = (
    "compatibility-evidence-v1-6468a0e4e6cbde077484687bfd2e45b64e1d0ce196f41a69ae845f7ac836d1b5"
)


def _baseline_reference() -> DiffReference:
    return DiffReference(
        kind=DiffReferenceKind.BASELINE,
        monitoring_run_id=None,
        source_run_id="train-run-baseline",
    )


def _previous_reference() -> DiffReference:
    return DiffReference(
        kind=DiffReferenceKind.PREVIOUS,
        monitoring_run_id="monitoring-run-previous",
        source_run_id="train-run-previous",
    )


def _diff(**overrides: object) -> Diff:
    values: dict[str, object] = {
        "diff_id": DIFF_ID_FIXTURE,
        "monitoring_run_id": "monitoring-run-current",
        "source_run_id": "train-run-current",
        "reference": _baseline_reference(),
        "metric_name": "accuracy",
        "current_value": 0.75,
        "reference_value": 0.5,
        "delta": 0.25,
    }
    values.update(overrides)
    return Diff(**values)  # type: ignore[arg-type]


def _compatibility_evidence(
    *,
    reason: ContractCheckReason | None = None,
) -> CompatibilityEvidence:
    return CompatibilityEvidence(
        compatibility_evidence_id=COMPATIBILITY_EVIDENCE_ID_FIXTURE,
        monitoring_run_id="monitoring-run-current",
        source_run_id="train-run-current",
        baseline_source_run_id="train-run-baseline",
        contract_id="default_permissive",
        contract_version="v0",
        reason=reason
        or ContractCheckReason(
            code="environment_mismatch",
            message="Execution environment does not match the baseline.",
            blocking=False,
        ),
    )


def test_diff_id_matches_fixed_fixture() -> None:
    assert (
        make_diff_id(
            monitoring_run_id="monitoring-run-current",
            source_run_id="train-run-current",
            reference=_baseline_reference(),
            metric_name="accuracy",
        )
        == DIFF_ID_FIXTURE
    )


def test_compatibility_evidence_id_matches_fixed_fixture() -> None:
    assert (
        make_compatibility_evidence_id(
            monitoring_run_id="monitoring-run-current",
            source_run_id="train-run-current",
            baseline_source_run_id="train-run-baseline",
            contract_id="default_permissive",
            contract_version="v0",
            reason_code="environment_mismatch",
        )
        == COMPATIBILITY_EVIDENCE_ID_FIXTURE
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"monitoring_run_id": "monitoring-run-other"},
        {"source_run_id": "train-run-other"},
        {"reference": _baseline_reference()},
        {
            "reference": DiffReference(
                kind=DiffReferenceKind.PREVIOUS,
                monitoring_run_id="monitoring-run-other",
                source_run_id="train-run-previous",
            )
        },
        {
            "reference": DiffReference(
                kind=DiffReferenceKind.PREVIOUS,
                monitoring_run_id="monitoring-run-previous",
                source_run_id="train-run-other",
            )
        },
        {"metric_name": "precision"},
    ],
)
def test_diff_id_changes_with_every_semantic_identity_component(
    overrides: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "monitoring_run_id": "monitoring-run-current",
        "source_run_id": "train-run-current",
        "reference": _previous_reference(),
        "metric_name": "accuracy",
    }
    original_id = make_diff_id(**values)  # type: ignore[arg-type]
    values.update(overrides)

    assert make_diff_id(**values) != original_id  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    [
        "monitoring_run_id",
        "source_run_id",
        "baseline_source_run_id",
        "contract_id",
        "contract_version",
        "reason_code",
    ],
)
def test_compatibility_evidence_id_changes_with_every_semantic_component(field: str) -> None:
    values = {
        "monitoring_run_id": "monitoring-run-current",
        "source_run_id": "train-run-current",
        "baseline_source_run_id": "train-run-baseline",
        "contract_id": "default_permissive",
        "contract_version": "v0",
        "reason_code": "environment_mismatch",
    }
    original_id = make_compatibility_evidence_id(**values)
    values[field] = f"different-{field}"

    assert make_compatibility_evidence_id(**values) != original_id


def test_diff_id_excludes_numeric_evidence_values() -> None:
    original = _diff()
    changed = replace(
        original,
        current_value=0.9,
        reference_value=0.4,
        delta=0.5,
    )

    assert _make_diff_id_for(original) == _make_diff_id_for(changed)


def test_compatibility_evidence_id_excludes_reason_content() -> None:
    original = _compatibility_evidence()
    changed = _compatibility_evidence(
        reason=ContractCheckReason(
            code=original.reason.code,
            message="Changed message under the same semantic identity.",
            blocking=True,
        )
    )

    assert _make_compatibility_evidence_id_for(original) == (
        _make_compatibility_evidence_id_for(changed)
    )


def test_canonical_encoding_does_not_conflate_delimited_values() -> None:
    first = make_compatibility_evidence_id(
        monitoring_run_id="monitoring|source",
        source_run_id="training",
        baseline_source_run_id="baseline",
        contract_id="contract",
        contract_version="v0",
        reason_code="environment_mismatch",
    )
    second = make_compatibility_evidence_id(
        monitoring_run_id="monitoring",
        source_run_id="source|training",
        baseline_source_run_id="baseline",
        contract_id="contract",
        contract_version="v0",
        reason_code="environment_mismatch",
    )

    assert first != second


@pytest.mark.parametrize("hash_seed", ["1", "8675309"])
def test_identity_fixtures_are_stable_across_processes(hash_seed: str) -> None:
    script = """
from mlflow_monitor.domain import DiffReference, DiffReferenceKind
from mlflow_monitor.identity import make_compatibility_evidence_id, make_diff_id

reference = DiffReference(
    kind=DiffReferenceKind.BASELINE,
    monitoring_run_id=None,
    source_run_id="train-run-baseline",
)
print(make_diff_id(
    monitoring_run_id="monitoring-run-current",
    source_run_id="train-run-current",
    reference=reference,
    metric_name="accuracy",
))
print(make_compatibility_evidence_id(
    monitoring_run_id="monitoring-run-current",
    source_run_id="train-run-current",
    baseline_source_run_id="train-run-baseline",
    contract_id="default_permissive",
    contract_version="v0",
    reason_code="environment_mismatch",
))
"""
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = hash_seed

    output = subprocess.check_output(  # noqa: S603
        [sys.executable, "-c", script],
        env=environment,
        text=True,
    )

    assert output.splitlines() == [DIFF_ID_FIXTURE, COMPATIBILITY_EVIDENCE_ID_FIXTURE]


def test_diff_identity_validation_accepts_derived_id() -> None:
    validate_diff_identity(_diff())


def test_diff_identity_validation_rejects_incorrect_id() -> None:
    with pytest.raises(InvariantViolation) as exc_info:
        validate_diff_identity(_diff(diff_id="diff-v1-incorrect"))

    error = exc_info.value
    assert error.code == "diff_identity_mismatch"
    assert error.entity == "Diff"
    assert error.field == "diff_id"


def test_compatibility_evidence_identity_validation_accepts_derived_id() -> None:
    validate_compatibility_evidence_identity(_compatibility_evidence())


def test_compatibility_evidence_identity_validation_rejects_incorrect_id() -> None:
    evidence = replace(
        _compatibility_evidence(),
        compatibility_evidence_id="compatibility-evidence-v1-incorrect",
    )

    with pytest.raises(InvariantViolation) as exc_info:
        validate_compatibility_evidence_identity(evidence)

    error = exc_info.value
    assert error.code == "compatibility_evidence_identity_mismatch"
    assert error.entity == "CompatibilityEvidence"
    assert error.field == "compatibility_evidence_id"


def test_diff_identity_consistency_accepts_identical_retries() -> None:
    diff = _diff()

    validate_diff_identity_consistency((diff, diff))


def test_diff_identity_consistency_rejects_changed_numeric_content() -> None:
    original = _diff()
    changed = replace(
        original,
        current_value=0.9,
        reference_value=0.4,
        delta=0.5,
    )

    with pytest.raises(InvariantViolation) as exc_info:
        validate_diff_identity_consistency((original, changed))

    error = exc_info.value
    assert error.code == "diff_identity_content_conflict"
    assert error.entity == "Diff"
    assert error.field == "diff_id"


def test_compatibility_evidence_identity_consistency_accepts_identical_retries() -> None:
    evidence = _compatibility_evidence()

    validate_compatibility_evidence_identity_consistency((evidence, evidence))


@pytest.mark.parametrize(
    "reason",
    [
        ContractCheckReason(
            code="environment_mismatch",
            message="Changed message under the same semantic identity.",
            blocking=False,
        ),
        ContractCheckReason(
            code="environment_mismatch",
            message="Execution environment does not match the baseline.",
            blocking=True,
        ),
    ],
)
def test_compatibility_evidence_identity_consistency_rejects_changed_reason_content(
    reason: ContractCheckReason,
) -> None:
    original = _compatibility_evidence()
    changed = _compatibility_evidence(reason=reason)

    with pytest.raises(InvariantViolation) as exc_info:
        validate_compatibility_evidence_identity_consistency((original, changed))

    error = exc_info.value
    assert error.code == "compatibility_evidence_identity_content_conflict"
    assert error.entity == "CompatibilityEvidence"
    assert error.field == "compatibility_evidence_id"


def _make_diff_id_for(diff: Diff) -> str:
    return make_diff_id(
        monitoring_run_id=diff.monitoring_run_id,
        source_run_id=diff.source_run_id,
        reference=diff.reference,
        metric_name=diff.metric_name,
    )


def _make_compatibility_evidence_id_for(evidence: CompatibilityEvidence) -> str:
    return make_compatibility_evidence_id(
        monitoring_run_id=evidence.monitoring_run_id,
        source_run_id=evidence.source_run_id,
        baseline_source_run_id=evidence.baseline_source_run_id,
        contract_id=evidence.contract_id,
        contract_version=evidence.contract_version,
        reason_code=evidence.reason.code,
    )
