"""Workflow lifecycle helpers for MLflow-Monitor v0.

This module contains backend-agnostic workflow logic for two responsibilities:

1. Prepare-stage context resolution before contract checking begins.
2. Contract checking and evaluation after prepare-stage context resolution.

Prepare-stage resolution combines caller inputs (run identity, compiled plan,
resolved contract, optional first-run baseline input) with gateway-resolved
state (timeline, source run, prior monitoring runs, and optional references).
The workflow layer decides what must be resolved for a run to proceed, while
the gateway owns all persistence-specific mechanics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from mlflow_monitor.contract_checker import (
    ContractChecker,
    make_contract_evaluation_context,
)
from mlflow_monitor.domain import (
    Contract,
    ContractCheckResult,
    DiffReferenceKind,
    LifecycleStatus,
    MonitoringRunReference,
)
from mlflow_monitor.errors import (
    PREPARE_BASELINE_OVERRIDE_EXISTING_BASELINE,
    CheckStageError,
    InvariantViolation,
    PreparedContextConsistencyViolation,
    PrepareStageError,
)
from mlflow_monitor.gateway import MonitoringGateway
from mlflow_monitor.gateway_models import TimelineState
from mlflow_monitor.invariant import validate_contract_check_result
from mlflow_monitor.recipe_compiler import CompiledRecipe, EffectiveRecipePlan
from mlflow_monitor.utils import canonical_json

PREPARED_CONTEXT_ARTIFACT_PATH = "state/prepared_context.json"
_PREPARED_CONTEXT_ARTIFACT_SCHEMA_VERSION = "v0"
_PREPARED_CONTEXT_FIELDS = frozenset(
    {
        "artifact_schema_version",
        "monitoring_run_id",
        "source_run_id",
        "subject_id",
        "timeline_id",
        "sequence_index",
        "baseline_source_run_id",
        "effective_recipe",
        "contract",
        "references",
    }
)
_PREPARED_CONTRACT_FIELDS = frozenset(
    {
        "contract_id",
        "contract_version",
        "schema_contract_ref",
        "feature_contract_ref",
        "metric_contract_ref",
        "data_scope_contract_ref",
        "execution_contract_ref",
    }
)
_PREPARED_REFERENCE_FIELDS = frozenset(
    {"kind", "monitoring_run_id", "source_run_id", "unavailable_reason"}
)
_REFERENCE_ORDER = {
    DiffReferenceKind.BASELINE: 0,
    DiffReferenceKind.PREVIOUS: 1,
    DiffReferenceKind.LKG: 2,
    DiffReferenceKind.CUSTOM: 3,
}
_PREPARE_REFERENCE_UNAVAILABLE_REASONS = {
    DiffReferenceKind.PREVIOUS: frozenset({"previous_reference_missing"}),
    DiffReferenceKind.LKG: frozenset(
        {
            "lkg_not_selected",
            "lkg_selection_inconsistent",
        }
    ),
}


@dataclass(frozen=True, slots=True)
class BaselineResolutionResult:
    """Result of baseline source run resolution for prepare-stage context.

    Attributes:
        baseline_source_run_id: Resolved baseline source run id.
    """

    baseline_source_run_id: str


@dataclass(frozen=True, slots=True)
class PreparedReferencePlanEntry:
    """One fixed reference group frozen during Prepare.

    Attributes:
        kind: Canonical reference kind for this group.
        reference: Resolved paired reference, or ``None`` when unavailable.
        unavailable_reason: Prepare-time reason when the reference is unavailable.
    """

    kind: DiffReferenceKind
    reference: MonitoringRunReference | None
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        """Validate resolved and unavailable plan-entry shapes."""
        try:
            kind = DiffReferenceKind(self.kind)
        except ValueError as exc:
            raise ValueError(f"Unsupported prepared reference kind {self.kind!r}.") from exc
        object.__setattr__(self, "kind", kind)

        if self.reference is not None:
            if not isinstance(self.reference, MonitoringRunReference):
                raise ValueError("Resolved prepared reference must be a MonitoringRunReference.")
            if self.reference.kind is not kind:
                raise ValueError("Prepared reference kind must match its resolved reference kind.")
            if self.unavailable_reason is not None:
                raise ValueError("Resolved prepared reference cannot have an unavailable reason.")
            return

        allowed_reasons = _PREPARE_REFERENCE_UNAVAILABLE_REASONS.get(kind, frozenset())
        if self.unavailable_reason not in allowed_reasons:
            raise ValueError(
                f"Unavailable {kind.value!r} prepared reference requires one of "
                f"{tuple(sorted(allowed_reasons))!r}."
            )

    def to_dict(self) -> dict[str, str | None]:
        """Serialize this fixed reference group deterministically."""
        reference = self.reference
        return {
            "kind": self.kind.value,
            "monitoring_run_id": None if reference is None else reference.monitoring_run_id,
            "source_run_id": None if reference is None else reference.source_run_id,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class PreparedContext:
    """Resolved prepare-stage context required before contract checking.

    Attributes:
        monitoring_run_id: Stable monitoring run identifier.
        source_run_id: Resolved source training run id.
        subject_id: Stable monitored subject identifier.
        timeline_id: Stable timeline identifier.
        sequence_index: Sequence index within the timeline.
        baseline_source_run_id: Resolved baseline source run id.
        effective_recipe: Resolved effective compiled Recipe.
        contract: Resolved contract.
        reference_plan: Fixed ordered reference plan, including unavailable groups.
    """

    monitoring_run_id: str
    source_run_id: str
    subject_id: str
    timeline_id: str
    sequence_index: int
    baseline_source_run_id: str
    effective_recipe: EffectiveRecipePlan
    contract: Contract
    reference_plan: tuple[PreparedReferencePlanEntry, ...]

    def __post_init__(self) -> None:
        """Freeze and validate the canonical ordered reference plan."""
        reference_plan = tuple(self.reference_plan)
        expected_kinds = (
            DiffReferenceKind.BASELINE,
            DiffReferenceKind.PREVIOUS,
            DiffReferenceKind.LKG,
        )
        actual_kinds = tuple(entry.kind for entry in reference_plan)
        if actual_kinds not in (expected_kinds, (*expected_kinds, DiffReferenceKind.CUSTOM)):
            raise ValueError(
                "PreparedContext reference_plan must contain baseline, previous, LKG, "
                "and optional custom groups in canonical order."
            )
        baseline_reference = reference_plan[0].reference
        if baseline_reference != MonitoringRunReference(
            kind=DiffReferenceKind.BASELINE,
            monitoring_run_id=None,
            source_run_id=self.baseline_source_run_id,
        ):
            raise ValueError(
                "PreparedContext baseline reference must match baseline_source_run_id."
            )
        object.__setattr__(self, "reference_plan", reference_plan)

    @property
    def references(self) -> tuple[MonitoringRunReference, ...]:
        """Return resolved references from the frozen plan in canonical order."""
        return tuple(
            entry.reference for entry in self.reference_plan if entry.reference is not None
        )

    @property
    def recipe_id(self) -> str:
        """Return the normalized Recipe identifier."""
        return self.effective_recipe.identity.recipe_id

    @property
    def recipe_version(self) -> str:
        """Return the normalized Recipe version."""
        return self.effective_recipe.identity.recipe_version

    @property
    def contract_id(self) -> str:
        """Return the resolved Contract identifier."""
        return self.contract.contract_id

    @property
    def source_experiment(self) -> str | None:
        """Return the normalized optional Source Training Run experiment filter."""
        return self.effective_recipe.source_requirements.source_experiment

    @property
    def required_metrics(self) -> tuple[str, ...]:
        """Return normalized required metric names."""
        return self.effective_recipe.source_requirements.required_metric_names

    @property
    def required_artifacts(self) -> tuple[str, ...]:
        """Return normalized required artifact paths."""
        return self.effective_recipe.source_requirements.required_artifact_paths

    @property
    def previous_monitoring_run_id(self) -> str | None:
        """Return the frozen previous Monitoring Run identifier, when resolved."""
        return self._reference_monitoring_run_id(DiffReferenceKind.PREVIOUS)

    @property
    def active_lkg_monitoring_run_id(self) -> str | None:
        """Return the frozen LKG Monitoring Run identifier, when resolved."""
        return self._reference_monitoring_run_id(DiffReferenceKind.LKG)

    @property
    def custom_reference_monitoring_run_id(self) -> str | None:
        """Return the frozen custom Monitoring Run identifier, when resolved."""
        return self._reference_monitoring_run_id(DiffReferenceKind.CUSTOM)

    def _reference_monitoring_run_id(self, kind: DiffReferenceKind) -> str | None:
        """Return one resolved Monitoring Run identifier by reference kind."""
        for reference in self.references:
            if reference.kind is kind:
                return reference.monitoring_run_id
        return None


def prepared_context_to_dict(context: PreparedContext) -> dict[str, object]:
    """Serialize one prepared context into its canonical artifact payload.

    Args:
        context: Typed prepared context to persist.

    Returns:
        JSON-compatible prepared-context artifact content.
    """
    contract = context.contract
    return {
        "artifact_schema_version": _PREPARED_CONTEXT_ARTIFACT_SCHEMA_VERSION,
        "monitoring_run_id": context.monitoring_run_id,
        "source_run_id": context.source_run_id,
        "subject_id": context.subject_id,
        "timeline_id": context.timeline_id,
        "sequence_index": context.sequence_index,
        "baseline_source_run_id": context.baseline_source_run_id,
        "effective_recipe": context.effective_recipe.to_dict(),
        "contract": {
            "contract_id": contract.contract_id,
            "contract_version": contract.contract_version,
            "schema_contract_ref": contract.schema_contract_ref,
            "feature_contract_ref": contract.feature_contract_ref,
            "metric_contract_ref": contract.metric_contract_ref,
            "data_scope_contract_ref": contract.data_scope_contract_ref,
            "execution_contract_ref": contract.execution_contract_ref,
        },
        "references": [entry.to_dict() for entry in context.reference_plan],
    }


def hydrate_prepared_context(
    raw: Mapping[str, object] | None,
    *,
    compiled_recipe: CompiledRecipe,
    monitoring_run_id: str,
    source_run_id: str,
    subject_id: str,
    timeline_id: str,
    sequence_index: int,
) -> PreparedContext:
    """Hydrate and validate one committed prepared-context artifact.

    Args:
        raw: Decoded prepared-context JSON object, or ``None`` when missing.
        compiled_recipe: Caller-supplied executable Recipe whose effective plan
            must exactly match the persisted plan.
        monitoring_run_id: Allocated Monitoring Run identity expected in the artifact.
        source_run_id: Immutable Source Training Run identity expected in the artifact.
        subject_id: Subject identity expected in the artifact.
        timeline_id: Allocated Timeline identity expected in the artifact.
        sequence_index: Allocated Timeline sequence expected in the artifact.

    Returns:
        Typed context reconstructed without live Prepare resolution.

    Raises:
        GatewayConsistencyViolation: If the artifact is malformed or contradicts
            the current allocation or compiled Recipe.
    """
    if raw is None:
        raise PreparedContextConsistencyViolation.missing_artifact(
            field="prepared_context",
        )
    _require_exact_prepared_fields(raw, _PREPARED_CONTEXT_FIELDS, section="prepared_context")
    if raw.get("artifact_schema_version") != _PREPARED_CONTEXT_ARTIFACT_SCHEMA_VERSION:
        raise PreparedContextConsistencyViolation.unsupported_artifact_schema_version(
            field="artifact_schema_version",
        )

    expected_identity: tuple[tuple[str, str | int], ...] = (
        ("monitoring_run_id", monitoring_run_id),
        ("source_run_id", source_run_id),
        ("subject_id", subject_id),
        ("timeline_id", timeline_id),
        ("sequence_index", sequence_index),
    )
    for field, expected in expected_identity:
        actual = raw.get(field)
        if isinstance(expected, int):
            valid_type = isinstance(actual, int) and not isinstance(actual, bool)
        else:
            valid_type = isinstance(actual, str) and bool(actual.strip())
        if not valid_type or actual != expected:
            raise PreparedContextConsistencyViolation.allocation_identity_mismatch(
                field=field,
            )

    baseline_source_run_id = _require_prepared_string(raw, "baseline_source_run_id")

    effective_recipe = raw.get("effective_recipe")
    if not isinstance(effective_recipe, Mapping):
        raise PreparedContextConsistencyViolation.invalid_field_type(
            field="effective_recipe",
        )

    try:
        persisted = canonical_json(dict(effective_recipe))
        expected = canonical_json(compiled_recipe.effective_plan.to_dict())
    except (TypeError, ValueError) as exc:
        raise PreparedContextConsistencyViolation.invalid_field_type(
            field="effective_recipe",
        ) from exc

    if persisted != expected:
        raise PreparedContextConsistencyViolation.effective_recipe_mismatch(
            field="effective_recipe",
        )

    contract = _hydrate_prepared_contract(raw.get("contract"))
    if contract != compiled_recipe.contract:
        raise PreparedContextConsistencyViolation.contract_mismatch(
            field="contract",
        )

    reference_plan = _hydrate_prepared_reference_plan(
        raw.get("references"),
        baseline_source_run_id=baseline_source_run_id,
    )
    try:
        return PreparedContext(
            monitoring_run_id=monitoring_run_id,
            source_run_id=source_run_id,
            subject_id=subject_id,
            timeline_id=timeline_id,
            sequence_index=sequence_index,
            baseline_source_run_id=baseline_source_run_id,
            effective_recipe=compiled_recipe.effective_plan,
            contract=contract,
            reference_plan=reference_plan,
        )
    except ValueError as exc:
        raise PreparedContextConsistencyViolation.noncanonical_references(
            field="references",
        ) from exc


def _hydrate_prepared_contract(raw: object) -> Contract:
    """Hydrate the complete Contract frozen in prepared state."""
    if not isinstance(raw, Mapping):
        raise PreparedContextConsistencyViolation.invalid_field_type(
            field="contract",
        )
    _require_exact_prepared_fields(raw, _PREPARED_CONTRACT_FIELDS, section="contract")
    return Contract(
        contract_id=_require_prepared_string(raw, "contract_id"),
        contract_version=_require_prepared_string(raw, "contract_version"),
        schema_contract_ref=_require_prepared_optional_string(raw, "schema_contract_ref"),
        feature_contract_ref=_require_prepared_optional_string(raw, "feature_contract_ref"),
        metric_contract_ref=_require_prepared_optional_string(raw, "metric_contract_ref"),
        data_scope_contract_ref=_require_prepared_optional_string(
            raw,
            "data_scope_contract_ref",
        ),
        execution_contract_ref=_require_prepared_optional_string(raw, "execution_contract_ref"),
    )


def _hydrate_prepared_reference_plan(
    raw: object,
    *,
    baseline_source_run_id: str,
) -> tuple[PreparedReferencePlanEntry, ...]:
    """Hydrate the canonical fixed Prepare-stage reference plan."""
    if not isinstance(raw, list):
        raise PreparedContextConsistencyViolation.invalid_field_type(
            field="references",
        )

    reference_plan: list[PreparedReferencePlanEntry] = []
    for index, item in enumerate(raw):
        field = f"references[{index}]"
        if not isinstance(item, Mapping):
            raise PreparedContextConsistencyViolation.invalid_field_type(field=field)
        _require_exact_prepared_fields(item, _PREPARED_REFERENCE_FIELDS, section=field)
        kind = item.get("kind")
        monitoring_run_id = item.get("monitoring_run_id")
        source_run_id = item.get("source_run_id")
        unavailable_reason = item.get("unavailable_reason")
        if not isinstance(kind, str):
            raise PreparedContextConsistencyViolation.invalid_field_type(field=f"{field}.kind")
        if monitoring_run_id is not None and not isinstance(monitoring_run_id, str):
            raise PreparedContextConsistencyViolation.invalid_field_type(
                field=f"{field}.monitoring_run_id"
            )
        if source_run_id is not None and not isinstance(source_run_id, str):
            raise PreparedContextConsistencyViolation.invalid_field_type(
                field=f"{field}.source_run_id"
            )
        if unavailable_reason is not None and not isinstance(unavailable_reason, str):
            raise PreparedContextConsistencyViolation.invalid_field_type(
                field=f"{field}.unavailable_reason"
            )
        try:
            reference_kind = DiffReferenceKind(kind)
            reference = (
                None
                if source_run_id is None
                else MonitoringRunReference(
                    kind=reference_kind,
                    monitoring_run_id=monitoring_run_id,
                    source_run_id=source_run_id,
                )
            )
            entry = PreparedReferencePlanEntry(
                kind=reference_kind,
                reference=reference,
                unavailable_reason=unavailable_reason,
            )
        except ValueError as exc:
            raise PreparedContextConsistencyViolation.invalid_reference(field=field) from exc
        reference_plan.append(entry)

    if not reference_plan or reference_plan[0].reference != MonitoringRunReference(
        kind=DiffReferenceKind.BASELINE,
        monitoring_run_id=None,
        source_run_id=baseline_source_run_id,
    ):
        raise PreparedContextConsistencyViolation.baseline_reference_mismatch(
            field="references",
        )

    reference_kinds = tuple(entry.kind for entry in reference_plan)
    if (
        len(reference_kinds) != len(set(reference_kinds))
        or tuple(sorted(reference_kinds, key=_REFERENCE_ORDER.__getitem__)) != reference_kinds
    ):
        raise PreparedContextConsistencyViolation.noncanonical_references(
            field="references",
        )
    return tuple(reference_plan)


def _require_exact_prepared_fields(
    raw: Mapping[str, object],
    expected: frozenset[str],
    *,
    section: str,
) -> None:
    """Require one prepared-artifact mapping to have exactly its canonical fields."""
    if set(raw) != expected:
        raise PreparedContextConsistencyViolation.invalid_fields(field=section)


def _require_prepared_string(raw: Mapping[str, object], field: str) -> str:
    """Return one required nonempty string from a prepared artifact mapping."""
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise PreparedContextConsistencyViolation.invalid_field_type(field=field)
    return value


def _require_prepared_optional_string(
    raw: Mapping[str, object],
    field: str,
) -> str | None:
    """Return one optional string from a prepared artifact mapping."""
    value = raw.get(field)
    if value is not None and not isinstance(value, str):
        raise PreparedContextConsistencyViolation.invalid_field_type(field=field)
    return value


def prepare_run_context(
    *,
    monitoring_run_id: str,
    subject_id: str,
    compiled_recipe: CompiledRecipe,
    gateway: MonitoringGateway,
    source_run_id: str,
    sequence_index: int,
    baseline_source_run_id: str | None = None,
    custom_reference_monitoring_run_id: str | None = None,
) -> PreparedContext:
    """Resolve prepare-stage references and validate required source-run inputs.

    Args:
        monitoring_run_id: Monitoring run identifier being prepared.
        subject_id: Stable monitored subject identifier.
        compiled_recipe: Execution-ready compiled Recipe.
        gateway: Gateway used for timeline and source-run reads.
        source_run_id: Invocation-owned Source Training Run identifier.
        sequence_index: Stable allocation order within the Timeline.
        baseline_source_run_id: Optional baseline source run id used to initialize or
            explicitly confirm the immutable timeline baseline.
        custom_reference_monitoring_run_id: Optional invocation-owned Monitoring
            Run selected as a custom reference.

    Raises:
        PrepareStageError: If required prepare-stage references or inputs are missing.

    Returns:
        Success-only prepared context for later workflow stages.
    """
    timeline_state = gateway.get_timeline_state(subject_id)
    baseline_resolution_result = _resolve_baseline_for_prepare(
        subject_id=subject_id,
        compiled_recipe=compiled_recipe,
        gateway=gateway,
        timeline_state=timeline_state,
        baseline_source_run_id=baseline_source_run_id,
    )

    resolved_source_run_id = gateway.resolve_source_run_id(
        subject_id=subject_id,
        source_experiment=compiled_recipe.source_requirements.source_experiment,
        source_run_id=source_run_id,
    )
    if resolved_source_run_id is None:
        raise PrepareStageError(
            code="prepare_source_run_not_found",
            message=(
                "Source training run could not be resolved for "
                f"subject_id={subject_id} and source_run_id={source_run_id!r}."
            ),
            details=(("subject_id", subject_id),),
        )

    missing_metrics = gateway.get_missing_source_run_metrics(
        source_run_id=resolved_source_run_id,
        required_metrics=compiled_recipe.source_requirements.required_metric_names,
    )
    if missing_metrics:
        missing_metric = missing_metrics[0]
        raise PrepareStageError(
            code="prepare_missing_required_metric",
            message=(
                f"Source run {resolved_source_run_id} is missing required metric {missing_metric}."
            ),
            details=(("source_run_id", resolved_source_run_id), ("metric", missing_metric)),
        )

    missing_artifacts = gateway.get_missing_source_run_artifacts(
        source_run_id=resolved_source_run_id,
        required_artifacts=compiled_recipe.source_requirements.required_artifact_paths,
    )
    if missing_artifacts:
        missing_artifact = missing_artifacts[0]
        raise PrepareStageError(
            code="prepare_missing_required_artifact",
            message=(
                f"Source run {resolved_source_run_id} is missing required artifact "
                f"{missing_artifact}."
            ),
            details=(
                ("source_run_id", resolved_source_run_id),
                ("artifact", missing_artifact),
            ),
        )

    custom_reference: MonitoringRunReference | None = None
    if custom_reference_monitoring_run_id is not None:
        resolved_custom_monitoring_run_id = gateway.resolve_timeline_monitoring_run_id(
            subject_id,
            custom_reference_monitoring_run_id,
        )
        if resolved_custom_monitoring_run_id is None:
            raise PrepareStageError(
                code="prepare_custom_reference_not_found",
                message=(
                    "Custom reference monitoring run could not be resolved on the subject timeline."
                ),
                details=(("subject_id", subject_id),),
            )
        custom_record = gateway.get_monitoring_run(
            subject_id,
            resolved_custom_monitoring_run_id,
        )
        if custom_record is None:
            raise PrepareStageError(
                code="prepare_custom_reference_not_found",
                message=(
                    "Custom reference monitoring run could not be resolved on the subject timeline."
                ),
                details=(("subject_id", subject_id),),
            )
        if custom_record.lifecycle_status is not LifecycleStatus.CLOSED:
            raise PrepareStageError(
                code="prepare_custom_reference_not_closed",
                message="Custom reference Monitoring Run must be closed.",
                details=(
                    (
                        "custom_reference_monitoring_run_id",
                        resolved_custom_monitoring_run_id,
                    ),
                ),
            )
        custom_reference = MonitoringRunReference(
            kind=DiffReferenceKind.CUSTOM,
            monitoring_run_id=custom_record.monitoring_run_id,
            source_run_id=custom_record.source_run_id,
        )

    reference_plan = [
        PreparedReferencePlanEntry(
            kind=DiffReferenceKind.BASELINE,
            reference=MonitoringRunReference(
                kind=DiffReferenceKind.BASELINE,
                monitoring_run_id=None,
                source_run_id=baseline_resolution_result.baseline_source_run_id,
            ),
            unavailable_reason=None,
        )
    ]

    previous = max(
        (
            record
            for record in gateway.list_timeline_monitoring_runs(subject_id)
            if record.lifecycle_status is LifecycleStatus.CLOSED
            and record.sequence_index < sequence_index
        ),
        key=lambda record: record.sequence_index,
        default=None,
    )
    if previous is not None:
        reference_plan.append(
            PreparedReferencePlanEntry(
                kind=DiffReferenceKind.PREVIOUS,
                reference=MonitoringRunReference(
                    kind=DiffReferenceKind.PREVIOUS,
                    monitoring_run_id=previous.monitoring_run_id,
                    source_run_id=previous.source_run_id,
                ),
                unavailable_reason=None,
            )
        )
    else:
        reference_plan.append(
            PreparedReferencePlanEntry(
                kind=DiffReferenceKind.PREVIOUS,
                reference=None,
                unavailable_reason="previous_reference_missing",
            )
        )

    reference_plan.append(
        PreparedReferencePlanEntry(
            kind=DiffReferenceKind.LKG,
            reference=None,
            unavailable_reason="lkg_not_selected",
        )
    )

    if custom_reference is not None:
        reference_plan.append(
            PreparedReferencePlanEntry(
                kind=DiffReferenceKind.CUSTOM,
                reference=custom_reference,
                unavailable_reason=None,
            )
        )

    timeline_state = gateway.reconcile_timeline_baseline(
        subject_id,
        monitoring_run_id,
        baseline_resolution_result.baseline_source_run_id,
    )

    # reconciled_baseline_source_run_id is guaranteed to be a str
    # if .reconcile_timeline_baseline() returns without raising an exception.
    # Therefore, adding an error check here is unnecessary, and we can safely cast it to str.
    reconciled_baseline_source_run_id = cast(str, timeline_state.baseline_source_run_id)

    return PreparedContext(
        monitoring_run_id=monitoring_run_id,
        source_run_id=resolved_source_run_id,
        subject_id=subject_id,
        timeline_id=timeline_state.timeline_id,
        sequence_index=sequence_index,
        baseline_source_run_id=reconciled_baseline_source_run_id,
        effective_recipe=compiled_recipe.effective_plan,
        contract=compiled_recipe.contract,
        reference_plan=tuple(reference_plan),
    )


def execute_contract_check(
    prepared_context: PreparedContext,
    gateway: MonitoringGateway,
    contract_checker: ContractChecker,
) -> ContractCheckResult:
    """Evaluate the prepared contract context and return the check result.

    Args:
        prepared_context: Resolved prepare-stage context for one contract evaluation.
        gateway: Gateway used to read source-run evidence.
        contract_checker: Checker implementation.

    Raises:
        CheckStageError: If required evidence is missing or the checker result
            violates invariants.

    Returns:
        Validated contract check result for the prepared context.
    """
    baseline_evidence = gateway.get_source_run_contract_evidence(
        source_run_id=prepared_context.baseline_source_run_id,
    )
    if baseline_evidence is None:
        raise CheckStageError(
            code="check_missing_baseline_evidence",
            message="Baseline contract evidence could not be resolved.",
            details=(("baseline_source_run_id", prepared_context.baseline_source_run_id),),
        )

    current_evidence = gateway.get_source_run_contract_evidence(
        source_run_id=prepared_context.source_run_id,
    )
    if current_evidence is None:
        raise CheckStageError(
            code="check_missing_current_evidence",
            message="Current contract evidence could not be resolved.",
            details=(("source_run_id", prepared_context.source_run_id),),
        )

    evaluation_context = make_contract_evaluation_context(
        subject_id=prepared_context.subject_id,
        source_run_id=prepared_context.source_run_id,
        baseline_source_run_id=prepared_context.baseline_source_run_id,
        baseline_context=baseline_evidence,
        current_context=current_evidence,
    )

    result = contract_checker.check(prepared_context.contract, evaluation_context)

    try:
        validate_contract_check_result(result)
    except InvariantViolation as exc:
        raise CheckStageError(
            code="check_result_invalid",
            message="Contract checker produced an invalid contract check result.",
            details=(("monitoring_run_id", prepared_context.monitoring_run_id),),
        ) from exc

    return result


def _resolve_baseline_for_prepare(
    subject_id: str,
    compiled_recipe: CompiledRecipe,
    gateway: MonitoringGateway,
    timeline_state: TimelineState | None,
    baseline_source_run_id: str | None = None,
) -> BaselineResolutionResult:
    """Resolve the immutable baseline source run for prepare.

    Args:
        subject_id: Stable monitored subject identifier.
        compiled_recipe: Execution-ready compiled Recipe.
        gateway: Gateway used for timeline and source-run reads.
        timeline_state: Timeline state for the subject, if it exists.
        baseline_source_run_id: Caller-supplied baseline source run id to resolve.

    Raises:
        PrepareStageError: If an uninitialized timeline has no valid baseline, or if
            the provided baseline attempts to override an established baseline.

    Returns:
        Baseline resolution result containing the resolved baseline information.
    """
    # The timeline does not exist yet, so we cannot resolve a baseline without it.
    if timeline_state is None:
        raise PrepareStageError(
            code="prepare_missing_timeline",
            message=(
                f"No timeline exists for the subject_id={subject_id!r} "
                "and baseline resolution cannot proceed. "
                "Consider allocating a monitoring run first."
            ),
            details=(("subject_id", subject_id),),
        )

    established_baseline = timeline_state.baseline_source_run_id

    if established_baseline is None:
        if baseline_source_run_id is None or baseline_source_run_id == "":
            raise PrepareStageError(
                code="prepare_missing_baseline_for_uninitialized_timeline",
                message=(
                    f"The timeline for subject_id={subject_id!r} has no pinned baseline "
                    "and no baseline_source_run_id was provided. "
                    "A valid baseline_source_run_id is required to bootstrap the timeline."
                ),
                details=(
                    ("subject_id", subject_id),
                    ("baseline_source_run_id", baseline_source_run_id),
                ),
            )

        resolved_baseline = gateway.resolve_source_run_id(
            subject_id=subject_id,
            source_experiment=compiled_recipe.source_requirements.source_experiment,
            source_run_id=baseline_source_run_id,
        )

        if resolved_baseline is None:
            raise PrepareStageError(
                code="prepare_invalid_bootstrap_baseline",
                message=(
                    f"Baseline source run could not be resolved for subject_id={subject_id!r}, "
                    f"source_experiment={compiled_recipe.source_requirements.source_experiment!r}, "
                    f"and baseline_source_run_id={baseline_source_run_id!r}."
                ),
                details=(
                    ("subject_id", subject_id),
                    (
                        "compiled_recipe.source_requirements.source_experiment",
                        compiled_recipe.source_requirements.source_experiment,
                    ),
                    ("baseline_source_run_id", baseline_source_run_id),
                ),
            )

        return BaselineResolutionResult(
            baseline_source_run_id=resolved_baseline,
        )

    if baseline_source_run_id is not None:
        resolved_baseline = gateway.resolve_source_run_id(
            subject_id=subject_id,
            source_experiment=compiled_recipe.source_requirements.source_experiment,
            source_run_id=baseline_source_run_id,
        )

        if resolved_baseline is None:
            raise PrepareStageError(
                code="prepare_invalid_baseline",
                message=(
                    f"Baseline source run could not be resolved for subject_id={subject_id!r}, "
                    f"compiled_recipe.source_requirements.source_experiment="
                    f"{compiled_recipe.source_requirements.source_experiment!r}, "
                    f"and baseline_source_run_id={baseline_source_run_id!r}."
                ),
                details=(
                    ("subject_id", subject_id),
                    (
                        "compiled_recipe.source_requirements.source_experiment",
                        compiled_recipe.source_requirements.source_experiment,
                    ),
                    ("baseline_source_run_id", baseline_source_run_id),
                ),
            )

        if resolved_baseline != established_baseline:
            raise PrepareStageError(
                code=PREPARE_BASELINE_OVERRIDE_EXISTING_BASELINE,
                message=(
                    f"Provided baseline_source_run_id={baseline_source_run_id!r} "
                    f"with resolved baseline_source_run_id={resolved_baseline!r} does not match "
                    f"existing timeline pinned baseline_source_run_id={established_baseline!r} "
                    f"for subject_id={subject_id!r}. "
                    "Overriding an existing timeline's baseline is not allowed."
                ),
                details=(
                    ("subject_id", subject_id),
                    ("baseline_source_run_id", baseline_source_run_id),
                ),
            )

    return BaselineResolutionResult(
        baseline_source_run_id=established_baseline,
    )
