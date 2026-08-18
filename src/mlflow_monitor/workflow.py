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

from mlflow_monitor.contract_checker import (
    ContractChecker,
    make_contract_evaluation_context,
)
from mlflow_monitor.domain import (
    Contract,
    ContractCheckResult,
    DiffReferenceKind,
    MonitoringRunReference,
)
from mlflow_monitor.errors import (
    CheckStageError,
    GatewayConsistencyViolation,
    InvariantViolation,
    PrepareStageError,
)
from mlflow_monitor.errors.workflow import PREPARED_BASELINE_OVERRIDE_EXISTING_BASELINE
from mlflow_monitor.gateway import MonitoringGateway, TimelineState
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
_PREPARED_REFERENCE_FIELDS = frozenset({"kind", "monitoring_run_id", "source_run_id"})
_REFERENCE_ORDER = {
    DiffReferenceKind.BASELINE: 0,
    DiffReferenceKind.PREVIOUS: 1,
    DiffReferenceKind.LKG: 2,
    DiffReferenceKind.CUSTOM: 3,
}


@dataclass(frozen=True, slots=True)
class BaselineResolutionResult:
    """Result of baseline source run resolution for prepare-stage context.

    Attributes:
        baseline_source_run_id: Resolved baseline source run id.
        requires_bootstrap: Whether the baseline source run must be bootstrapped.
    """

    baseline_source_run_id: str
    requires_bootstrap: bool


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
        references: Tuple of resolved monitoring run references.
    """

    monitoring_run_id: str
    source_run_id: str
    subject_id: str
    timeline_id: str
    sequence_index: int
    baseline_source_run_id: str
    effective_recipe: EffectiveRecipePlan
    contract: Contract
    references: tuple[MonitoringRunReference, ...]

    def __post_init__(self) -> None:
        """Freeze the ordered resolved references."""
        object.__setattr__(self, "references", tuple(self.references))

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
        "references": [reference.to_dict() for reference in context.references],
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
        raise GatewayConsistencyViolation.prepared_context_inconsistent(
            reason="missing_artifact",
            field="prepared_context",
        )
    _require_exact_prepared_fields(raw, _PREPARED_CONTEXT_FIELDS, section="prepared_context")
    if raw.get("artifact_schema_version") != _PREPARED_CONTEXT_ARTIFACT_SCHEMA_VERSION:
        raise GatewayConsistencyViolation.prepared_context_inconsistent(
            reason="unsupported_artifact_schema_version",
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
            raise GatewayConsistencyViolation.prepared_context_inconsistent(
                reason="allocation_identity_mismatch",
                field=field,
            )

    baseline_source_run_id = _require_prepared_string(raw, "baseline_source_run_id")

    effective_recipe = raw.get("effective_recipe")
    if not isinstance(effective_recipe, Mapping):
        raise GatewayConsistencyViolation.prepared_context_inconsistent(
            reason="invalid_field_type",
            field="effective_recipe",
        )

    try:
        persisted = canonical_json(dict(effective_recipe))
        expected = canonical_json(compiled_recipe.effective_plan.to_dict())
    except (TypeError, ValueError) as exc:
        raise GatewayConsistencyViolation.prepared_context_inconsistent(
            reason="invalide_field_type",
            field="effective_recipe",
        ) from exc

    if persisted != expected:
        raise GatewayConsistencyViolation.prepared_context_inconsistent(
            reason="effective_recipe_mismatch",
            field="effective_recipe",
        )

    contract = _hydrate_prepared_contract(raw.get("contract"))
    if contract != compiled_recipe.contract:
        raise GatewayConsistencyViolation.prepared_context_inconsistent(
            reason="contract_mismatch",
            field="contract",
        )

    references = _hydrate_prepared_references(
        raw.get("references"),
        baseline_source_run_id=baseline_source_run_id,
    )
    return PreparedContext(
        monitoring_run_id=monitoring_run_id,
        source_run_id=source_run_id,
        subject_id=subject_id,
        timeline_id=timeline_id,
        sequence_index=sequence_index,
        baseline_source_run_id=baseline_source_run_id,
        effective_recipe=compiled_recipe.effective_plan,
        contract=contract,
        references=references,
    )


def _hydrate_prepared_contract(raw: object) -> Contract:
    """Hydrate the complete Contract frozen in prepared state."""
    if not isinstance(raw, Mapping):
        raise GatewayConsistencyViolation.prepared_context_inconsistent(
            reason="invalid_field_type", field="contract"
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


def _hydrate_prepared_references(
    raw: object,
    *,
    baseline_source_run_id: str,
) -> tuple[MonitoringRunReference, ...]:
    """Hydrate canonical resolved Monitoring Run references."""
    if not isinstance(raw, list):
        raise GatewayConsistencyViolation.prepared_context_inconsistent(
            reason="invalid_field_type", field="references"
        )

    references: list[MonitoringRunReference] = []
    for index, item in enumerate(raw):
        field = f"references[{index}]"
        if not isinstance(item, Mapping):
            raise GatewayConsistencyViolation.prepared_context_inconsistent(
                reason="invalid_field_type", field=field
            )
        _require_exact_prepared_fields(item, _PREPARED_REFERENCE_FIELDS, section=field)
        kind = item.get("kind")
        monitoring_run_id = item.get("monitoring_run_id")
        source_run_id = item.get("source_run_id")
        if not isinstance(kind, str):
            raise GatewayConsistencyViolation.prepared_context_inconsistent(
                reason="invalid_field_type", field=f"{field}.kind"
            )
        if monitoring_run_id is not None and not isinstance(monitoring_run_id, str):
            raise GatewayConsistencyViolation.prepared_context_inconsistent(
                reason="invalid_field_type",
                field=f"{field}.monitoring_run_id",
            )
        if not isinstance(source_run_id, str):
            raise GatewayConsistencyViolation.prepared_context_inconsistent(
                reason="invalid_field_type",
                field=f"{field}.source_run_id",
            )
        try:
            reference = MonitoringRunReference(
                kind=DiffReferenceKind(kind),
                monitoring_run_id=monitoring_run_id,
                source_run_id=source_run_id,
            )
        except ValueError as exc:
            raise GatewayConsistencyViolation.prepared_context_inconsistent(
                reason="invalid_reference",
                field=field,
            ) from exc
        references.append(reference)

    if not references or references[0] != MonitoringRunReference(
        kind=DiffReferenceKind.BASELINE,
        monitoring_run_id=None,
        source_run_id=baseline_source_run_id,
    ):
        raise GatewayConsistencyViolation.prepared_context_inconsistent(
            reason="baseline_reference_mismatch",
            field="references",
        )

    reference_kinds = tuple(reference.kind for reference in references)
    if (
        len(reference_kinds) != len(set(reference_kinds))
        or tuple(sorted(reference_kinds, key=_REFERENCE_ORDER.__getitem__)) != reference_kinds
    ):
        raise GatewayConsistencyViolation.prepared_context_inconsistent(
            reason="noncanonical_references",
            field="references",
        )
    return tuple(references)


def _require_exact_prepared_fields(
    raw: Mapping[str, object],
    expected: frozenset[str],
    *,
    section: str,
) -> None:
    """Require one prepared-artifact mapping to have exactly its canonical fields."""
    if set(raw) != expected:
        raise GatewayConsistencyViolation.prepared_context_inconsistent(
            reason="invalid_fields", field=section
        )


def _require_prepared_string(raw: Mapping[str, object], field: str) -> str:
    """Return one required nonempty string from a prepared artifact mapping."""
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise GatewayConsistencyViolation.prepared_context_inconsistent(
            reason="invalid_field_type", field=field
        )
    return value


def _require_prepared_optional_string(
    raw: Mapping[str, object],
    field: str,
) -> str | None:
    """Return one optional string from a prepared artifact mapping."""
    value = raw.get(field)
    if value is not None and not isinstance(value, str):
        raise GatewayConsistencyViolation.prepared_context_inconsistent(
            reason="invalid_field_type", field=field
        )
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
        baseline_source_run_id: Optional baseline source run id used to bootstrap (pin)
            timeline baseline, or to explicitly confirm the baseline for an existing timeline.
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

    if custom_reference_monitoring_run_id is not None:
        custom_reference_monitoring_run_id = gateway.resolve_timeline_monitoring_run_id(
            subject_id,
            custom_reference_monitoring_run_id,
        )
        if custom_reference_monitoring_run_id is None:
            raise PrepareStageError(
                code="prepare_custom_reference_not_found",
                message=(
                    "Custom reference monitoring run could not be resolved on the subject timeline."
                ),
                details=(("subject_id", subject_id),),
            )

    if baseline_resolution_result.requires_bootstrap:
        # race handling
        timeline_pin_baseline_result = gateway.pin_timeline_baseline(
            subject_id,
            baseline_resolution_result.baseline_source_run_id,
        )

        timeline_state = gateway.get_timeline_state(subject_id)
        if timeline_state is None:
            _id = subject_id
            raise PrepareStageError(
                code="prepare_timeline_initialization_failed",
                message=(
                    f"Timeline initialization did not materialize state for subject_id={_id}."
                ),
                details=(("subject_id", subject_id),),
            )
        if timeline_pin_baseline_result.baseline_pinned:
            if (
                timeline_state.baseline_source_run_id
                != baseline_resolution_result.baseline_source_run_id
            ):
                _id = subject_id
                raise PrepareStageError(
                    code="prepare_timeline_pin_failed",
                    message=(f"Timeline pinning did not materialize state for subject_id={_id}."),
                    details=(("subject_id", subject_id),),
                )
        elif (
            timeline_state.baseline_source_run_id
            != baseline_resolution_result.baseline_source_run_id
        ):
            _id = subject_id
            _provided_baseline = baseline_source_run_id
            _resolved_baseline = baseline_resolution_result.baseline_source_run_id
            _existing_baseline = timeline_state.baseline_source_run_id
            raise PrepareStageError(
                code=PREPARED_BASELINE_OVERRIDE_EXISTING_BASELINE,
                message=(
                    f"Provided baseline_source_run_id={_provided_baseline!r} "
                    f"with resolved_baseline_source_run_id={_resolved_baseline!r} "
                    "does not match existing timeline "
                    f"baseline_source_run_id={_existing_baseline!r} for subject_id={_id}. "
                    "Overriding an existing timeline's baseline is not allowed."
                ),
                details=(
                    ("subject_id", subject_id),
                    ("baseline_source_run_id", _provided_baseline),
                ),
            )
    else:
        timeline_state = gateway.get_timeline_state(subject_id)

    if timeline_state is None:
        _id = subject_id
        raise PrepareStageError(
            code="prepare_timeline_initialization_failed",
            message=(f"Timeline initialization did not materialize state for subject_id={_id}."),
            details=(("subject_id", subject_id),),
        )

    # This should be impossible to happen, adding this check as a safeguard.
    if not timeline_state.baseline_source_run_id:
        raise PrepareStageError(
            code="prepare_baseline_missing",
            message=(f"Timeline for subject_id={subject_id} does not have a pinned baseline."),
            details=(("subject_id", subject_id),),
        )

    references = [
        MonitoringRunReference(
            kind=DiffReferenceKind.BASELINE,
            monitoring_run_id=None,
            source_run_id=timeline_state.baseline_source_run_id,
        )
    ]

    timeline_runs = gateway.list_timeline_monitoring_runs(subject_id, exclude_failed=True)
    if timeline_runs:
        previous = timeline_runs[-1]
        references.append(
            MonitoringRunReference(
                kind=DiffReferenceKind.PREVIOUS,
                monitoring_run_id=previous.monitoring_run_id,
                source_run_id=previous.source_run_id,
            )
        )

    active_lkg_monitoring_run_id = gateway.resolve_active_lkg_monitoring_run_id(subject_id)
    if active_lkg_monitoring_run_id is not None:
        references.append(
            _hydrate_monitoring_run_reference(
                kind=DiffReferenceKind.LKG,
                subject_id=subject_id,
                monitoring_run_id=active_lkg_monitoring_run_id,
                gateway=gateway,
            )
        )

    if custom_reference_monitoring_run_id is not None:
        references.append(
            _hydrate_monitoring_run_reference(
                kind=DiffReferenceKind.CUSTOM,
                subject_id=subject_id,
                monitoring_run_id=custom_reference_monitoring_run_id,
                gateway=gateway,
            )
        )

    return PreparedContext(
        monitoring_run_id=monitoring_run_id,
        source_run_id=resolved_source_run_id,
        subject_id=subject_id,
        timeline_id=timeline_state.timeline_id,
        sequence_index=sequence_index,
        baseline_source_run_id=timeline_state.baseline_source_run_id,
        effective_recipe=compiled_recipe.effective_plan,
        contract=compiled_recipe.contract,
        references=tuple(references),
    )


def _hydrate_monitoring_run_reference(
    *,
    kind: DiffReferenceKind,
    subject_id: str,
    monitoring_run_id: str,
    gateway: MonitoringGateway,
) -> MonitoringRunReference:
    """Freeze a complete reference pair from a resolved Monitoring Run pointer."""
    record = gateway.get_monitoring_run(subject_id, monitoring_run_id)
    if record is None:
        raise GatewayConsistencyViolation.monitoring_reference_inconsistent(
            kind=kind,
            monitoring_run_id=monitoring_run_id,
        )

    return MonitoringRunReference(
        kind=kind,
        monitoring_run_id=record.monitoring_run_id,
        source_run_id=record.source_run_id,
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
    """Resolve baseline source run for prepare when no timeline exists.

    Handle races of timeline initialization and baseline bootstrapping.

    Args:
        subject_id: Stable monitored subject identifier.
        compiled_recipe: Execution-ready compiled Recipe.
        gateway: Gateway used for timeline and source-run reads.
        timeline_state: Timeline state for the subject, if it exists.
        baseline_source_run_id: Caller-supplied baseline source run id to resolve.

    Raises:
        PrepareStageError: If a new timeline must be bootstrapped but the provided baseline
                           is invalid or missing,
                           or if the provided baseline attempts to override an existing timeline.

    Returns:
        Baseline resolution result containing timeline and resolved baseline information.
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

    # The timeline exists, so we can check for a pinned baseline.
    pinned_baseline = timeline_state.baseline_source_run_id

    # If there is no pinned baseline, we will need to bootstrap the baseline
    if pinned_baseline is None:
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
            requires_bootstrap=True,
        )

    # If the timeline exists, we do not need to bootstrap the baseline.
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

        if resolved_baseline != pinned_baseline:
            raise PrepareStageError(
                code=PREPARED_BASELINE_OVERRIDE_EXISTING_BASELINE,
                message=(
                    f"Provided baseline_source_run_id={baseline_source_run_id!r} "
                    f"with resolved baseline_source_run_id={resolved_baseline!r} does not match "
                    f"existing timeline pinned baseline_source_run_id={pinned_baseline!r} "
                    f"for subject_id={subject_id!r}. "
                    "Overriding an existing timeline's baseline is not allowed."
                ),
                details=(
                    ("subject_id", subject_id),
                    ("baseline_source_run_id", baseline_source_run_id),
                ),
            )

    return BaselineResolutionResult(
        baseline_source_run_id=pinned_baseline,
        requires_bootstrap=False,
    )
