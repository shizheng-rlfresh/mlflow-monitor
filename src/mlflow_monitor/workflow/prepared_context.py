"""Prepared context module for mlflow-monitor v0."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from mlflow_monitor.domain import Contract, DiffReferenceKind, MonitoringRunReference
from mlflow_monitor.errors import PreparedContextConsistencyViolation
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
        "reference_plan",
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
class PreparedReferencePlanEntry:
    """One fixed reference group frozen during Prepare.

    Attributes:
        kind: Canonical reference kind for this group.
        reference: Resolved paired reference, or null when unavailable.
        unavailable_reason: Prepare-time reason when the reference is unavailable
                                for null when reference is available.
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
            if self.reference.kind != kind:
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
        "reference_plan": [entry.to_dict() for entry in context.reference_plan],
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
        raw.get("reference_plan"),
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
            field="reference_plan",
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
            field="reference_plan",
        )

    reference_plan: list[PreparedReferencePlanEntry] = []
    for index, item in enumerate(raw):
        field = f"reference_plan[{index}]"
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
        if source_run_id is None and monitoring_run_id is not None:
            raise PreparedContextConsistencyViolation.invalid_reference(field=field)
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
            field="reference_plan",
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
