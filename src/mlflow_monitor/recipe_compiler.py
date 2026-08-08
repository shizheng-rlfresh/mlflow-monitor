"""Side-effect-free compilation for strict v0 Recipes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from mlflow_monitor.builtins import (
    SYSTEM_COMPATIBILITY_FINDING_POLICY,
    SYSTEM_COMPATIBILITY_FINDING_POLICY_ID,
    SYSTEM_COMPATIBILITY_FINDING_POLICY_VERSION,
    SYSTEM_DEFAULT_CONTRACT_ID,
)
from mlflow_monitor.contract import resolve_contract_v0
from mlflow_monitor.domain import Contract
from mlflow_monitor.errors import RecipeValidationError, RecipeValidationIssue
from mlflow_monitor.finding_policy import (
    FindingPolicy,
    FrozenFindingPolicyParameters,
    FrozenJSONValue,
    JSONValue,
)
from mlflow_monitor.recipe import (
    FrozenRecipeJSONValue,
    Recipe,
    RecipeAnalysis,
    RecipeContractBinding,
    RecipeFindingPolicyBinding,
    RecipeIdentity,
    RecipeSourceRequirements,
    build_system_default_recipe,
    parse_recipe,
)


@dataclass(frozen=True, slots=True)
class EffectiveFindingPolicyBinding:
    """Serializable normalized Finding-policy binding.

    Attributes:
        finding_policy_id: Resolved policy identifier.
        finding_policy_version: Resolved exact policy version.
        parameters: Validated immutable effective parameters.
    """

    finding_policy_id: str
    finding_policy_version: str
    parameters: FrozenFindingPolicyParameters

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable binding mapping.

        Returns:
            Normalized binding data without the executable policy.
        """
        return {
            "finding_policy_id": self.finding_policy_id,
            "finding_policy_version": self.finding_policy_version,
            "parameters": _json_mapping_to_dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class EffectiveRecipeAnalysis:
    """Normalized analysis selection persisted for replay.

    Attributes:
        metric_names: Three-state normalized metric selection.
        finding_policy_bindings: Expanded, canonically ordered policy bindings.
    """

    metric_names: tuple[str, ...] | None
    finding_policy_bindings: tuple[EffectiveFindingPolicyBinding, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable normalized analysis data.

        Returns:
            Normalized metric and policy selections.
        """
        return {
            "metric_names": None if self.metric_names is None else list(self.metric_names),
            "finding_policy_bindings": [
                binding.to_dict() for binding in self.finding_policy_bindings
            ],
        }


@dataclass(frozen=True, slots=True)
class EffectiveRecipePlan:
    """Normalized serializable Recipe plan without executable components.

    Attributes:
        recipe_schema_version: Exact Recipe schema version.
        identity: Stable Recipe identity.
        source_requirements: Canonically ordered source requirements.
        contract: Exact resolved Contract binding identity.
        analysis: Normalized effective analysis selection.
    """

    recipe_schema_version: str
    identity: RecipeIdentity
    source_requirements: RecipeSourceRequirements
    contract: RecipeContractBinding
    analysis: EffectiveRecipeAnalysis

    def to_dict(self) -> dict[str, object]:
        """Return the complete normalized plan as JSON-compatible data.

        Returns:
            Effective Recipe data without executable objects.
        """
        return {
            "recipe_schema_version": self.recipe_schema_version,
            "identity": {
                "recipe_id": self.identity.recipe_id,
                "recipe_version": self.identity.recipe_version,
            },
            "source_requirements": {
                "source_experiment": self.source_requirements.source_experiment,
                "required_metric_names": list(self.source_requirements.required_metric_names),
                "required_artifact_paths": list(self.source_requirements.required_artifact_paths),
            },
            "contract": {
                "contract_id": self.contract.contract_id,
                "contract_version": self.contract.contract_version,
            },
            "analysis": self.analysis.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CompiledFindingPolicyBinding:
    """Validated parameters paired with one process-local policy implementation.

    Attributes:
        finding_policy_id: Resolved policy identifier.
        finding_policy_version: Resolved exact policy version.
        parameters: Validated immutable effective parameters.
        policy: Process-local executable policy implementation.
    """

    finding_policy_id: str
    finding_policy_version: str
    parameters: FrozenFindingPolicyParameters
    policy: FindingPolicy = field(compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class CompiledRecipe:
    """Immutable execution-ready Recipe with resolved system components.

    Attributes:
        effective_plan: Serializable normalized Recipe plan.
        contract: Resolved system Contract.
        finding_policy_bindings: Resolved process-local policy bindings.
    """

    effective_plan: EffectiveRecipePlan
    contract: Contract
    finding_policy_bindings: tuple[CompiledFindingPolicyBinding, ...]

    def __post_init__(self) -> None:
        """Ensure the plan and contract represent same id and version."""
        expcted = (
            self.effective_plan.contract.contract_id,
            self.effective_plan.contract.contract_version,
        )
        actual = (self.contract.contract_id, self.contract.version)
        if expcted != actual:
            raise ValueError(
                f"Mismatch between effective plan contract {expcted} and resolved contract {actual}"
            )

    @property
    def identity(self) -> RecipeIdentity:
        """Return the compiled Recipe identity."""
        return self.effective_plan.identity

    @property
    def source_requirements(self) -> RecipeSourceRequirements:
        """Return normalized source-evidence requirements."""
        return self.effective_plan.source_requirements

    @property
    def analysis(self) -> EffectiveRecipeAnalysis:
        """Return normalized analysis selections."""
        return self.effective_plan.analysis


@dataclass(frozen=True, slots=True, init=False)
class ComponentRegistry:
    """Minimal immutable registry containing the v0 system components.

    Attributes:
        contracts: System Contracts indexed by exact identity and version.
        finding_policies: System policies indexed by exact identity and version.
    """

    contracts: Mapping[tuple[str, str], Contract]
    finding_policies: Mapping[tuple[str, str], FindingPolicy]

    def __init__(self) -> None:
        """Construct the fixed system-only component registry."""
        contract = resolve_contract_v0(SYSTEM_DEFAULT_CONTRACT_ID)
        object.__setattr__(
            self,
            "contracts",
            MappingProxyType({(contract.contract_id, contract.version): contract}),
        )
        object.__setattr__(
            self,
            "finding_policies",
            MappingProxyType(
                {
                    (
                        SYSTEM_COMPATIBILITY_FINDING_POLICY_ID,
                        SYSTEM_COMPATIBILITY_FINDING_POLICY_VERSION,
                    ): SYSTEM_COMPATIBILITY_FINDING_POLICY
                }
            ),
        )


SYSTEM_COMPONENT_REGISTRY = ComponentRegistry()


def _recipe_json_to_authoring(value: object) -> object:
    """Thaw typed Recipe JSON containers without accepting invalid values."""
    if isinstance(value, Mapping):
        return {key: _recipe_json_to_authoring(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_recipe_json_to_authoring(item) for item in value]
    return value


def _recipe_identity_to_authoring(value: object) -> object:
    """Convert a typed Recipe identity while preserving malformed values."""
    if not isinstance(value, RecipeIdentity):
        return value
    return {
        "recipe_id": value.recipe_id,
        "recipe_version": value.recipe_version,
    }


def _recipe_source_requirements_to_authoring(value: object) -> object:
    """Convert typed source requirements while preserving malformed values."""
    if not isinstance(value, RecipeSourceRequirements):
        return value
    source_requirements: dict[str, object] = {
        "required_metric_names": _recipe_json_to_authoring(value.required_metric_names),
        "required_artifact_paths": _recipe_json_to_authoring(value.required_artifact_paths),
    }
    if value.source_experiment is not None:
        source_requirements["source_experiment"] = value.source_experiment
    return source_requirements


def _recipe_contract_to_authoring(value: object) -> object:
    """Convert a typed Contract binding while preserving malformed values."""
    if not isinstance(value, RecipeContractBinding):
        return value
    return {
        "contract_id": value.contract_id,
        "contract_version": value.contract_version,
    }


def _recipe_policy_binding_to_authoring(value: object) -> object:
    """Convert a typed policy binding while preserving malformed values."""
    if not isinstance(value, RecipeFindingPolicyBinding):
        return value
    return {
        "finding_policy_id": value.finding_policy_id,
        "finding_policy_version": value.finding_policy_version,
        "parameters": _recipe_json_to_authoring(value.parameters),
    }


def _recipe_analysis_to_authoring(value: object) -> object:
    """Convert typed analysis authoring while preserving three-state selections."""
    if not isinstance(value, RecipeAnalysis):
        return value
    analysis: dict[str, object] = {}
    if value.metric_names is not None:
        analysis["metric_names"] = _recipe_json_to_authoring(value.metric_names)
    if value.finding_policy_bindings is not None:
        bindings = value.finding_policy_bindings
        analysis["finding_policy_bindings"] = (
            [_recipe_policy_binding_to_authoring(binding) for binding in bindings]
            if isinstance(bindings, tuple | list)
            else bindings
        )
    return analysis


def _recipe_to_authoring_mapping(recipe: Recipe) -> dict[str, object]:
    """Convert a typed Recipe back to its strict authoring shape."""
    return {
        "recipe_schema_version": recipe.recipe_schema_version,
        "identity": _recipe_identity_to_authoring(recipe.identity),
        "source_requirements": _recipe_source_requirements_to_authoring(recipe.source_requirements),
        "contract": _recipe_contract_to_authoring(recipe.contract),
        "analysis": _recipe_analysis_to_authoring(recipe.analysis),
    }


def compile_recipe(
    recipe: Recipe | Mapping[str, object] | None = None,
    *,
    component_registry: ComponentRegistry | None = None,
) -> CompiledRecipe:
    """Compile one Recipe without reads, persistence, or run allocation.

    Args:
        recipe: Parsed Recipe or raw Mapping. Omission selects the system-default
            Recipe.
        component_registry: Immutable component registry. Omission selects the
            fixed system registry.

    Returns:
        An immutable execution-ready Compiled Recipe.

    Raises:
        RecipeValidationError: If parsing, component resolution, or parameter
            validation fails.
    """
    if recipe is None:
        parsed = parse_recipe(build_system_default_recipe())
    elif isinstance(recipe, Recipe):
        parsed = parse_recipe(_recipe_to_authoring_mapping(recipe))
    else:
        parsed = parse_recipe(recipe)
    registry = SYSTEM_COMPONENT_REGISTRY if component_registry is None else component_registry

    contract_key = (parsed.contract.contract_id, parsed.contract.contract_version)
    contract = registry.contracts.get(contract_key)
    if contract is None:
        raise RecipeValidationError(
            issues=(
                RecipeValidationIssue(
                    code="unknown_component",
                    section="contract",
                    field="contract_id",
                    message="Recipe Contract identity/version is not registered.",
                ),
            )
        )

    authored_bindings = parsed.analysis.finding_policy_bindings
    if authored_bindings is None:
        authored_bindings = (
            RecipeFindingPolicyBinding(
                finding_policy_id=SYSTEM_COMPATIBILITY_FINDING_POLICY_ID,
                finding_policy_version=SYSTEM_COMPATIBILITY_FINDING_POLICY_VERSION,
                parameters=MappingProxyType({}),
            ),
        )

    compiled_bindings = _compile_policy_bindings(authored_bindings, registry)
    effective_bindings = tuple(
        EffectiveFindingPolicyBinding(
            finding_policy_id=binding.finding_policy_id,
            finding_policy_version=binding.finding_policy_version,
            parameters=binding.parameters,
        )
        for binding in compiled_bindings
    )
    effective_plan = EffectiveRecipePlan(
        recipe_schema_version=parsed.recipe_schema_version,
        identity=parsed.identity,
        source_requirements=RecipeSourceRequirements(
            source_experiment=parsed.source_requirements.source_experiment,
            required_metric_names=tuple(sorted(parsed.source_requirements.required_metric_names)),
            required_artifact_paths=tuple(
                sorted(parsed.source_requirements.required_artifact_paths)
            ),
        ),
        contract=parsed.contract,
        analysis=EffectiveRecipeAnalysis(
            metric_names=(
                None
                if parsed.analysis.metric_names is None
                else tuple(sorted(parsed.analysis.metric_names))
            ),
            finding_policy_bindings=effective_bindings,
        ),
    )
    return CompiledRecipe(
        effective_plan=effective_plan,
        contract=contract,
        finding_policy_bindings=compiled_bindings,
    )


def _compile_policy_bindings(
    bindings: tuple[RecipeFindingPolicyBinding, ...],
    registry: ComponentRegistry,
) -> tuple[CompiledFindingPolicyBinding, ...]:
    """Resolve, validate, and canonically order authored policy bindings."""
    compiled: list[CompiledFindingPolicyBinding] = []
    issues: list[RecipeValidationIssue] = []
    for index, binding in enumerate(bindings):
        policy_key = (binding.finding_policy_id, binding.finding_policy_version)
        policy = registry.finding_policies.get(policy_key)
        field = f"finding_policy_bindings[{index}]"
        if policy is None:
            issues.append(
                RecipeValidationIssue(
                    code="unknown_component",
                    section="analysis",
                    field=field,
                    message="Recipe Finding-policy identity/version is not registered.",
                )
            )
            continue
        try:
            validated = policy.validate_parameters(
                _recipe_parameters_to_json_mapping(binding.parameters)
            )
            frozen = _freeze_validated_parameters(validated)
        except (TypeError, ValueError):
            issues.append(
                RecipeValidationIssue(
                    code="invalid_policy_parameters",
                    section="analysis",
                    field=f"{field}.parameters",
                    message="Recipe Finding-policy parameters are invalid.",
                )
            )
            continue
        compiled.append(
            CompiledFindingPolicyBinding(
                finding_policy_id=binding.finding_policy_id,
                finding_policy_version=binding.finding_policy_version,
                parameters=frozen,
                policy=policy,
            )
        )
    if issues:
        raise RecipeValidationError(issues=tuple(issues))
    return tuple(
        sorted(
            compiled,
            key=lambda binding: (
                binding.finding_policy_id,
                binding.finding_policy_version,
            ),
        )
    )


def _recipe_parameters_to_json_mapping(
    parameters: Mapping[str, FrozenRecipeJSONValue],
) -> Mapping[str, JSONValue]:
    """Return a defensive JSON-compatible parameter copy for validation."""
    return {key: _recipe_json_value_to_json(value) for key, value in sorted(parameters.items())}


def _recipe_json_value_to_json(value: FrozenRecipeJSONValue) -> JSONValue:
    """Thaw one immutable Recipe JSON value for policy validation."""
    if isinstance(value, Mapping):
        return {key: _recipe_json_value_to_json(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_recipe_json_value_to_json(item) for item in value]
    return value


def _freeze_validated_parameters(
    parameters: Mapping[str, FrozenJSONValue],
) -> FrozenFindingPolicyParameters:
    """Defensively freeze and validate one policy's effective parameters."""
    return MappingProxyType(
        {key: _freeze_validated_json_value(value) for key, value in sorted(parameters.items())}
    )


def _freeze_validated_json_value(value: FrozenJSONValue) -> FrozenJSONValue:
    """Validate finite JSON data and defensively freeze its containers."""
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Finding-policy parameters must contain finite numbers.")
        return value
    if isinstance(value, tuple):
        return tuple(_freeze_validated_json_value(item) for item in value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Finding-policy parameter object keys must be strings.")
        return MappingProxyType(
            {key: _freeze_validated_json_value(value[key]) for key in sorted(value)}
        )
    raise TypeError("Finding-policy parameters must be frozen JSON-compatible values.")


def _json_mapping_to_dict(
    parameters: Mapping[str, FrozenJSONValue],
) -> dict[str, object]:
    """Convert frozen JSON parameters to serializable dictionaries and lists."""
    return {key: _frozen_json_to_serializable(value) for key, value in sorted(parameters.items())}


def _frozen_json_to_serializable(value: FrozenJSONValue) -> object:
    """Convert one frozen JSON value to standard JSON containers."""
    if isinstance(value, Mapping):
        return _json_mapping_to_dict(value)
    if isinstance(value, tuple):
        return [_frozen_json_to_serializable(item) for item in value]
    return value


SYSTEM_DEFAULT_COMPILED_RECIPE = compile_recipe(
    build_system_default_recipe(),
    component_registry=SYSTEM_COMPONENT_REGISTRY,
)
