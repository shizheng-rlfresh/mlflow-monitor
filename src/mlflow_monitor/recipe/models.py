"""Recipe models module for mlflow-monitor v0."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

RECIPE_SCHEMA_VERSION = "v0"
SYSTEM_DEFAULT_RECIPE_VERSION = "v0"
SYSTEM_DEFAULT_CONTRACT_VERSION = "v0"

type RecipeJSONScalar = str | int | float | bool | None
type FrozenRecipeJSONValue = (
    RecipeJSONScalar | tuple[FrozenRecipeJSONValue, ...] | Mapping[str, FrozenRecipeJSONValue]
)
type FrozenRecipeParameters = Mapping[str, FrozenRecipeJSONValue]


@dataclass(frozen=True, slots=True)
class RecipeIdentity:
    """Stable user-authored Recipe identity.

    Attributes:
        recipe_id: Stable Recipe identifier.
        recipe_version: Exact user-authored Recipe version.
    """

    recipe_id: str
    recipe_version: str


@dataclass(frozen=True, slots=True)
class RecipeSourceRequirements:
    """Source Training Run requirements authored by a Recipe.

    Attributes:
        source_experiment: Optional owning experiment constraint.
        required_metric_names: Metrics that Prepare requires on the source.
        required_artifact_paths: Artifact paths that Prepare requires on the source.
    """

    source_experiment: str | None = None
    required_metric_names: tuple[str, ...] = ()
    required_artifact_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecipeContractBinding:
    """Exact system Contract binding authored by a Recipe.

    Attributes:
        contract_id: Registered system Contract identifier.
        contract_version: Exact registered Contract version.
    """

    contract_id: str
    contract_version: str


@dataclass(frozen=True, slots=True)
class RecipeFindingPolicyBinding:
    """Exact Finding-policy binding and opaque JSON parameters.

    Attributes:
        finding_policy_id: Registered Finding-policy identifier.
        finding_policy_version: Exact registered policy version.
        parameters: Immutable structurally validated JSON parameters.
    """

    finding_policy_id: str
    finding_policy_version: str
    parameters: FrozenRecipeParameters


@dataclass(frozen=True, slots=True)
class RecipeAnalysis:
    """Three-state metric and Finding-policy analysis authoring.

    Attributes:
        metric_names: ``None`` for omitted, an empty tuple for none, or exact names.
        finding_policy_bindings: ``None`` for defaults, an empty tuple for none,
            or exact authored bindings.
    """

    metric_names: tuple[str, ...] | None = None
    finding_policy_bindings: tuple[RecipeFindingPolicyBinding, ...] | None = None


@dataclass(frozen=True, slots=True)
class Recipe:
    """Typed immutable representation of one structurally valid v0 Recipe.

    Attributes:
        recipe_schema_version: Exact Recipe schema version.
        identity: Stable Recipe identity.
        source_requirements: Source Training Run preconditions.
        contract: Exact system Contract binding.
        analysis: Three-state analysis authoring.
    """

    recipe_schema_version: str
    identity: RecipeIdentity
    source_requirements: RecipeSourceRequirements
    contract: RecipeContractBinding
    analysis: RecipeAnalysis
