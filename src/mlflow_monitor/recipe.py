"""Recipe schema and parser for MLflow-Monitor v0-lite."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

_REQUIRED_TOP_LEVEL_SECTIONS = {
    "identity",
    "input_binding",
    "contract_binding",
    "metrics_slices",
    "finding_policy",
    "output_binding",
}


@dataclass(frozen=True, slots=True)
class RecipeIdentity:
    """Identity and version metadata for a recipe."""

    recipe_id: str
    version: str


@dataclass(frozen=True, slots=True)
class RecipeInputBinding:
    """Input binding section for source run selection and required evidence."""

    run_selector: str


@dataclass(frozen=True, slots=True)
class RecipeContractBinding:
    """Contract binding section for resolved contract selection."""

    contract_id: str


@dataclass(frozen=True, slots=True)
class RecipeMetricsSlices:
    """Metrics and slices section for analysis selection."""


@dataclass(frozen=True, slots=True)
class RecipeFindingPolicy:
    """Finding policy section for interpreted output behavior."""


@dataclass(frozen=True, slots=True)
class RecipeOutputBinding:
    """Output binding section for summary rendering options."""


@dataclass(frozen=True, slots=True)
class RecipeV0Lite:
    """Canonical v0-lite recipe runtime model."""

    identity: RecipeIdentity
    input_binding: RecipeInputBinding
    contract_binding: RecipeContractBinding
    metrics_slices: RecipeMetricsSlices
    finding_policy: RecipeFindingPolicy
    output_binding: RecipeOutputBinding


def parse_recipe_v0_lite(raw: Mapping[str, object]) -> RecipeV0Lite:
    """Parse a mapping into the canonical v0-lite recipe model."""
    unknown_sections = sorted(set(raw.keys()) - _REQUIRED_TOP_LEVEL_SECTIONS)
    if unknown_sections:
        unknown = ", ".join(unknown_sections)
        raise ValueError(f"Unknown/disallowed recipe section(s): {unknown}")

    return RecipeV0Lite(
        identity=RecipeIdentity(
            recipe_id=raw["identity"]["recipe_id"],  # type: ignore[index]
            version=raw["identity"]["version"],  # type: ignore[index]
        ),
        input_binding=RecipeInputBinding(
            run_selector=raw["input_binding"]["run_selector"],  # type: ignore[index]
        ),
        contract_binding=RecipeContractBinding(
            contract_id=raw["contract_binding"]["contract_id"],  # type: ignore[index]
        ),
        metrics_slices=RecipeMetricsSlices(),
        finding_policy=RecipeFindingPolicy(),
        output_binding=RecipeOutputBinding(),
    )
