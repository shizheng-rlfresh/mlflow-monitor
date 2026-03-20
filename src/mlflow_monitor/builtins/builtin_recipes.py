"""System-defined raw recipe mappings shipped with MLflow-Monitor v0."""

from collections.abc import Mapping
from types import MappingProxyType

from mlflow_monitor.builtins.builtin_contract import SYSTEM_DEFAULT_CONTRACT_ID

SYSTEM_DEFAULT_RECIPE_ID = "system_default"
SYSTEM_DEFAULT_RUN_SELECTOR_TOKEN = "__RUNTIME_SOURCE_RUN_ID__"


def build_system_default_recipe_raw() -> Mapping[str, object]:
    """Build the raw mapping for the system default recipe."""
    return MappingProxyType(
        {
            "identity": MappingProxyType({"recipe_id": SYSTEM_DEFAULT_RECIPE_ID, "version": "v0"}),
            "input_binding": MappingProxyType({"run_selector": SYSTEM_DEFAULT_RUN_SELECTOR_TOKEN}),
            "contract_binding": MappingProxyType({"contract_id": SYSTEM_DEFAULT_CONTRACT_ID}),
            "metrics_slices": MappingProxyType({}),
            "finding_policy": MappingProxyType({}),
            "output_binding": MappingProxyType({}),
        }
    )
