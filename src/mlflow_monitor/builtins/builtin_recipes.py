"""System-defined raw recipe mappings shipped with MLflow-Monitor v0."""

from mlflow_monitor.builtins.builtin_contract import SYSTEM_DEFAULT_CONTRACT_ID

SYSTEM_DEFAULT_RECIPE_ID = "system_default"
SYSTEM_DEFAULT_RUN_SELECTOR_TOKEN = "__RUNTIME_SOURCE_RUN_ID__"

SYSTEM_DEFAULT_RECIPE_RAW = {
    "identity": {"recipe_id": SYSTEM_DEFAULT_RECIPE_ID, "version": "v0"},
    "input_binding": {"run_selector": SYSTEM_DEFAULT_RUN_SELECTOR_TOKEN},
    "contract_binding": {"contract_id": SYSTEM_DEFAULT_CONTRACT_ID},
    "metrics_slices": {},
    "finding_policy": {},
    "output_binding": {},
}
