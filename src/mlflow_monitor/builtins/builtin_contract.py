"""System-defined raw contract mappings shipped with MLflow-Monitor v0."""

SYSTEM_DEFAULT_CONTRACT_ID = "default_permissive"
SYSTEM_DEFAULT_SCHEMA_CONTRACT_REF = "system_default_schema"
SYSTEM_DEFAULT_FEATURE_CONTRACT_REF = "system_default_feature"
SYSTEM_DEFAULT_DATA_SCOPE_CONTRACT_REF = "system_default_data_scope"
SYSTEM_DEFAULT_EXECUTION_CONTRACT_REF = "system_default_execution"

SYSTEM_DEFAULT_PERMISSIVE_CONTRACT_RAW = {
    "contract_id": SYSTEM_DEFAULT_CONTRACT_ID,
    "version": "v0",
    "schema_contract_ref": SYSTEM_DEFAULT_SCHEMA_CONTRACT_REF,
    "feature_contract_ref": SYSTEM_DEFAULT_FEATURE_CONTRACT_REF,
    "metric_contract_ref": None,
    "data_scope_contract_ref": SYSTEM_DEFAULT_DATA_SCOPE_CONTRACT_REF,
    "execution_contract_ref": SYSTEM_DEFAULT_EXECUTION_CONTRACT_REF,
}
