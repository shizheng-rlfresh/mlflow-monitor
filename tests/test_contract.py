"""Unit tests for built-in contract resolution."""

import pytest

from mlflow_monitor.builtins import (
    SYSTEM_DEFAULT_DATA_SCOPE_CONTRACT_REF,
    SYSTEM_DEFAULT_EXECUTION_CONTRACT_REF,
    SYSTEM_DEFAULT_FEATURE_CONTRACT_REF,
    SYSTEM_DEFAULT_SCHEMA_CONTRACT_REF,
)
from mlflow_monitor.contract import (
    SYSTEM_DEFAULT_CONTRACT_ID,
    parse_contract_v0,
    resolve_contract_v0,
)
from mlflow_monitor.domain import Contract
from mlflow_monitor.errors import ContractResolutionError

system_default_permissive_contract = Contract(
    contract_id=SYSTEM_DEFAULT_CONTRACT_ID,
    version="v0",
    schema_contract_ref=SYSTEM_DEFAULT_SCHEMA_CONTRACT_REF,
    feature_contract_ref=SYSTEM_DEFAULT_FEATURE_CONTRACT_REF,
    metric_contract_ref=None,
    data_scope_contract_ref=SYSTEM_DEFAULT_DATA_SCOPE_CONTRACT_REF,
    execution_contract_ref=SYSTEM_DEFAULT_EXECUTION_CONTRACT_REF,
)


def test_resolve_contract_v0_returns_default_permissive_contract() -> None:
    """System default binding should resolve to the built-in runtime contract."""
    resolved = resolve_contract_v0(SYSTEM_DEFAULT_CONTRACT_ID)

    assert resolved == system_default_permissive_contract
    assert resolved.contract_id == SYSTEM_DEFAULT_CONTRACT_ID
    assert resolved.version == "v0"
    assert resolved.schema_contract_ref is not None
    assert resolved.feature_contract_ref is not None
    assert resolved.data_scope_contract_ref is not None
    assert resolved.execution_contract_ref is not None
    assert resolved.metric_contract_ref is None


def test_resolve_contract_v0_rejects_unknown_binding() -> None:
    """Unknown contract binding IDs should fail explicitly."""
    with pytest.raises(ContractResolutionError) as exc:
        resolve_contract_v0("missing_contract")

    error = exc.value
    assert error.code == "unknown_contract_binding"
    assert error.details == (("contract_id", "missing_contract"),)


def test_parse_contract_v0_rejects_non_string_optional_ref() -> None:
    """Malformed built-in contract mappings should fail explicitly."""
    with pytest.raises(ContractResolutionError) as exc:
        parse_contract_v0(
            {
                "contract_id": SYSTEM_DEFAULT_CONTRACT_ID,
                "version": "v0",
                "schema_contract_ref": "schema",
                "feature_contract_ref": "feature",
                "metric_contract_ref": None,
                "data_scope_contract_ref": 123,
                "execution_contract_ref": "execution",
            }
        )

    error = exc.value
    assert error.code == "invalid_contract_definition"
    assert error.details == (("field", "data_scope_contract_ref"),)
