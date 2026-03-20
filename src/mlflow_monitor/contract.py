"""Runtime contract parsing and binding resolution for MLflow-Monitor v0.

This module owns the internal bridge from recipe-selected contract binding IDs
to resolved runtime ``Contract`` values. It does not evaluate contracts; that
responsibility stays in ``contract_checker.py``.
"""

from __future__ import annotations

from collections.abc import Mapping

from mlflow_monitor.builtins.builtin_contract import (
    SYSTEM_DEFAULT_CONTRACT_ID,
    SYSTEM_DEFAULT_PERMISSIVE_CONTRACT_RAW,
)
from mlflow_monitor.domain import Contract
from mlflow_monitor.errors import ContractResolutionError


def _require_string(raw: Mapping[str, object], field: str) -> str:
    """Return one required string field from a raw contract mapping.

    Args:
        raw: Raw mapping defining one built-in contract.
        field: The required field name to extract from the raw mapping.

    Returns:
        The string value of the required field.

    Raises:
        ContractResolutionError: If the field is missing or not a string.
    """
    value = raw.get(field)
    if not isinstance(value, str):
        raise ContractResolutionError(
            code="invalid_contract_definition",
            message=f"Contract field '{field}' must be a string.",
            details=(("field", field),),
        )
    return value


def _optional_string(raw: Mapping[str, object], field: str) -> str | None:
    """Return one optional string field from a raw contract mapping.

    Args:
        raw: Raw mapping defining one built-in contract.
        field: The optional field name to extract from the raw mapping.

    Returns:
        The string value of the optional field, or None if the field is missing.

    Raises:
        ContractResolutionError: If the field is not a string or null.
    """
    value = raw.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ContractResolutionError(
            code="invalid_contract_definition",
            message=f"Contract field '{field}' must be a string or null.",
            details=(("field", field),),
        )
    return value


def parse_contract_v0(raw: Mapping[str, object]) -> Contract:
    """Parse one raw v0 contract mapping into the runtime ``Contract`` model.

    Args:
        raw: Raw mapping defining one built-in contract.

    Returns:
        Parsed runtime contract.

    Raises:
        ContractResolutionError: If the raw mapping is malformed.
    """
    if not isinstance(raw, Mapping):
        raise ContractResolutionError(
            code="invalid_contract_definition",
            message="Contract definition must be a mapping.",
        )

    return Contract(
        contract_id=_require_string(raw, "contract_id"),
        version=_require_string(raw, "version"),
        schema_contract_ref=_optional_string(raw, "schema_contract_ref"),
        feature_contract_ref=_optional_string(raw, "feature_contract_ref"),
        metric_contract_ref=_optional_string(raw, "metric_contract_ref"),
        data_scope_contract_ref=_optional_string(raw, "data_scope_contract_ref"),
        execution_contract_ref=_optional_string(raw, "execution_contract_ref"),
    )


def resolve_contract_v0(contract_id: str) -> Contract:
    """Resolve one supported v0 contract binding ID into a runtime contract.

    Args:
        contract_id: Recipe-selected contract binding ID.

    Returns:
        The resolved runtime ``Contract`` consumed by workflow code.

    Raises:
        ContractResolutionError: If the contract binding ID is unknown.
    """
    if contract_id == SYSTEM_DEFAULT_CONTRACT_ID:
        resolved = parse_contract_v0(SYSTEM_DEFAULT_PERMISSIVE_CONTRACT_RAW)
        if resolved.contract_id != contract_id:
            raise ContractResolutionError(
                code="contract_binding_resolution_mismatch",
                message=(
                    "Resolved contract does not match requested contract binding "
                    f"({resolved.contract_id!r} != {contract_id!r})."
                ),
                details=(
                    ("contract_id", contract_id),
                    ("resolved_contract_id", resolved.contract_id),
                ),
            )
        return resolved

    raise ContractResolutionError(
        code="unknown_contract_binding",
        message=f"Unknown contract binding: {contract_id}.",
        details=(("contract_id", contract_id),),
    )
