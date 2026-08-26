"""Workflow lifecycle helpers for MLflow-Monitor v0.

This module contains backend-agnostic workflow logic for two responsibilities:

1. Prepare-stage context resolution before contract checking begins.
2. Contract checking and evaluation after prepare-stage context resolution.

Prepare-stage resolution combines caller inputs (Monitoring Run identity, compiled plan,
resolved contract, optional first-run baseline input) with gateway-resolved
state (Timeline, Source Training Run, prior Monitoring Runs, and optional references).
The workflow layer decides what must be resolved for a run to proceed, while
the gateway owns all persistence-specific mechanics.
"""

from .check import (
    CONTRACT_CHECK_ARTIFACT_PATH,
    contract_check_result_to_dict,
    execute_contract_check,
    hydrate_contract_check_result,
)
from .prepare import (
    BaselineResolutionResult,
    prepare_run_context,
)
from .prepared_context import (
    PREPARED_CONTEXT_ARTIFACT_PATH,
    PreparedContext,
    PreparedReferencePlanEntry,
    hydrate_prepared_context,
    prepared_context_to_dict,
)

__all__ = [
    "PREPARED_CONTEXT_ARTIFACT_PATH",
    "PreparedContext",
    "PreparedReferencePlanEntry",
    "hydrate_prepared_context",
    "prepared_context_to_dict",
    "BaselineResolutionResult",
    "prepare_run_context",
    "CONTRACT_CHECK_ARTIFACT_PATH",
    "contract_check_result_to_dict",
    "hydrate_contract_check_result",
    "execute_contract_check",
]
