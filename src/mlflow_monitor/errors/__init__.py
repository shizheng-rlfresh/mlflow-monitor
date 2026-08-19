"""Custom exception types for MLflow-Monitor v0."""

from .gateway import (
    AllocationConsistencyViolation,
    GatewayConsistencyViolation,
    GatewayNamespaceViolation,
    PreparedContextConsistencyViolation,
    TimelineConsistencyViolation,
    TrainingRunMutationViolation,
)
from .invariant import InvariantViolation
from .recipe import (
    ContractResolutionError,
    RecipeValidationError,
    RecipeValidationIssue,
)
from .workflow import (
    PREPARE_BASELINE_OVERRIDE_EXISTING_BASELINE,
    CheckStageError,
    PrepareStageError,
    TerminalRunRetryError,
)

__all__ = [
    "CheckStageError",
    "ContractResolutionError",
    "GatewayConsistencyViolation",
    "GatewayNamespaceViolation",
    "InvariantViolation",
    "PrepareStageError",
    "RecipeValidationError",
    "RecipeValidationIssue",
    "TerminalRunRetryError",
    "TrainingRunMutationViolation",
    "AllocationConsistencyViolation",
    "PreparedContextConsistencyViolation",
    "TimelineConsistencyViolation",
    "PREPARE_BASELINE_OVERRIDE_EXISTING_BASELINE",
]
