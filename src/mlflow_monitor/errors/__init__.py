"""Custom exception types for MLflow-Monitor v0."""

from .gateway import (
    AllocationConsistencyViolation,
    GatewayConsistencyViolation,
    GatewayNamespaceViolation,
    PreparedContextConsistencyViolation,
    TrainingRunMutationViolation,
)
from .invariant import InvariantViolation
from .recipe import (
    ContractResolutionError,
    RecipeValidationError,
    RecipeValidationIssue,
)
from .workflow import (
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
]
