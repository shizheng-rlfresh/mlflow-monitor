"""Custom exception types for MLflow-Monitor v0."""

from .allocation import AllocationConsistencyViolation
from .gateway import (
    GatewayConsistencyViolation,
    GatewayNamespaceViolation,
    TrainingRunMutationViolation,
)
from .invariant import InvariantViolation
from .prepared_context import PreparedContextConsistencyViolation
from .recipe import (
    ContractResolutionError,
    RecipeValidationError,
    RecipeValidationIssue,
)
from .timeline import TimelineConsistencyViolation
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
