"""Custom exception types for MLflow-Monitor v0."""

from .allocation import AllocationConsistencyViolation, AllocationInconsistentReason
from .gateway import (
    GatewayConsistencyCode,
    GatewayConsistencyViolation,
    GatewayNamespaceViolation,
    TrainingRunMutationViolation,
)
from .invariant import InvariantViolation
from .prepared_context import PreparedContextConsistencyViolation, PreparedContextInconsistentReason
from .recipe import (
    ContractResolutionError,
    RecipeValidationError,
    RecipeValidationIssue,
)
from .timeline import TimelineConsistencyViolation, TimelineInconsistentReason
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
    "TimelineInconsistentReason",
    "PREPARE_BASELINE_OVERRIDE_EXISTING_BASELINE",
    "GatewayConsistencyCode",
    "AllocationInconsistentReason",
    "PreparedContextInconsistentReason",
]
