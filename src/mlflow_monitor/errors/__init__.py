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
    ANALYZE_FINDING_POLICY_EVALUATION_FAILED,
    ANALYZE_FINDING_POLICY_OUTPUT_INCONSISTENT,
    ANALYZE_FINDING_POLICY_OUTPUT_INVALID,
    ANALYZE_MISSING_CURRENT_SOURCE_RUN,
    PREPARE_BASELINE_OVERRIDE_EXISTING_BASELINE,
    AnalyzeStageError,
    CheckStageError,
    PrepareStageError,
    TerminalRunRetryError,
)

__all__ = [
    "ANALYZE_FINDING_POLICY_EVALUATION_FAILED",
    "ANALYZE_FINDING_POLICY_OUTPUT_INCONSISTENT",
    "ANALYZE_FINDING_POLICY_OUTPUT_INVALID",
    "ANALYZE_MISSING_CURRENT_SOURCE_RUN",
    "AnalyzeStageError",
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
