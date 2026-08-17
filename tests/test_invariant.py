"""Unit tests for invariants in mlflow_monitor."""

import pytest

from mlflow_monitor.domain import (
    ComparabilityStatus,
    Contract,
    ContractCheckReason,
    ContractCheckResult,
)
from mlflow_monitor.errors import InvariantViolation
from mlflow_monitor.invariant import (
    validate_contract_check_result,
)

CONTRACT = Contract(
    contract_id="default",
    contract_version="v0",
    schema_contract_ref=None,
    feature_contract_ref=None,
    metric_contract_ref=None,
    data_scope_contract_ref=None,
    execution_contract_ref=None,
)


class TestInvariantContractCheckResult:
    def test_contract_check_result_accepts_pass_without_reasons(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.PASS,
            reasons=(),
        )

        validate_contract_check_result(result)

    def test_contract_check_result_accepts_warn_with_environment_mismatch(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.WARN,
            reasons=(
                ContractCheckReason(
                    code="environment_mismatch",
                    message="Execution environment does not match the baseline.",
                    blocking=False,
                ),
            ),
        )

        validate_contract_check_result(result)

    def test_contract_check_result_accepts_fail_with_blocking_reason(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.FAIL,
            reasons=(
                ContractCheckReason(
                    code="schema_mismatch",
                    message="Data schema does not match the baseline.",
                    blocking=True,
                ),
            ),
        )

        validate_contract_check_result(result)

    def test_contract_check_result_rejects_duplicate_reason_codes(self) -> None:
        reason = ContractCheckReason(
            code="environment_mismatch",
            message="Execution environment does not match the baseline.",
            blocking=False,
        )
        result = ContractCheckResult(
            status=ComparabilityStatus.WARN,
            reasons=(reason, reason),
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_contract_check_result(result)

        error = exc_info.value
        assert error.code == "contract_check_reason_code_duplicate"
        assert error.entity == "ContractCheckResult"
        assert error.field == "reasons"

    def test_duplicate_reason_code_takes_precedence_over_changed_content(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.WARN,
            reasons=(
                ContractCheckReason(
                    code="environment_mismatch",
                    message="Execution environment does not match the baseline.",
                    blocking=False,
                ),
                ContractCheckReason(
                    code="environment_mismatch",
                    message="Changed duplicate content.",
                    blocking=True,
                ),
            ),
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_contract_check_result(result)

        assert exc_info.value.code == "contract_check_reason_code_duplicate"

    def test_contract_check_result_rejects_unknown_status(self) -> None:
        result = ContractCheckResult(
            status="unknown_status",  # type: ignore
            reasons=(),
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_contract_check_result(result)

        error = exc_info.value
        assert error.code == "contract_check_status_unknown"
        assert error.entity == "ContractCheckResult"
        assert error.field == "status"

    def test_contract_check_result_rejects_pass_with_reasons(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.PASS,
            reasons=(
                ContractCheckReason(
                    code="environment_mismatch",
                    message="Execution environment does not match the baseline.",
                    blocking=False,
                ),
            ),
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_contract_check_result(result)

        error = exc_info.value
        assert error.code == "contract_check_status_reason_mismatch"
        assert error.entity == "ContractCheckResult"
        assert error.field == "status"

    def test_contract_check_result_rejects_warn_with_blocking_reason(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.WARN,
            reasons=(
                ContractCheckReason(
                    code="schema_mismatch",
                    message="Data schema does not match the baseline.",
                    blocking=True,
                ),
            ),
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_contract_check_result(result)

        error = exc_info.value
        assert error.code == "contract_check_status_reason_mismatch"
        assert error.entity == "ContractCheckResult"
        assert error.field == "status"

    def test_contract_check_result_rejects_warn_with_no_non_blocking_reason(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.WARN,
            reasons=(),
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_contract_check_result(result)

        error = exc_info.value
        assert error.code == "contract_check_status_reason_mismatch"
        assert error.entity == "ContractCheckResult"
        assert error.field == "status"

    def test_contract_check_result_rejects_fail_with_only_non_blocking_reasons(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.FAIL,
            reasons=(
                ContractCheckReason(
                    code="environment_mismatch",
                    message="Execution environment does not match the baseline.",
                    blocking=False,
                ),
            ),
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_contract_check_result(result)

        error = exc_info.value
        assert error.code == "contract_check_status_reason_mismatch"
        assert error.entity == "ContractCheckResult"
        assert error.field == "status"

    def test_contract_check_result_rejects_unknown_reason_code(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.WARN,
            reasons=(
                ContractCheckReason(
                    code="metric_mismatch",
                    message="Metric definition differs.",
                    blocking=False,
                ),
            ),
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_contract_check_result(result)

        error = exc_info.value
        assert error.code == "contract_check_reason_code_unknown"
        assert error.entity == "ContractCheckReason"
        assert error.field == "code"

    def test_contract_check_result_rejects_environment_mismatch_with_blocking_flag(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.WARN,
            reasons=(
                ContractCheckReason(
                    code="environment_mismatch",
                    message="Python version differs.",
                    blocking=True,
                ),
            ),
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_contract_check_result(result)

        error = exc_info.value
        assert error.code == "contract_check_reason_blocking_mismatch"
        assert error.entity == "ContractCheckReason"
        assert error.field == "blocking"

    def test_contract_check_result_rejects_schema_mismatch_without_blocking_flag(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.FAIL,
            reasons=(
                ContractCheckReason(
                    code="schema_mismatch",
                    message="Schema differs.",
                    blocking=False,
                ),
            ),
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_contract_check_result(result)

        error = exc_info.value
        assert error.code == "contract_check_reason_blocking_mismatch"
        assert error.entity == "ContractCheckReason"
        assert error.field == "blocking"
