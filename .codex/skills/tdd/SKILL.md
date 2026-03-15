---
name: tdd
description: Use this skill when implementing or changing code in this repository with a test-driven workflow. It enforces red-green-refactor with pytest first, then minimal code changes, then cleanup and linting.
---

# TDD Workflow

## Overview

Use this skill for feature work, bug fixes, and refactors that change behavior in this repo.
It keeps development strict: write a failing test first, implement minimal code to pass, then refactor while keeping tests green.

## When To Use

Use this skill when:
1. Adding behavior in `src/mlflow_monitor/`.
2. Fixing regressions or bugs.
3. Changing interfaces, contracts, or output formats.
4. Refactoring logic with behavior-preservation requirements.

Skip this skill when:
1. Editing docs only.
2. Pure formatting/comment cleanup with no behavior changes.
3. One-off exploratory work where tests are intentionally deferred.

## Repository Defaults

1. Use Python 3.12+ and `uv`.
2. Put runtime code in `src/mlflow_monitor/`
3. Put tests in `tests/` using `test_*.py` and `test_<behavior>()`.
4. Run:
   - `uv run pytest`
   - `uv run ruff check .`
5. Use deterministic, network-independent unit tests unless explicitly needed.

## Workflow (Red -> Green -> Refactor)

### 1. Red: Specify Behavior First

1. Identify the smallest externally visible behavior change.
2. Write or update one failing test that captures only that behavior.
3. Confirm failure explicitly:
   - `uv run pytest tests/<target_test_file>.py -k <test_name>`
4. If the test does not fail for the expected reason, fix the test before code changes.

### 2. Green: Implement Minimal Change

1. Implement only what is needed to make the failing test pass.
2. Re-run the focused test first.
3. Then run adjacent tests likely impacted.
4. Avoid broad refactors in this phase.

### 3. Refactor: Improve Without Behavior Drift

1. Refactor naming/structure only after tests are green.
2. Keep changes small and re-run tests frequently.
3. Re-run full suite for confidence:
   - `uv run pytest`
4. Ensure lint still passes:
   - `uv run ruff check .`

## Test Scope Rules

1. Start with the narrowest failing unit test.
2. Add integration-style tests only when behavior crosses module boundaries.
3. Do not overfit tests to internal implementation details.
4. Prefer explicit fixtures/test data over hidden globals.

## Change Checklist

Before finishing:
1. At least one failing test was observed before implementation.
2. New/updated behavior is covered by tests.
3. All relevant tests pass.
4. Ruff check passes.
5. Notes include what behavior changed and why.

## Common Commands

1. Run one test file:
   - `uv run pytest tests/test_<name>.py`
2. Run one test:
   - `uv run pytest tests/test_<name>.py -k <test_case>`
3. Run all tests:
   - `uv run pytest`
4. Lint:
   - `uv run ruff check .`

## Output Expectations

When this skill is used, outputs should include:
1. The failing test that drove the change.
2. The minimal implementation change summary.
3. Evidence that tests and lint pass (or explicit blockers).
