---
name: tdd
description: Use this skill when implementing or changing code in this repository with a test-driven workflow. It enforces red-green-refactor with pytest first, then minimal code changes, then cleanup and linting.
---

# TDD Workflow

## Overview

Use this skill for feature work, bug fixes, and behavior-preserving refactors in this repo.
Start behavior changes with a failing test. For a behavior-preserving refactor,
establish a green test baseline before editing and keep it green.

## When To Use

Use this skill when:
1. Adding behavior in `src/mlflow_monitor/`.
2. Fixing regressions or bugs.
3. Changing interfaces, contracts, or output formats.
4. Refactoring logic with behavior-preservation requirements.

Skip this skill when:
1. Editing docs only.
2. Pure formatting/comment cleanup with no behavior changes.
3. One-off exploration that produces no deliverable code changes.

## Repository Defaults

1. Use Python 3.12+ and `uv`.
2. Put runtime code in `src/mlflow_monitor/`
3. Put tests in `tests/` using `test_*.py` and `test_<behavior>()`.
4. Run:
   - `uv run poe test`
   - `uv run poe lint`
   - `uv run poe format-check`
   - `uv run poe typecheck`
5. Use deterministic, network-independent unit tests unless explicitly needed.

## Workflow (Red -> Green -> Refactor)

For a behavior-preserving refactor, run the relevant existing tests first. Add
passing characterization coverage if needed, then make the smallest refactor
and rerun those tests. Do not create an artificial failing test.

### 1. Red: Specify Behavior First

For features, bug fixes, and other behavior changes:

1. Identify the smallest externally visible behavior change.
2. Write or update one failing test that captures only that behavior.
3. Confirm failure explicitly:
   - `uv run poe test tests/<target_test_file>.py -k <test_name>`
4. If the test does not fail for the expected reason, fix the test before code changes.

### 2. Green: Implement Minimal Change

For behavior changes:

1. Implement only what is needed to make the failing test pass.
2. Re-run the focused test first.
3. Then run adjacent tests likely impacted.
4. Avoid broad refactors in this phase.

### 3. Refactor: Improve Without Behavior Drift

1. Refactor naming/structure only after tests are green.
2. Keep changes small and re-run tests frequently.
3. Re-run full suite for confidence:
   - `uv run poe test`
4. Ensure lint still passes:
   - `uv run poe lint`
5. Before closing the ticket, run the complete validation gate defined in
   `AGENTS.md`, including format check, Pyright, and build.

## Test Scope Rules

1. Start behavior changes with the narrowest failing unit test; start
   behavior-preserving refactors with focused passing tests.
2. Add integration-style tests only when behavior crosses module boundaries.
3. Do not overfit tests to internal implementation details.
4. Prefer explicit fixtures/test data over hidden globals.

## Change Checklist

Before finishing:
1. A relevant failing test was observed before a behavior change, or a green
   test baseline was established before a behavior-preserving refactor.
2. Changed behavior is covered by tests; refactors preserve existing or
   characterization coverage.
3. All relevant tests pass.
4. Ruff check passes.
5. The complete `AGENTS.md` ticket-close gate passes.
6. Notes include what changed and why.

## Common Commands

1. Run one test file:
   - `uv run poe test tests/test_<name>.py`
2. Run one test:
   - `uv run poe test tests/test_<name>.py -k <test_case>`
3. Run all tests:
   - `uv run poe test`
4. Lint:
   - `uv run poe lint`
5. Check formatting:
   - `uv run poe format-check`
6. Type-check:
   - `uv run poe typecheck`

## Output Expectations

When this skill is used, outputs should include:
1. The failing test that drove a behavior change, or the green baseline used
   for a behavior-preserving refactor.
2. The minimal implementation or refactor summary.
3. Evidence that tests and lint pass (or explicit blockers).
4. Evidence that the complete ticket-close gate passes before the ticket is
   declared complete.
