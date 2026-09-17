---
name: mlflow-monitor-commit-discipline
description: Use for commit follow-through in MLflow-Monitor after a verified ticket, review, or documentation slice is ready. Enforces repository-specific validation, staging review, and private-file exclusions.
---

# MLflow-Monitor Commit Discipline

Use this skill after implementation or review work in this repo when changes are ready to commit.

## Purpose

Enforce the `AGENTS.md` rule: commit frequently at coherent green, reviewable
slices.

## When To Use

Use this skill when:
1. A behavior-changing ticket slice is green.
2. A review fix is complete and verified.
3. A coherent docs+code slice is ready and reviewable.

Do not use this skill when:
1. Work is still in a red or unstable state.
2. The user explicitly says not to commit yet.
3. The task is planning-only or pure exploration.
4. The worktree contains unrelated user changes that cannot be safely separated.

## Workflow

1. Confirm scope.
   - Run `git status --short` before staging.
   - Commit only the active ticket or review slice.
   - Do not bundle unrelated files.
   - For maintainer release or ticket work, read the private maintainer overlay
     when present.
   - Never stage private design, ADR, context, or overlay files excluded by the
     repository instructions.
   - Do not alter unrelated user staging. If the active slice cannot be isolated
     safely, report the staged files as a blocker.
2. Verify readiness.
   - Before each green ticket commit, run focused tests plus the applicable Ruff
     and Pyright checks required by `AGENTS.md`.
   - For documentation- or instruction-only changes, run the narrow validator
     that covers the changed artifact; do not invent runtime tests.
   - Before closing a ticket, and for release transitions or final acceptance,
     run the complete validation gate required by `AGENTS.md`.
   - Do not commit with known red-phase failures in the slice.
3. Review the unstaged diff.
   - Inspect `git diff --stat` and a path-limited diff for the active slice.
   - Confirm each changed file belongs to the slice before staging it.
4. Stage and review the index.
   - Stage only explicit files from the active slice.
   - Re-run `git status --short`.
   - Inspect `git diff --cached --stat` and the complete cached diff.
   - Confirm the index contains no unrelated or private files.
   - Confirm the commit is small and reviewable.
5. Commit by default.
   - Commit after each coherent green slice.
   - Prefer multiple small commits over one large commit.
6. Report clearly.
   - State the commit created and what slice it covers.
   - If no commit was made, state the blocker.

## Commit Rules

1. Default unit of commit: one green, reviewable ticket slice.
2. Preferred timing: after focused tests are green and before starting the next sub-slice.
3. Hold the commit only when:
   - the user asked to wait
   - unrelated worktree changes make the commit unsafe
   - the slice is still red or unstable
4. Never amend or rewrite history unless explicitly requested.
5. Never include unrelated user changes.
6. Avoid interactive git flows.

## Commit Messages

Use concise plain-language subjects, for example:

- `Add paired monitoring run identity`
- `Persist canonical prepared context`
- `Document MVP release transition`

Rules:
1. Do not require ticket, Plane, category, or Conventional Commit prefixes.
2. Keep the subject short and factual.
3. Use plain language; use specialized terms only when they improve precision.
4. Mention tests in the title only for test-only commits.

## Relationship To Other Skills

1. Use `seeking-design-truth` before implementation or review when behavior or scope needs grounding.
2. Use `tdd` for behavior-changing work.
3. Use this skill after the slice is green to handle commit follow-through.
