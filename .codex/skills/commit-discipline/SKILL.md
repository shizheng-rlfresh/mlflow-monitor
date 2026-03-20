---
name: commit-discipline
description: Use this skill when doing code-changing work in this repository that should end with a small, reviewable commit, especially after a green TDD slice or completed ticket slice.
---

# Commit Discipline

Use this skill after implementation or review work in this repo when changes are ready to commit.

## Purpose

Enforce the `AGENTS.md` rule: commit very frequently by default.

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
   - Commit only the active ticket or review slice.
   - Do not bundle unrelated files.
2. Verify readiness.
   - Run the touched-area tests first.
   - Run broader required checks when the slice warrants it.
   - Do not commit with known red-phase failures in the slice.
3. Review the diff.
   - Inspect `git diff --stat` and a narrow diff.
   - Confirm the commit is small and reviewable.
4. Commit by default.
   - Commit after each coherent green slice.
   - Prefer multiple small commits over one large commit.
5. Report clearly.
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

Prefer the cleaner ticket-oriented style used in this repo:

- `[MM-9] W-002 add prepare-stage workflow context`
- `[MM-18] R-003 add recipe compiler`

Rules:
1. Include the ticket id when known.
2. Infer ticket id, e.g., `MM-11` from branch name, e.g., `MM-11/xxx`
2. Keep the subject short and factual.
3. Use imperative/lowercase style.
4. Mention tests in the title only for test-only commits.

## Relationship To Other Skills

1. Use `seeking-design-truth` before implementation or review when behavior or scope needs grounding.
2. Use `tdd` for behavior-changing work.
3. Use this skill after the slice is green to handle commit follow-through.
