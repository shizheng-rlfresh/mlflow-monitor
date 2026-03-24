---
name: seeking-design-truth
description: Use this skill when investigating design intent, ticket scope, architecture questions, behavioral ambiguity, or conflicting assumptions in this repository. It grounds reasoning in the repo's planning and design docs, then verifies conclusions against code and tests.
---

# Seeking Design Truth

## Overview

Use this skill when the task is to understand what the system is intended to do, what it currently does, or where design intent is unclear.
The goal is to ground work in the repo's planning and design docs, then separate observed facts from inference and make uncertainty explicit.

## When To Use

Use this skill when:
1. Investigating behavioral ambiguity or regressions.
2. Explaining architecture or design intent.
3. Implementing or planning ticket-driven work.
4. Reviewing code where expected behavior or scope is unclear.
5. Comparing implementation, tests, and docs for contradictions.
6. Planning changes that depend on understanding current semantics.

Skip this skill when:
1. Making straightforward mechanical edits with no ambiguity.
2. Editing docs or formatting only.
3. The user already provided the exact behavior to implement and no verification is needed.

## Workflow

### 1. Ground In Planning And Design Docs

1. For current MVP work, start with `docs/v0/mvp/mvp_design_v0.md`.
2. For current MVP tickets, read `docs/v0/mvp/mvp_tickets.md`.
3. Read `docs/v0/m1_closeout_v0.md` to understand the intended M1 scope and what is already considered "done" for M1.
4. Read broader v0 docs such as `docs/v0/design_doc_v0.md` and `docs/v0/ticket_breakdown_v0.md` only as supporting context, historical reference, or contradiction checks.
5. Read `docs/v0/tickets_m1_v0.md` only when you need specific M1 ticket details and execution order.
6. Read only the additional docs needed for the specific task.


### 2. Inspect Current Implementation

1. Read the relevant code paths after reviewing the applicable planning and design docs.
2. Read nearby tests before inferring intent from implementation alone.
3. Prefer the narrowest files and symbols that can resolve the question.
4. Use focused test runs or minimal reproductions when they can settle ambiguity quickly.

### 3. Distinguish Fact From Inference

1. State what is directly observed from docs, code, tests, command output, or repo configuration.
2. Label inferences explicitly when behavior is implied rather than proven.
3. If multiple interpretations remain plausible, say so instead of collapsing them into one answer.

### 4. Resolve Contradictions

1. Treat code, tests, and docs as separate evidence sources that may disagree.
2. Surface contradictions explicitly.
3. When sources conflict, distinguish between:
   - intended design in current MVP docs
   - intended design in older repo docs
   - current behavior in code
   - enforced behavior in tests
4. For MVP tickets, treat `docs/v0/mvp/` as the current intended source unless code/tests clearly enforce something else.
5. If the conflict changes implementation risk, stop and call it out.

### 5. Report Clearly

Outputs using this skill should include:
1. Confirmed facts.
2. Important inferences or assumptions.
3. Open questions or contradictions.
4. The files or tests that support the conclusion.

## References

- `docs/v0/ticket_breakdown_v0.md`: ticket list, scope boundaries, and where to find the relevant ticket.
- `docs/v0/mvp/mvp_design_v0.md`: current MVP design and execution boundary.
- `docs/v0/mvp/mvp_tickets.md`: current MVP ticket list and sequencing.
- `docs/v0/tickets_m1_v0.md`: M1 cycle ticket details and execution order.
- `docs/v0/design_doc_v0.md`: intended architecture and monitoring semantics.
- `docs/v0/workflow_v0.md`: lifecycle and workflow behavior.
- `docs/v0/recipe_v0.md`: recipe behavior and user-facing flows.
- `docs/v0/mlflow_mapping_v0.md`: MLflow mapping and persistence semantics.
- `docs/v0/cast_v0.md`: cast concepts and terminology.
- `docs/v0/mlflow_glossary_v0.md`: glossary and term disambiguation.

## Heuristics

1. Do not present guesses as settled design.
2. Do not rely on a single source when the repo provides stronger evidence.
3. For ticket implementation, start from ticket docs, then design docs, then code and tests.
4. For review, inspect code and tests first, then pull design docs when expected behavior or scope is unclear.
5. If confidence is low, say what evidence is missing.
