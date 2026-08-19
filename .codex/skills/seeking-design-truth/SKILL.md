---
name: seeking-design-truth
description: Use this skill when investigating design intent, ticket scope, architecture questions, behavioral ambiguity, or conflicting assumptions in this repository. It grounds reasoning in the repo's planning and design docs, then verifies conclusions against code and tests.
---

# Seeking Design Truth

## Overview

Use this skill when the task is to understand what the system is intended to do, what it currently does, or where design intent is unclear.
The goal is to ground work in the repo's planning and design docs, then separate observed facts from inference and make uncertainty explicit.

Read `AGENTS.md` first. If `.agents/v0-development.md` exists, read it completely
and obey its private authority, sequencing, terminology, and publication rules.

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

1. For private maintainer work, read the current ticket and dependencies in
   `docs/v0/design_docs/ticket_breakdown_post_mvp_v0.md`.
2. Read the relevant accepted ADRs, terms in `CONTEXT.md`, and sections of
   `docs/v0/design_docs/design_doc_post_mvp_v0.md`.
3. Read the M1 and MVP closeouts to establish delivered historical behavior.
4. Use `design_doc_v0.md` and `ticket_breakdown_v0.md` only for historical intent
   or contradiction checks; never restore their superseded behavior.
5. If the private overlay or private design record is absent, use the available
   public ticket, documentation, code, and tests without inferring or requesting
   the private plan.
6. Read only the additional material needed for the current task.

### 2. Inspect Current Implementation

1. Read the relevant code paths after reviewing the applicable planning and design docs.
2. Read nearby tests before inferring intent from implementation alone.
3. Prefer the narrowest files and symbols that can resolve the question.
4. Use focused test runs or minimal reproductions when they can settle ambiguity quickly.

### 3. Classify Current-To-Target Changes

For ticket plans and implementation plans:

1. Inventory only affected behaviors, interfaces, persisted representations,
   tests, test helpers and doubles, and documentation surfaces.
2. Classify each affected surface:
   - `add`: Introduce a new surface.
   - `modify`: Keep the surface while changing or extending its behavior.
   - `replace`: Remove an existing surface and introduce a successor for its role.
   - `remove`: Eliminate a surface without a successor.
3. Separately list unchanged invariants at realistic risk of regression. Do not
   classify every unaffected surface as retained.
4. State compatibility separately as `none`, `API compatibility`, or
   `persisted-data migration`. For a migration, describe how legacy state is
   read, repaired, migrated, or rejected.
5. For replaced or removed surfaces, identify obsolete call sites, tests, test
   helpers and doubles, and documentation that must be updated or deleted.
6. Do not preserve an obsolete interface through a compatibility alias when the
   alias cannot satisfy the new invariant.

### 4. Distinguish Fact From Inference

1. State what is directly observed from docs, code, tests, command output, or repo configuration.
2. Label inferences explicitly when behavior is implied rather than proven.
3. If multiple interpretations remain plausible, say so instead of collapsing them into one answer.

### 5. Resolve Contradictions

1. Treat code, tests, and docs as separate evidence sources that may disagree.
2. Surface contradictions explicitly.
3. When sources conflict, distinguish between:
   - intended design in current MVP docs
   - intended design in older repo docs
   - current behavior in code
   - enforced behavior in tests
4. If the conflict changes implementation risk, stop and call it out.
5. When the private record exists, apply its authority order. Stop if an accepted
   ADR and the canonical post-MVP design disagree.

### 6. Report Clearly

Outputs using this skill should include:
1. Confirmed facts.
2. Important inferences or assumptions.
3. Open questions or contradictions.
4. The files or tests that support the conclusion.

Ticket plans and implementation plans should also include:

| Surface | Change | Compatibility | Implementation action | Test/docs action |
| --- | --- | --- | --- | --- |

List protected unchanged invariants immediately after the table. Keep the table
limited to affected surfaces.

## Heuristics

1. Do not present guesses as settled design.
2. Do not rely on a single source when the repo provides stronger evidence.
3. For ticket implementation, establish ticket scope and dependencies, then apply
   the private design-authority order before inspecting code and tests.
4. For review, inspect code and tests first, then pull design docs when expected behavior or scope is unclear.
5. Never stage, publish, or publicly link private design files discovered through
   this workflow.
6. If confidence is low, say what evidence is missing.
