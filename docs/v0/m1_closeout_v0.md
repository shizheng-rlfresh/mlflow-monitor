# MLflow-Monitor v0: M1 Closeout and Post-M1 Handoff

Date: 2026-03-20

## Summary

M1 coding is complete for the intended core slice: synchronous create/prepare/check execution with minimal run/check persistence and a canonical synchronous result.

This document is the canonical post-M1 summary for implementers picking up the next phase of work. It records what M1 set out to deliver, what actually landed, the final implemented boundary, the tickets that were effectively closed in the process, and what remains as post-M1 backlog rather than unfinished M1 coding.

## What M1 Set Out To Do

M1 was the "core skeleton" milestone for v0. Its exit criterion was that runs can prepare, check, and persist status with recipe binding.

At a product level, M1 aimed to make the following path real:

- caller invokes `monitor.run(...)`
- the system resolves recipe and contract internally
- workflow executes `prepare -> check`
- minimal run/check state is persisted through the gateway
- the canonical synchronous result is returned

M1 was never intended to finish the full v0 design. Analyze, close, diff/finding generation, query support, and promotion behavior remained later-cycle work.

## What M1 Actually Delivered

The implemented M1 slice now includes:

- canonical runtime/domain models and invariant validation
- contract check result taxonomy with `pass` / `warn` / `fail`
- a concrete built-in contract checker boundary and default permissive contract
- a system default recipe plus deterministic recipe validation and compilation
- prepare-stage resolution for timeline, baseline, previous run, LKG, custom reference, source run, and required evidence
- first-run baseline bootstrap plus later baseline immutability enforcement
- orchestration for the synchronous create/prepare/check flow
- minimal gateway-backed persistence for timeline state and monitoring run state
- a canonical synchronous `MonitorRunResult`

The resulting runtime path is:

`monitor.run(...)` -> internal recipe resolution -> runtime contract resolution -> orchestration -> `prepare` -> `check` -> minimal run/check persistence -> synchronous `MonitorRunResult`

## Final Implemented M1 Boundary

The public M1 API remains intentionally small:

- `monitor.py` is a thin public facade
- recipe resolution remains internal
- contract resolution remains internal
- the caller provides `subject_id`, `source_run_id`, and first-run `baseline_source_run_id`

The implemented execution boundary ends at:

- create/prepare/check lifecycle progression
- comparability result production
- minimal run/check persistence through the gateway
- synchronous return of `MonitorRunResult`

The implemented M1 boundary does not include:

- analyze-stage branching
- diff generation
- finding generation
- close-stage persistence completeness
- promotion/LKG policy evaluation
- query APIs or anchor-window traversal
- recipe file-format loading beyond in-memory mapping inputs

## Ticket Closeout Summary

The following M1 tickets are now effectively closed in code and docs:

- `D-001`, `D-002`, `D-003`, `D-004`, `D-004D`
- `R-000`, `R-000a`, `R-001`, `R-002`, `R-003`
- `W-001`, `W-001a`, `W-002`, `W-002a`, `W-003`
- `P-001`

Additional closeout notes for tickets that were previously still called out as remaining:

- `R-000a` is satisfied by the shipped system default recipe, default runtime selector token, default contract binding, and tests that make those defaults observable.
- `D-004E` is satisfied by the design now reflected in `contract.py`, the distinction between recipe-side contract binding and runtime `Contract`, and the explicit ownership of binding resolution by the runtime contract module.
- `D-004F` is satisfied by the implemented `resolve_contract_v0(...)` path, the built-in default permissive contract, and explicit failure for unknown bindings.
- `W-003` is satisfied for the M1-scoped create/prepare/check orchestration and persistence slice.
- `O-001` is satisfied for the M1 closeout pass, while remaining an ongoing maintenance discipline for later milestones.

## Architectural Decisions That Matter Going Forward

The most important M1 architectural decisions to preserve are:

- `workflow.py` remains the home for pure workflow components and stage helpers such as transition logic, `prepare_run_context()`, and `execute_contract_check()`.
- orchestration owns use-case assembly: run allocation, lifecycle progression, gateway upserts, idempotent replay handling, and result assembly.
- `monitor.py` stays thin and should not absorb orchestration logic.
- contract handling in v0 is binding resolution, not a second compilation pipeline.
- recipe and contract remain internal runtime concerns for the M1 public API.
- the gateway remains the only MLflow-facing read/write boundary.

These decisions keep the public SDK surface small, preserve testable workflow components, and leave room for M2 analysis/close work without collapsing the layering established in M1.

## What Remains Unsolved After M1

What remains is no longer "remaining M1 coding." It is post-M1 backlog and unresolved future work.

### Later-cycle work already expected in M2/M3

- analyze-stage branching after check
- diff generation across baseline / previous / LKG / custom reference
- finding generation and summary hydration
- close-stage persistence completeness
- query and anchor-window behavior
- promotion/LKG decision workflows
- broader acceptance-scenario coverage for later lifecycle stages

### Explicitly deferred tickets

- `R-001D` recipe JSON/YAML file-format support
- `D-003D` `metric_mismatch` semantics
- `D-004DD` richer contract-check reason messages
- `W-001b` inline finding hydration in synchronous results

### Documentation and acceptance follow-through

- keep the full-v0 planning/design docs aligned with the now-completed M1 boundary
- preserve the distinction between "implemented M1 slice" and "planned full v0 system"
- continue acceptance/documentation work as M2 capabilities land

## Recommended Next Work

The next work should treat M1 as a completed foundation and move into M2 planning/execution rather than reopening M1.

Recommended direction:

1. We should aim for a MVP rather than chasing completeness of M2 from the original design.
2. A MVP should clear one goal: we can clear a path in real MLflow setting, even if it is just demo.
3. After that, we can switch back to M2 and make iterative devleopment in steady pace.

For future planning, use this document as the closeout reference for M1 and use the ticket docs for milestone/backlog structure rather than as the source of truth for what M1 still lacks.
