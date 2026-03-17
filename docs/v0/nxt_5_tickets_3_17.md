# Next 5 Ticket Proposal

## Summary

Reviewing the updated ticket docs against the implemented code and the dependency graph, the fastest coherent path is still to finish the remaining M1 execution chain before pulling in deferred enhancements or M2 analysis work.

Recommended next 5 tickets, in order:

1. `R-003` Build recipe compilation pipeline
2. `W-002` Implement prepare stage
3. `W-002a` Implement first-run baseline bootstrap rule
4. `D-004D` Concrete contract checker
5. `W-003` Implement comparability check stage

This sequence is the shortest path to “real run can prepare and check against resolved context,” which is the core missing capability between the current skeleton and a minimally usable product.

## Recommended Tickets

### 1. `R-003` Recipe compilation pipeline

Why first:

- It is the explicit dependency for `W-002`.
- It is the missing contract between today’s validated recipe model and the workflow prepare/check stages.
- It is also one of the main currently unclear areas in the design, so settling it unlocks later work cleanly.

What it unlocks:

- A stable run-plan shape for prepare-stage inputs.
- Deterministic default resolution without workflow-side reinterpretation.

### 2. `W-002` Prepare stage

Why second:

- This is the first real runtime orchestration ticket.
- It resolves the baseline, previous run, LKG, custom reference, source run, and resolved contract, which every later step depends on.
- It turns the current skeleton into an actual execution path.

What it unlocks:

- Prepared runtime context for contract checking.
- First meaningful end-to-end monitoring behavior.

### 3. `W-002a` First-run baseline bootstrap rule

Why third:

- It is a concrete, high-risk part of prepare semantics.
- The design is explicit that bootstrap must not be hidden or inferred.
- It completes the critical first-run path and baseline pinning semantics before comparability logic builds on top of it.

What it unlocks:

- Correct first-run behavior.
- Strong baseline immutability semantics at the workflow boundary.

### 4. `D-004D` Concrete contract checker

Why fourth:

- `D-004` gives only the boundary; the real checker is still missing.
- `W-003` without a concrete checker would be mostly wiring with no substantive comparability behavior.
- The deferred ticket exists precisely because the provisional checker protocol is not enough for a real check stage.

What it unlocks:

- Actual comparability evaluation logic rather than a placeholder interface.
- Clearer baseline-side evidence expectations before wiring `W-003`.

### 5. `W-003` Comparability check stage

Why fifth:

- It depends naturally on `W-002` plus a real checker.
- It completes the core M1 “prepare + check” behavior.
- It is the ticket that turns the system from structural scaffolding into an actual comparability-first monitor.

What it unlocks:

- Executed contract checking in workflow.
- Persisted check reasons and explicit comparability state.
- Clean handoff to later M2 analysis work.

## Why These 5 Over The Others

Not recommended in the next 5:

- `D-003D`
  - Important, but `metric_mismatch` is explicitly deferred and not required to establish the first real check path.
- `R-001D`
  - Purely deferred format support; not relevant to minimal usable v0.
- `E-001`, `E-002`, `W-004`, `P-002`, `Q-*`
  - These are M2 and should come after prepare/check is real.
- `R-000a`
  - This ticket is still open in docs, but the current code and tests already cover much of its practical behavior through the system-default recipe, default contract ID, and runtime selector token. I would not prioritize it ahead of the execution-chain blockers unless you want to explicitly close its remaining acceptance gaps first.

## Test / Validation Focus

For this 5-ticket slice, acceptance should be:

- `R-003`
  - same recipe/version compiles to the same run plan
  - compiled output contains everything `W-002` and `W-003` need
- `W-002`
  - successful prepare resolves all required references
  - missing required metrics/artifacts fails explicitly
- `W-002a`
  - first run without `baseline_source_run_id` fails explicitly
  - first run with explicit baseline pins timeline state deterministically
- `D-004D`
  - checker can evaluate prepared evidence against resolved contract with deterministic reasons
- `W-003`
  - no analysis before check
  - pass/warn/fail outcomes are produced from real prepared context and checker logic

## Assumptions

- We are optimizing for shortest path to a minimally usable v0, not for polishing deferred P1/P2 work.
- The newly added deferred tickets (`R-001D`, `D-003D`, `D-004D`) should not automatically jump ahead of the core M1 execution chain; only `D-004D` does because it directly blocks a meaningful `W-003`.
- `R-000a` appears substantially implemented already, so I am treating it as lower priority than the remaining blocked execution tickets unless you want a stricter doc-driven closeout of that ticket before moving forward.
