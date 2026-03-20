# MLflow-Monitor v0: Remaining M1 Tickets as of 2026-03-20

## Summary

This document records the remaining M1 work after the 2026-03-20 planning review.

M1 remains focused on one synchronous core execution path:

- caller invokes `monitor.run(...)`
- the system resolves recipe and contract internally
- workflow executes `prepare -> check`
- minimal run/check state is persisted through the gateway
- the canonical synchronous result is returned

The public API for M1 stays aligned with the design doc and remains intentionally small. Recipe resolution, contract binding resolution, run allocation, lifecycle mutation, and gateway upsert behavior remain internal implementation concerns.

## Architectural Decision

For M1, orchestration should not live in `monitor.py` and should not be added to `workflow.py`.

The intended layering is:

- `workflow.py`
  - pure workflow components and stage helpers
  - examples: transition rules, `prepare_run_context()`, `execute_contract_check()`
- orchestration module
  - internal use-case layer that assembles the execution slice
  - owns run allocation, lifecycle progression, gateway upserts, and result assembly
- `monitor.py`
  - thin public facade only
  - delegates to orchestration and returns the result

This keeps the SDK boundary thin, preserves `workflow.py` as a pure component module, and avoids mixing user-facing API code with execution orchestration.

## Remaining M1 Tickets

### `R-000a` Define zero-config default behavior details

Purpose:
Make the built-in default behavior explicit and testable as runtime rules rather than leaving it implicit in scattered implementation details.

Remaining scope:

- lock the default source-run selection behavior
- lock the default contract behavior
- make behavior for optional missing evidence explicit
- ensure tests capture those defaults as observable behavior

Notes:

- parts of this already exist through the system default recipe and runtime selector token
- the remaining work is to normalize and explicitly close the ticket

### `D-004E` Design contract model and binding resolution for v0

Purpose:
Define the internal bridge from recipe-side contract binding to workflow-facing runtime `Contract`.

Remaining scope:

- define what runtime `Contract` represents in v0
- define how recipe-selected contract binding maps to runtime `Contract`
- assign runtime contract resolution responsibility to `contract.py`
- define the built-in default permissive contract as a real system concept
- define failure behavior for unknown contract bindings

Notes:

- this is an internal architecture ticket
- M1 does not expose runtime contract objects in the public API
- M1 does not introduce a contract compiler; contract handling is binding resolution only

### `D-004F` Implement contract binding resolution for v0

Purpose:
Implement the resolver that turns recipe contract binding into a workflow-ready runtime `Contract`.

Remaining scope:

- add runtime contract resolution in app code
- place the resolver in `contract.py`
- support the system default permissive contract
- fail explicitly on unknown contract bindings
- keep workflow consuming resolved `Contract`, not raw recipe binding

Notes:

- current workflow already expects a resolved `Contract`
- this resolver is required for real orchestration

### `W-003` Remaining orchestration and persistence slice

Status:
Partially implemented.

Already done:

- gateway support was extended for check-stage needs
- workflow exposes `execute_contract_check()`
- prepared context can be checked and can produce `ContractCheckResult`

Remaining scope:

- add the internal orchestration slice that assembles `prepare -> check`
- progress lifecycle state at run level across the execution slice
- persist prepared/checked run state through gateway upserts
- wire the orchestration through the public synchronous path
- assemble the canonical `MonitorRunResult`

Notes:

- `workflow.py` should remain a pure components module
- the remaining `W-003` work belongs in a dedicated orchestration module, not in `workflow.py`
- `monitor.py` remains a thin facade and should not own execution orchestration

### `O-001` Keep design docs synchronized

Purpose:
Keep the v0 docs aligned with the actual implemented M1 boundary.

Remaining scope:

- update docs to reflect the thin public API
- clarify that recipe and contract resolution are internal in M1
- clarify that M1 ends at checked result plus persisted run/check status
- record deferrals to M2 where relevant

## Explicitly Deferred From M1

The following tickets are not part of M1:

- `R-001D` Support JSON and YAML in recipe
- `D-003D` define and implement `metric_mismatch`
- `D-004DD` support richer contract-check reason messages
- `W-001b` hydrate synchronous result with inline findings
- all M2 analysis-stage work such as `E-001`, `E-002`, `W-004`, `P-002`, and query tickets

## M1 Boundary

M1 is complete when a caller can synchronously invoke:

```python
monitor.run(
    subject_id="churn_model",
    source_run_id="train_run_2026_03_10",
    baseline_source_run_id="train_run_2026_03_01",
)
```

and the system will internally:

- resolve the default recipe
- compile it
- resolve the runtime contract
- allocate the monitoring run
- prepare context
- execute the comparability check
- persist minimal run/check state through the gateway
- return the canonical `MonitorRunResult`

without exposing recipe objects or contract objects in the public API.

## Assumptions

- the M1 public API remains aligned with the design doc
- recipe and contract stay internal runtime concerns for M1
- JSON/YAML recipe support is deferred to M2
- findings, diffs, analyze-stage behavior, and inline finding hydration are out of scope for M1
