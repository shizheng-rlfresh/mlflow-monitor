# MLflow-Monitor Product Runtime Diagram

## Purpose

This document accompanies the Excalidraw source at [mlflow_monitor_product_runtime.excalidraw](/Users/zhengshi/src/mlflow-monitor/assets/arch_v0/mlflow_monitor_product_runtime.excalidraw).

It shows two things at once:

- The final intended MLflow-Monitor product structure and runtime flow.
- The current implementation status in this repository as of today.

The diagram is product-first, not ticket-first. This file is the authoritative place for detailed status notes, unclear areas, and ticket mappings for unfinished work.

## How To Read The Diagram

The main path runs left to right:

1. A caller invokes MLflow-Monitor through the SDK, with a future CLI expected to wrap the same flow.
2. The system captures the run request and resolves the effective recipe.
3. The workflow prepares runtime context: timeline, baseline, previous run, LKG, source training run, and contract inputs.
4. The workflow runs comparability checks before any analysis.
5. Analysis produces diffs and findings.
6. Persistence writes monitoring-owned state into MLflow.
7. The system returns a synchronous result, and may later evaluate promotion to update LKG.

The lower band shows the architecture layers:

- `Domain` owns semantic truth and invariants.
- `Workflow` owns execution order and lifecycle.
- `Recipe` owns configuration and run-plan intent.
- `Gateway` owns all MLflow-facing reads and writes.
- `MLflow` is the external persistence system and training-run source.

The diagram uses four status labels:

- `Done`: implemented and covered by current code/tests.
- `Partial`: part of the final subsystem exists, but the end-to-end product behavior is incomplete.
- `Not Done`: planned behavior is documented but not implemented yet.
- `Unclear`: the design intent exists, but the final implementation shape is not yet fixed enough to claim a settled interface or storage model.

## What Is Clear / Known / Done

### Domain And Contract Foundations

These parts are implemented and backed by tests:

- Canonical runtime models for run, timeline, baseline, LKG, contract, diff, and finding are defined in [domain.py](/Users/zhengshi/src/mlflow-monitor/src/mlflow_monitor/domain.py).
- Core invariant checks are implemented in [invariant.py](/Users/zhengshi/src/mlflow-monitor/src/mlflow_monitor/invariant.py).
- The contract-check result taxonomy and validation rules are implemented, including `pass`, `warn`, and `fail`, with `metric_mismatch` explicitly deferred.
- The workflow-facing contract checker boundary and prepared evaluation context are implemented in [contract_checker.py](/Users/zhengshi/src/mlflow-monitor/src/mlflow_monitor/contract_checker.py).

Role in the final product:

- These pieces define what the monitoring outputs mean.
- They prevent invalid ownership, baseline mutation, bad evidence linkage, and invalid contract-check results.

### Recipe Foundations

The recipe layer is partly implemented and its current scope is clear:

- The v0-lite recipe schema is implemented in [recipe.py](/Users/zhengshi/src/mlflow-monitor/src/mlflow_monitor/recipe.py).
- The built-in system default recipe is implemented.
- Recipe validation is implemented, including structural validation, referential checks, and constraint validation.

Role in the final product:

- Recipes are the customization surface for monitoring intent.
- The default recipe keeps zero-config behavior explicit instead of hiding it in workflow code.

### Workflow Skeleton

The workflow skeleton is implemented, but not the actual stage execution:

- Lifecycle transitions are implemented in [workflow.py](/Users/zhengshi/src/mlflow-monitor/src/mlflow_monitor/workflow.py).
- The canonical synchronous result envelope is implemented in [result_contract.py](/Users/zhengshi/src/mlflow-monitor/src/mlflow_monitor/result_contract.py).

Role in the final product:

- These pieces establish the lifecycle contract and the public shape of a monitoring result.
- They give the runtime a stable envelope for success and failure responses.

### Gateway Abstraction

The gateway abstraction exists and its intended boundary is clear:

- The gateway protocol and in-memory implementation are in [gateway.py](/Users/zhengshi/src/mlflow-monitor/src/mlflow_monitor/gateway.py).
- Namespace rules and training-run immutability protections are already enforced in the in-memory gateway.

Role in the final product:

- The gateway is the only boundary allowed to touch MLflow.
- It enforces “read training experiments, write only to monitoring namespace.”

## What Is Clear But Not Done Yet

### Recipe Pipeline

- Recipe compilation into an executable run plan is clearly intended but not implemented. Ticket: `R-003`.
- The design and docs clearly expect workflow to consume a compiled plan rather than reinterpret raw recipe config. Ticket: `R-003`.

### Workflow Orchestration

- The prepare stage is clearly part of the final flow and should resolve timeline, baseline, previous run, LKG, source run, and contract inputs. Ticket: `W-002`.
- First-run baseline bootstrap behavior is clearly required, including the explicit `baseline_source_run_id` rule when no timeline exists. Ticket: `W-002a`.
- Executed comparability check as a workflow stage is planned and clearly scoped, but not wired into runtime flow. Ticket: `W-003`.
- Analysis branching between comparable and non-comparable paths is clearly described in the design docs, but not implemented. Ticket: `W-004`.
- The result envelope is expected to include richer inline findings in M2, beyond the current ID-only references. Ticket: `W-001b`.
- The optional LKG promotion stage is planned and clearly part of the final product story, but not implemented. Ticket: `W-005`.

### Analysis Engines

- The diff engine is clearly required for baseline, previous, LKG, and optional custom-reference comparison. Ticket: `E-001`.
- The finding engine is clearly required to convert evidence into severity-ranked outputs. Ticket: `E-002`.
- An initial packaged finding-policy layer is planned after the basic engine exists. Ticket: `E-003`.

### Persistence And Query

- Stage-aligned persistence writes are not implemented. Ticket: `P-002`.
- Canonical persisted artifact schemas for diffs, findings, summaries, references, and metadata are not implemented. Ticket: `P-002a`.
- LKG metadata persistence and deterministic retrieval semantics are not implemented. Ticket: `P-003`.
- Timeline traversal and anchor-window query APIs are clearly planned, but not implemented. Tickets: `Q-001`, `Q-002`.
- Preset baseline-anchored and LKG-anchored queries are planned for later. Ticket: `Q-003`.

### Validation And Ops

- The acceptance scenario harness covering end-to-end behavior is not implemented. Ticket: `T-001`.
- The invariant regression suite is planned later, beyond the current unit coverage. Ticket: `T-002`.
- Failure diagnosis runbook and pilot onboarding checklist are planned documentation deliverables. Tickets: `O-002`, `O-003`.

## What Is Not Clear Yet

These areas are not just unfinished. Their final implementation shape is not yet fixed enough in the repo to treat them as settled contracts.

### Compiled Run-Plan Shape

The design consistently says recipes should compile into an executable run plan, but the concrete run-plan schema is not defined in code or docs with enough precision to count as fixed.

Relevant ticket:

- `R-003`

### Persisted Artifact Contract

The system clearly intends to persist diffs, findings, summaries, and references, but the final artifact schemas and storage/read contract are not yet concretely defined.

Relevant tickets:

- `P-002`
- `P-002a`

### Inline Findings In Public Result

The current result contract is implemented, but the future M2 shape for inline structured findings is still only described at ticket level.

Relevant ticket:

- `W-001b`

### Promotion Semantics

The product story clearly includes LKG promotion, but the precise gate inputs, persisted decision model, and retrieval semantics are not settled in code.

Relevant tickets:

- `P-003`
- `W-005`

### Query Read Model

Timeline traversal and anchor-window behavior are well-motivated, but the exact output contract for these APIs is still not concretely defined in runtime code.

Relevant tickets:

- `Q-001`
- `Q-002`
- `Q-003`

## Subsystem Status Table

| Subsystem | Final Role | Today | Tickets |
| --- | --- | --- | --- |
| Domain models | Canonical monitoring entities and vocabularies | Done | `D-001` |
| Invariants | Enforce semantic safety rules | Done | `D-002` |
| Contract result model | Stable comparability taxonomy | Done, with one deferred reason | `D-003` |
| Checker boundary | Workflow-facing comparability interface | Done | `D-004` |
| Recipe schema/default/validation | Define and validate monitoring intent | Mostly done | `R-001`, `R-000`, `R-000a`, `R-002` |
| Recipe compilation | Convert recipe into executable run plan | Not done, shape still unclear | `R-003` |
| Workflow lifecycle | Stage transitions and lifecycle contract | Done | `W-001` |
| Result envelope | Public synchronous run result | Done for basic shape, richer payload not done | `W-001a`, `W-001b` |
| Prepare stage | Resolve execution context and references | Not done | `W-002`, `W-002a` |
| Comparability check stage | Execute checker before analysis | Not done | `W-003` |
| Diff engine | Produce comparison evidence | Not done | `E-001` |
| Finding engine | Produce actionable interpreted outputs | Not done | `E-002`, `E-003` |
| Analysis branching | Comparable and non-comparable execution paths | Not done | `W-004` |
| Gateway abstraction | MLflow-facing persistence boundary | Done as abstraction, partial as full product behavior | `P-001`, `P-002`, `P-002a`, `P-003` |
| Query APIs | Read timeline and anchor-window views | Not done, contract still unclear | `Q-001`, `Q-002`, `Q-003` |
| Promotion flow | Evaluate and update active LKG | Not done, semantics still unclear | `P-003`, `W-005` |
| Acceptance and ops | End-to-end validation and operator docs | Not done | `T-001`, `T-002`, `O-002`, `O-003` |

## Ticket Map For Not-Done Areas

- Recipe compilation: `R-003`
- Prepare stage: `W-002`
- First-run baseline bootstrap: `W-002a`
- Comparability check stage: `W-003`
- Diff engine: `E-001`
- Finding engine: `E-002`
- Analysis branching: `W-004`
- Inline findings in result envelope: `W-001b`
- Stage-aligned persistence writes: `P-002`
- Persisted artifact schemas: `P-002a`
- Timeline traversal API: `Q-001`
- Anchor-window query API: `Q-002`
- Acceptance harness: `T-001`
- Recipe version policy: `R-004`
- Initial finding policy package: `E-003`
- LKG metadata semantics: `P-003`
- Optional promotion stage: `W-005`
- Preset baseline/LKG queries: `Q-003`
- Invariant regression suite: `T-002`
- Failure runbook: `O-002`
- Pilot onboarding checklist: `O-003`

## Notes On Current Reality

The largest gap between intended product and current implementation is the actual runtime execution path.

- [monitor.py](/Users/zhengshi/src/mlflow-monitor/src/mlflow_monitor/monitor.py) still raises `NotImplementedError`.
- There is no implemented end-to-end run orchestration yet.
- There is no CLI implementation in the current repo.
- The README describes capabilities that are still only partially built or still planned.

That is why the diagram shows the final product flow, but marks several central runtime boxes as `Not Done` or `Partial`.
