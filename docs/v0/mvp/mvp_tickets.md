# MLflow-Monitor MVP Tickets

**Exit criteria:** the M1 create -> prepare -> check slice works against real MLflow through the SDK, preserves read-only treatment of training runs, demonstrates `pass` / `warn` / `fail`, and is backed by a reproducible local demo plus integration tests.

## MVP Cycle: Real-MLflow M1 Slice

### Workflow & Naming

**MVP-01: Rename monitoring-side `run_id` to `monitoring_run_id`** (P0)

Goal:
Remove identifier ambiguity before real MLflow integration introduces both training run IDs and monitoring run IDs into the same execution path.

Required deliverables:

- Rename monitoring-side `run_id` fields, parameters, and local variables to `monitoring_run_id` across the M1 codebase.
- Rename `MonitorRunResult.run_id` to `MonitorRunResult.monitoring_run_id` as part of the same monitoring-side clarification pass.
- Keep training-side identifiers such as `source_run_id` and `baseline_source_run_id` unchanged.
- Update affected unit tests to use the renamed monitoring-side vocabulary.

Non-goals:

- No behavior changes beyond the public-surface field rename needed for naming clarity.
- No MLflow gateway implementation in this ticket.
- No rename of training-side identifiers.

Acceptance criteria:

- The codebase no longer uses ambiguous monitoring-side `run_id` names in touched runtime and test code.
- SDK/result-contract tests are updated to assert `MonitorRunResult.monitoring_run_id`.
- Existing in-memory unit tests still pass after the rename.
- The rename does not alter public M1 behavior beyond identifier naming clarity in code/tests.

Dependencies: none

**MVP-02: Refit gateway/orchestration contract for MLflow-assigned monitoring run IDs** (P0)

Goal:
Change the current in-memory-shaped protocol so the gateway can own create-or-reuse of monitoring runs when the backing store assigns the monitoring run ID at persistence time.

Required deliverables:

- Replace the preallocated monitoring-run-ID assumption in the gateway/orchestration boundary.
- Define a gateway-owned create-or-reuse flow that returns the `monitoring_run_id` and the sequence/replay context orchestration needs.
- Make the gateway own monitoring run ID creation after the refit for both MLflow and in-memory implementations.
- Be prepared for the current `get_or_create_idempotent_run_id(...)` plus `upsert_monitoring_run(...)` split to collapse into a single create-or-reuse operation on the MLflow-backed path.
- Remove the caller-supplied run ID factory input from `run_orchestration(...)`.
- Preserve current rerun and replay semantics for already-checked and failed monitoring runs.
- Update the in-memory gateway to self-generate monitoring run IDs and satisfy the refit contract.

Non-goals:

- No real MLflow API usage yet.
- No workflow-stage semantic changes beyond what the protocol refit requires.
- No analyze/close-stage expansion.

Acceptance criteria:

- Orchestration no longer relies on allocating a monitoring run ID outside the gateway.
- Orchestration no longer relies on a caller-supplied run ID factory.
- Both MLflow-shaped and in-memory gateways satisfy the same create-or-reuse contract.
- The in-memory gateway still supports first run, later run, checked rerun, and failed rerun behavior.
- Existing orchestration and gateway tests stay green after the refit.

Dependencies: MVP-01

### Persistence Gateway

**MVP-03: Add `MonitorMLflowClient` adapter** (P0)

Goal:
Introduce one thin MLflow client adapter that encapsulates all direct `MlflowClient` usage needed by the MVP slice.

Required deliverables:

- Add `MonitorMLflowClient` as the only runtime module that imports `MlflowClient`.
- Implement experiment lookup/create, experiment tag read/write, run create/get/terminate, run tag write, run metrics/params/tags reads, artifact listing, and JSON artifact logging.
- Support terminating monitoring runs as `FINISHED` or `FAILED`.
- Normalize MLflow API quirks and missing-run behavior into deterministic adapter behavior suitable for the gateway.

Non-goals:

- No workflow or orchestration logic in the adapter.
- No `search_runs()` support.
- No broader query API support beyond what the MVP slice requires.

Acceptance criteria:

- Adapter tests cover create/get patterns, experiment tag access, run reads, tag writes, termination behavior, artifact listing, and JSON artifact logging against a local MLflow store.
- Runtime code outside the adapter does not import `MlflowClient` directly.

Dependencies: MVP-02

**MVP-04: Implement `MLflowMonitoringGateway` with direct experiment-tag indexing** (P0)

Goal:
Provide a real-MLflow gateway implementation for the existing M1 create -> prepare -> check slice using direct lookup and experiment-tag indexing only.

Required deliverables:

- Implement timeline bootstrap via monitoring experiment creation and experiment tags.
- Implement idempotency via `training.{source_run_id}.monitoring_run_id`.
- Implement ordered timeline traversal via `monitoring.next_sequence_index` plus `monitoring.run.{index}` tags.
- Implement previous-run lookup via `monitoring.latest_run_id`.
- Implement contract evidence reads from source training runs using metrics, params, and tags only.
- Persist lifecycle/comparability state and final `outputs/result.json` to monitoring runs.
- Terminate monitoring runs through the adapter so successful runs end as `FINISHED` and owned failures end as `FAILED`.
- Enforce read-only treatment of training runs.

Non-goals:

- No `search_runs()` usage.
- No analyze, close, findings, diff, query, or promotion behavior.
- No concurrency hardening beyond the documented MVP assumptions.

Acceptance criteria:

- First-run bootstrap creates the monitoring experiment and baseline timeline state.
- Later runs reuse the pinned baseline and resolve the previous monitoring run correctly.
- Timeline listing works through experiment tags and `get_run()`, not `search_runs()`.
- Training runs are never written to by the gateway.

Dependencies: MVP-02, MVP-03

### SDK & Validation

**MVP-05: Update the SDK default path to real MLflow** (P0)

Goal:
Make `monitor.run(...)` use the MLflow-backed gateway by default while preserving explicit gateway injection for tests and controlled use.

Required deliverables:

- Update `monitor.py` to accept an optional gateway parameter.
- Default the public SDK path to the MLflow-backed gateway when no gateway is supplied.
- Instantiate a fresh `MLflowMonitoringGateway` per `monitor.run(...)` call unless an explicit gateway is supplied.
- Preserve the current canonical synchronous result contract.

Non-goals:

- No CLI in this ticket.
- No changes to the result envelope shape beyond what the gateway/orchestration refit requires.

Acceptance criteria:

- SDK callers can run against real MLflow with no explicit gateway argument.
- Tests can still inject the in-memory gateway explicitly.
- Existing result-contract expectations remain valid for the M1 slice.

Dependencies: MVP-04

**MVP-06: Add real-MLflow integration coverage for the M1 slice** (P0)

Goal:
Validate the SDK, adapter, gateway, and direct-lookup indexing behavior end to end against a local MLflow store.

Required deliverables:

- Add integration fixtures for a dedicated local MLflow store.
- Add adapter, gateway, and end-to-end tests for the real-MLflow path.
- Cover first-run bootstrap, later-run baseline reuse, baseline immutability rejection, `pass`, `warn`, `fail`, idempotent rerun, timeline traversal, namespace isolation, `outputs/result.json`, and read-only treatment of training runs.

Non-goals:

- No remote-server testing.
- No performance or concurrency testing beyond the MVP assumptions.

Acceptance criteria:

- Integration tests pass against a local file-backed MLflow store.
- The direct experiment-tag indexing path is exercised in tests.
- Integration tests verify `outputs/result.json` exists on monitoring runs and is valid JSON.
- Integration tests verify monitoring runs terminate with final MLflow status rather than remaining `RUNNING`.
- Real-MLflow tests prove training experiments remain unmodified.

Dependencies: MVP-04, MVP-05

### Demo & Optional CLI

**MVP-07: Add local demo setup and walkthrough** (P0)

Goal:
Create a reproducible local demo that shows the real-MLflow M1 slice working in the MLflow UI.

Required deliverables:

- Add a demo seeding script that creates synthetic training runs for baseline, comparable, environment-mismatch, and schema-mismatch scenarios.
- Document the local MLflow setup and SDK-driven demo loop.
- Update the user-facing README/public quick-start material so it matches the real-MLflow MVP path once the SDK flow is shipped.
- Ensure the demo visibly produces monitoring runs, experiment tags, and final result artifacts.

Non-goals:

- No CLI dependency for demo success.
- No broader public-doc cleanup outside MVP demo materials.

Acceptance criteria:

- A local user can seed the demo data, run the SDK flow, and inspect monitoring results in the MLflow UI.
- The demo covers `pass`, `warn`, and `fail`.
- The monitoring experiment and final `outputs/result.json` artifact are visible after the demo flow.

Dependencies: MVP-05, MVP-06

**MVP-08: Add optional CLI wrapper** (P1)

Goal:
Provide a thin CLI wrapper over the SDK if the required SDK path, demo, and tests are already green.

Required deliverables:

- Add `mlflow-monitor run --subject ... --source-run ...` as a thin SDK wrapper.
- Emit JSON to stdout and return non-zero exit on failure.
- Add packaging entry point wiring only if the CLI ships this week.

Non-goals:

- No separate business logic from the SDK.
- No CLI-first development path.

Acceptance criteria:

- CLI output matches the canonical synchronous result shape from the SDK.
- CLI failure exit behavior is covered by focused tests.
- This ticket is deferred automatically if MVP-01 through MVP-07 are not yet green.

Dependencies: MVP-05, MVP-06
