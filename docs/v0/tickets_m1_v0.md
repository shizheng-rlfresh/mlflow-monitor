## Cycle M1: Core skeleton

**Exit criteria:** runs can prepare, check, and persist status with recipe binding.

### Domain & Contracts

**D-001: Define canonical domain models** (P0) **DONE**

Goal:
Define the canonical v0 runtime/domain entities used across workflow, persistence, and query code.

Required deliverables:

- Runtime types for `Run`, `Baseline`, `Timeline`, `LKG`, `Contract`, `ContractCheckReason`, `ContractCheckResult`, `Diff`, and `Finding`.
- Fixed enum vocabularies for lifecycle, comparability, diff reference kind, and finding severity.
- Required fields for lifecycle, comparability, visibility, lineage, and evidence linkage.

Non-goals:

- No workflow logic.
- No persistence implementation.
- No MLflow-specific code in the domain layer.

Acceptance criteria:

- Domain entities can be constructed with valid field shapes.
- Related entities compose correctly (`Run` references `Contract`, `Timeline` owns `Baseline`, etc.).
- Baseline snapshot fields are immutable after construction.

Dependencies: none

**D-002: Implement invariant validation layer** (P0) **DONE**

Goal:
Provide deterministic domain guardrails for the core invariants already established in D-001.

Required deliverables:

- Validation utilities for:
  - timeline ownership
  - baseline immutability
  - LKG membership within the same timeline
  - finding-to-diff evidence linkage
- Structured invariant failures using a stable error shape.

Non-goals:

- No persistence-time enforcement.
- No workflow transition validation beyond the invariants explicitly listed in this ticket.

Acceptance criteria:

- Each invariant has direct unit-test coverage for success and failure cases.
- Violations raise deterministic structured errors with code, message, entity, and field.

Dependencies: D-001

**D-003: Implement contract check result schema and reason taxonomy** (P0) **DONE, deferred `metric_mismatch`; see Plane deferred ticket list**

Goal:
Define the canonical output of comparability checking before any real checker implementation exists.

Required deliverables:

- `pass` / `warn` / `fail` comparability result model.
- Structured reason model with machine-readable `code`, human-readable `message`, and `blocking` flag.
- Validation rules for the initial v0 reason taxonomy:
  - `environment_mismatch` => non-blocking / `warn`
  - `schema_mismatch` => blocking / `fail`
  - `feature_mismatch` => blocking / `fail`
  - `data_scope_mismatch` => blocking / `fail`

Non-goals:

- No checker execution logic.
- No `metric_mismatch` support in this ticket.
- No policy layer beyond the blocking/non-blocking distinction already required by v0.

Acceptance criteria:

- Invalid reason codes are rejected.
- Invalid blocking flags are rejected.
- Result status must match the supplied reason set.

Dependencies: D-001

**D-004: Define resolved contract model and checker boundary** (P0)

Goal:
Define the minimal workflow-facing API needed for W-003 to execute comparability checks against one resolved contract.

Required deliverables:

- Keep `Contract` as the only contract-domain model in code for v0.
- Clarify in domain docs that `Contract` is the resolved effective contract attached to `Run` and `Timeline`, not a recipe-facing profile object.
- Add a single workflow-facing checker boundary that accepts:
  - a resolved `Contract`
  - a prepared contract-evaluation context
  - and returns `ContractCheckResult`
- Keep the checker boundary platform-agnostic and independent of MLflow client objects.

Non-goals:

- No first-class `ContractProfile` type in this ticket.
- No real comparability logic yet.
- No recipe/profile resolution logic.
- No `metric_mismatch` behavior.

Acceptance criteria:

- The runtime code exposes one checker boundary for workflow use.
- Tests prove a stub checker can satisfy the boundary.
- Tests prove the prepared evaluation context contains resolved evidence only, not recipe objects or MLflow primitives.

Dependencies: D-001

### Recipe Pipeline

**R-001: Define recipe schema (v0-lite)** (P0)

Goal:
Define the minimal declarative recipe shape that workflow can validate and compile in M1.

Required deliverables:

- A canonical v0-lite recipe schema with sections for:
  - identity/version
  - input binding
  - contract binding
  - metrics/slices
  - finding policy
  - output binding
- A runtime representation for parsed recipe data.
- Explicit rejection of unknown or disallowed top-level sections.

Non-goals:

- No recipe execution in this ticket.
- No recipe version compatibility policy beyond storing identity/version.
- No complex plugin-style extension points.

Acceptance criteria:

- Valid recipes parse into a stable in-memory shape.
- Unknown/disallowed sections fail validation with actionable errors.
- Required sections and required fields are explicit and test-covered.

Dependencies: D-001, D-004

**R-000: Ship system default recipe** (P0)

Goal:
Provide a built-in zero-config recipe so users can run monitoring without authoring a custom recipe.

Required deliverables:

- A system default recipe representation available to recipe resolution code.
- Default behavior that:
  - requires no user-authored recipe
  - requires no additional metrics/artifacts beyond what the system reads by default
  - binds to a default permissive contract instead of bypassing contract checks

Non-goals:

- No custom user recipe loading in this ticket.
- No special-case bypass of the contract check stage.

Acceptance criteria:

- A caller with no recipe specified resolves to the system default recipe.
- The default recipe is stable and testable as a first-class recipe, not hidden workflow behavior.
- The default recipe keeps contract checking in the flow.

Dependencies: R-001, D-004

**R-000a: Define zero-config default behavior details** (P0)

Goal:
Make the system default recipe behavior explicit enough that later workflow code can consume it deterministically.

Required deliverables:

- Locked defaults for:
  - source-run selection behavior
  - default contract behavior
  - behavior when optional evidence is absent
- Tests that capture those defaults as observable behavior.

Non-goals:

- No implementation of all recipe compilation rules.
- No MLflow retrieval logic.

Acceptance criteria:

- The default recipe behavior is written as executable rules, not just doc prose.
- Tests prove the same default recipe/version resolves to the same effective behavior.
- Missing optional evidence behavior is explicit and not left to ad hoc workflow decisions.

Dependencies: R-000, D-004

**R-002: Build recipe validation pipeline** (P0)

Goal:
Validate recipe structure and references before any workflow stage consumes them.

Required deliverables:

- Structural validation for recipe shape.
- Referential validation for contract/policy references required by v0-lite.
- Constraint validation for v0 recipe invariants.
- Actionable error messages for invalid recipes.

Non-goals:

- No recipe execution.
- No workflow state transitions.

Acceptance criteria:

- Invalid recipes fail before compile/run stages.
- Validation errors identify the failing section/field clearly.
- Tests cover structural, referential, and constraint failures.

Dependencies: R-001, R-000

**R-003: Build recipe compilation pipeline** (P0)

Goal:
Convert a validated recipe into the executable run-plan shape consumed by workflow prepare/check stages.

Required deliverables:

- Compilation step that normalizes validated recipe input into a deterministic run plan.
- Default resolution using R-000/R-000a rules.
- A stable compiled output shape for workflow consumption.

Non-goals:

- No MLflow resolution in this ticket.
- No workflow stage execution.
- No historical version selection policy beyond exact input resolution.

Acceptance criteria:

- Same recipe/version compiles to the same run plan.
- The compiled run plan contains the fields W-002 and W-003 need, with no workflow-side re-interpretation required.
- Compilation is covered by focused unit tests.

Dependencies: R-002, R-000a

### Workflow Orchestration

**W-001: Build run state machine** (P0)

Goal:
Define and enforce the core execution lifecycle for monitoring runs.

Required deliverables:

- Runtime state machine for:
  - `created -> prepared -> checked -> analyzed -> closed`
  - `failed` as a terminal state reachable from any active stage
- Illegal transition protection.
- Transition recording sufficient for audit/debug purposes.

Non-goals:

- No prepare/check/analyze/persist business logic in this ticket.
- No promotion stage.

Acceptance criteria:

- Valid transitions succeed.
- Illegal transitions fail deterministically.
- Transition history is inspectable and test-covered.

Dependencies: D-001

**W-001a: Define synchronous SDK/CLI result contract** (P0)

Goal:
Define the stable result envelope returned by `monitor.run(...)` and mirrored by the CLI.

Required deliverables:

- Canonical result type or schema containing:
  - lifecycle status
  - comparability status
  - summary
  - findings
  - references/identifiers
  - structured error information
- One consistent shape shared by SDK and CLI layers.

Non-goals:

- No actual CLI implementation.
- No persistence formatting.
- No rich reporting output.

Acceptance criteria:

- Success and failure results share one predictable contract.
- The result shape is sufficient for downstream callers to inspect status and outcomes without reading MLflow directly.
- Tests cover basic success/failure envelope construction.

Dependencies: D-001, W-001

**W-001b: Hydrate synchronous SDK/CLI result with inline findings** (P1, M2)

Goal:
Make the synchronous result contract self-sufficient for downstream automation decisions.

Required deliverables:

- Inline structured finding payloads in the result envelope.
- Backward-compatible ID fields retained.
- Deterministic serialization for SDK/CLI parity.

Non-goals:

- No diff-payload hydration unless separately ticketed.

Acceptance criteria:

- Callers can gate on finding severity using the returned object only.
- Success and failure envelopes preserve stable top-level keys.

Dependencies: W-004, E-001, E-002, W-001a

**W-002: Implement prepare stage** (P0)

Goal:
Resolve all run inputs and references needed before contract checking begins.

Required deliverables:

- Prepare-stage logic that consumes the compiled run plan.
- Resolution of:
  - timeline
  - pinned baseline
  - previous run
  - active LKG
  - custom reference, if configured
  - resolved contract
  - source training run reference
- Validation that required metrics/artifacts exist on the source run.
- Actionable failure messages for missing required inputs.

Non-goals:

- No contract checking.
- No diff/finding generation.
- No persistence beyond what is strictly needed to represent prepare-stage state.

Acceptance criteria:

- Successful prepare produces a complete prepared context for W-003.
- Missing required metrics/artifacts fail prepare explicitly.
- Prepare does not start contract checking or analysis.

Dependencies: W-001, D-001, D-004, R-003

**W-002a: Implement first-run baseline bootstrap rule** (P0)

Goal:
Enforce the v0 bootstrap rule that the first run on a new timeline must pin an explicit baseline.

Required deliverables:

- Detection of “no existing timeline” during prepare.
- Requirement that first run include `baseline_source_run_id`.
- Creation of the initial timeline/sentinel baseline state through the persistence gateway.
- Rejection of later attempts to mutate the pinned baseline.

Non-goals:

- No automatic baseline inference.
- No baseline reset behavior.

Acceptance criteria:

- First run without explicit baseline fails with an actionable error.
- First run with explicit baseline initializes baseline state once.
- Later runs resolve baseline from timeline state instead of accepting a new one.

Dependencies: W-002, P-001

**W-003: Implement comparability check stage** (P0)

Goal:
Execute the resolved contract check after prepare and produce the canonical comparability result for the run.

Required deliverables:

- Check-stage orchestration that invokes the D-004 checker boundary using prepared evidence.
- Recording of `ContractCheckResult` on the run/check output.
- Explicit guarantee that analysis cannot begin before check completes.

Non-goals:

- No diff/finding generation.
- No policy-driven promotion behavior.
- No analysis-stage branching beyond emitting the comparability result needed by later stages.

Acceptance criteria:

- Prepared runs can execute contract check and produce `pass` / `warn` / `fail`.
- Check reasons are preserved in structured form for persistence.
- Workflow ordering prevents analyze-stage behavior before check completion.

Dependencies: W-002, D-003, D-004

### Persistence Gateway

**P-001: Implement persistence gateway abstraction** (P0)

Goal:
Create the only MLflow-facing read/write boundary for v0 workflow code.

Required deliverables:

- Central gateway API for monitoring reads/writes.
- Enforcement of configurable monitoring namespace prefix.
- Read-only access pattern for training experiments.
- Support for:
  - timeline/sentinel initialization
  - sequence index assignment
  - idempotency key lookup
  - active LKG resolution
  - visibility-aware timeline reads

Non-goals:

- No direct MLflow access outside the gateway.
- No stage-aligned persistence completeness guarantees beyond the abstraction needed by M1.
- No query API beyond the gateway primitives required by M1 tickets.

Acceptance criteria:

- Workflow code can resolve and persist required M1 state without touching MLflow directly.
- Namespace and training-run immutability rules are enforced through the gateway.
- Gateway behavior is unit-testable with deterministic fakes/stubs.

Dependencies: W-001, D-001

### Docs & Ops Readiness

**O-001: Keep design docs synchronized** (P0, ongoing)

Goal:
Keep the v0 design docs aligned with implemented behavior as tickets land.

Required deliverables:

- Doc updates for any ticket that changes behavior, interfaces, assumptions, or scope.
- Explicit notes when planned behavior is deferred or narrowed.

Non-goals:

- No standalone documentation redesign.
- No speculative documentation for tickets not yet implemented.

Acceptance criteria:

- Behavior changes in code are reflected in the relevant v0 docs in the same development slice.
- Deferred items are called out explicitly where needed to avoid design drift.

Dependencies: none
