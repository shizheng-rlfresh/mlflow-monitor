# MLflow-Monitor v0: Ticket Breakdown

## Plane Setup

### Modules
1. Domain & Contracts
2. Workflow Orchestration
3. Recipe Pipeline
4. Diff & Finding Engines
5. Persistence Gateway
6. Query & Anchor Window
7. Validation & Acceptance
8. Docs & Ops Readiness

### Cycles
1. **M1: Core skeleton** — runs can prepare, check, and persist with recipe binding
2. **M2: Analysis complete** — comparable and non-comparable paths work end-to-end
3. **M3: Operational trust** — LKG promotion, preset windows, invariant regression

### Priority Mapping
- **P0** — required for minimal usable v0
- **P1** — required for robust v0
- **P2** — deferred optimization/polish

---

## Cycle M1: Core skeleton

**Exit criteria:** runs can prepare, check, and persist status with recipe binding.

### Domain & Contracts

**D-001: Define canonical domain models** (P0)
- Types/structures for: Run, Baseline, Timeline, LKG, Contract/Profile, Contract check result, Diff, Finding
- Required fields for lifecycle, comparability, visibility, and evidence linkage
- Invariants encoded as validation utilities
- Dependencies: none

**D-002: Implement invariant validation layer** (P0)
- Timeline ownership
- Baseline immutability
- LKG-in-same-timeline
- Finding-to-diff evidence checks
- Violations produce deterministic structured errors
- Dependencies: D-001

**D-003: Implement contract check result schema and reason taxonomy** (P0)
- `pass` / `warn` / `fail` status model with structured reasons
- Environment mismatch = warn
- Schema / feature / data-scope mismatch = fail
- Blocking vs non-blocking flags
- Dependencies: D-001

**D-004: Define contract/profile model and checker interface** (P0)
- Canonical Contract/Profile domain model
- Checker interface / protocol for comparability evaluation
- Clear boundary between recipe binding and workflow execution
- Dependencies: D-001

### Recipe Pipeline

**R-001: Define recipe schema (v0-lite)** (P0)
- Sections: identity/version, input binding, contract binding, metrics/slices, finding policy, output binding
- Unknown/disallowed sections rejected
- Dependencies: D-001, D-004

**R-000: Ship system default recipe** (P0)
- Zero-config default recipe
- No required metrics/artifacts
- Reads whatever MLflow already has
- Depends on a system default permissive contract/profile rather than bypassing contract check
- Dependencies: R-001, D-004

**R-000a: Define zero-config default behavior details** (P0)
- Default source-run selection behavior
- Default contract/profile behavior
- Default handling when expected evidence is missing
- Make zero-config behavior explicit and testable
- Dependencies: R-000, D-004

**R-002: Build recipe validation pipeline** (P0)
- Structural, referential, and constraint validation
- Fails fast with specific, actionable error messages
- Dependencies: R-001, R-000

**R-003: Build recipe compilation pipeline** (P0)
- Normalize recipe to executable run plan
- Resolve defaults deterministically for same recipe/version
- Output is the prepared input contract consumed by workflow
- Dependencies: R-002, R-000a

### Workflow Orchestration

**W-001: Build run state machine** (P0)
- `created → prepared → checked → analyzed → closed`
- `failed` terminal state reachable from any stage
- Illegal transitions blocked
- Transition history auditable
- Dependencies: D-001

**W-001a: Define synchronous SDK/CLI result contract** (P0)
- Canonical return object for `monitor.run(...)` and CLI output envelope
- Status, comparability, summary, findings, references, and identifiers
- Error/result shape consistent across SDK and CLI
- Dependencies: D-001, W-001

**W-002: Implement prepare stage** (P0)
- Consume compiled recipe/run plan
- Resolve timeline, baseline, previous run, LKG, custom reference, contract/profile, source training run
- Validate required metrics/artifacts on source run
- Actionable error messages on failure
- Dependencies: W-001, D-001, D-004, R-003

**W-002a: Implement first-run baseline bootstrap rule** (P0)
- If no timeline exists, require explicit `baseline_source_run_id`
- Create sentinel/timeline config with pinned baseline reference
- Fail prepare with actionable error if omitted
- Reject later baseline mutation attempts
- Dependencies: W-002, P-001

**W-003: Implement comparability check stage** (P0)
- Execute contract checker via checker interface
- Persist check reasons
- No analysis before check completion
- Dependencies: W-002, D-003, D-004

### Persistence Gateway

**P-001: Implement persistence gateway abstraction** (P0)
- Central write/read API
- Enforce configurable namespace prefix for all writes
- Enforce read-only access to training experiments
- Sentinel/timeline initialization support
- Sequence index assignment
- Idempotency key check
- LKG resolution query
- Read visibility rule foundation
- Dependencies: W-001, D-001

### Docs & Ops Readiness

**O-001: Keep design docs synchronized** (P0, ongoing)
- Update docs for any behavior deltas
- Dependencies: none

---

## Cycle M2: Analysis complete

**Exit criteria:** comparable and non-comparable analysis paths complete. Anchor-window query works. Acceptance scenarios A-F, I, J pass.

### Diff & Finding Engines

**E-001: Implement diff engine core** (P0)
- Comparison modes: baseline, previous, LKG, structural compatibility
- Missing references handled as unavailable, not failure
- Non-comparable runs skip metric diffs
- Dependencies: D-001, W-003

**E-002: Implement finding engine core** (P0)
- Diff-to-finding mapping
- Severity assignment
- Evidence linkage
- Every finding references evidence
- Dependencies: E-001, D-001

### Workflow Orchestration

**W-004: Implement analysis branching logic** (P0)
- Comparable path: diffs + findings
- Non-comparable path: compatibility records only
- Both paths produce summary outputs
- Dependencies: W-003, E-001, E-002, W-001a

### Persistence Gateway

**P-002: Stage-aligned write implementation** (P0)
- Writes for prepare, check, analyze, close stages
- Partial failure handling explicit
- Dependencies: P-001, W-004

**P-002a: Define canonical persisted artifact schemas** (P0)
- Canonical schemas for diffs, findings, summaries, references, and metadata
- Stable read/write contract for persistence and query layers
- Dependencies: P-001, E-001, E-002, W-001a

### Query & Anchor Window

**Q-001: Timeline traversal API** (P0)
- Ordered runs by `sequence_index`
- Lifecycle and comparability statuses included
- Non-comparable closed runs visible
- Failed runs excluded from default views unless explicitly requested
- Dependencies: P-002, P-002a

**Q-002: Anchor-window query API** (P0)
- Resolve anchor, return runs from anchor onward
- Aggregate from stored outputs, no recomputation
- Dependencies: Q-001, P-002a

### Validation & Acceptance

**T-001: Build acceptance scenario test harness** (P0)
- Implement scenarios A-J
- Deterministic assertions, actionable diagnostics
- Dependencies: all P0 tasks

---

## Cycle M3: Operational trust

**Exit criteria:** LKG promotion flow stable. Preset windows stable. Scenarios G/H pass. Invariant regression suite active.

### Recipe Pipeline

**R-004: Recipe version compatibility policy** (P1)
- Version identity and selection rules
- Historical reproducibility preserved
- Dependencies: R-003

### Diff & Finding Engines

**E-003: Initial finding policy package** (P1)
- Minimal categories
- Default severity rules
- Default recommendation templates
- Dependencies: E-002, R-003

### Persistence Gateway

**P-003: LKG metadata and retrieval semantics** (P1)
- Promotion persistence
- Atomic active/superseded tag update
- Deterministic LKG retrieval
- Dependencies: P-002

### Workflow Orchestration

**W-005: Implement optional LKG promotion stage** (P1)
- Policy evaluation hook
- Promote/hold decision
- LKG pointer update
- Promotion allowed only for closed runs
- Dependencies: W-004, P-003

### Query & Anchor Window

**Q-003: LKG and baseline preset window queries** (P1)
- Baseline-anchored and LKG-anchored presets
- Outputs match anchor-window semantics
- Dependencies: Q-002, P-003

### Validation & Acceptance

**T-002: Invariant regression suite** (P1)
- Automated invariant checks
- CI integration
- Dependencies: T-001, D-002

### Docs & Ops Readiness

**O-002: Runbook for failure diagnosis** (P1)
- Lifecycle, comparability, and persistence failure diagnostics
- Dependencies: W-004, P-002, T-001

**O-003: Pilot onboarding checklist** (P1)
- Recipe setup
- Baseline pinning
- Contract/profile setup
- Interpretation guide
- Dependencies: R-003, W-004, W-002a

---

## Post-v0

**T-003: Load and stress validation** (P2)
- Concurrency and timeline-size stress tests
- Dependencies: T-001, Q-002

---

## Acceptance Scenario Index (A–J)

**A. First run bootstrap succeeds**
- No existing timeline
- Caller provides `baseline_source_run_id`
- Sentinel/timeline created
- Baseline pinned

**B. First run bootstrap fails without explicit baseline**
- No existing timeline
- Caller omits `baseline_source_run_id`
- Prepare fails with actionable error

**C. Comparable run analyzes successfully**
- Comparable source/baseline
- Check passes/warns appropriately
- Diffs + findings produced
- Run closes successfully

**D. Non-comparable run records compatibility outcome only**
- Contract fails comparability
- No metric diff analysis
- Compatibility record and summary persisted
- Run closes successfully

**E. Previous-run reference missing is non-fatal**
- No previous closed run
- Analysis still succeeds with unavailable previous reference

**F. LKG reference missing is non-fatal before promotion**
- No active LKG yet
- Analysis still succeeds with unavailable LKG reference

**G. LKG promotion updates active pointer correctly**
- Closed run eligible for promotion
- Promote/hold decision applied
- Active LKG pointer updated atomically

**H. Superseding LKG preserves deterministic retrieval**
- New promoted run supersedes prior active LKG
- Retrieval returns only the active LKG deterministically

**I. Default query views exclude failed runs**
- Failed runs may exist in storage
- Default timeline / anchor-window views exclude them
- Explicit retrieval can still request them

**J. Baseline remains immutable after bootstrap**
- Later attempts to change baseline are rejected
- Timeline still resolves original pinned baseline

---

## Dependency Graph Summary

```text
Foundation (M1):
  D-001 → D-002, D-003, D-004
  R-001 → R-000 → R-000a → R-002 → R-003
  W-001 → W-001a
  W-001 + D-001 + D-004 + R-003 → W-002 → W-002a
  W-002 + D-003 + D-004 → W-003
  W-001 + D-001 → P-001

Analysis (M2):
  W-003 + E-001 + E-002 + W-001a → W-004
  P-001 + W-004 → P-002
  P-001 + E-001 + E-002 + W-001a → P-002a
  P-002 + P-002a → Q-001 → Q-002
  All P0 → T-001

Promotion (M3):
  R-003 → R-004
  E-002 + R-003 → E-003
  P-002 → P-003
  W-004 + P-003 → W-005
  Q-002 + P-003 → Q-003
  T-001 + D-002 → T-002


## v0 "Ready" Criteria

1. All P0 tasks complete
2. Acceptance scenarios A-F, I, J pass
3. No critical invariant violations
4. Documentation reflects implemented behavior
5. At least one pilot workflow runs end-to-end