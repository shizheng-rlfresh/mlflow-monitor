# MLflow-Monitor v0 MLflow Mapping Strategy

Date: 2026-03-09
Status: Draft (Planning)

## 1. Purpose

This document defines how MLflow-Monitor v0 uses MLflow as both a read source and a write sink while preserving CAST domain semantics and workflow guarantees.

It focuses on:

1. MLflow's two distinct roles in the system.
2. Namespace and naming strategy.
3. Entity-to-storage strategy.
4. Write/read patterns.
5. Traceability and consistency rules.
6. Scope boundaries and risks.

This is not a low-level field dictionary. It is an architectural mapping contract.

## 2. MLflow's Two Distinct Roles

MLflow plays two distinct and strictly separated roles in MLflow-Monitor:

### 2.1 Role 1: Read Source (Training Experiments)

MLflow training experiments are the input data source for monitoring runs.

Rules:

1. Training runs are read-only. MLflow-Monitor never writes to, modifies, or appends to any training run.
2. MLflow-Monitor reads metrics, params, tags, and artifacts from existing training runs.
3. If a training run does not have the inputs required by the recipe, the monitoring run fails explicitly at prepare stage.
4. No partial reads. Either the required inputs are present or monitoring cannot proceed.

### 2.2 Role 2: Write Sink (Monitoring Experiments)

MLflow-Monitor owns a dedicated experiment namespace for all monitoring outputs.

Rules:

1. All monitoring outputs are written to system-owned experiments, never to training experiments.
2. The monitoring experiment namespace is strictly separated from training experiments.
3. Users never manually name or create monitoring experiments.
4. The system derives experiment names deterministically from subject identity.

### 2.3 Why This Separation Matters

1. Training runs remain sacred and untouched — teams trust their MLflow training history is never modified.
2. Backfill is clean — multiple monitoring runs against the same training run coexist without collision or overwrite.
3. MLflow experiment lists stay clean — no monitoring clutter mixed with training experiments.
4. Naming entropy is eliminated — monitoring experiments are system-owned and deterministic.

## 3. Namespace and Naming Strategy

### 3.1 Monitoring Experiment Naming Convention

```
mlflow_monitor/{subject_id}

Examples:
  mlflow_monitor/churn_model
  mlflow_monitor/fraud_scorer
  mlflow_monitor/demand_forecast
```

### 3.2 Naming Principles

1. System owns all monitoring experiment names. Users never type monitoring experiment names.
2. One subject maps to exactly one monitoring experiment. No duplicates, no versioned suffixes.
3. The `mlflow_monitor/` prefix is reserved and enforced by the gateway module.
4. Subject IDs are the only naming decision users make — they should be stable, agreed-upon identifiers.

### 3.3 Why System-Owned Naming

1. Human-managed MLflow naming breaks down at scale — conventions drift, teams diverge, experiments proliferate.
2. Deterministic naming makes any subject's monitoring history discoverable without documentation.
3. Grouping under `mlflow_monitor/` prefix makes monitoring experiments visually distinct in the MLflow UI.

### 3.4 Backfill Semantics

Multiple monitoring runs against the same training run are valid and expected:

```
mlflow_monitor/churn_model
  Run: monitor_abc123_2026-03-01    <- original monitoring run
  Run: monitor_abc123_2026-03-09    <- backfill run, coexists cleanly
```

No overwrite. No collision. Each monitoring run is a separate record referencing the same source training run.

## 4. Mapping Design Principles

1. Domain-first semantics: CAST meaning is primary; MLflow is representation.
2. Training runs are sacred: never modified by MLflow-Monitor.
3. Explicit lineage: every monitoring run is traceable to its source training run.
4. Deterministic retrieval: timeline and state queries must be reproducible.
5. Non-ambiguous status: comparability and lifecycle outcomes must be directly queryable.
6. Artifact-centric evidence: complex outputs stay in structured artifacts, not overloaded tags.
7. System-owned namespace: monitoring experiments are never manually created or named.

## 5. Conceptual Entity-to-MLflow Mapping

### 5.1 Run (Monitoring Run)

Mapped as:

1. A new MLflow run in the `mlflow_monitor/{subject_id}` experiment.
2. Tag `source_run_id` pointing to the source training run.
3. Tag `source_experiment` pointing to the source training experiment name.
4. Tags for lifecycle and comparability state.
5. Artifacts for contract outputs, diffs, findings, and summaries.

The source training run is never modified.

### 5.2 Baseline

Mapped as:

1. A dedicated sentinel MLflow run in `mlflow_monitor/{subject_id}` with tag `role=timeline_config`.
2. The sentinel run holds tag `baseline_source_run_id` pointing to the designated training run.
3. A baseline metadata artifact on the sentinel run captures the snapshot context at pin time.

Sentinel run rules:

1. Created once when the timeline is initialized.
2. Baseline is pinned by writing `baseline_source_run_id` to the sentinel run.
3. Once pinned, the sentinel run's baseline reference is immutable.

Reasoning:

1. A stable sentinel run is the single source of truth for baseline resolution at prepare stage.
2. Avoids scattering baseline metadata across individual monitoring runs.

### 5.3 Timeline

Mapped as:

1. The `mlflow_monitor/{subject_id}` experiment represents the timeline.
2. A monotonic `sequence_index` tag on each monitoring run provides deterministic temporal ordering.
3. Sequence index is assigned by the gateway under serialized finalization per subject.

Timeline ordering rule:

1. Each new monitoring run gets `sequence_index = max(existing sequence_index for subject) + 1`.
2. Assignment is serialized per subject at gateway level. Concurrent finalization is not allowed in v0.
3. MLflow run IDs and timestamps are never used for ordering — only `sequence_index`.

Reasoning:

1. MLflow run IDs are UUIDs (unordered). Timestamps can collide under concurrent triggers.
2. Explicit sequence index gives deterministic, unambiguous ordering.

### 5.4 LKG

Mapped as:

1. Tag `lkg_status=active` on the currently promoted monitoring run.
2. Tag `lkg_status=superseded` on previously promoted monitoring runs.

LKG resolution rule:

1. At any time there is at most one monitoring run with `lkg_status=active` per subject experiment.
2. When a new run is promoted, the gateway atomically sets the new run to `active` and the previous LKG to `superseded`.
3. If no run has `lkg_status=active`, LKG is absent for this timeline.
4. LKG resolution always queries for the single run with `lkg_status=active` — no ambiguity.

Reasoning:

1. Single explicit active tag avoids ambiguity in LKG resolution.
2. Superseded tag preserves full promotion history for auditability.

### 5.5 Contract

Mapped as:

1. Contract identity and version in run metadata tags.
2. Full contract check outputs in artifacts.

### 5.6 Diff and Finding

Mapped as:

1. Structured artifacts per reference mode and summary layer.
2. Optional summary-level metadata tags for quick filtering.

Reasoning:

1. Diffs and findings are potentially high-volume structured records not suited for flat metadata fields.

## 6. Input Resolution at Prepare Stage

### 6.1 Source Run Resolution

At prepare stage, the gateway resolves the source training run from MLflow:

1. Reads `source_experiment` and the compiled source-run selector from the compiled recipe.
2. Interprets current selector semantics as:
   - raw source run ID for user-authored recipes
   - reserved system-default runtime token for the built-in default recipe
3. Generic selector modes such as `latest` are not currently part of implemented v0 behavior.
4. Validates that all `required_metrics` and `required_artifacts` from the recipe are present on that run.
5. If any required input is missing: monitoring run transitions to `failed` with explicit missing-input reason. No partial analysis proceeds.

When prepare also receives an explicit `baseline_source_run_id` for timeline bootstrap or caller confirmation of an existing pinned baseline, the gateway validates that baseline reference separately from the monitored source run selector. The baseline must resolve to the same subject and satisfy the same compiled `source_experiment` filter before it can initialize or confirm timeline state.

### 6.2 Input Availability Principle

MLflow-Monitor is only as good as what is already in MLflow. If the source training run does not have the required inputs, monitoring fails loudly at prepare — never silently or partially.

## 7. Write Path Strategy (Execution-Time Persistence)

### 7.1 Stage-Aligned Writes

Writes to `mlflow_monitor/{subject_id}` reflect workflow stages:

1. On prepare: `source_run_id`, `source_experiment`, `sequence_index`, `baseline_source_run_id` reference, recipe version.
2. On check: comparability status tag + check artifacts.
3. On analyze: diff/finding artifacts + summary artifact.
4. On close: terminal lifecycle status tag + completion metadata.
5. On promotion: `lkg_status` tag update (active/superseded).

### 7.2 Idempotency

Idempotency key: `(subject_id, source_run_id, recipe_id, recipe_version)`.

1. Gateway checks for existing monitoring run matching the idempotency key before creating a new one.
2. Retry of the same monitoring intent does not create duplicate run records.
3. Explicit backfill (different recipe version or intentional re-run) creates a new run record deliberately.

### 7.3 Partial Failure Handling

1. If analysis completes but persistence partially fails, run enters `failed` terminal state.
2. Failure details indicate which persistence phase failed.
3. Prevents false "success" for incompletely persisted runs.

## 8. Read Path Strategy (Query-Time Semantics)

### 8.1 Required Query Classes

1. Timeline traversal: retrieve monitoring runs for a subject ordered by `sequence_index`.
2. Resolve pinned baseline: read `baseline_source_run_id` from sentinel run in subject experiment. The pinned baseline was written only after satisfying the subject and `source_experiment` constraints enforced at prepare time.
3. Resolve current active LKG: query for `lkg_status=active` in subject experiment.
4. Retrieve run-level outputs by monitoring run id.
5. Build anchor-window views from selected anchor run onward.
6. Multi-subject query: retrieve timelines for multiple subjects for side-by-side comparison (View 2).

### 8.2 Read Visibility Rule

1. `closed` and `failed` monitoring runs are included in timeline traversal and anchor-window queries.
2. Runs in intermediate lifecycle states (`created`, `prepared`, `checked`, `analyzed`) are not visible in read queries.
3. `failed` runs are visible in timeline queries with explicit failed status — never silently dropped.

Reasoning:

1. Prevents in-progress runs from polluting trajectory views.
2. Failed runs remain visible for diagnostics and auditability.

### 8.3 Determinism Requirements

1. Timeline ordering is always by `sequence_index` — never by timestamp or run ID.
2. Active LKG resolution returns at most one run deterministically.
3. Non-comparable runs remain visible in timeline queries with explicit comparability status.

### 8.4 Performance Posture (v0)

1. Optimize for correctness and clarity first.
2. Accept moderate query inefficiency in v0 if semantics remain stable.
3. Defer specialized indexing/caching until usage patterns are clear.

## 9. Metadata vs Artifact Partitioning

### 9.1 Metadata/Tags (for filtering and state)

Use tags for:

1. Lifecycle status.
2. Comparability status.
3. Sequence index (timeline ordering).
4. Subject identity.
5. Source run ID and source experiment name.
6. Baseline source run ID reference.
7. LKG status (`active` / `superseded`).
8. Recipe ID and version.
9. Contract ID and version.
10. Mapping version.
11. Idempotency key fields.

### 9.2 Artifacts (for structured evidence)

Use artifacts for:

1. Contract check details and reason set.
2. Diff records (per reference mode).
3. Finding records.
4. Run summaries and lineage snapshots.
5. Error details and diagnostics.
6. Baseline metadata snapshot (on sentinel run).

Reasoning:

1. Tags enable fast coarse retrieval and filtering.
2. Artifacts preserve rich structured outputs without flattening complexity into flat fields.

## 10. Consistency Rules

1. Every closed monitoring run must have a corresponding summary artifact.
2. Every finding must be traceable to diff evidence artifacts.
3. Comparability `fail` runs must not contain regular metric diff artifacts.
4. Timeline references must remain self-consistent within a monitoring run record.
5. LKG promotion records must reference monitoring runs in the same subject experiment.
6. Source training run ID must be present and valid on every monitoring run.

## 11. Mapping Contract Versioning

1. Introduce explicit mapping version tag (`mapping_version=v0`) on all monitoring runs.
2. Mapping version is queryable per monitoring run.
3. Any future schema/representation changes require a version bump.
4. Retrieval logic branches by mapping version when needed.

## 12. Security and Data Hygiene Considerations

1. Avoid storing sensitive payloads in easily exposed metadata tags.
2. Keep sensitive detailed outputs in controlled artifacts.
3. Minimize duplication of large payloads across runs.
4. Ensure references (source_run_id, baseline_source_run_id) are stable and non-ambiguous.

## 13. Operational Risks and Controls

### 13.1 Risk: Convention Drift

If naming and representation are inconsistent, retrieval breaks.

Control:

1. Centralize all write/read conventions in the gateway module.
2. `mlflow_monitor/` prefix is enforced by gateway — no writes outside this namespace are allowed.
3. Validate required mapping elements before close.

### 13.2 Risk: Retrieval Ambiguity

If sequence or LKG resolution is implicit, different clients may disagree.

Control:

1. Explicit `sequence_index` tag for ordering — timestamps never used for ordering.
2. Single authoritative LKG resolution rule via `lkg_status=active` tag.
3. Serialized finalization per subject at gateway level.

### 13.3 Risk: Artifact Inflation

Large diff/finding payloads can grow quickly.

Control:

1. Structured compact formats for artifacts.
2. Optional aggregation/sampling views for heavy timelines (deferred).

### 13.4 Risk: Tight Coupling to MLflow

Future storage evolution becomes costly.

Control:

1. Keep domain and workflow logic MLflow-agnostic.
2. Encapsulate all MLflow operations behind the gateway module.
3. Gateway is the only layer that knows about MLflow primitives.

### 13.5 Risk: Training Run Mutation

Accidental writes to training experiments would break team trust.

Control:

1. Gateway enforces write-only to `mlflow_monitor/` namespace.
2. Any write attempted outside this namespace is a gateway-level error, never a silent failure.

## 14. v0 Scope for MLflow Mapping

In scope:

1. MLflow as read source for training run inputs (read-only).
2. MLflow as write sink for monitoring outputs under `mlflow_monitor/{subject_id}`.
3. System-owned deterministic experiment naming.
4. Stage-aligned monitoring run writes.
5. Explicit sequence-index-based timeline ordering.
6. Sentinel run for baseline pointer storage.
7. LKG active/superseded tag semantics with single authoritative resolution rule.
8. Read visibility rule (only closed runs in timeline/window queries).
9. Idempotency key strategy for retry safety.
10. Anchor-window read support over persisted monitoring run outputs.

Deferred:

1. Multi-backend persistence abstraction runtime.
2. Rich query acceleration subsystem.
3. Cross-store synchronization/replication.

## 15. Acceptance Scenarios

Scenario 1: Comparable monitoring run

1. Source training run found in training experiment with all required inputs.
2. Monitoring run created in `mlflow_monitor/{subject_id}` with correct tags.
3. Run closes with comparable status.
4. Summary and diff/finding artifacts retrievable.
5. Timeline traversal includes run at correct `sequence_index`.
6. Source training run is unmodified.

Scenario 2: Non-comparable monitoring run

1. Source training run found but contract check fails.
2. Monitoring run created in `mlflow_monitor/{subject_id}`.
3. Comparability fail tag written at check stage.
4. Contract failure artifact present; regular metric diff artifacts absent.
5. Run visible in timeline query with non-comparable status.
6. Source training run is unmodified.

Scenario 3: LKG promotion

1. Promotion metadata recorded on promoted monitoring run (`lkg_status=active`).
2. Previous LKG run updated to `lkg_status=superseded`.
3. Active LKG resolution returns promoted run deterministically.
4. Subsequent monitoring runs can reference new LKG.

Scenario 4: Anchor-window query

1. Anchor monitoring run resolved.
2. All later closed runs returned ordered by `sequence_index`.
3. Mixed comparable/non-comparable runs visible with explicit status.

Scenario 5: Backfill

1. Two monitoring runs created against same source training run.
2. Both coexist in `mlflow_monitor/{subject_id}` without collision.
3. Each has a distinct `sequence_index`.
4. Source training run is unmodified.

Scenario 6: Missing required input at prepare

1. Recipe specifies `required_metrics: [auc, f1]`.
2. Source training run only has `auc` logged.
3. Monitoring run transitions to `failed` at prepare stage.
4. Failure reason explicitly states missing `f1` metric.
5. No analysis artifacts produced.

## 16. Relationship to Other v0 Docs

This mapping strategy depends on:

1. `cast_v0.md` for semantics.
2. `workflow_v0.md` for stage behavior.
3. `recipe_v0.md` for configuration provenance and input binding.

And it informs:

1. `acceptance_v0.md` test plan.
2. Implementation-level metadata/artifact conventions (separate spec).

## 17. Summary

MLflow-first mapping in v0 is architecturally sound when:

1. MLflow plays two explicit and separated roles: read source (training experiments) and write sink (monitoring experiments).
2. Training runs are never modified.
3. Monitoring experiments are system-owned under `mlflow_monitor/{subject_id}`.
4. Timeline ordering uses explicit sequence index — never timestamps.
5. Baseline is pinned on a stable sentinel run per subject.
6. LKG uses active/superseded tag semantics with a single authoritative resolution rule.
7. Read visibility rule keeps in-progress runs out of trajectory views.
8. Gateway module enforces all conventions — no direct MLflow writes outside the gateway.
