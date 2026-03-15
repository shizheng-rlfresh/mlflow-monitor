# CAST v0: Core Actors and State Transitions

## Purpose

  Define the v0 mental model for MLflow-Monitor:

- baseline-aware monitoring
- comparability-first workflow
- actionable diffs/findings over time

## Core Value Outcomes

  1. Baseline-aware monitoring.
  2. Comparable runs with explicit contracts.
  3. Actionable diffs and prioritized findings.

## First-Class Citizens

  1. Run

- Atomic monitoring execution event.
- Contains inputs, context, outputs, and lifecycle status.

  1. Baseline

- Frozen reference snapshot for comparison.
- Includes model/data/config/metrics/environment context.

  1. Timeline

- Ordered sequence of runs for one monitored subject.

  1. LKG (Last Known Good)

- Promoted trusted run on a timeline.

  1. Contract

- Rules defining comparability and diff semantics.

  1. Diff

- Objective change record between run and reference(s).

  1. Finding

- Interpreted, prioritized issue derived from one or more diffs.

## Relationship Model

- A Timeline has many Runs.
- A Timeline has one pinned Baseline (v0).
- A Timeline has at most one active LKG pointer.
- Each Run is evaluated against one active Contract.
- Each Run produces Diffs.
- Findings are derived from Diffs.

## Baseline Model

  Baseline is a snapshot contract, not just a model pointer.

  Conceptual contents:

  1. Model identity/version and parameter fingerprint.
  2. Data snapshot/scope reference.
  3. Run config (feature set, thresholds, metric spec).
  4. Baseline observed offline metrics.
  5. Code/environment reproducibility context.

## Contract Model

  Contract defines whether runs are comparable and how diffs are computed:

  1. Schema contract.
  2. Feature contract.
  3. Metric contract.
  4. Data-scope contract.
  5. Execution contract (runtime/dependency/service constraints).
  6. Compatibility outcome: `pass | warn | fail` with machine-readable reasons.

## Diff vs Finding

- Diff answers: “What changed?”
- Finding answers: “So what should we do?”

## Subject Model

  A subject corresponds to an MLflow training experiment (or a stable tagged subset of one).
  The subject ID is the only naming decision users make. It determines:

  1. Which MLflow training experiment is the read source.
  2. The monitoring experiment namespace: `mlflow_monitor/{subject_id}`.
  3. The timeline identity — one subject, one timeline, one monitoring experiment.

  Subject IDs should be stable, team-agreed identifiers (e.g. `churn_model`, `fraud_scorer`).
  The system derives all experiment naming from subject ID. Users never manually name monitoring experiments.

## Anchor Window (Generalized Timeline View)

  User selects any run on a timeline as an anchor.
  System returns diffs/findings from that run onward.

  Presets:

  1. Baseline-anchored: anchor is the baseline run. Shows full trajectory from the beginning of the timeline.
  2. LKG-anchored: anchor is the current LKG run. Shows performance since the last trusted state.
  3. Custom: user specifies any run on the timeline as anchor.

  Rules:

  1. One anchor per query/view.
  2. Window outputs are derived from run-level outputs.
  3. Non-comparable runs are explicitly listed, not silently dropped.
  4. Anchor must belong to the same timeline.

## v0 Scope

  1. Single timeline per subject.
  2. Single active contract version per timeline.
  3. One pinned baseline anchor.
  4. One movable LKG pointer.
  5. Fixed default run comparisons:

- run vs baseline (always, if comparable)
- run vs previous run (if exists)
- run vs LKG (if exists)

  1. Optional custom reference run:

- User can specify any run on the same timeline as an additional reference point.
- Default behavior (baseline + LKG + previous) still always executes.
- Cross-timeline custom references are not supported in v0.

  1. Major incompatibility handling:

- mark run as non-comparable
- require explicit user decision (update contract or start new timeline)

## Intended Outcome Views (v0)

  The system is designed to support these user-facing outcome views:

### View 1: Single Model Trajectory

  How did one model perform over time across monitoring runs.

- X axis: time / run sequence
- Y axis: metric values
- Reference lines: baseline, LKG
- Status markers: pass / warn / fail per run

### View 2: Multi-Model Side-by-Side

  Same time window, multiple subjects compared.

- Implemented by querying multiple timelines in parallel.
- Each timeline is one subject.

### View 3: Counterfactual Lines (Deferred to v1)

  What would a retired model have scored on more recent evaluation windows.

- Requires model identity and evaluation window to be decoupled.
- Not supported in v0 because MLflow input provides one evaluation snapshot per training run.
- Design note: keep model identity and evaluation window as separate conceptual fields to avoid designing this out.

## v1 Scope (Deferred)

  1. Timeline branching/epochs.
  2. Advanced compatibility transitions.
  3. Custom comparison profiles.

## v0 Invariants

  1. Every Run belongs to exactly one Timeline.
  2. Every Timeline has exactly one pinned Baseline.
  3. Every Run is evaluated against exactly one active Contract.
  4. Every Run produces comparability status: `pass | warn | fail`.
  5. If status is `fail`, run is non-comparable for metric diffs.
  6. Every comparable Run produces baseline diff output.
  7. Previous-run diff exists when a previous run exists.
  8. LKG diff exists when an LKG exists.
  9. Every Finding references one or more supporting Diffs.
  10. A Timeline has at most one active LKG pointer.
  11. LKG must reference an existing Run in the same Timeline.
  12. Baseline is immutable once pinned to a Timeline.
  13. Contract changes are explicit and versioned.
  14. v0 has no branching inside a timeline.
  15. Major incompatibility requires explicit user action.

## Non-Goals (v0)

  1. Full branching semantics.
  2. Automatic conflict resolution across incompatible contracts.
  3. Overly generic workflow engine abstractions.
