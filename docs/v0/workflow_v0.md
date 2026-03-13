## 1. Purpose
  This document defines the deterministic execution workflow for MLflow-Monitor v0.
  It specifies:
  1. Run lifecycle states.
  2. Stage-by-stage behavior.
  3. Decision rules.
  4. Failure handling.
  5. Required outputs per stage.
  6. How timeline/baseline/LKG/context are applied.

  This is the canonical behavioral spec for v0 execution.

  ## 2. Workflow Design Goals
  1. Deterministic: same inputs and context should produce same decision path.
  2. Comparable-first: no metric diff interpretation before comparability check.
  3. Transparent: each decision and status transition must be explainable.
  4. Reproducible: outputs must retain enough context to replay reasoning.
  5. Operationally useful: findings should be actionable and traceable to evidence.

  ## 3. Key Concepts Used in Workflow
  1. `Run`: atomic execution event.
  2. `Timeline`: ordered context for runs.
  3. `Baseline`: pinned reference snapshot.
  4. `LKG`: optional trusted run pointer.
  5. `Contract`: comparability rules.
  6. `Diff`: objective delta record.
  7. `Finding`: prioritized interpretation.

  ## 4. Lifecycle State Machine

  ### 4.1 Run Lifecycle States
  1. `created`
  - Run intent exists but no context resolved.

  2. `prepared`
  - Timeline, baseline, references, and inputs resolved.

  3. `checked`
  - Contract checks executed; comparability status decided.

  4. `analyzed`
  - Diff and finding generation completed according to comparability path.

  5. `closed`
  - Outputs persisted; run complete.

  6. `failed`
  - Terminal failure path due to unrecoverable error.

  ### 4.2 Comparability Status (Orthogonal to lifecycle)
  1. `pass`
  2. `warn`
  3. `fail`

  Notes:
  1. Lifecycle and comparability are separate dimensions.
  2. A run can be `closed` with comparability `fail`.

  ## 5. Stage Overview

  ### 5.1 Stage A: Create
  Inputs:
  1. Subject identity.
  2. Requested recipe/config.
  3. Trigger metadata (manual/scheduled).

  Actions:
  1. Allocate run identity.
  2. Record initial request context.
  3. Set lifecycle state to `created`.

  Outputs:
  1. Initial run record.
  2. Request metadata snapshot.

  ### 5.2 Stage B: Prepare
  Inputs:
  1. Run request.
  2. Timeline resolution inputs.
  3. Active recipe/config.

  Actions:
  1. Resolve timeline for subject.
  2. Resolve pinned baseline.
  3. Resolve previous run (if exists).
  4. Resolve current LKG (if exists).
  5. Resolve custom reference run (if specified in recipe — must be on same timeline).
  6. Resolve active contract.
  7. Resolve source MLflow training run from recipe input binding.
  8. Validate required metrics and artifacts exist on source training run.
  9. Resolve execution context fingerprint.

  Transition:
  1. On success: lifecycle -> `prepared`.
  2. On missing required metrics/artifacts: lifecycle -> `failed` with explicit validation error.
  3. On unrecoverable failure: lifecycle -> `failed`.

  Outputs:
  1. Lineage references (timeline/baseline/prev/LKG/custom ref).
  2. Resolved source training run reference.
  3. Effective contract reference.
  4. Effective run context snapshot.

  ### 5.3 Stage C: Contract Check
  Precondition:
  1. Run state is `prepared`.

  Actions:
  1. Execute contract sections:
  - schema checks
  - feature checks
  - metric definition checks
  - data scope checks
  - execution compatibility checks
  2. Aggregate check outcomes into comparability status.
  3. Persist machine-readable reasons.

  Decision:
  1. `pass`: comparable.
  2. `warn`: comparable-with-warnings (still computable in v0 unless policy blocks).
  3. `fail`: non-comparable for metric diff computation.

  Transition:
  1. lifecycle -> `checked`.

  Outputs:
  1. Comparability status.
  2. Structured check reason set.
  3. Blocking/non-blocking flags.

  ### 5.4 Stage D: Analyze
  Precondition:
  1. Run state is `checked`.

  Path 1: Comparable path (`pass` or allowed `warn`)
  1. Compute diffs:
  - run vs baseline
  - run vs previous (if exists)
  - run vs LKG (if exists)
  2. Generate findings from diffs.
  3. Rank findings by severity and impact.
  4. Build run summary.

  Path 2: Non-comparable path (`fail`)
  1. Skip metric diff computation.
  2. Emit explicit non-comparable records.
  3. Generate compatibility findings (if policy/rules define them).
  4. Build run summary with non-comparable emphasis.

  Transition:
  1. lifecycle -> `analyzed`.

  Outputs:
  1. Diff sets or non-comparable records.
  2. Finding set.
  3. Analysis summary.

  ### 5.5 Stage E: Persist
  Precondition:
  1. Run state is `analyzed`.

  Actions:
  1. Persist run metadata and states.
  2. Persist contract check outputs.
  3. Persist diffs/findings.
  4. Persist lineage and context references.
  5. Persist summary bundle.

  Transition:
  1. On success: lifecycle -> `closed`.
  2. On failure: lifecycle -> `failed`.

  Outputs:
  1. Durable run record.
  2. Durable analysis artifacts.

  ### 5.6 Stage F: Optional LKG Decision
  Trigger:
  1. Run is `closed`.
  2. Promotion policy is enabled for timeline.

  Actions:
  1. Evaluate promotion policy inputs:
  - comparability result
  - findings severity profile
  - configured gates
  2. Decide promote/hold.
  3. If promote:
  - mark run as new LKG pointer for timeline.

  Outputs:
  1. Promotion decision record.
  2. Updated LKG pointer (if promoted).

  Notes:
  1. Promotion is policy-driven; no hardcoded domain thresholds in core.
  2. Promotion does not re-run analysis.

  ## 6. Decision Tables

  ### 6.1 Comparability Decision Matrix

  Case 1:
  1. All contract checks pass.
  2. Result: `pass`.
  3. Action: compute diffs and findings normally.

  Case 2:
  1. Non-blocking deviations detected.
  2. Result: `warn`.
  3. Action: compute diffs and findings, attach warnings.

  Case 3:
  1. Blocking incompatibility detected.
  2. Result: `fail`.
  3. Action: mark non-comparable, skip metric deltas, emit compatibility outputs.

  ### 6.2 Reference Comparison Matrix

  Default (always attempted when comparable):
  1. Baseline comparison.
  2. Previous comparison only if previous run exists.
  3. LKG comparison only if LKG exists.

  Optional (user-specified):
  4. Custom reference run comparison if `reference_run_id` specified in recipe.
  - Must be a run on the same timeline.
  - Cross-timeline custom references are not supported in v0.

  For non-comparable runs:
  1. All metric diff comparisons skipped.
  2. Structural/compatibility records still emitted.

  ### 6.3 LKG Promotion Matrix (Policy-Driven)

  Case 1:
  1. Policy disabled.
  2. Action: no promotion evaluation.

  Case 2:
  1. Policy enabled and gates pass.
  2. Action: promote run to LKG.

  Case 3:
  1. Policy enabled and gates fail.
  2. Action: hold current LKG.

  Case 4:
  1. Run non-comparable.
  2. Default action: hold current LKG.

  ## 7. Failure Handling Model

  ### 7.1 Preparation Failures
  Examples:
  1. Timeline cannot be resolved.
  2. Baseline missing for required path.
  3. Contract reference invalid.
  4. Required data reference missing.

  Behavior:
  1. Record failure reason.
  2. Transition to `failed`.
  3. Emit partial summary if possible.

  ### 7.2 Contract Check Failures (Execution errors, not logical fail result)
  Examples:
  1. Checker runtime exception.
  2. Invalid checker configuration.

  Behavior:
  1. Transition to `failed`.
  2. Persist execution error details.
  3. Do not produce analysis outputs.

  Note:
  1. Logical contract `fail` is not lifecycle failure; it is a valid checked outcome.

  ### 7.3 Analyze Failures
  Examples:
  1. Diff engine exception.
  2. Finding engine exception.

  Behavior:
  1. Transition to `failed`.
  2. Persist available partial artifacts with explicit incompleteness marker.

  ### 7.4 Persistence Failures
  Examples:
  1. Storage write failure.
  2. Artifact write failure.

  Behavior:
  1. Transition to `failed`.
  2. Persist local emergency error log if possible.

  ## 8. Output Contracts by Stage

  ### 8.1 Required After Prepare
  1. Resolved lineage:
  - timeline
  - baseline
  - previous (optional)
  - LKG (optional)
  2. Effective contract reference.
  3. Effective context fingerprint.

  ### 8.2 Required After Check
  1. Comparability status.
  2. Check reason records.
  3. Blocking/non-blocking classification.

  ### 8.3 Required After Analyze
  Comparable path:
  1. Diff outputs by reference mode.
  2. Findings set.
  3. Summary metrics.

  Non-comparable path:
  1. Compatibility break records.
  2. Findings (if generated).
  3. Summary with explicit non-comparable status.

  ### 8.4 Required After Close
  1. Final lifecycle status.
  2. Final comparability status.
  3. Final artifacts/record references.
  4. Optional promotion decision output.

  ## 9. Anchor-Window Workflow Integration
  Anchor-window analysis is read-time behavior over closed runs.

  Steps:
  1. Resolve requested timeline.
  2. Resolve anchor run within timeline.
  3. Collect runs from anchor onward.
  4. Aggregate existing run-level outputs.
  5. Present:
  - trajectory summaries
  - finding trends
  - non-comparable intervals

  Rules:
  1. No recomputation of canonical run outputs during window read.
  2. Non-comparable runs remain visible and flagged.

  ## 10. Concurrency and Ordering Assumptions (v0)
  1. Timeline sequence is append-only in run completion order.
  2. LKG promotion acts on closed runs only.
  3. Concurrent run triggers on same timeline are allowed only if sequence ordering remains deterministic.
  4. If deterministic ordering cannot be guaranteed, v0 should serialize run finalization per timeline.

  Reasoning:
  1. Temporal ambiguity undermines timeline semantics and anchor-window consistency.

  ## 11. Observability Requirements
  Every run should expose:
  1. Current lifecycle state.
  2. Comparability status.
  3. Stage timing metrics.
  4. Error reason (if failed).
  5. Artifact and lineage pointers.

  Reasoning:
  1. Workflow systems fail in opaque ways without stage-level observability.

  ## 12. v0 Invariants for Workflow
  1. No analysis before comparability check.
  2. No close without persistence attempt.
  3. Every finding links to evidence diffs or explicit compatibility records.
  4. Contract `fail` never silently produces metric deltas.
  5. LKG promotion cannot target a non-closed run.
  6. LKG always belongs to same timeline.

  ## 13. Boundaries and Exclusions
  Not in v0 workflow:
  1. Branch/merge timelines.
  2. Multi-contract simultaneous evaluation in one run.
  3. Automated compatibility migration.
  4. Re-anchoring timeline topology inside a run.

  ## 14. Acceptance Scenarios

  Scenario 1: Normal comparable run
  1. Prepare succeeds.
  2. Contract returns pass.
  3. Diffs generated for available references.
  4. Findings generated.
  5. Run closes successfully.

  Scenario 2: Comparable with warnings
  1. Contract returns warn.
  2. Diffs/findings generated with warning context.
  3. Run closes.
  4. Optional policy may still block LKG promotion.

  Scenario 3: Non-comparable run
  1. Contract returns fail.
  2. Metric diffs skipped.
  3. Compatibility records + findings emitted.
  4. Run closes as non-comparable.

  Scenario 4: Missing previous/LKG references
  1. Baseline comparison still runs.
  2. Missing optional refs logged as not available.
  3. Run closes successfully.

  Scenario 5: Persistence failure
  1. Analysis may complete.
  2. Persistence fails.
  3. Run transitions to failed.
  4. Error details recorded.

  ## 15. Next Document Dependencies
  This workflow spec should be paired with:
  1. `cast_v0.md` (domain semantics)
  2. `recipe_model_v0.md` (recipe compilation/validation)
  3. `mlflow_mapping_v0.md` (storage conventions)
  4. `acceptance_v0.md` (test plan and pass criteria)