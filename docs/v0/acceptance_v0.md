# MLflow-Monitor v0 Acceptance Criteria and Test Plan
  Date: 2026-03-09
  Status: Draft (Planning)

  ## 1. Purpose
  This document defines how we validate that v0 is complete and correct at a system level before wider rollout.

  It translates the v0 design into:
  1. Acceptance criteria.
  2. Scenario-based tests.
  3. Pass/fail conditions.
  4. Coverage matrix across CAST, workflow, recipe, and MLflow mapping.

  This is a behavioral validation plan, not an implementation test framework specification.

  ## 2. Acceptance Philosophy
  v0 is accepted when:
  1. Core value outcomes are demonstrably satisfied.
  2. System invariants hold under normal and failure paths.
  3. Outputs are reproducible, explainable, and actionable.
  4. Scope boundaries are respected (no hidden v1 complexity).

  ## 3. Core Value Acceptance Criteria

  ### 3.1 Baseline-Aware Monitoring
  Pass if:
  1. Every run can resolve and report baseline reference context.
  2. Baseline comparison output is produced for comparable runs.
  3. Baseline immutability and timeline pin semantics are preserved.

  Fail if:
  1. Baseline reference is implicit/ambiguous.
  2. Comparable run lacks baseline diff outputs.
  3. Baseline mutates without explicit timeline reset behavior.

  ### 3.2 Comparability-First Workflow
  Pass if:
  1. Contract checks always execute before diff/finding generation.
  2. Every run has explicit comparability status (`pass|warn|fail`).
  3. `fail` runs do not produce normal metric delta outputs.
  4. Check reasons are persisted and queryable.

  Fail if:
  1. Any path allows analysis before comparability check.
  2. Comparability status is missing or ambiguous.
  3. Non-comparable runs silently produce misleading deltas.

  ### 3.3 Actionable Output Model
  Pass if:
  1. Diffs are objective and structured.
  2. Findings are derived from evidence and include severity.
  3. Every finding links to supporting diff/compatibility evidence.
  4. Run summary clearly reflects top findings and status.

  Fail if:
  1. Findings appear without traceable evidence.
  2. Diff/finding boundaries are mixed or inconsistent.
  3. Outputs are technically complete but not operationally interpretable.

  ## 4. Invariant Acceptance Criteria
  All v0 invariants from `cast_v0.md` and `workflow_v0.md` must hold.

  Critical invariants:
  1. One run belongs to exactly one timeline.
  2. One timeline has one pinned baseline in v0.
  3. One run has one comparability status.
  4. `fail` comparability implies non-comparable metric outputs.
  5. LKG points to run in same timeline.
  6. Baseline is immutable once pinned.
  7. Findings reference evidence.

  Any invariant violation is release-blocking for v0.

  ## 5. Scenario-Based Acceptance Tests

  ## 5.1 Scenario A: First Comparable Run on New Timeline
  Setup:
  1. New subject and timeline.
  2. Baseline established and pinned.
  3. Valid recipe and contract.

  Expected:
  1. Run transitions through created -> prepared -> checked -> analyzed -> closed.
  2. Comparability status is pass or warn.
  3. Baseline diff outputs exist.
  4. Findings and summary exist.
  5. Timeline contains run with deterministic ordering.

  Pass condition:
  1. All expected outputs present and invariant-safe.

  ## 5.2 Scenario B: Regular Iteration Run with Previous and LKG
  Setup:
  1. Existing timeline with baseline.
  2. Previous run exists.
  3. LKG exists.

  Expected:
  1. Comparable run computes all available reference diff modes:
  - baseline
  - previous
  - LKG
  2. Findings include evidence links.
  3. Optional promotion decision recorded.

  Pass condition:
  1. Reference-specific outputs are correct and traceable.

  ## 5.3 Scenario C: Contract Warning Path
  Setup:
  1. Non-blocking contract deviation introduced.

  Expected:
  1. Comparability status = warn.
  2. Diffs/findings still generated.
  3. Warning reasons persisted and visible.

  Pass condition:
  1. Warn path remains fully analyzable and explicitly marked.

  ## 5.4 Scenario D: Contract Fail Path (Non-Comparable)
  Setup:
  1. Blocking incompatibility introduced.

  Expected:
  1. Comparability status = fail.
  2. Metric diff generation skipped.
  3. Compatibility evidence and findings persisted.
  4. Run closes as non-comparable, not silently dropped.

  Pass condition:
  1. No misleading metric deltas.
  2. Non-comparable state is explicit in timeline and summaries.

  ## 5.5 Scenario E: Missing Optional References
  Setup:
  1. Baseline exists.
  2. Previous and/or LKG absent.

  Expected:
  1. Run remains valid.
  2. Available comparisons execute.
  3. Missing references are explicitly recorded as unavailable.

  Pass condition:
  1. No false failures due to optional reference absence.

  ## 5.6 Scenario F: Anchor-Window Analysis from Arbitrary Run
  Setup:
  1. Timeline with multiple runs including mixed statuses.

  Expected:
  1. Anchor run resolves.
  2. Window includes all runs from anchor onward.
  3. Comparable and non-comparable runs are both visible.
  4. Aggregated view derives from stored run outputs.

  Pass condition:
  1. Deterministic ordering and status fidelity in window output.

  ## 5.7 Scenario G: LKG Promotion Success
  Setup:
  1. Closed comparable run meeting policy.
  2. Promotion policy enabled.

  Expected:
  1. Promotion decision recorded.
  2. Run marked/promoted as current LKG.
  3. Subsequent runs can resolve this LKG as active reference.

  Pass condition:
  1. LKG pointer update is deterministic and auditable.

  ## 5.8 Scenario H: LKG Promotion Rejection
  Setup:
  1. Closed run not meeting policy or non-comparable run.

  Expected:
  1. Promotion decision = hold/reject.
  2. Existing LKG remains unchanged.
  3. Decision reason recorded.

  Pass condition:
  1. Trust state remains stable and explainable.

  ## 5.9 Scenario I: Recipe Validation Failure
  Setup:
  1. Invalid recipe (missing or conflicting required sections).

  Expected:
  1. Run cannot enter normal workflow execution.
  2. Validation errors are explicit and actionable.
  3. No partial misleading analysis artifacts.

  Pass condition:
  1. Fail-fast behavior with clear diagnostics.

  ## 5.10 Scenario J: Persistence Failure During Close
  Setup:
  1. Simulate storage write failure near finalization.

  Expected:
  1. Run transitions to failed terminal state.
  2. Failure reason persisted where possible.
  3. System does not report false close success.

  Pass condition:
  1. Failure is explicit and recoverable in diagnostics.

  ## 6. Coverage Matrix

  Coverage by domain entity:
  1. Run: A, B, C, D, E, I, J
  2. Baseline: A, B, D
  3. Timeline: A, B, F
  4. LKG: B, G, H
  5. Contract: C, D
  6. Diff: A, B, C, D
  7. Finding: A, B, C, D

  Coverage by workflow stages:
  1. Create/Prepare: A, B, E
  2. Check: C, D
  3. Analyze: A, B, C, D
  4. Persist/Close: A, B, J
  5. Promotion: G, H

  Coverage by recipe:
  1. Valid binding: A, B
  2. Invalid binding: I

  Coverage by MLflow mapping behavior:
  1. Standard write/read: A, B, C
  2. Non-comparable representation: D
  3. Timeline/anchor retrieval: F
  4. Promotion metadata/read semantics: G, H
  5. Failure semantics: J

  ## 7. Exit Criteria for v0
  v0 is accepted only if:
  1. All critical scenarios pass.
  2. No invariant violations occur in tested paths.
  5. LKG behavior is auditable and stable.
  6. Recipe validation prevents invalid workflow execution.
  7. Persistence failure behavior avoids false-success states.

  ## 8. Nice-to-Have (Non-Blocking) Criteria
  1. Performance optimizations for large timelines.
  2. Richer summary visualization polish.
  3. Additional warning categorization granularity.
  4. Extended diagnostics packaging.

  These may be deferred if core acceptance criteria are met.

  ## 9. Open Validation Questions
  1. What minimum scenario count is required before declaring readiness?
  2. Which scenarios must be repeated under load/concurrency conditions for v0 confidence?
  3. What threshold of manual review is acceptable for findings quality before policy tuning?

  ## 10. Review and Sign-Off
  Recommended sign-off sequence:
  1. Product/Domain sign-off:
  - value outcomes and scenario relevance.
  2. Architecture sign-off:
  - invariants, boundaries, and scope lock.
  3. Implementation sign-off:
  - readiness to convert docs into executable backlog.