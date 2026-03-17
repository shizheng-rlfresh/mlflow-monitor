The current design is highly aligned with your PDF’s core idea, but it is now more concrete and opinionated in system behavior.

  The PDF said:

- The problem is workflow-level, not metric-level.
- Baseline/comparability/actionability are the core pains.
- MLflow should be optional/not the center.

  Your current v0 design says:

- Make baseline/comparability/actionability first-class through explicit entities and lifecycle rules.
- Use timeline and LKG to preserve temporal decision context.
- Use recipe as the customization surface.
- For v0 execution practicality, use MLflow-first storage.

  So conceptually, this is the same product thesis, but now translated into implementable system structure.

  ———

  Where It Is Similar to the PDF

  1. Problem framing is unchanged

- You are still solving “monitoring workflow reliability,” not building another metric library.

  1. Core value focus is unchanged

- Baseline-aware comparisons.
- Comparability checks before interpretation.
- Actionable outputs (diffs/findings), not raw logs.

  1. Modular philosophy is unchanged

- There is still a “core framework + reusable modules + higher-level assembly” concept.

  1. Practitioner-first intent is unchanged

- The design still optimizes for teams currently doing ad-hoc scripts and manual glue.

  ———

  Where It Is Different from the PDF, and Why

  1. More explicit temporal model

- You added Timeline, LKG, and Anchor Window.
- Why: the PDF identified repeated monitoring pain, but did not fully formalize ongoing run evolution semantics.

  1. Stronger type boundaries

- You explicitly separated Diff from Finding.
- Why: this prevents factual data from being mixed with interpretation, which improves trust and auditability.

  1. Recipe is now an explicit architecture layer

- Not just “templates later,” but a clear composition boundary.
- Why: this is the practical mechanism to support user-specific workflows without polluting core logic.

  1. v0 is deliberately stricter

- Single timeline, single active contract, no branching.
- Why: reduces complexity and edge-case explosion while preserving end-to-end value.

  1. MLflow positioning is pragmatically adjusted

- You originally preferred optional sink. Current v0 plan moves toward MLflow-first persistence.
- Why: faster delivery and less infra burden in v0.
- Tradeoff: stronger coupling to MLflow model. This is acceptable if explicit and revisitable in v1+.

  ———

  Detailed System Design Elaboration

  1. Domain Model (What exists and what it means)

  1. Run

- Atomic execution event.
- Purpose: immutable record of one monitoring evaluation.
- Why it matters: all decisions and history are anchored in concrete run units.

  1. Baseline

- Pinned reference snapshot.
- Purpose: stable long-horizon anchor for comparisons.
- Why it matters: without a stable anchor, degradation reasoning becomes ambiguous over time.

  1. Timeline

- Ordered sequence of runs for one subject.
- Purpose: represent model/system evolution over time.
- Why it matters: monitoring is a trajectory problem, not isolated run snapshots.

  1. LKG

- A promoted trusted run on timeline.
- Purpose: operational safety point for promotion/rollback style decisions.
- Why it matters: baseline and “currently trusted state” are not the same thing.

  1. Contract

- Rules for comparability and calculation semantics.
- Purpose: prevent apples-to-oranges comparisons.
- Why it matters: comparability failures are one of the highest-cost failure modes in monitoring.

  1. Diff

- Objective change records.
- Purpose: encode what changed in machine/human-readable form.
- Why it matters: this is evidence, not opinion.

  1. Finding

- Interpreted and prioritized issue records.
- Purpose: map diffs to action.
- Why it matters: teams act on findings, not raw deltas.

  ———

  1. Workflow Model (How system behaves)

  1. Prepare

- Resolve timeline, baseline, contract, references.
- Why first: all later outputs are invalid if context resolution is wrong.

  1. Comparability Check

- Run contract checks before metric interpretation.
- Why: strict gatekeeping avoids misleading outputs and false confidence.

  1. Compute Diffs

- Compare against baseline, previous run, LKG where available.
- Why: these three references answer different questions:
- Baseline = long-term drift/regression.
- Previous = iteration-to-iteration movement.
- LKG = operational trust gap.

  1. Generate Findings

- Turn diffs into severity-ranked actionable issues.
- Why: closes the gap between technical signals and operational decisions.

  1. Persist

- Save run state, lineage, and outputs.
- Why: reproducibility and historical analysis require stable persisted artifacts.

  1. Promotion Decision

- Optionally update LKG based on policy.
- Why: keeps trust state explicit and auditable.

  1. Analyze by Anchor

- Query from any anchor run onward.
- Why: gives flexible trajectory analysis without requiring branching complexity in v0.

  ———

  1. Recipe Layer (Customization boundary)

  1. What recipe does

- Binds data inputs, contract profile, metric set, finding policy, and output options into one executable configuration.

  1. Why recipe must be separate

- Core should own correctness and invariants.
- Recipe should own use-case variance.
- This separation keeps extensibility without destabilizing core behavior.

  1. v0 recipe scope

- Config-first, versioned, validated, compilable to normalized execution plan.
- Why: enough for real workflows, avoids overbuilding authoring UX too early.

  ———

  1. Architecture Boundaries (System-level separation)

  1. Domain Layer

- Owns semantics and invariants.

  1. Workflow Layer

- Owns execution order and decision logic.

  1. Platform/Integration Layer

- Owns persistence, retrieval, and external adapters.

  1. Key boundary principle

- Integration cannot bypass core invariants.
- Why: prevents storage/tooling choices from changing correctness behavior.

  ———

  1. Why chosen

- Faster practical delivery.
- Lower operational overhead.
- Immediate fit with existing ML team habits.

  1. What to watch

- Coupling risk to MLflow metadata/artifact conventions.
- Query model limitations vs dedicated DB.
- Need clear namespace/versioning to keep future migration feasible.

  1. How to keep it safe architecturally

- Keep domain/workflow logic MLflow-agnostic in code structure.
- Treat MLflow as persistence adapter at design boundary, even if it is the only adapter in v0 runtime.

  ———

  1. v0 Constraint Philosophy

  1. Single timeline per subject.
  1. Single active contract version.
  1. No branching.
  1. Explicit handling of major incompatibility via non-comparable status and user decision.

  Why this is right now:

- Complexity in monitoring comes from hidden ambiguity.
- v0 should optimize for deterministic behavior and user trust, not maximal flexibility.

  ———

  1. What this means for “value delivered”
  If implemented as designed, v0 should concretely deliver:

  1. No ambiguous baseline reference.
  1. No silent invalid comparisons.
  1. Clear difference between evidence and interpretation.
  1. Time-aware monitoring context through timeline + anchor windows.
  1. Practical operational fit via MLflow-backed execution.

  This directly validates your strongest original claims from the proposal.

  ———

  If you want, next I can turn this into a document-ready version with exact sections matching your repo style, so you can drop it directly into docs/v0/system_v0.md and keep cast_v0.md as the
  concise companion spec.

› yes please, with a lot of details.

• # MLflow-Monitor v0 System Overview (Detailed Draft)

## 1. Purpose

  This document is a detailed system-level overview for v0. It explains:

  1. How the current design relates to the proposal PDF.
  2. What changed and why.
  3. The full conceptual architecture.
  4. The domain model (CAST).
  5. The workflow mental model.
  6. The recipe layer.
  7. The MLflow-first platform strategy.
  8. v0 scope boundaries, risks, and next design checkpoints.

  The intent is to align product meaning, engineering structure, and implementation priorities before coding.

## Integration and Usage Model

### How MLflow-Monitor Connects to Your Stack

  MLflow-Monitor is a post-hoc analysis layer that sits on top of existing MLflow runs. It does not participate in training, does not own data ingestion, and does not touch raw data.

  The flow is:

  ```
  [Team trains model]
        ↓
  [MLflow logs run: metrics, params, artifacts]
        ↓
  [MLflow-Monitor kicks off a monitoring run]
        ↓
  [Reads from source MLflow experiment — never modifies it]
        ↓
  [Writes findings/diffs/summaries to mlflow_monitor/{subject_id}]
  ```

### What "Subject" Means Concretely

  A subject maps to an MLflow experiment the team already owns. The subject ID is the stable identity used to name the monitoring experiment and organize the timeline.

  Example: subject `churn_model` → monitors experiment `churn_model` → writes to `mlflow_monitor/churn_model`.

### What MLflow-Monitor Reads

  From the source training run:

  1. Logged metrics (e.g. auc, f1, recall).
  2. Logged params and tags.
  3. Artifacts produced during evaluation (e.g. eval_report.json, confusion_matrix.json).

  Hard rule: if the source run lacks required metrics or artifacts, the monitoring run fails at prepare stage with a clear error. No partial analysis.

### Trigger Patterns

  Three natural trigger patterns, all supported by the same interface:

  1. Manual trigger (v0 primary):
  `monitor.run(subject="churn_model", recipe="churn_v1")`
  Good for exploration, debugging, one-off analysis.

  2. Post-training hook (most common production use):
  Called after a training pipeline step closes a training run. Can be a CI step, Airflow task, or callback. Recipe specifies which upstream run to treat as current.

  3. Scheduled trigger:
  A cron job runs against the latest available training run for a subject. Recipe defines the subject; monitor resolves "latest" at execution time.

  v0 recommendation: implement manual trigger only, but design the interface so hooks can wrap it without changes.

### What Users See

  After several monitoring runs accumulate:

  1. Single model trajectory view — how did this model perform over time, relative to baseline and LKG.
  2. Multi-model comparison — query multiple subjects side by side over the same time window.
  3. Per-run detail — diffs, findings, comparability status, lineage for any individual run.

### What MLflow-Monitor Does Not Do

  1. Does not ingest raw data.
  2. Does not modify or append to training runs or training experiments.
  3. Does not manage custom databases or storage infrastructure.
  4. Does not run during training.
  5. Does not re-evaluate models on new data (counterfactual analysis — deferred to v1).

  ———

  The proposal framed the product as a monitoring workflow framework, not a metric toolkit. The key problems were:

  1. Baseline handling is weak and inconsistent.
  2. Comparability is brittle and often accidental.
  3. Outputs are not action-oriented.
  4. Evidence is often insufficient for debugging and trust.
  5. Integration into systems like MLflow is awkward and repetitive.
  6. Templates and reusable components are needed to reduce boilerplate.

  The proposal also emphasized that MLflow should be treated as a sink/viewer rather than the conceptual center.

## 3. How Current Design Maps to the PDF

  The current design keeps the same core intent, but now makes it operationally explicit.

### 3.1 What is the same

  1. The primary product value remains baseline-aware, comparability-first, actionable monitoring.
  2. The system still prioritizes workflow correctness over adding many metrics.
  3. Modularity remains central: core logic separated from use-case customization.
  4. The design still targets practical team pain from real production workflows.

### 3.2 What changed

  1. The time model is now explicit (Timeline, LKG, anchor-window analysis).
  2. The output semantics are explicit (Diff is factual, Finding is interpreted).
  3. The customization boundary is explicit (Recipe layer).
  4. v0 constraints are explicit (single timeline, no branching, explicit non-comparable handling).
  5. Storage strategy is pragmatic for v0 (MLflow-first), even if long-term architecture stays conceptually decoupled.

### 3.3 Why these changes were necessary

  1. The PDF articulated pain and value, but not enough execution semantics.
  2. Temporal behavior needed concrete representation to avoid implicit/fragile logic.
  3. Teams need deterministic lifecycle and status outputs to trust the framework.
  4. v0 must avoid edge-case explosion and premature generalization.

## 4. Product Thesis for v0

  MLflow-Monitor v0 is a deterministic offline monitoring workflow engine that:

  1. Anchors every run in explicit reference context.
  2. Performs comparability checks before computing metric deltas.
  3. Produces both objective diffs and actionable findings.
  4. Preserves longitudinal context with timeline-based analysis.
  5. Uses MLflow as the operational storage backbone in v0.

## 5. Design Principles

  1. Correctness before flexibility.
  2. Explicit states over implicit behavior.
  3. Evidence before interpretation.
  4. Stable core semantics with configurable policy.
  5. Practical v0 constraints over ambitious generalization.

  Each principle is deliberate:

  1. Correctness before flexibility prevents misleading outputs.
  2. Explicit states improve observability and debugging.
  3. Evidence before interpretation preserves trust and auditability.
  4. Stable core semantics enable cross-team consistency.
  5. Practical constraints reduce time-to-value.

## 6. Domain Model (CAST v0)

  First-class citizens:

  1. Run
  2. Baseline
  3. Timeline
  4. LKG
  5. Contract
  6. Diff
  7. Finding

### 6.1 Run

  1. Definition: one monitoring execution event.
  2. Role: the atomic unit of history and evaluation.
  3. Must include: input context, execution context, outcomes, status.
  4. Why first-class: all other entities either configure runs or are derived from runs.

### 6.2 Baseline

  1. Definition: frozen reference snapshot.
  2. Role: long-horizon anchor for comparability.
  3. Contents conceptually include: model ref, data ref, metric spec, observed metrics, execution context.
  4. Why first-class: baseline ambiguity is one of the root pains.

### 6.3 Timeline

  1. Definition: ordered set of runs for one subject.
  2. Role: preserve evolution and trajectory.
  3. Why first-class: run-to-run and run-to-reference analysis requires temporal structure.

### 6.4 LKG (Last Known Good)

  1. Definition: promoted trusted run on a timeline.
  2. Role: operational trust reference.
  3. Why distinct from baseline: baseline is a stable anchor; LKG is a dynamic trust pointer.

### 6.5 Contract

  1. Definition: explicit comparability and computation rules.
  2. Scope includes: schema, features, metrics, data scope, execution constraints.
  3. Output: comparability status with reasons.
  4. Why first-class: prevents accidental apples-to-oranges comparisons.

### 6.6 Diff

  1. Definition: objective change record.
  2. Role: capture what changed with no interpretation.
  3. Why first-class: factual substrate for all downstream reasoning.

### 6.7 Finding

  1. Definition: prioritized interpretation of one or more diffs.
  2. Role: actionable output for teams.
  3. Why first-class: turns technical signals into operational decisions.

## 7. Core Semantic Boundaries

  1. Diff is objective, Finding is interpretive.
  2. Baseline is immutable anchor, LKG is movable trust point.
  3. Contract governs comparability, not business decision policy.
  4. Run status and comparability status are related but separate concepts.
  5. Non-comparable is explicit and visible, never silent.

  These boundaries reduce confusion and implementation drift.

## 8. Workflow Mental Model (Detailed)

  Lifecycle stages:

### 8.1 Prepare

  1. Resolve subject and timeline.
  2. Resolve pinned baseline.
  3. Resolve previous run and current LKG if available.
  4. Resolve active contract.
  5. Resolve run inputs and execution context.

  Purpose:

  1. Build deterministic context before any evaluation.

### 8.2 Comparability Check

  1. Execute contract sections.
  2. Produce pass, warn, or fail.
  3. Persist reason set.

  Purpose:

  1. Decide if metric comparisons are valid.

### 8.3 Compute Diffs

  If comparable:

  1. Compute run vs baseline.
  2. Compute run vs previous when previous exists.
  3. Compute run vs LKG when LKG exists.

  If fail:

  1. Mark run non-comparable.
  2. Skip metric delta computations.
  3. Emit explicit comparability break outputs.

  Purpose:

  1. Guarantee that numeric deltas only exist under valid comparability.

### 8.4 Generate Findings

  1. Transform diff records into issue records.
  2. Assign severity and category.
  3. Attach action guidance.
  4. Link each finding to supporting diffs.

  Purpose:

  1. Make outputs operationally useful, not just descriptive.

### 8.5 Persist Outputs

  1. Persist run state and metadata.
  2. Persist contract outputs.
  3. Persist diffs and findings.
  4. Persist lineage links.

  Purpose:

  1. Ensure reproducibility, traceability, and historical analysis.

### 8.6 Promotion Decision

  1. Evaluate configurable promotion policy.
  2. Promote run to LKG when policy passes.
  3. Keep prior LKG otherwise.

  Purpose:

  1. Separate monitoring computation from trust state updates.

### 8.7 Timeline Analysis

  1. Support anchor-window queries from any run.
  2. Surface trend and issue trajectories from chosen anchor onward.
  3. Explicitly include non-comparable intervals.

  Purpose:

  1. Provide coherent temporal reasoning without branching in v0.

## 9. Anchor Window Model

  Anchor window is a generalized analysis concept:

  1. Choose an anchor run in timeline.
  2. Return outputs for runs from anchor forward.
  3. Baseline view and LKG view are just anchor presets.

  Benefits:

  1. Unified temporal analysis model.
  2. Flexible retrospective and forward-looking inspection.
  3. No need for v0 branching complexity.

## 10. Recipe Layer (Why and How)

  Recipe is a dedicated layer for use-case composition.

### 10.1 Why recipe exists

  1. Different teams need different metric/slice/rule combinations.
  2. Core lifecycle must remain fixed and deterministic.
  3. Customization should happen by composition, not core rewrites.

### 10.2 Recipe responsibilities

  1. Bind data inputs and scope.
  2. Select contract profile.
  3. Select metric/slice definitions.
  4. Select finding policy.
  5. Select output/publish behavior.

### 10.3 Recipe constraints

  1. Recipe cannot bypass comparability checks.
  2. Recipe cannot redefine core entity semantics.
  3. Recipe outputs must map to canonical diff/finding structures.

### 10.4 v0 recipe posture

  1. Config-first recipe definition.
  2. Versioned recipe identity.
  3. Validate and compile recipe before execution.
  4. Defer rich template marketplace to v1.

## 11. System Architecture (Layered)

  Three core layers plus recipe composition layer:

  1. Domain Layer

- Owns entity semantics and invariants.

  1. Workflow Layer

- Owns orchestration and stage transitions.

  1. Recipe Layer

- Owns use-case assembly and configuration binding.

  1. Platform/Integration Layer

- Owns persistence and external system interaction.

  Boundary rule:

  1. Domain and workflow define truth and correctness.
  2. Platform defines how/where persisted.
  3. Recipe defines configuration, not semantics.

## 12. MLflow-First v0 Strategy

  You selected MLflow as primary storage for v0.

### 12.1 Why this is reasonable now

  1. Faster implementation with existing ecosystem.
  2. Lower operational burden than introducing separate storage infra.
  3. Familiar workflow for target users.

### 12.2 Risks to acknowledge

  1. Storage coupling can constrain future flexibility.
  2. Complex timeline queries may be less ergonomic.
  3. Schema conventions must be disciplined to avoid entropy.

### 12.3 Mitigation

  1. Keep conceptual model MLflow-agnostic.
  2. Keep stable internal naming conventions.
  3. Keep retrieval and persistence logic behind a clean gateway boundary.
  4. Version metadata conventions from day one.

## 13. v0 Scope Definition

  In scope:

  1. Single timeline per subject.
  2. Single active contract per timeline.
  3. Pinned baseline.
  4. Optional LKG promotion.
  5. Three comparison modes (baseline, previous, LKG).
  6. Anchor-window analysis.
  7. Recipe-lite composition.

  Out of scope:

  1. Branching/epoch models.
  2. Automatic compatibility migration.
  3. Complex profile composition engines.
  4. Full template marketplace.

## 14. v0 Invariants

  1. Every run belongs to exactly one timeline.
  2. Every timeline has exactly one pinned baseline.
  3. Every run has explicit comparability status.
  4. Comparability fail implies non-comparable metric outputs.
  5. Every finding must reference supporting diff evidence.
  6. LKG points to a run in the same timeline.
  7. Baseline is immutable once pinned.
  8. Contract changes are explicit and versioned.
  9. Non-comparable runs remain visible in timeline history.

## 15. Failure Behavior Model

  1. Contract fail

- Mark non-comparable.
- Persist reasons.
- Skip metric deltas.

  1. Missing optional references

- Missing previous or LKG is non-fatal.
- Continue with available comparisons.

  1. Persistence errors

- Run enters failed terminal state with explicit failure reason.

  1. Invalid recipe

- Fail before run computation.
- Persist validation errors.

## 16. Why This Design Is Valuable

  1. It directly targets the highest pain points from your real experience.
  2. It converts abstract workflow pain into explicit objects and states.
  3. It prevents misleading analytics through comparability gating.
  4. It supports iterative releases without losing reference context.
  5. It is constrained enough for v0 delivery and robust enough for real use.

## 17. What Is Still Open (Intentional)

  1. Exact promotion policy language and defaults.
  2. Severity taxonomy and action recommendation baseline set.
  3. Supported data modalities for first release boundary.
  4. Query/report ergonomics over MLflow-backed storage.

  These are valid next-step design topics after this system overview is accepted.

## 18. Recommended Next Planning Steps

  1. Finalize workflow_v0.md as deterministic state machine plus decision table.
  2. Finalize recipe_model_v0.md with config contract and validation rules.
  3. Finalize mlflow_mapping_v0.md with metadata/artifact conventions.
  4. Add acceptance_v0.md with scenario-based pass criteria.
