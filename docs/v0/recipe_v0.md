# MLflow-Monitor v0 Recipe Model

  Date: 2026-03-09
  Status: Draft (Planning)

## 1. Purpose

  This document defines the Recipe model for v0:

  1. What a recipe is and is not.
  2. Why recipe is a separate layer.
  3. What recipe can configure in v0.
  4. How recipe is validated and compiled.
  5. How recipe binds to run execution.
  6. Recipe invariants, failure behavior, and versioning.

  This is the canonical v0 customization surface.

## 2. Why Recipe Exists

### 2.1 Problem Context

  The core pain from the proposal is workflow fragmentation:

  1. Teams hand-stitch data loading, checks, diffs, and reporting.
  2. Monitoring logic becomes script-specific and hard to reuse.
  3. Comparability guarantees are often inconsistent across teams/runs.

### 2.2 Role of Recipe

  Recipe is the compositional unit that captures use-case intent without altering core workflow semantics.

  Recipe answers:

  1. Which data scope do we evaluate?
  2. Which contract profile do we enforce?
  3. Which metrics/slices do we compute?
  4. Which finding policies do we apply?
  5. Which outputs/sinks do we emit?

### 2.3 Default Recipe

  MLflow-Monitor ships with a system default recipe. This means:

  1. Every run always references exactly one recipe version — the invariant holds unconditionally.
  2. Users do not need to write any recipe to get basic monitoring value.
  3. User recipes are optional overrides, not prerequisites.

  Resolution rule:

  1. If user specifies a recipe → use that recipe.
  2. If user specifies nothing → use the system default recipe.

  The default recipe provides sensible baseline analysis using whatever MLflow already has logged — no required metrics, no required artifacts, no contract gates.

## 3. Input Layer Model

  Inputs to MLflow-Monitor operate in three layers:

### Layer 1: System Defaults (always active)

  MLflow-Monitor reads standard MLflow logged content automatically — whatever metrics, params, and tags exist on the resolved training run. No configuration required.

### Layer 2: Recipe (optional, user-defined enrichment)

  Recipe author specifies additional required metrics, artifacts, or evaluation scope. If a required item is absent from the source training run, the monitoring run fails at prepare stage with an explicit error. No partial analysis.

### Layer 3: Contract (optional, user-defined comparability rules)

  User defines stricter comparability gates. If a contract requires things that don't exist in the source run, it fails explicitly. Writing a contract that requires unavailable inputs is the user's responsibility — the system reports it clearly.

  Guiding principle:
  > Zero config gives you something useful. More config gives you more precision. Bad config fails loudly.

## 3. What Recipe Is (and Is Not)

### 3.1 Recipe Is

  1. A versioned, declarative monitoring assembly specification.
  2. A compile-time input to produce an executable run plan.
  3. A reproducible record of monitoring intent at run time.

### 3.2 Recipe Is Not

  1. Not a replacement for CAST domain semantics.
  2. Not an alternate run lifecycle engine.
  3. Not a free-form script runtime that can bypass comparability gates.
  4. Not a branching timeline strategy mechanism in v0.

## 4. Recipe Placement in System Architecture

  Layer ordering:

  1. Domain layer (semantic truth)
  2. Workflow layer (deterministic lifecycle)
  3. Recipe layer (customization/config composition)
  4. Platform layer (persistence/integration)

  Boundary rule:

  1. Recipe configures what to run.
  2. Workflow decides how it runs.
  3. Domain defines what outputs mean.
  4. Platform defines where outputs are persisted.

## 5. Recipe Responsibilities (v0)

### 5.1 Input Binding

  1. Source experiment name (which MLflow training experiment to read from).
  2. Run selector:

      - current implemented behavior supports a raw source run ID for user-authored recipes
      - the built-in system default recipe uses a reserved runtime token that is resolved later by workflow/gateway logic
      - generic selector modes such as `latest` are not currently supported in v0 code
  3. Required metrics (optional — fails prepare if absent from source run).
  4. Required artifacts (optional — fails prepare if absent from source run).
  5. Optional evaluation window / slice selectors.
  6. Optional custom reference run ID:

      - Any run on the same timeline can be specified as an additional comparison reference.
      - Defaults (baseline, LKG, previous) still always execute when available.
      - Cross-timeline custom references are not supported in v0.

  Reasoning:

  1. Different teams monitor different entities, windows, and artifact types.
  2. Input variance should be explicit and versioned.
  3. Custom reference enables flexible retrospective comparisons (e.g. "compare against the model from 3 months ago") without requiring counterfactual infrastructure.

### 5.2 Contract Binding

  1. Contract profile reference.
  2. Contract mode options allowed in v0 (strictness policy envelope).

  Reasoning:

  1. Comparability rules are use-case dependent.
  2. Contract profile should be resolved before run check stage.

### 5.3 Metric and Slice Binding

  1. Metric set selection.
  2. Slice dimensions and scope.

  Reasoning:

  1. Actionability depends on meaningful segmentation.
  2. Metric/slice definitions should be stable per recipe version.

### 5.4 Finding Policy Binding

  1. Severity mapping profile.
  2. Category mapping profile.
  3. Recommendation rule profile.

  Reasoning:

  1. Diff alone is factual, not actionable.
  2. Finding interpretation needs explicit policy context.

### 5.5 Output Binding

  1. Summary/report options.
  2. Sink behavior options (MLflow-first in v0).
  3. Optional artifact detail levels.

  Reasoning:

  1. Different consumers need different output granularity.
  2. Storage output consistency remains required by workflow contracts.

## 6. Recipe Shape (Conceptual Sections)

  A recipe should conceptually contain:

  1. `identity`

- recipe id
- version
- owner/team metadata

  1. `subject`

- subject type/id strategy
- timeline targeting rules

  1. `inputs`

- data sources/references
- evaluation scope

  1. `contract_binding`

- contract profile ref
- compatibility policy options

  1. `metrics_and_slices`

- metric set
- slice set

  1. `finding_policy`

- severity mapping
- issue categorization
- recommendation mapping

  1. `outputs`

- summary options
- persistence/sink options

  1. `execution_defaults`

- retry policy defaults
- timeout defaults
- non-comparable handling options allowed by v0 policy

  1. `metadata`

- changelog note
- creation/update metadata

## 7. Recipe Lifecycle (Authoring to Execution)

### 7.1 Author

  1. User/team defines recipe in declarative form.
  2. Recipe assigned stable id + version.

### 7.2 Validate

  Validation categories:

  1. Structural validation (required sections present).
  2. Referential validation (contract/metric policy refs resolvable).
  3. Constraint validation (v0 invariants not violated).
  4. Compatibility validation (recipe options consistent with workflow stage model).

### 7.3 Compile

  1. Normalize recipe into executable run plan.
  2. Resolve defaults.
  3. Freeze effective plan snapshot for run.

### 7.4 Bind to Run

  1. Run references exact recipe id+version.
  2. Effective compiled plan is attached to run context.

### 7.5 Execute

  1. Workflow consumes compiled plan deterministically.
  2. Recipe cannot mutate stage order or skip required checks.

### 7.6 Audit

  1. Run output can always be traced to recipe version.
  2. Recipe changes are trackable over time.

## 8. Recipe Invariants (v0)

  1. Every run must reference exactly one resolved recipe version.
  Reason:

- Guarantees reproducibility and run intent traceability.

  1. Recipe cannot bypass contract check stage.
  Reason:

- Comparability-first is a safety invariant.

  1. Recipe cannot redefine CAST entity semantics.
  Reason:

- Domain model consistency must be global.

  1. Recipe cannot alter required workflow stage ordering.
  Reason:

- Lifecycle determinism is central to trust.

  1. Recipe outputs must map to canonical diff/finding contracts.
  Reason:

- Cross-recipe analysis must remain consistent.

  1. Recipe cannot introduce branching timeline behavior in v0.
  Reason:

- Branching is out-of-scope and would break v0 simplification.

## 9. Recipe Versioning Policy (v0)

### 9.1 Version Identity

  1. Recipe has semantic version or monotonic version id.
  2. Run stores exact recipe version used.

### 9.2 Change Classes

  1. Patch change:

- Non-semantic metadata updates.
- No behavior change expected.

  1. Minor change:

- Additive options/slices/metrics without breaking existing assumptions.

  1. Major change:

- Behavioral or contract-binding changes that affect comparability expectations.

### 9.3 Execution Rule

  1. Existing runs never rebind to newer recipe versions.
  2. New runs use explicitly selected or default-latest version per policy.

## 10. Validation Error Model

### 10.1 Structural Errors

  Examples:

  1. Missing required section.
  2. Invalid field type.

  Behavior:

  1. Validation fail before run creation/prepare.

### 10.2 Referential Errors

  Examples:

  1. Unknown contract profile.
  2. Unknown metric/finding policy reference.

  Behavior:

  1. Validation fail; no compiled plan.

### 10.3 Constraint Errors

  Examples:

  1. Recipe attempts disallowed v0 behavior.
  2. Recipe options conflict with workflow invariants.

  Behavior:

  1. Validation fail with explicit reason set.

### 10.4 Compile Errors

  Examples:

  1. Ambiguous defaults.
  2. Inconsistent derived execution options.

  Behavior:

  1. Compile fail; run cannot proceed.

## 11. Recipe and Workflow Interaction Contract

### 11.1 What Workflow Expects from Recipe

  1. Resolved input spec.
  2. Resolved contract binding.
  3. Resolved metrics/slices spec.
  4. Resolved finding policy.
  5. Resolved output options.

### 11.2 What Recipe Can Expect from Workflow

  1. Deterministic stage execution.
  2. Stable lifecycle states.
  3. Standardized output schema semantics.
  4. Explicit failure/status reporting.

## 12. Recipe and CAST Interaction

### 12.1 Recipe influences

  1. How Run is configured.
  2. Which Contract is applied.
  3. Which Diffs are computed (content and granularity, not semantic type).
  4. Which Findings are generated (policy-driven interpretation).

### 12.2 Recipe does not influence

  1. Baseline immutability.
  2. Timeline ownership semantics.
  3. LKG pointer semantics.
  4. Core lifecycle stage guarantees.

## 13. v0 Recipe Scope and Deferred Features

### 13.1 In Scope

  1. Declarative recipe spec.
  2. Versioned recipe identity.
  3. Validation + compilation pipeline.
  4. Execution binding and run traceability.

### 13.2 Deferred to v1

  1. Recipe inheritance/composition graphs.
  2. GUI recipe builder.
  3. Template marketplace/distribution model.
  4. Advanced dynamic policy plugins.

## 14. Security and Safety Considerations (v0)

  1. Recipe should be treated as configuration, not arbitrary code execution.
  2. External reference resolution should be allowlisted/validated.
  3. Invalid recipe behavior should fail fast pre-execution.
  4. Auditability of recipe change history should be preserved.

## 15. Observability Requirements for Recipe

  For every run:

  1. recipe id
  2. recipe version
  3. recipe hash/signature (optional v0, recommended)
  4. compile status
  5. validation warnings/errors (if any)

  Reasoning:

  1. Troubleshooting without recipe traceability is incomplete.

## 16. Acceptance Scenarios

  Scenario 1: Valid recipe, normal run

  1. Recipe validates and compiles.
  2. Run executes full workflow.
  3. Outputs include recipe version trace.

  Scenario 2: Recipe references missing contract profile

  1. Validation fails.
  2. Run does not enter workflow execution.

  Scenario 3: Recipe attempts disallowed v0 behavior (branch option)

  1. Constraint validation fails.
  2. Explicit error explains v0 scope boundary.

  Scenario 4: Recipe version update

  1. New runs can use new version by policy.
  2. Old runs remain bound to old version.
  3. Historical reproducibility preserved.

## 17. Open Questions

  1. Exact recipe schema format and file conventions.
  2. Whether recipe policy references are inline or registry-based in v0.
  3. Minimal default finding policy package for first release.
  4. How strict v0 should be on warning-level recipe validation.

## 18. Summary

  Recipe is the v0 customization surface that:

  1. Encodes use-case variability.
  2. Preserves core workflow correctness.
  3. Improves reproducibility through versioned configuration.
  4. Enables future expansion (templates/marketplace) without destabilizing core design.
