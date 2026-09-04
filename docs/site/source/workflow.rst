Workflow and Stages
===================

Prepare Stage
-------------

Prepare freezes a fixed reference plan in this order: baseline, previous, LKG,
and optional custom. The baseline is always resolved. Previous selects the
closed Monitoring Run with the greatest lower sequence index; checked, failed,
and nonterminal Monitoring Runs are ineligible regardless of comparability.
When no previous Monitoring Run is eligible, its plan group records
``previous_reference_missing``.

LKG selection history is not yet integrated into Prepare. The legacy mutable
LKG pointer is non-authoritative and is not consulted, so current Prepare paths
record ``lkg_not_selected``. Prepared-context hydration also validates the
nonfatal ``lkg_selection_inconsistent`` shape for later selection integration.

``custom_reference_monitoring_run_id`` remains invocation-owned. When supplied,
it must identify a closed Monitoring Run on the same Timeline or Prepare fails.
Committed prepared context is replayed as written without resolving Timeline or
reference state again.

.. automodule:: mlflow_monitor.workflow.prepare
   :members:
   :show-inheritance:

Prepared Context
----------------

.. automodule:: mlflow_monitor.workflow.prepared_context
   :members:
   :show-inheritance:

Check Stage
-----------

After Check succeeds, the complete result is committed to
``outputs/contract_check.json`` before the Monitoring Run advances to
``checked``. The artifact records the Monitoring Run and Source Training Run
identities, Contract identity and version, comparability status, and the
ordered reason codes, messages, and blocking flags. It is the authoritative
Check output; the MLflow comparability tag is only a projection of its status.

If execution resumes from ``prepared`` and a partial Check artifact already
exists, the artifact is validated before Check runs again. Identical output is
reused, while malformed or conflicting output fails closed without changing
the Monitoring Run. Fresh checker output continues to use normal owned Check
failure handling.

Replaying a ``checked`` Monitoring Run hydrates both committed prepared context
and the authoritative Check artifact. It does not resolve current Timeline or
reference state, reload Source Training Run evidence, or invoke the checker.
An omitted baseline or the baseline identity stored in prepared context is
accepted without resolving the Source Training Run again; a different identity
is rejected. Missing or malformed artifacts, duplicate persisted reason codes,
identity conflicts, and comparability-projection disagreement raise a Gateway
consistency violation without mutating committed state. Consequently, legacy
``checked`` records that predate the authoritative artifact cannot be replayed.

Internal Analyze Execution
--------------------------

``mlflow_monitor.workflow.analyze.execute_analyze()`` composes the Analyze
building blocks using committed prepared context and Check output. The supplied
compiled Recipe must match the persisted effective plan and Contract. This
internal boundary does not change the public ``monitor.run()`` checked-result
workflow, persist artifacts, or terminalize a Monitoring Run.

PASS and WARN resolve omitted metric selection from sorted current metric keys;
explicit selection uses exactly its normalized names, and empty selection
compares none. Analyze reads each distinct ``source_run_id`` once and reuses the
detached observation when references share a source, including self-comparisons.
An existing source with empty metrics is valid. A missing current source raises
an owned Analyze error; a missing reference source produces unavailable coverage
with its frozen reference retained. Empty selections still distinguish missing
sources from sources with no metric values.

FAIL performs no metric reads. Resolved reference groups are skipped with
``current_not_comparable`` while absent groups retain their unavailability reason.
All branches materialize Compatibility Evidence from Check reasons and execute
the compiled Finding policies. A comparability FAIL is not an execution failure.

Analyze Finding Policy Execution
--------------------------------

``execute_finding_policies()`` is the pure, backend-independent policy execution
boundary used by Analyze. It accepts the current Monitoring Run and Source
Training Run identities, compiled Finding-policy bindings, and the Diffs,
Compatibility Evidence, and Reference Comparison Coverage already produced by
the current Analyze execution.

Bindings execute in canonical policy-identity/version order. Every policy receives
its already-validated frozen parameters and the same immutable evidence tuples;
policies do not receive drafts or Findings produced by another policy. A policy
returns only transient ``FindingDraft`` values. MLflow-Monitor validates that every
cited evidence identity exists in the supplied current output, attaches the exact
Monitoring Run, Source Training Run, and policy identity, and returns materialized
Findings ordered by deterministic ``finding_id``. Empty binding lists and empty
draft tuples both produce an empty Finding tuple.

A policy exception, invalid draft or evidence reference, or conflicting content
under one deterministic Finding identity raises ``AnalyzeStageError`` and publishes
no partial result. This helper does not persist artifacts, advance lifecycle, or
commit a terminal failure. Artifact persistence, replay, and lifecycle integration
are delivered by V0-021; end-to-end custom-policy guidance is deferred to V0-033.

Built-in Compatibility Finding Policy
-------------------------------------

``system-compatibility-findings@v0`` interprets Compatibility Evidence only. It
produces one ``HIGH`` Finding draft in category ``compatibility`` per evidence
record. The draft summary is the committed reason message, it cites exactly that
record's Compatibility Evidence identity, and it cites no Metric Diff identities.

.. list-table:: Built-in compatibility mappings
   :header-rows: 1
   :widths: 20 30 50

   * - Reason code
     - Finding rule ID
     - Recommendation
   * - ``environment_mismatch``
     - ``compatibility.environment_mismatch``
     - Review the execution-environment differences and confirm that the current
       evidence is comparable with the baseline before relying on metric comparisons.
   * - ``schema_mismatch``
     - ``compatibility.schema_mismatch``
     - Review the schema changes and either restore baseline-compatible data or
       intentionally update the Contract for a future Monitoring Run.
   * - ``feature_mismatch``
     - ``compatibility.feature_mismatch``
     - Review the feature-set changes and either restore baseline-compatible features
       or intentionally update the Contract for a future Monitoring Run.
   * - ``data_scope_mismatch``
     - ``compatibility.data_scope_mismatch``
     - Confirm the intended data population and either restore the baseline-compatible
       scope or intentionally update the Contract for a future Monitoring Run.

The reason's ``blocking`` flag controls whether metric analysis proceeds; it does
not determine Finding severity. Consequently, a nonblocking environment mismatch
still produces a ``HIGH`` Finding. Metric Diffs and Reference Comparison Coverage
do not affect this built-in policy's output. An unsupported reason code fails
closed rather than producing a partially interpreted result.

.. automodule:: mlflow_monitor.workflow
   :members:
   :show-inheritance:
