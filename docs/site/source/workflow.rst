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

.. automodule:: mlflow_monitor.workflow
   :members:
   :show-inheritance:
