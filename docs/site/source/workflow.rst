Workflow and Stages
===================

Workflow
--------

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

.. automodule:: mlflow_monitor.workflow
   :members:
   :show-inheritance:
