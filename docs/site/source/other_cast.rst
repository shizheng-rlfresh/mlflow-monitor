Invariant, Identity, and Other CAST
===================================

Invariant
---------

.. automodule:: mlflow_monitor.invariant
   :members:
   :show-inheritance:

Result Contract
---------------

.. automodule:: mlflow_monitor.result_contract
   :members:
   :show-inheritance:

Identity
--------

.. automodule:: mlflow_monitor.identity
   :members:
   :show-inheritance:

Compatibility Evidence Materialization
--------------------------------------

Compatibility Evidence materialization is a pure Analyze building block. It
converts the ordered reasons from one committed Contract Check result into
identified records using the same committed prepared context, and projects those
records into the canonical self-contained JSON payload. It does not persist the
payload or advance the Monitoring Run lifecycle; Analyze integration owns those
responsibilities.

Typical usage keeps the committed prepared context paired with its hydrated Check
result through both operations:

.. code-block:: python

   evidence_records = materialize_compatibility_evidence(
       prepared_context,
       contract_check_result,
   )
   artifact_payload = compatibility_evidence_to_dict(
       prepared_context,
       evidence_records,
   )

Populated Compatibility Evidence records already carry the same lineage fields,
but the prepared context remains an explicit projection input because a PASS
result produces an empty record tuple. The canonical empty payload still requires
the complete Monitoring Run, Source Training Run, Baseline Source Run, and
Contract envelope.

The payload's ``artifact_schema_version="v0"`` and the deterministic evidence-ID
scheme are independently versioned. Compatibility Evidence IDs retain the
``compatibility-evidence-v1-...`` prefix defined by the shared identity helpers;
the artifact version does not change or override that identity contract.

.. automodule:: mlflow_monitor.compatibility
   :members:
   :show-inheritance:

Finding Policy
--------------

.. automodule:: mlflow_monitor.finding_policy
   :members:
   :show-inheritance:


Diff and Coverage Computation
-----------------------------

.. automodule:: mlflow_monitor.differ
   :members:
   :show-inheritance:
