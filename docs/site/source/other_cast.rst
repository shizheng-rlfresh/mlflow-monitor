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
Contract envelope. Pairing the records with that same prepared context is a
precondition of this pure projection; Analyze integration validates the
cross-stage binding before persistence.

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

Analyze Output Artifacts
------------------------

The internal ``AnalyzeOutput`` collects Compatibility Evidence, atomic Diffs,
Reference Comparison Coverage, and Findings. It has no independent identity.
Its canonical projection consists of three ``artifact_schema_version="v0"``
artifacts; there is no separate coverage artifact or stage-transition log.

All three envelopes contain ``monitoring_run_id`` and ``source_run_id``.
Compatibility Evidence also carries the Baseline Source Run and Contract lineage
and exactly reproduces the committed Check reasons in their original order.
``diffs.json`` contains ordered ``reference_groups``: each group retains its
reference kind, nullable paired reference, status, reason, nested ``diffs``, and
``metric_unavailability`` rows. Atomic Diff rows inherit current identity from
the envelope and reference identity from their group. Coverage Diff IDs are
reconstructed from those rows rather than stored a second time.

``findings.json`` contains Findings ordered by deterministic identity. Each row
inherits the envelope's current identity and stores the exact policy identity
and version, rule, severity, category, summary, recommendation, and both evidence
ID collections. Hydration validates policy bindings and evidence citations but
does not execute policies or reinterpret saved conclusions.

Validation rejects extra or missing fields, conflicting identities, duplicate
rows, orphaned or unknown evidence, reference-plan disagreement, incompatible
observations of a shared source, and incomplete explicit metric selection.
Every selected metric in a completed group has exactly one Diff or metric-level
unavailability entry. For omitted selection, replay checks agreement among
completed groups; it does not rediscover current metric keys from the live source.

.. automodule:: mlflow_monitor.workflow.analyze_artifacts
   :members:

.. automodule:: mlflow_monitor.workflow.analyze_hydration
   :members:
