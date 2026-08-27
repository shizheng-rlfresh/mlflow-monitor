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
