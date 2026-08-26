MLflow-Monitor Errors
=====================

MLflow-Monitor exceptions separate human-readable diagnostics from structured
error data. Treat ``message`` as text for operators and users. When an exception
provides ``code`` and ``details``, use those fields for programmatic handling
instead of parsing the message.

Gateway consistency failures own their codes, messages, and detail formatting.
Create them through the named constructors below and pass only the semantic
fields that describe the failure. Catch ``GatewayConsistencyViolation`` when one
handler should cover every gateway consistency failure.

For example:

.. code-block:: python

   from mlflow_monitor.errors import AllocationConsistencyViolation

   raise AllocationConsistencyViolation.duplicate_sequence(
       sequence_index=2,
       first_monitoring_run_id="monitoring-run-1",
       second_monitoring_run_id="monitoring-run-2",
   )


Gateway Namespace Violation
---------------------------

.. autoclass:: mlflow_monitor.errors.GatewayNamespaceViolation
   :members:
   :show-inheritance:

Training Run Mutation Violation Code
------------------------------------

.. autoclass:: mlflow_monitor.errors.TrainingRunMutationViolation
   :members:
   :show-inheritance:

Gateway Consistency Violation
------------------------------


Generic Gateway Consistency Violation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: mlflow_monitor.errors.GatewayConsistencyViolation
   :members:
   :show-inheritance:

Allocation Consistency Violation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: mlflow_monitor.errors.AllocationConsistencyViolation
   :members:
   :show-inheritance:

Prepared-Context Consistency Violation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: mlflow_monitor.errors.PreparedContextConsistencyViolation
   :members:
   :show-inheritance:

Timeline Consistency Violation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: mlflow_monitor.errors.TimelineConsistencyViolation
   :members:
   :show-inheritance:


Workflow Errors
---------------

.. autoclass:: mlflow_monitor.errors.PrepareStageError
   :members:
   :show-inheritance:

.. autoclass:: mlflow_monitor.errors.CheckStageError
   :members:
   :show-inheritance:

.. autoclass:: mlflow_monitor.errors.TerminalRunRetryError
   :members:
   :show-inheritance:

Recipe Errors
-------------

.. autoclass:: mlflow_monitor.errors.ContractResolutionError
   :members:
   :show-inheritance:

.. autoclass:: mlflow_monitor.errors.RecipeValidationIssue
   :members:

.. autoclass:: mlflow_monitor.errors.RecipeValidationError
   :members:
   :show-inheritance:

Invariant Errors
----------------

.. autoclass:: mlflow_monitor.errors.InvariantViolation
   :members:
   :show-inheritance:
