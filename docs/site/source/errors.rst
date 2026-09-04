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

Analyze policy execution uses one ``AnalyzeStageError`` class with three stable
codes:

- ``analyze_finding_policy_evaluation_failed`` for an ordinary exception raised
  by policy evaluation;
- ``analyze_finding_policy_output_invalid`` for a nonconforming draft tuple,
  invalid draft, or unknown/cross-Monitoring-Run evidence reference;
- ``analyze_finding_policy_output_inconsistent`` for conflicting content under
  one deterministic Finding identity.

The message is bounded and ``details`` contains exactly ``finding_policy_id`` and
``finding_policy_version``. Raw policy parameters and arbitrary exception text are
not exposed. Process interruptions such as ``KeyboardInterrupt`` are not converted.
The synchronous Finding-policy interface defines no separate policy-cancellation
signal, so exceptions raised by policy evaluation are bounded as Analyze failures.
Internal Analyze execution propagates these bounded errors without terminalizing
the Monitoring Run. Durable failed-result persistence and replay belong to the
later terminal-failure integration; the public workflow still ends at Check.

``analyze_missing_current_source_run`` identifies a missing current Source Training
Run during Analyze metric collection. Its bounded details contain only
``source_run_id``. This differs from an existing source with an empty metric map,
which is valid, and a missing reference source, which produces unavailable coverage.

Analyze commit validates saved artifacts and their cross-stage bindings before
advancing the lifecycle. Malformed or conflicting output raises
``GatewayConsistencyViolation`` with code
``monitoring_run_json_artifact_inconsistent`` and only the Monitoring Run identity
and artifact path in its details. An incompatible lifecycle update raises
``monitoring_run_upsert_field_override``. These are consistency failures, not
policy failures, and do not create a terminal failed result. Interrupted writes
can leave partial artifacts while the stage remains ``checked``; validated
identical output can be reused on retry.

.. autoclass:: mlflow_monitor.errors.AnalyzeStageError
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
