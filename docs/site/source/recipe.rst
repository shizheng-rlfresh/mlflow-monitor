Recipe and Compiler
===================

Recipes are strict JSON-compatible, versioned descriptions of monitoring
behavior. Recipe parsing and compilation are side-effect-free preflight work:
they do not read MLflow, allocate a Monitoring Run, resolve execution identities,
or persist state.

Authoring Contract
------------------

A v0 Recipe requires ``recipe_schema_version``, ``identity``, and an exact
``contract`` binding. ``source_requirements`` and ``analysis`` are optional:

.. code-block:: json

   {
     "recipe_schema_version": "v0",
     "identity": {
       "recipe_id": "churn-monitoring",
       "recipe_version": "3"
     },
     "source_requirements": {
       "source_experiment": "training/churn",
       "required_metric_names": ["accuracy"],
       "required_artifact_paths": ["model/MLmodel"]
     },
     "contract": {
       "contract_id": "default_permissive",
       "contract_version": "v0"
     },
     "analysis": {
       "metric_names": ["accuracy", "latency_p95"],
       "finding_policy_bindings": []
     }
   }

``analysis.metric_names`` has three states: omission selects all current scalar
metrics, an empty list selects none, and a nonempty list selects exactly those
case-sensitive names. ``analysis.finding_policy_bindings`` follows the same
three-state pattern: omission selects the schema-defined default, an empty list
selects no policy, and a nonempty list selects exactly those bindings.

Unknown fields, explicit nulls where omission has meaning, invalid types,
duplicate or empty names, duplicate policy identities, and non-finite parameter
values are rejected with located validation issues. Recipes cannot contain Source
Training Run or Monitoring Run identities, selectors, slices, output bindings,
promotion, scheduling, deployment, import paths, or executable code.

Parsing and Compilation
-----------------------

``parse_recipe()`` accepts a Python mapping. ``load_recipe_json()`` explicitly
decodes a UTF-8 JSON file and feeds the result through the same parser. Both
produce an immutable typed ``Recipe`` while preserving omitted, empty, and
nonempty analysis selections.

``compile_recipe()`` accepts a mapping, a typed ``Recipe``, or no Recipe for the
zero-configuration default. Typed values are revalidated through the strict
parser. Compilation expands defaults, canonicalizes ordering, resolves the exact
Contract and Finding-policy bindings, validates policy parameters, and returns an
immutable ``CompiledRecipe`` with a JSON-serializable effective plan.

The current ``ComponentRegistry`` intentionally contains only the packaged system
Contract and ``system-compatibility-findings@v0`` policy. It accepts no user
components; custom Contracts are outside the v0 boundary.


Recipe Compiler
---------------

.. automodule:: mlflow_monitor.recipe_compiler
   :members:
   :show-inheritance:

Recipe
------

.. automodule:: mlflow_monitor.recipe
   :members:
   :show-inheritance:
