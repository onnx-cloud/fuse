# TODO — prioritized short-term items (developer-focused)

## Output-type inference improvements (high priority) 🔧
- Implement schema-driven type-constraint inference for operators:
  - Parse operator schema `type_constraints` to map type variables (e.g., `T`) to concrete input types when available.
  - Use inferred bindings to compute output types deterministically for more ops beyond the current heuristic categories.
  - Add tests that exercise type-constraint cases (e.g., Binary ops with `T`, Reduce ops, MatMul-like ops where output dtype follows inputs).
  - Status: In progress — extracted schema-driven inference into `src/lowering/schema_inference.py`; added unit tests for helper and lowered examples (`Concat`, `Add`, `MatMul`, `ReduceSum`).

- Expand operator categories and add dedicated tests:
  - Logical/comparison (already handled) + more: `And/Or/Xor` -> bool output tests.
  - Reductions -> single-element/shape-reduced output types tests.
  - Sequence-producing ops (e.g., `Sequence`/`Scan`) and shape-sensitive ops (e.g., `Reshape`) to ensure safe fallbacks.

## Lowering robustness & regressions (medium priority) ⚠️
- Add regression tests that cover these real-world edge cases found during work:
  - Inline-lambda deduplication (ensure identical lambdas generate the same helper node name and behave deterministically).
  - Loop subgraph initializer handling (ensure initializers do not become extra positional inputs when `_preserve_local_input_names` is set).
  - Output ordering preservation for nested GraphProto (ensure helper lowering doesn't reorder outputs and mismatch expected positional outputs).
- Deduplicate generated helper functions across modules (global dedup) to minimize emitted subgraphs for large codebases.
- Improve integration with onnxruntime: run a small ORT smoke test during CI for selected golden examples to catch type-inference/load errors sooner.
- do not allow lowering when output type cannot be inferred - rather we fail when encountering dynamic/ambiguous types. 

- small refactors:
  - Centralize GraphProto post-processing (qualification, identity-wrapping) for reuse and clearer invariants.
  - Encapsulate schema inspection logic (wrap require_op_schema + constraint parsing) into a helper module for easier unit testing.

## Tests & CI (short tasks) ✅
- Add unit tests for newly added `_infer_output_type` behavior and type-constraint cases.
- Extend end-to-end golden tests covering lowered models that use the newly inferred types to catch runtime mismatches early (ONNX Runtime / ReferenceEvaluator).

