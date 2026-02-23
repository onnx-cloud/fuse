Lowering tests — PLAN

Goals

- Verify correctness of IR generation and ONNX emission from the typed AST.
- Ensure emitted ONNX graphs validate (`onnx.checker.check_model`) and follow invariants.

Scope

- Unit tests for `GraphContext`, `FuseLowerer`, and `ImportManager` behaviors.
- Integration tests: full-file lowering from `.fuse` source → ONNX and validation via `onnx.checker`.
- Performance/lightweight smoke tests for common patterns (conv, control flow).

Approach

- Use small source inputs and in-memory lowering; avoid filesystem writes when possible.
- Snapshot important graph-level properties: fn counts, initializer shapes, op types, and deterministic fn names.
- Include negative tests for unsupported lowering constructs and informative errors.

Fixtures & DI

- Inject `NameAllocator`, `ImportManager`, and an `ONNXEmitter` abstraction that can be swapped for an in-memory verifier.
- Provide a `make_graph_context()` test factory that starts from a minimal typed AST.

Files to consult

- `src/lowering/` (lowerer implementations)
- `src/graph_context.py`