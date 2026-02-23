AST tests — PLAN

Goals

- Verify AST transformations, name resolution, and early constant folding.
- Ensure SSA/stable naming invariants and determinism in transformations.

Scope

- Transformer unit tests: node-level transforms, const folding, param → initializer conversion.
- Resolved AST tests: scoping, imports, symbol resolution, and simple type/shape annotation checks.
- Negative tests: ambiguous names, duplicate symbols, invalid import references.

Approach

- Use builder factories to create AST fragments (`make_function`, `make_const`) so tests remain concise.
- Validate that transforms are pure functions: AST in → AST out with no hidden global state changes.
- Snapshot small AST JSON for stable cases and assert structural equality (not textual formatting).

Fixtures & DI

- Inject `NameAllocator` / `StableNamer` into transformers so tests can use a deterministic namer.
- Provide an `InMemoryImportManager` to resolve imports without file I/O.

Files to consult

- `src/parser.py` (AST emitter)
- `src/lowering.py` (interfaces between AST and IR)