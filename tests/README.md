# Tests & Fixtures

Useful fixtures available in `tests/conftest.py`:

- `stable_name_allocator()` — returns a fresh `StableNameAllocator(scope_prefix="test")` for deterministic naming in lowering tests.
- `inmemory_emitter()` — returns an `InMemoryONNXEmitter` for emitting models to memory without touching disk.
- `in_memory_imports()` — in-memory import manager for test imports.
- `graph_context_factory()` — factory for constructing `GraphContext` instances with injected kwargs.

When writing tests, prefer these fixtures to get consistent behavior and faster runs.
