Debug tests & harness — PLAN

Goals

- Provide reproducible debugging snapshots and developer tools that help reproduce failures quickly.

Scope

- Tests for helper debug tools under `src/` (e.g., AST/IR pretty printers, manifest debugging utilities).
- Ensure debug output is stable and includes necessary metadata (seed, namer state, import manifest).

Approach

- Create a `tests/debug/fixtures/` directory with failing-case artifacts and a reproducer script.
- Add unit tests that assert debug output contains key fields and stable formatting.

Fixtures & DI

- Expose a `DebugDumper` interface that accepts injection of the current `NameAllocator` and `ImportManager` state for deterministic dumps.