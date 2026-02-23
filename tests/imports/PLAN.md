Imports tests — PLAN

Goals

- Validate import resolution (ONNX import, HF imports, remote imports) and metadata mapping into AST/IR.

Scope

- Unit tests for `ImportManager` behavior: locating, caching, and resolving imported symbols.
- Tests for remote import fallbacks and error messages.

Approach

- Provide `InMemoryImportManager` to simulate local and remote artifacts.
- For ONNX imports, validate that imported graphs produce expected AST fragments or typed IR shapes.
- Use recorded responses for remote APIs (VCR-style fixtures) or explicit mocks.

Fixtures & DI

- Inject a pluggable storage layer into `ImportManager` so tests provide local files, zip entries, or mocked HTTP responses.

Files to consult

- `src/import_fusion.py`, `src/remote_imports.py`