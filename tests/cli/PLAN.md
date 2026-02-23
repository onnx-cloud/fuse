CLI tests — PLAN

Goals

- Exercise end-to-end flows (`fuse onnx`, `fuse ast`, `fuse import`, `fuse zoo`) with deterministic outputs.
- Ensure CLI error handling and exit codes are stable and testable.

Scope

- Functional tests that run CLI entrypoints with injected I/O streams (capture stdout/stderr) and mocked filesystem where needed.
- Tests for flags: `--no-ns`, `--wasm`, `-f` input globs, and `-o` output paths.

Approach

- Refactor CLI entrypoints to expose a programmatic `main(argv, in_stream, out_stream, err_stream, fs)` so tests can call without spawning subprocesses.
- Use `pyfakefs` or an in-memory filesystem adapter to avoid touching real disk.
- Mock network/download steps (ORT Web) and assert the CLI calls the expected hooks.

Fixtures & DI

- Inject a `Runner` object that accepts a `BackendInstaller` and `ONNXEmitter` so tests can provide no-op or in-memory implementations.

Files to consult

- `src/__main__.py` and `src/fuse.py` (CLI plumbing)