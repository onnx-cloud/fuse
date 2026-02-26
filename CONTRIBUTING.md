# Contributing to Fuse

## Writing deterministic tests ✅

Determinism is important to keep tests reliable and fast. When contributing tests:

- Prefer in-memory APIs and fixtures to avoid filesystem or environment dependence. Use `inmemory_emitter` and `in_memory_imports` fixtures provided in `tests/conftest.py`.
- Use `stable_name_allocator` fixture for deterministic node names in lowering tests. This prevents flakiness caused by global counters or varying dict iteration orders.
- Seed any random sources and document the seed in tests.
- Avoid relying on external network or platform-specific resources; mock or use in-memory alternatives.

If you add a test that depends on specific node names or serialized bytes, add a snapshot assertion (e.g., in `tests/name_allocator/`) and explain in the test comment why the snapshot should remain stable.

Pre-commit hooks (optional but recommended):

- A `.pre-commit-config.yaml` is provided which includes local hooks such as `check-lint-json-schema` (lint JSON) and `check-opcodes-checksum` (OpCodes checksum). Install pre-commit and run `pre-commit install` to enable the hooks locally.
- You can run the checks manually with `bash scripts/check_lint_json_schema.sh` and `bash scripts/check_opcodes_checksum.sh`.

Sanitizer configuration

- The sanitizer loads `schemas/training_optimizers.json` to provide config-driven training validation rules (optimizer state expectations). To override sanitizer behaviour, set an alternate sanitizer config via environment variable `FUSE_SANITIZER_CONFIG` pointing to a TOML file containing e.g.:


Updating golden ONNX models

- If you make changes that affect code generation, run `./scripts/regenerate_golds.sh` and validate output (the script runs `onnx.checker`).
- After regeneration, run the full test suite (`make test`), review changed `.onnx` files, and commit them if they reflect intended changes.
- CI runs a verification workflow that regenerates goldens and fails the PR if generated artifacts differ from the committed `onnx/` artifacts.

```toml
[tool.fuse.sanitizer]
enable_training_state_checks = false
```

This allows team-level toggles and opt-outs for conservative pre-lowering checks.