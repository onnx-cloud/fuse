# fuse - cognitive compiler

Fuse is a **compiler** that makes building neural networks as simple as writing math equations. 

Design, train, and deploy models in an interactive notebook environment with AI-powered assistance.

- [Quick Start](#-quick-start-60-seconds) 
- [Prerequisites](#-prerequisites-check)
- [CLI Tools](#command-line-tools)
- [Development](#development--scripts)
- [Documentation](https://github.com/onnx-cloud/fuse/tree/main/docs)

---

### Option 1: Jupyter (Recommended)

```bash
# One command to build & run everything
make jupyter

# Open your browser to:
http://localhost:8888
```

**That's it!** You now have a complete ML development environment with ONNX operators, AI chat assistant, and cookbook recipes.

📖 **New to Fuse?** See [docker/jupyter/README.md](docker/jupyter/README.md) for detailed Jupyter setup guide.

### Option 2: Command Line

```bash
# Setup development environment
make setup
source .venv/bin/activate

# Verify installation
fuse verify -f ./examples/golden/

# Compile Fuse to ONNX
fuse onnx -f ./examples/golden/ -o ./tmp/onnx/ -ns=my.ns

# View AST
fuse ast -f ./examples/golden/golden.fuse -o ./tmp/golden.ast.json

# Build standalone CLI binary (linux/macOS host)
make cli      # depends on make install, output in dist/
```

---

## ✅ Prerequisites Check

Run this to verify your environment is ready:

```bash
./scripts/onboarding_checklist.sh
```

This checks:
- ✓ Docker installed & running
- ✓ Python 3.11+
- ✓ Port 8888 available
- ✓ Sufficient disk space (2GB+)
- ✓ Sufficient RAM (8GB+)

---

## Command line tools

Alternatively, Fuse can be used in scripts and command line.

```bash
fuse verify -f ./examples/golden/
fuse onnx -f ./examples/golden/ -o ./tmp/onnx/ -ns=my.ns
fuse ast -f ./examples/golden/golden.fuse -o ./tmp/golden.ast.json
fuse import example.onnx --hf org/project | -f ./examples/golden/
fuse lsp          # run the bundled Fuse language server for editors
fuse dot --dot=./tmp/dot/ 
fuse zoo -f ./examples/golden -ns=my.ns
```

Troubleshooting:

- If `pip` is missing: `python3 -m ensureuv pip--upgrade` or use your OS package manager to install pip.
- Avoid using `sudo uv pip install` — prefer virtualenv or `--user` installs.
- Use the helper script to check your environment: `python tools/check_env.py`.
- If `pytest` reports “command not found”, either activate the project virtualenv (`source .venv/bin/activate`) or run tests  with the venv pytest - m `./scripts/run_tests.sh`.

Makefile shortcuts (convenience):

- `make setup-dev` — create `.venv` and install development & test dependencies (runs `scripts/setup_dev.sh`).
- `make venv` — create just the `.venv` (no package installs).
- `make test` — run the full test suite (`pytest`).
- `make test-lowering` — run only `tests/lowering` for fast, focused feedback during lowering work.- `make cli` — build a standalone command‑line executable using PyInstaller (built binary placed in `dist-exe/`; requires host OS build).
Example:

```bash
# create venv and install dev deps
make setup-dev
source .venv/bin/activate
# run the full test suite
make test
# or just quick lowering tests
make test-lowering
```
```


## Development & Scripts

- See `SPEC.md` 

Quick dev setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m uv pip install --upgrade pip
uv pip install -r requirements.txt
```

Scripts are provided in the `scripts/` directory to simplify common tasks:

- `scripts/setup_dev.sh` — create `.venv` and install development dependencies (existing helper).
- `scripts/check_env.sh` — verify Python version and presence of critical packages (lark-parser, onnx, pytest).
- `scripts/run_examples.sh [--validate]` — lower `examples/golden/` to ONNX (output to `onnx/`). Use `--validate` to run `onnx.checker` on the generated models.
- `scripts/validate_onnx.sh <model.onnx>` — validate a single ONNX model using `onnx.checker`.
- `scripts/run_tests.sh` — run the test suite via `pytest`.
- `scripts/supported_ops.py` — generate deterministic operator catalog `ONNX_OPS.json` using local `onnx` schemas (usage: `python -m scripts.supported_ops --output ONNX_OPS.json`).

### Regenerating golden ONNX models 🔁

- Use `./scripts/regenerate_golds.sh` to regenerate all goldens into the `onnx/` directory and validate them via `onnx.checker`.
- Use `./scripts/check_golden_consistency.sh <generated_dir> <expected_dir>` to compare generated models to committed artifacts. This is the script run by CI to ensure golden artifacts are reproducible.

Example local workflow:

```bash
# regenerate goldens (writes into 'onnx/') and validate
./scripts/regenerate_golds.sh
# run the test suite and inspect changed files
make test
git add onnx/*.onnx && git commit -m "Regenerate golden ONNX models"
```

If CI detects differences on a PR, regenerate locally, inspect changes, and submit a follow-up commit updating the goldens.
- `scripts/build_exe.sh` — build a standalone executable via PyInstaller (build on target OS).  The script now automatically injects the correct project version and build timestamp into `src/__init__.py` before packaging; run with `--patch-only` to apply the patch without doing a full build (useful for testing).
- `scripts/build_wheel.sh` — build a wheel into `dist/` (requires the `build` package; script installs it if missing).
- `scripts/format.sh` / `scripts/lint.sh` — run formatters/linters if installed (black, ruff).
- `scripts/help.sh` — list available scripts and basic usage.

Examples:

```bash
# quick env check
./scripts/check_env.sh

# run examples and validate produced ONNX models
./scripts/run_examples.sh --validate

# run tests
./scripts/run_tests.sh

# build a standalone executable (Linux/macOS binary on Linux/macOS)
./scripts/build_exe.sh

# build wheel
./scripts/build_wheel.sh
```

These scripts are intentionally lightweight and safe for CI usage; they return non-zero status on failure so they can be used in automation.


## Invariants (hard rules)

* Every emitted ONNX graph validates
* Same source → identical bytes
* No hidden rewrites
* No backend-specific lowering

