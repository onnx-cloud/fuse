golden-onnx:
	@echo "Exporting all examples/golden/*.fuse to tmp/onnx/*.onnx ..."
	$(PY) scripts/golden_onnx_export.py
	@echo "✅ Golden ONNX export complete."
# Makefile for common tasks
UV ?= python -m uv
PY ?= python
# Prefer using the project venv's Python/uv when available so targets like
# `make gold` run against the created environment.
ifneq ("$(wildcard .venv/bin/python)","")
PY := .venv/bin/python
UV := $(PY) -m uv
endif
PIP ?= $(UV) pip install
FUSE_CMD ?= ./.venv/bin/fuse
EXAMPLES_DIR = examples
JUPYTER_DIR = jupyter
JUPYTER_COOKBOOK_DIR = $(JUPYTER_DIR)/cookbook

.PHONY: help install test lint lint-all \
	lint-examples lint-examples-all lint-examples-% \
	notebook-run build-md clean export-cookbook snippets build onnx-ops gold gold \
	# Additional targets
	dev smoke-test test-parsing test-golden test-jupyter test-decompile test-server test-all package jupyter-docker examples
	@echo "╚═══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "   make setup          Setup development environment"
	@echo "   make jupyter        Build & run Jupyter (one command!)"
	@echo "   make test           Run all tests"
	@echo ""
	@echo "📦 Setup & Installation:"
	@echo "   make venv           Create virtualenv (.venv)"
	@echo "   make setup          Install dev dependencies"
	@echo "   make dev            Complete dev setup (setup + install)"
	@echo "   make install        Build wheel + install package"
	@echo ""
	@echo "🧪 Testing:"
	@echo "   make test           Full test suite"
	@echo "   make test-lowering  Lowering tests (fast)"
	@echo "   make test-parsing   Parser tests"
	@echo "   make test-jupyter   Jupyter tests"
	@echo "   make test-golden    Golden examples"
	@echo "   make test-all       All test suites combined"
	@echo "   make smoke-test     Quick smoke tests"
	@echo ""
	@echo "🏗️  Building:"
	@echo "   make build          Build wheel package"
	@echo "   make gold           Clean build + full tests"
	@echo "   make package        Clean + build + test"
	@echo ""
	@echo "📓 Jupyter:"
	@echo "   make jupyter        Build image + start container"
	@echo "   make jupyter-stop   Stop Jupyter container"
	@echo "   make jupyter-shell  Shell into container"
	@echo "   make jupyter-image  Build Docker image only"
	@echo "   make jupyter-clean  Remove Docker image"
	@echo ""
	@echo "   Quick access: http://localhost:8888"
	@echo "   See: docker/jupyter/README.md for details"
	@echo ""
	@echo "🔍 Linting:"
	@echo "   make lint           Lint Python code"
	@echo "   make lint-examples  Lint Fuse examples"
	@echo "   make lint-all       Lint code + examples"
	@echo ""
	@echo "📚 Documentation:"
	@echo "   make export-cookbook Export cookbook notebooks"
	@echo "   make build-md       Build markdown docs"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "   make clean          Remove build artifacts"
	@echo ""
	@echo "💡 Pro Tips:"
	@echo "   • Run 'make jupyter' for one-command Jupyter setup"
	@echo "   • Use 'make test-lowering' for fast iteration"
	@echo "   • See README.md for CLI usage examples"
	@echo "   • Check docker/jupyter/README.md for Jupyter help"
	@echo ""

venv:
	@echo "Creating virtualenv .venv"
	python3 -m venv .venv
	@echo "Activate with: source .venv/bin/activate"

setup:
	@echo "Installing developer dependencies into .venv"
	$(UV) pip install -r requirements.txt
	$(UV) pip install -e ".[dev]"

# Default install: core requirements and wheel
install: build test
	@echo "Installing core requirements and package wheel using uv"
	$(UV) pip install --upgrade pip setuptools wheel || true
	$(UV) pip install -r requirements.txt --no-deps
	$(UV) pip install dist/*.whl

# Optional: install PyTorch-related dependencies
.PHONY: pytorch
pytorch:
	@echo "Installing PyTorch and onnx2pytorch (optional, for PyTorch export support)"
	$(PIP)  torch onnx2pytorch

# Optional: install TensorFlow-related dependencies
.PHONY: tensorflow
tensorflow:
	@echo "Installing TensorFlow and onnx-tf (optional, for TensorFlow export support)"
	$(PIP)  tensorflow onnx-tf

test: ensure-venv
	$(PY) -m pytest -q

# Build the package (wheel)
build:
	@echo "Building wheel..."
	./scripts/build_wheel.sh

# Gold: run full test-suite then build artifacts
.PHONY: onnx-ops
onnx-ops:
	@echo "Updating ONNX operator catalog (ONNX_OPS.json)..."
	@$(PY) -m scripts.update_onnx_ops --output ONNX_OPS.json

.PHONY: ops
ops:
	@echo "Generating operator catalog (OPS.json)..."
	@$(PY) -m scripts.supported_ops --output OPS.json

# Ensure dev deps are available for trace builds too
gold: setup
	@echo "Running gold steps with full trace..."
	@$(PY) scripts/gold.py --trace
	@$(PY) tests/test_gold_exists.py --trace

meta:
	@echo "Exporting meta and artifacts for examples/golden/*.fuse to tmp/onnx (no graphviz)"
	@$(PY) scripts/golden_onnx_export.py --meta --ttl --no-dot --out-dir tmp/onnx || (echo "ERROR: 'make meta' failed"; exit 1)
	@echo "✅ meta exported to tmp/onnx"

.PHONY: examples
examples: ensure-venv
	@echo "Exporting all examples/golden/*.fuse to tmp/onnx (fail-fast)"
	@set -e; for f in $(EXAMPLES_DIR)/golden/*.fuse; do \
		echo "Processing $$f"; \
		$(PY) -m scripts.golden_onnx_export --process-file "$$f" --out-dir tmp/onnx || { echo "ERROR: export failed for $$f"; exit 1; }; \
	done
	@echo "✅ All examples exported successfully."

.PHONY: benchmark
benchmark:
	@echo "Running bench: fuse vs py goldens"
	$(PY) scripts/benchmark_fuse_vs_py.py --out benchmark

# Run only lowering related tests (fast feedback loop)
test-lowering:
	$(PY) -m pytest -q tests/lowering

# Quick smoke tests (add small, fast tests under tests/smoke/)
smoke-test:
	@echo "Running smoke tests (fast feedback)"
	$(PY) -m pytest -q tests/smoke

# Parsing-only tests
test-parsing:
	$(PY) -m pytest -q tests/parsing

# Golden-related tests (exporters, onnx golden checks)
test-golden:
	@echo "Running golden tests (scripts + onnx related tests)"
	$(PY) -m pytest -q tests/scripts tests/onnx

# Jupyter-related tests
test-jupyter:
	$(PY) -m pytest -q tests/jupyter

# Decompilation/audit tests
test-decompile:
	$(PY) -m pytest -q tests/cli/test_decompile_alias.py

# Server tests (non-blocking server unit tests)
test-server:
	$(PY) -m pytest -q tests/server

# Combined quick 'all' runner (composes smaller suites)
test-all:
	@echo "Running full test run (composed)":
	$(MAKE) test-lowering
	$(MAKE) test-parsing
	$(MAKE) test-jupyter
	$(MAKE) test-decompile
	$(MAKE) test-server
	$(MAKE) test
	@echo "✅ test-all completed."

# Development setup target
dev: setup dev-install
	@echo "Development environment prepared. Activate with: source .venv/bin/activate"

# Backwards-compatible dev-install target (some automation depends on this name)
.PHONY: dev-install
dev-install:
	@$(MAKE) setup

# Build/package flow (clean + build + tests)
package: clean build test
	@echo "Package artifact built in dist/"

lint:
	$(PY) -m flake8 src examples tests

# Lint Fuse examples (per-directory and aggregate targets)
# Usage: `make lint-examples-<set>` or `make lint-examples-all`
lint-examples-%:
	@echo "🔎 Linting examples/$(subst lint-examples-,,$@)"
	@$(FUSE_CMD) lint -f $(EXAMPLES_DIR)/$(subst lint-examples-,,$@) || (echo "fuse lint failed in examples/$(subst lint-examples-,,$@)"; exit 2)
	@$(FUSE_CMD) lint -f $(JUPYTER_COOKBOOK_DIR)/$(subst lint-cookbook-,,$@) || (echo "fuse lint failed in examples/$(subst lint-examples-,,$@)"; exit 2)

lint-examples:
	@echo "Run 'make lint-examples-all' or 'make lint-examples-<set>' (golden, cookbook, advanced, fail, runner)"

lint-examples-all: lint-examples-golden lint-examples-cookbook lint-examples-advanced lint-examples-fail lint-examples-runner
	@echo "✅ All example sets linted"

# Combined lint that checks code + examples
lint-all: lint lint-examples-all
	@echo "✅ Code + examples linted"

export-cookbook:
	FUSE_CMD=${FUSE_CMD:-./.venv/bin/fuse} ./scripts/export_cookbook.sh


notebook:
	papermill docs/cookbook/autoencoder_training.ipynb docs/cookbook/autoencoder_training-executed.ipynb

build-md:
	jupyter nbconvert --to markdown docs/cookbook/autoencoder_training-executed.ipynb --output docs/cookbook/autoencoder_training.md

# ----- Jupyter Docker helpers -----
.PHONY: jupyter-image jupyter-run jupyter-stop jupyter-shell jupyter-clean
JUPY_IMAGE ?= fused:local
JUPY_DOCKERFILE ?= docker/jupyter/Dockerfile
JUPY_PORT ?= 8888
JUPY_CONTAINER ?= fuse
# Optional: enable running notebooks and/or cleaning examples during image build
JUPY_RUN_NOTEBOOKS ?= 0
JUPY_CLEAN_EXAMPLES ?= 0

# Ensure the project virtualenv is present for local development flows.
# This enforces a consistent, reproducible dev environment and avoids
# accidental uses of the system Python when running make targets.
.PHONY: .venv-check ensure-venv
.venv-check:
	@if [ ! -x ".venv/bin/python" ]; then \
		echo "ERROR: missing .venv. Run 'make venv' and then 'make setup' before running this target."; \
		exit 1; \
	fi

# Ensure the project virtualenv is active (will fail if not activated)
ensure-venv: .venv-check
	@echo "Checking virtualenv activation..."
	@scripts/ensure-venv.sh
	@echo "Virtualenv seems active."

# Convenience: build jupyter docker image locally
jupyter-docker: .venv-check jupyter-image
	@echo "Building Jupyter Docker image (local): fuse-jupyter:local"
	DOCKER_BUILDKIT=1 docker build -f docker/jupyter/Dockerfile -t fuse-jupyter:local .

jupyter-image: .venv-check
	@echo "Building Jupyter Docker image: $(JUPY_IMAGE) (editable install) using $(JUPY_DOCKERFILE)"
	@echo "Note: pass JUPY_RUN_NOTEBOOKS=1 or JUPY_CLEAN_EXAMPLES=1 to the make command to enable optional notebook run/cleanup"
	DOCKER_BUILDKIT=1 docker build \
		--build-arg RUN_NOTEBOOKS=${JUPY_RUN_NOTEBOOKS:-0} \
		--build-arg CLEAN_EXAMPLES=${JUPY_CLEAN_EXAMPLES:-0} \
		--progress=plain \
		-f $(JUPY_DOCKERFILE) -t $(JUPY_IMAGE) .

jupyter: jupyter-stop jupyter-image
	@echo "Running Jupyter container (port $(JUPY_PORT))"
	docker run --rm -d -p $(JUPY_PORT):8888 -v $(PWD):/fused -w /fused --name $(JUPY_CONTAINER) $(JUPY_IMAGE)
	@docker exec $(JUPY_CONTAINER) python -m scripts.enable_jupyter_extension 
	@docker exec $(JUPY_CONTAINER) jupyter server extension list --sys-prefix 

jupyter-start: jupyter-stop 
	docker run --rm -d -p $(JUPY_PORT):8888 -v $(PWD):/fused -w /fused --name $(JUPY_CONTAINER) $(JUPY_IMAGE)


jupyter-stop:
	@echo "Stopping Jupyter container '$(JUPY_CONTAINER)'"
	-docker rm -f $(JUPY_CONTAINER) >/dev/null 2>&1 || true

jupyter-shell:
	@echo "Opening shell in running container $(JUPY_CONTAINER)"
	docker exec -it $(JUPY_CONTAINER) /bin/bash

jupyter-clean:
	@echo "Removing local image $(JUPY_IMAGE)"
	-docker rmi $(JUPY_IMAGE) || true

clean:
	rm -rf onnx/cookbook onnx_tmp
	rm -f docs/cookbook/autoencoder_training-executed.ipynb
	rm -f docs/cookbook/autoencoder_training.md
