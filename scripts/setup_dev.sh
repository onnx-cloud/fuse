#!/usr/bin/env bash
set -euo pipefail

# Create a local virtualenv and install test/dev requirements without sudo
VE_DIR=.venv
PYTHON=${PYTHON:-python3}

echo "Setting up development virtualenv in ${VE_DIR} using ${PYTHON}"
${PYTHON} -m venv ${VE_DIR}
source ${VE_DIR}/bin/activate
python -m uv pip install --upgrade pip
uv pip install -r requirements.txt

if [[ -f requirements-dev.txt ]]; then
	echo "Installing dev requirements (requirements-dev.txt)"
	uv pip install -r requirements-dev.txt
	# Perform an editable install with dev extras so entrypoints and optional
	# extras (e.g. lark, onnxruntime for runtime tests) are available.
	if python -c "import importlib.util,sys; sys.exit(importlib.util.find_spec('pip') is None)"; then
		:
	fi
	if ! uv pip install -e ".[dev]"; then
		echo "Editable install of extras failed; trying editable install without extras..."
		uv pip install -e . || echo "Editable install failed; continuing (you can run: uv pip install -e .)"
	fi
fi

echo "Done. Activate with: source ${VE_DIR}/bin/activate"
echo "Run tests: pytest (or python tests/run_simple_tests.py without pytest)"
