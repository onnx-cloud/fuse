#!/usr/bin/env bash
set -euo pipefail

# Check SHA256 checksum of src/OpCodes.json matches expected value in tests
python - <<'PY'
import hashlib
from pathlib import Path

p = Path('src/OpCodes.json')
if not p.exists():
    print('src/OpCodes.json not found')
    raise SystemExit(1)

h = hashlib.sha256(p.read_bytes()).hexdigest()
# The test `tests/test_opcodes_immutable.py` computes the expected checksum; run pytest to rely on that
import subprocess
subprocess.run(['python', '-m', 'pytest', 'tests/test_opcodes_immutable.py', '-q'], check=True)
PY
