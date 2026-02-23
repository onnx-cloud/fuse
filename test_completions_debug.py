#!/usr/bin/env python
"""Test completion system after fixes."""
import sys

# Clear cache
mods_to_clear = [m for m in list(sys.modules.keys()) if 'introspection' in m or 'jupyter.server' in m]
for m in mods_to_clear:
    sys.modules.pop(m, None)

# Import fresh
from src.jupyter.server import completions

# Test
results = completions('')
print(f"Total results: {len(results)}")
print(f"First 10: {[r['label'] for r in results[:10]]}")

# Check for 'Add'
has_add = any(r['label'] == 'Add' for r in results)
print(f"Has 'Add': {has_add}")
