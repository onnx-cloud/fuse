#!/usr/bin/env python3
"""Codemod: convert angle-scalar types `f32[...]` to shorthand `f32[...]`.

Runs in-place over .fuse/.md/.txt/.py files (where relevant) and prints a brief
summary. Intended as a one-off developer tool invoked from the repository root.
"""

import re
from pathlib import Path

# Prefer running via the project's virtualenv Python if available
try:
    import os, sys
    from pathlib import Path as _Path
    _here = _Path(__file__).resolve().parents[1]
    _venv_py = _here / ".venv" / "bin" / "python"
    if _venv_py.exists():
        try:
            if _Path(sys.executable).resolve() != _venv_py.resolve():
                os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)
        except Exception:
            pass
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
PAT = re.compile(
    r"<(?P<typ>f32|f64|i64|i32|i16|i8|u64|u32|u16|u8|bool|bf16|f16|complex64|complex128)>\["
)

exts = {".fuse", ".md", ".txt", ".py", ".rst", ".markdown"}
files = [
    p
    for p in ROOT.rglob("*")
    if p.suffix in exts and "third_party" not in p.parts
]

changed = []
for p in files:
    try:
        s = p.read_text(encoding="utf-8")
    except Exception:
        continue
    new = PAT.sub(lambda m: f"{m.group('typ')}[", s)
    if new != s:
        p.write_text(new, encoding="utf-8")
        changed.append(p.relative_to(ROOT))

print(f"Converted angle-scalar → shorthand in {len(changed)} files")
for p in changed:
    print(p)
