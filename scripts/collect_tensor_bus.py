#!/usr/bin/env python3
"""Collect tensor-bus style fully-qualified tensor names from a .fuse file.

Usage: python scripts/collect_tensor_bus.py examples/golden/clip.fuse

Produces a JSON mapping: {
  "module.func.name": {"name": <graph-name>, "tensor": "f32[1,768]"}, ...
}

This is a static analysis helper that does not require full lowering.
"""
import json
import sys
from pathlib import Path
from src.parser import fuse_parser

# Prefer running via the project's virtualenv Python if available
try:
    import os
    from pathlib import Path as _Path
    _here = _Path(__file__).resolve().parents[1]
    _venv_py = _here / ".venv" / "bin" / "python"
    if _venv_py.exists():
        try:
            import sys as _sys
            if _Path(_sys.executable).resolve() != _venv_py.resolve():
                os.execv(str(_venv_py), [str(_venv_py)] + _sys.argv)
        except Exception:
            pass
except Exception:
    pass


def type_to_str(t):
    if not t:
        return ""
    scalar = t.get("scalar")
    dims = t.get("dims") or []
    dim_str = ",".join(str(d) for d in dims) if dims else ""
    return f"{scalar}[{dim_str}]" if dim_str else f"{scalar}[]"


def collect(path: Path):
    src = path.read_text()
    # Accept legacy `@namespace` shorthand used in a few helper scripts/tests
    # (normalize to the canonical `@domain` directive for parsing).
    if "@namespace" in src:
        src = src.replace("@namespace", "@domain")
    ast = fuse_parser.parse(src)
    module = None
    # collect top-level types for weights/consts
    globals_types = {}
    decls = [d for d in ast if isinstance(d, dict)]
    for d in decls:
        if d.get("type") == "meta" and d.get("name") in ("domain", "module"):
            module = str(d.get("value"))
        if d.get("type") == "param":
            # weight/param
            name = d.get("name")
            t = d.get("type_decl") or d.get("type")
            globals_types[name] = t if isinstance(t, dict) else {"scalar": t, "dims": []}
        if d.get("type") == "const":
            name = d.get("name")
            t = d.get("type_decl") or d.get("type")
            globals_types[name] = t if isinstance(t, dict) else {"scalar": t, "dims": []}

    bus = {}
    func_calls = {}
    func_locals = {}
    for d in decls:
        if d.get("type") in ("node", "model", "export"):
            func = d.get("name")
            func_locals[func] = {}
            # params
            for p in d.get("params", []):
                pname = p.get("name")
                ptype = p.get("type") or p.get("type_decl") or globals_types.get(pname)
                key = f"{module}.{func}.{pname}" if module else f"{func}.{pname}"
                func_locals[func][key] = {"name": pname, "tensor": type_to_str(ptype)}
            # scan body for references to global names and calls
            body = d.get("body") or []
            calls = set()
            def walk(node):
                if isinstance(node, str):
                    if node in globals_types:
                        key = f"{module}.{func}.{node}" if module else f"{func}.{node}"
                        func_locals[func][key] = {"name": node, "tensor": type_to_str(globals_types.get(node))}
                elif isinstance(node, dict):
                    # capture call targets
                    if node.get("call") and isinstance(node.get("call"), str):
                        calls.add(node.get("call"))
                    for v in node.values():
                        walk(v)
                elif isinstance(node, list):
                    for v in node:
                        walk(v)
            walk(body)
            func_calls[func] = calls
    # transitively expand called functions' locals into callers and
    # also add aliases where the callee-scoped key is replicated under the
    # caller's namespace (so `mod.callee.P` -> `mod.caller.P`). This makes
    # it easy for consumers to bind to `module.func.name` regardless of where
    # the tensor was originally declared.
    changed = True
    while changed:
        changed = False
        for f, calls in func_calls.items():
            for callee in list(calls):
                if callee in func_locals:
                    for k, v in func_locals[callee].items():
                        if k not in func_locals[f]:
                            func_locals[f][k] = v
                            changed = True
                        # create an alias key by replacing the callee name
                        try:
                            parts = k.split(".")
                            if len(parts) >= 3:
                                parts[-2] = f
                                alias = ".".join(parts)
                                if alias not in func_locals[f]:
                                    func_locals[f][alias] = v
                                    changed = True
                        except Exception:
                            # be robust to unexpected key shapes
                            pass
    # flatten into bus. Also expose callee entries under caller's
    # namespace so bindings can refer to `module.func.param` even when the
    # tensor is declared in a callee function.
    for func, entries in func_locals.items():
        for k, v in entries.items():
            bus[k] = v
            # If key contains the callee function name, replicate under caller
            # namespace when possible (e.g., copy encode_text.P_txt -> clip_demo.P_txt)
            try:
                parts = k.split(".")
                if len(parts) >= 3:
                    mod = ".".join(parts[:-2])
                    callee = parts[-2]
                    localname = parts[-1]
                    # For each caller that transitively includes callee, also
                    # expose a variant under the caller's namespace.
                    for caller, calls in func_calls.items():
                        if callee == caller:
                            continue
                        # if caller uses callee transitively, create mapping
                        if callee in func_calls.get(caller, set()) or any(
                            callee in func_calls.get(c, set()) for c in func_calls.get(caller, set())
                        ):
                            new_key = f"{mod}.{caller}.{localname}" if mod else f"{caller}.{localname}"
                            if new_key not in bus:
                                bus[new_key] = {"name": localname, "tensor": v.get("tensor")}
            except Exception:
                pass
    return bus


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: collect_tensor_bus.py <path.fuse>")
        raise SystemExit(1)
    p = Path(sys.argv[1])
    out = collect(p)
    print(json.dumps(out, indent=2))
