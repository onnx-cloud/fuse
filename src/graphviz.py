"""Simple deterministic DOT emitter and optional renderer for Fuse/ONNX graphs.

Exports:
- model_to_dot(model: onnx.ModelProto) -> str
- write_dot(dot: str, path: str) -> None
- render_dot(dot: str, out_path: str, fmt: str = "svg") -> bool

Design goals:
- Deterministic ordering (sorted by names) for stable DOT outputs
- Minimal runtime deps (try python `graphviz`, else `dot` binary via subprocess)
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List
import json

import onnx


def _safe_name(s: str) -> str:
    # sanitize names for DOT identifiers
    if s is None:
        return ""
    return s.replace("/", "_").replace(".", "_").replace("\n", "_")


def _sorted_nodes(graph: onnx.GraphProto) -> List[onnx.NodeProto]:
    # deterministic order by node.name (fallback to op_type + index)
    nodes = list(graph.node)

    def key(n):
        return n.name or f"{n.op_type}:{list(graph.node).index(n)}"

    return sorted(nodes, key=key)


def model_to_dot(model: onnx.ModelProto) -> str:
    graph = model.graph
    lines = [
        "digraph G {",
        "  rankdir=LR;",
        "  node [shape=record,fontname=Helvetica];",
    ]

    # inputs
    inputs = sorted((vi.name for vi in graph.input))
    for name in inputs:
        lines.append(
            f'  "{_safe_name(name)}" [label="{name}: input", shape=oval];'
        )

    # outputs
    outputs = sorted((vo.name for vo in graph.output))
    for name in outputs:
        lines.append(
            f'  "{_safe_name(name)}" [label="{name}: output", shape=oval];'
        )

    # nodes
    for n in _sorted_nodes(graph):
        label = n.name or n.op_type
        label = f"{n.op_type}\n{label}" if n.name else n.op_type
        nid = _safe_name(n.name or f"{n.op_type}_{id(n)}")
        lines.append(f'  "{nid}" [label="{label}"];')
        # edges from inputs
        for inp in sorted(n.input):
            src = _safe_name(inp)
            if src:
                lines.append(f'  "{src}" -> "{nid}";')
        # edges to outputs
        for out in sorted(n.output):
            dst = _safe_name(out)
            if dst:
                lines.append(f'  "{nid}" -> "{dst}";')

    lines.append("}")
    return "\n".join(lines) + "\n"


def write_dot(dot: str, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dot, encoding="utf-8")


def render_dot(dot: str, out_path: str, fmt: str = "svg") -> bool:
    """Render DOT to `out_path` with format `fmt` (svg or png).

    Returns True if render succeeded; False otherwise (DOT written only).
    """
    # Prefer python `graphviz` package if available
    try:
        import graphviz

        src = graphviz.Source(dot)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # graphviz.Source.render appends format-specific extension unless cleanup
        out_file = str(out.with_suffix(""))
        # use format option
        src.format = fmt
        src.render(filename=out_file, cleanup=True)
        return True
    except Exception:
        # fallback to `dot` binary
        dot_exec = shutil.which("dot")
        if not dot_exec:
            return False
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            p = subprocess.run(
                [dot_exec, f"-T{fmt}"],
                input=dot.encode("utf-8"),
                stdout=subprocess.PIPE,
                check=True,
            )
            out.write_bytes(p.stdout)
            return True
        except Exception:
            return False


def render_dot_safe(dot: str, out_path: str, fmt: str = "svg", timeout: int = 10) -> bool:
    """Render DOT to `out_path` in a subprocess-isolated manner with timeout.

    Attempts to use the system `dot` binary first (via subprocess in the
    current process). If `dot` is not available, falls back to invoking a
    short isolated Python runner that uses python-graphviz if present.

    On failure an error file is written next to the expected output file
    (e.g., `graph.svg.error.txt`) and False is returned.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Try system 'dot' first (fast and isolated)
    dot_exec = shutil.which("dot")
    if dot_exec:
        try:
            subprocess.run(
                [dot_exec, f"-T{fmt}", "-o", str(out)],
                input=dot.encode("utf-8"),
                check=True,
                timeout=timeout,
            )
            return out.exists()
        except Exception as e:
            err = out.with_suffix(out.suffix + ".error.txt")
            try:
                err.write_text(str(e), encoding="utf-8")
            except Exception:
                pass
            return False

    # Fallback: spawn a short Python helper that uses python-graphviz if available
    helper = (
        "import sys, json; "
        "from pathlib import Path; "
        f"dot = {json.dumps(dot)}; "
        f"out = Path({json.dumps(str(out))}); "
        f"fmt = {json.dumps(fmt)}; "
        "try:\n"
        "    import graphviz\n"
        "    src = graphviz.Source(dot)\n"
        "    src.format = fmt\n"
        "    src.render(filename=str(out.with_suffix('')), cleanup=True)\n"
        "    sys.exit(0)\n"
        "except Exception as e:\n"
        "    try:\n"
        "        out_e = out.with_suffix(out.suffix + '.error.txt')\n"
        "        out_e.write_text(str(e), encoding='utf-8')\n"
        "    except Exception:\n"
        "        pass\n"
        "    sys.exit(1)"
    )
    try:
        res = subprocess.run([sys.executable, "-c", helper], check=False, timeout=timeout)
        return out.exists()
    except subprocess.TimeoutExpired as e:
        err = out.with_suffix(out.suffix + ".error.txt")
        try:
            err.write_text(f"render timeout: {e}", encoding="utf-8")
        except Exception:
            pass
        return False
