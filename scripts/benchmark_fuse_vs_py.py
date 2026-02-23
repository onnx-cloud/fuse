#!/usr/bin/env python3
"""Benchmark Fuse (.fuse) vs Python (.py) golden pairs.

For each pair found under `examples/golden/<name>.fuse` and
`examples/golden/<name>.py` this script writes a per-pair JSON file with
metrics and creates PNG charts (lines/bytes/complexity) in the output
folder (default: `./benchmark`). Also writes a summary
`comparison.jsonl` with one entry per pair.

Usage: python scripts/benchmark_fuse_vs_py.py [--out <out_dir>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import ast
import sys

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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    print("matplotlib is required to generate charts. Install dev deps in .venv and retry.", file=sys.stderr)
    raise

from src.parser import fuse_parser


def analyze_py(path: Path):
    txt = path.read_text(encoding="utf-8")
    node = ast.parse(txt)
    func_count = sum(1 for n in ast.walk(node) if isinstance(n, ast.FunctionDef))
    import_count = sum(1 for n in ast.walk(node) if isinstance(n, (ast.Import, ast.ImportFrom)))
    ast_nodes = sum(1 for _ in ast.walk(node))
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "lines": len(txt.splitlines()),
        "func_count": func_count,
        "import_count": import_count,
        "ast_nodes": ast_nodes,
    }


def analyze_fuse(path: Path):
    txt = path.read_text(encoding="utf-8")
    try:
        decs = fuse_parser.parse(txt)
        decl_count = len([d for d in decs if isinstance(d, dict) and d.get("type")])
    except Exception:
        decl_count = -1
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "lines": len(txt.splitlines()),
        "decl_count": decl_count,
    }


def write_json(pair_out: Path, name: str, fuse_info: dict, python_info: dict):
    """Write per-pair JSON using explicit keys `fuse` and `python`."""
    pair_out.mkdir(parents=True, exist_ok=True)
    p = pair_out / f"{name}.json"
    with p.open("w", encoding="utf-8") as f:
        json.dump({"name": name, "fuse": fuse_info, "python": python_info}, f, sort_keys=True, indent=2)
    return p


def make_charts(entries: list[dict], out: Path):
    names = [e["name"] for e in entries]
    fuse_lines = [e["fuse"]["lines"] for e in entries]
    python_lines = [e["python"]["lines"] for e in entries]
    fuse_bytes = [e["fuse"]["bytes"] for e in entries]
    python_bytes = [e["python"]["bytes"] for e in entries]
    fuse_decl = [e["fuse"]["decl_count"] for e in entries]
    python_ast = [e["python"]["ast_nodes"] for e in entries]

    x = range(len(names))
    width = 0.35

    # Lines comparison
    plt.figure(figsize=(max(6, len(names) * 0.5), 4))
    plt.bar([i - width / 2 for i in x], fuse_lines, width, label="Fuse (.fuse) lines")
    plt.bar([i + width / 2 for i in x], python_lines, width, label="Python (.py) lines")
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("Lines")
    plt.title("Lines: Fuse (.fuse) vs Python (.py)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "lines_comparison.png")
    plt.close()

    # Bytes comparison
    plt.figure(figsize=(max(6, len(names) * 0.5), 4))
    plt.bar([i - width / 2 for i in x], fuse_bytes, width, label="Fuse (.fuse) bytes")
    plt.bar([i + width / 2 for i in x], python_bytes, width, label="Python (.py) bytes")
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("Bytes")
    plt.title("Bytes: Fuse (.fuse) vs Python (.py)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "bytes_comparison.png")
    plt.close()

    # Complexity scatter (fuse decl_count vs python ast_nodes)
    plt.figure(figsize=(6, 6))
    plt.scatter(fuse_decl, python_ast)
    for i, n in enumerate(names):
        plt.annotate(n, (fuse_decl[i], python_ast[i]))
    plt.xlabel("Fuse decl_count")
    plt.ylabel("Python AST node count")
    plt.title("Complexity: Fuse declarations vs Python AST nodes")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out / "complexity_scatter.png")
    plt.close()

    # Normalized ratios: use (python - fuse) / (python + fuse) to get -1..1 scale
    def normalized(a, b):
        out = []
        for ai, bi in zip(a, b):
            s = ai + bi
            if s == 0:
                out.append(0.0)
            else:
                out.append((ai - bi) / s)
        return out

    nr_lines = normalized(python_lines, fuse_lines)
    nr_bytes = normalized(python_bytes, fuse_bytes)

    plt.figure(figsize=(max(6, len(names) * 0.5), 4))
    plt.bar([i - width / 2 for i in x], nr_lines, width, label="norm lines (Python vs Fuse)")
    plt.bar([i + width / 2 for i in x], nr_bytes, width, label="norm bytes (Python vs Fuse)")
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("Normalized"); plt.ylim(-1, 1)
    plt.title("Normalized Differences: Python vs Fuse (range -1..1) \n(normalized = (python - fuse)/(python + fuse))")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "normalized_comparison.png")
    plt.close()

    # Density comparison: python AST nodes per line vs fuse decls per line -> ratio
    python_nodes_per_line = [pa / max(1, pl) for pa, pl in zip(python_ast, python_lines)]
    fuse_decls_per_line = [fd / max(1, fl) for fd, fl in zip(fuse_decl, fuse_lines)]
    density_ratio = [pn / max(1e-6, fd) for pn, fd in zip(python_nodes_per_line, fuse_decls_per_line)]

    plt.figure(figsize=(max(6, len(names) * 0.5), 4))
    plt.bar(x, density_ratio)
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("Density ratio (python_nodes_per_line / fuse_decls_per_line)")
    plt.title("Density Comparison (Python vs Fuse): AST nodes per line vs decls per line")
    plt.tight_layout()
    plt.savefig(out / "density_comparison.png")
    plt.close()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="benchmark")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    roots = Path("examples/golden")
    entries = []
    # Iterate python files first (subset) so we only benchmark pairs that have a generator
    for py_path in sorted(roots.glob("*.py")):
        stem = py_path.stem
        fuse_path = roots / f"{stem}.fuse"
        if not fuse_path.exists():
            # skip python generators that do not have a corresponding .fuse
            continue
        fuse_info = analyze_fuse(fuse_path)
        py_info = analyze_py(py_path)
        # computed comparison metrics
        # Avoid division by zero: use max(1, denom)
        lines_ratio = py_info["lines"] / max(1, fuse_info["lines"])
        bytes_ratio = py_info["bytes"] / max(1, fuse_info["bytes"])
        # complexity: compare py AST nodes to fuse decl_count (fallback 1)
        complexity_ratio = py_info.get("ast_nodes", 0) / max(1, fuse_info.get("decl_count", 1))
            # log2 ratios for better scale handling
        import math
        log2_lines = math.log2(lines_ratio) if lines_ratio > 0 else 0
        log2_bytes = math.log2(bytes_ratio) if bytes_ratio > 0 else 0
        normalized_lines = (py_info["lines"] - fuse_info["lines"]) / max(1, (py_info["lines"] + fuse_info["lines"]))
        normalized_bytes = (py_info["bytes"] - fuse_info["bytes"]) / max(1, (py_info["bytes"] + fuse_info["bytes"]))
        metrics = {
            "lines_ratio": lines_ratio,
            "bytes_ratio": bytes_ratio,
            "complexity_ratio": complexity_ratio,
            "log2_lines_ratio": log2_lines,
            "log2_bytes_ratio": log2_bytes,
            "normalized_lines": normalized_lines,
            "normalized_bytes": normalized_bytes,
        }
        e = {"name": stem, "fuse": fuse_info, "python": py_info, "meta": metrics}
        entries.append(e)

        # write per-pair JSON including computed metrics into dedicated pair folder
        pair_out = out / stem
        write_json(pair_out, stem, fuse_info, py_info)
        # create per-pair charts that compare Python vs Fuse (lines, bytes, complexity, normalized, density)
        try:
            # Lines
            plt.figure(figsize=(4,3))
            values = [fuse_info["lines"], py_info["lines"]]
            labels = ["Fuse", "Python"]
            plt.bar(labels, values, color=["#2f7bdc", "#f7b32b"])
            plt.title(f"{stem}: Lines (Python vs Fuse)")
            plt.tight_layout()
            plt.savefig(pair_out / "lines.png")
            plt.close()

            # Bytes
            plt.figure(figsize=(4,3))
            values = [fuse_info["bytes"], py_info["bytes"]]
            plt.bar(labels, values, color=["#2f7bdc", "#f7b32b"])
            plt.title(f"{stem}: Bytes (Python vs Fuse)")
            plt.tight_layout()
            plt.savefig(pair_out / "bytes.png")
            plt.close()

            # Complexity: decls vs AST nodes
            plt.figure(figsize=(4,3))
            values = [fuse_info.get("decl_count", 0) or 0, py_info.get("ast_nodes", 0) or 0]
            plt.bar(["Fuse decls", "Python AST nodes"], values, color=["#2f7bdc", "#f7b32b"])
            plt.title(f"{stem}: Complexity (Fuse decls vs Python AST nodes)")
            plt.tight_layout()
            plt.savefig(pair_out / "complexity.png")
            plt.close()

            # Normalized differences (two bars: lines and bytes)
            nl = (py_info["lines"] - fuse_info["lines"]) / max(1, (py_info["lines"] + fuse_info["lines"]))
            nb = (py_info["bytes"] - fuse_info["bytes"]) / max(1, (py_info["bytes"] + fuse_info["bytes"]))
            plt.figure(figsize=(4,3))
            plt.bar(["norm_lines", "norm_bytes"], [nl, nb], color=["#2f7bdc", "#f7b32b"])
            plt.ylim(-1, 1)
            plt.title(f"{stem}: Normalized Differences (Python vs Fuse)")
            plt.tight_layout()
            plt.savefig(pair_out / "normalized.png")
            plt.close()

            # Density: AST nodes per line vs decls per line (as two bars)
            pnpl = (py_info.get("ast_nodes", 0) / max(1, py_info.get("lines", 1)))
            fdpl = (fuse_info.get("decl_count", 0) / max(1, fuse_info.get("lines", 1)))
            plt.figure(figsize=(4,3))
            plt.bar(["python_nodes_per_line", "fuse_decls_per_line"], [pnpl, fdpl], color=["#2f7bdc", "#f7b32b"])
            plt.title(f"{stem}: Density (nodes/line)")
            plt.tight_layout()
            plt.savefig(pair_out / "density.png")
            plt.close()
        except Exception:
            # best-effort: don't fail overall benchmarking due to per-pair charting
            pass

    # write summary jsonl
    summary = out / "comparison.jsonl"
    with summary.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, sort_keys=True))
            f.write("\n")

    # We intentionally produce only per-pair comparisons (one folder per pair
    # containing the comparison JSON and bar-chart PNGs). Do not generate
    # aggregate charts to keep benchmarks focused.
    pass

    print(f"Wrote {len(entries)} entries to {out} (json + png)")


if __name__ == "__main__":
    main()
