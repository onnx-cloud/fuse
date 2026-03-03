from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import html as _html
from typing import Any, Dict, List, Optional

import numpy as np
from IPython.core.magic import (Magics, magics_class, cell_magic, line_magic, needs_local_scope)
from IPython.display import HTML, display

from src.parser import fuse_parser
from src.lowering import FuseLowerer
from src.sandbox import LocalSandbox


def _dtype_str(elem_type: int) -> str:
    """Convert ONNX TensorProto.DataType to readable string."""
    _type_map = {
        1: "f32", 2: "u8", 3: "i8", 4: "u16", 5: "i16",
        6: "i32", 7: "i64", 8: "str", 9: "bool", 10: "f16",
        11: "f64", 12: "u32", 13: "u64", 14: "c64", 15: "c128",
        16: "bf16", 17: "f8e4m3fn", 18: "f8e4m3fnuz",
        19: "f8e5m2", 20: "f8e5m2fnuz",
    }
    return _type_map.get(elem_type, f"type{elem_type}")


def _shape_str(dims) -> str:
    """Format shape dimensions for display."""
    parts = []
    for d in dims:
        if d.dim_value:
            parts.append(str(d.dim_value))
        elif d.dim_param:
            parts.append(d.dim_param)
        else:
            parts.append("?")
    return "[" + ", ".join(parts) + "]"


def _render_dot_to_svg(dot: str, timeout: int = 5) -> Optional[str]:
    """Render DOT to SVG string. Returns None on failure."""
    dot_exec = shutil.which("dot")
    if dot_exec:
        try:
            result = subprocess.run(
                [dot_exec, "-Tsvg"],
                input=dot.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=timeout,
            )
            svg = result.stdout.decode("utf-8")
            # Strip XML declaration for embedding
            if svg.startswith("<?xml"):
                svg = svg.split("?>", 1)[-1].strip()
            return svg
        except Exception:
            pass
    # Fallback: python graphviz
    try:
        import graphviz
        src = graphviz.Source(dot)
        return src.pipe(format="svg").decode("utf-8")
    except Exception:
        pass
    return None


class Model:
    """Wrapper around compiled ONNX models with rich Jupyter display."""

    def __init__(self, name: str, model_proto):
        self.name = name
        self.model = model_proto
        self._sandbox = LocalSandbox()
        self._cached_svg: Optional[str] = None

    def run(self, inputs: Dict[str, Any], provider: str = "reference"):
        """Run inference with the model."""
        # resolve input names: allow users to specify short names when the
        # model's graph inputs are qualified (e.g. 'id.x').
        feeds = {}
        graph_inputs = [vi.name for vi in self.model.graph.input]
        for k, v in (inputs or {}).items():
            target = k
            if k not in graph_inputs:
                # try suffix match after last dot
                for gi in graph_inputs:
                    if gi.endswith(f".{k}"):
                        target = gi
                        break
            if isinstance(v, np.ndarray):
                feeds[target] = v
            else:
                feeds[target] = np.asarray(v)
        res = self._sandbox.run(self.model, feeds, runtime=provider)
        return {k: v for k, v in res.outputs.items()}

    def show(self):
        """Display model visualization in notebook."""
        display(HTML(self._repr_html_()))
        return self

    def to_onnx(self):
        """Return the underlying ONNX ModelProto."""
        return self.model

    def _get_graph_svg(self) -> Optional[str]:
        """Generate SVG for the model graph."""
        if self._cached_svg is not None:
            return self._cached_svg
        try:
            from src.graphviz import model_to_dot
            dot = model_to_dot(self.model)
            self._cached_svg = _render_dot_to_svg(dot)
            return self._cached_svg
        except Exception:
            return None

    def _get_metadata(self) -> Dict[str, Any]:
        """Extract model metadata for display."""
        graph = self.model.graph
        
        # Inputs with types
        inputs = []
        for vi in graph.input:
            tt = vi.type.tensor_type
            dtype = _dtype_str(tt.elem_type)
            shape = _shape_str(tt.shape.dim) if tt.HasField("shape") else "?"
            inputs.append({"name": vi.name, "dtype": dtype, "shape": shape})
        
        # Outputs with types
        outputs = []
        for vo in graph.output:
            tt = vo.type.tensor_type
            dtype = _dtype_str(tt.elem_type)
            shape = _shape_str(tt.shape.dim) if tt.HasField("shape") else "?"
            outputs.append({"name": vo.name, "dtype": dtype, "shape": shape})
        
        # Op counts
        op_counts: Dict[str, int] = {}
        for node in graph.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        
        # Initializers (params/consts)
        initializers = [i.name for i in graph.initializer]
        
        # Model metadata props
        meta_props = {p.key: p.value for p in self.model.metadata_props}
        
        # Trainable params (from metadata)
        trainables = []
        if "trainables" in meta_props:
            try:
                trainables = list(json.loads(meta_props["trainables"]).keys())
            except Exception:
                pass
        
        return {
            "name": self.name,
            "graph_name": graph.name or "(anonymous)",
            "inputs": inputs,
            "outputs": outputs,
            "nodes": len(graph.node),
            "op_counts": op_counts,
            "initializers": initializers,
            "trainables": trainables,
            "opset": self.model.opset_import[0].version if self.model.opset_import else "?",
            "domain": self.model.opset_import[0].domain or "ai.onnx" if self.model.opset_import else "?",
            "meta_props": meta_props,
        }

    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks."""
        meta = self._get_metadata()
        svg = self._get_graph_svg()
        
        # Style definitions
        style = """
        <style>
        .fuse-model-card {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            border: 1px solid #e1e4e8;
            border-radius: 8px;
            background: linear-gradient(135deg, #667eea11 0%, #764ba211 100%);
            padding: 0;
            margin: 8px 0;
            max-width: 100%;
            overflow: hidden;
        }
        .fuse-model-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .fuse-model-title {
            font-size: 16px;
            font-weight: 600;
            margin: 0;
        }
        .fuse-model-badge {
            background: rgba(255,255,255,0.2);
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
        }
        .fuse-model-body {
            padding: 16px;
        }
        .fuse-model-stats {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }
        .fuse-stat {
            background: white;
            border-radius: 6px;
            padding: 10px 14px;
            min-width: 90px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .fuse-stat-value {
            font-size: 20px;
            font-weight: 700;
            color: #667eea;
        }
        .fuse-stat-label {
            font-size: 11px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .fuse-io-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 16px;
        }
        .fuse-io-box {
            background: white;
            border-radius: 6px;
            padding: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .fuse-io-title {
            font-size: 12px;
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .fuse-io-item {
            font-family: 'SF Mono', Monaco, 'Consolas', monospace;
            font-size: 12px;
            padding: 4px 8px;
            background: #f6f8fa;
            border-radius: 4px;
            margin: 4px 0;
            display: flex;
            justify-content: space-between;
        }
        .fuse-io-name { color: #24292e; font-weight: 500; }
        .fuse-io-type { color: #6a737d; }
        .fuse-graph-container {
            background: white;
            border-radius: 6px;
            padding: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            overflow-x: auto;
        }
        .fuse-graph-title {
            font-size: 12px;
            font-weight: 600;
            color: #333;
            margin-bottom: 8px;
        }
        .fuse-graph-svg {
            max-width: 100%;
            height: auto;
        }
        .fuse-graph-svg svg {
            max-width: 100%;
            height: auto;
            max-height: 300px;
        }
        .fuse-ops-list {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 8px;
        }
        .fuse-op-chip {
            background: #e1e4e8;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-family: 'SF Mono', Monaco, monospace;
        }
        .fuse-op-count {
            background: #667eea;
            color: white;
            padding: 1px 5px;
            border-radius: 8px;
            margin-left: 4px;
            font-size: 10px;
        }
        .fuse-details {
            margin-top: 12px;
        }
        .fuse-details summary {
            cursor: pointer;
            font-size: 12px;
            color: #667eea;
            font-weight: 500;
        }
        .fuse-details-content {
            margin-top: 8px;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 11px;
            background: #f6f8fa;
            padding: 8px;
            border-radius: 4px;
            overflow-x: auto;
        }
        .fuse-toolbar {
            display: flex;
            gap: 8px;
            padding: 10px 16px;
            background: rgba(102, 126, 234, 0.08);
            border-bottom: 1px solid rgba(102, 126, 234, 0.1);
        }
        .fuse-toolbar-btn {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 6px 12px;
            background: white;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
            color: #333;
            cursor: pointer;
            transition: all 0.15s ease;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        .fuse-toolbar-btn:hover {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        .fuse-toolbar-btn:active {
            transform: scale(0.98);
        }
        .fuse-toolbar-btn .icon {
            font-size: 14px;
        }
        </style>
        """
        
        # Header
        header = f"""
        <div class="fuse-model-header">
            <h3 class="fuse-model-title">🔥 {_html.escape(meta['name'])}</h3>
            <span class="fuse-model-badge">opset {meta['opset']}</span>
        </div>
        """
        
        # Toolbar with action buttons
        # Generate unique ID for this model card
        import hashlib
        hashlib.md5(f"{meta['name']}{id(self)}".encode()).hexdigest()[:8]
        
        toolbar = f"""
        <div class="fuse-toolbar">
            <button class="fuse-toolbar-btn" onclick="IPython.notebook.kernel.execute('from src.jupyter.inspect import graph; display(graph({self.name!r}))')" title="View graph visualization">
                <span class="icon">📊</span> Graph
            </button>
            <button class="fuse-toolbar-btn" onclick="IPython.notebook.kernel.execute('from src.jupyter.inspect import weights; display(weights({self.name!r}))')" title="View model weights">
                <span class="icon">⚖️</span> Weights
            </button>
            <button class="fuse-toolbar-btn" onclick="IPython.notebook.kernel.execute('from src.jupyter.inspect import trace; display(trace({self.name!r}))')" title="Profile execution">
                <span class="icon">⏱️</span> Trace
            </button>
            <button class="fuse-toolbar-btn" onclick="IPython.notebook.kernel.execute('from src.jupyter.inspect import report; display(report({self.name!r}))')" title="Generate full report">
                <span class="icon">📄</span> Report
            </button>
        </div>
        """
        
        # Stats row
        stats = f"""
        <div class="fuse-model-stats">
            <div class="fuse-stat">
                <div class="fuse-stat-value">{meta['nodes']}</div>
                <div class="fuse-stat-label">Nodes</div>
            </div>
            <div class="fuse-stat">
                <div class="fuse-stat-value">{len(meta['inputs'])}</div>
                <div class="fuse-stat-label">Inputs</div>
            </div>
            <div class="fuse-stat">
                <div class="fuse-stat-value">{len(meta['outputs'])}</div>
                <div class="fuse-stat-label">Outputs</div>
            </div>
            <div class="fuse-stat">
                <div class="fuse-stat-value">{len(meta['initializers'])}</div>
                <div class="fuse-stat-label">Params</div>
            </div>
            <div class="fuse-stat">
                <div class="fuse-stat-value">{len(meta['trainables'])}</div>
                <div class="fuse-stat-label">Trainable</div>
            </div>
        </div>
        """
        
        # Inputs/Outputs
        inputs_html = "\n".join(
            f'<div class="fuse-io-item"><span class="fuse-io-name">{_html.escape(i["name"])}</span><span class="fuse-io-type">{i["dtype"]}{i["shape"]}</span></div>'
            for i in meta["inputs"]
        )
        outputs_html = "\n".join(
            f'<div class="fuse-io-item"><span class="fuse-io-name">{_html.escape(o["name"])}</span><span class="fuse-io-type">{o["dtype"]}{o["shape"]}</span></div>'
            for o in meta["outputs"]
        )
        
        io_section = f"""
        <div class="fuse-io-section">
            <div class="fuse-io-box">
                <div class="fuse-io-title">📥 Inputs</div>
                {inputs_html if inputs_html else '<div class="fuse-io-item">None</div>'}
            </div>
            <div class="fuse-io-box">
                <div class="fuse-io-title">📤 Outputs</div>
                {outputs_html if outputs_html else '<div class="fuse-io-item">None</div>'}
            </div>
        </div>
        """
        
        # Graph visualization
        if svg:
            graph_html = f"""
            <div class="fuse-graph-container">
                <div class="fuse-graph-title">📊 Computation Graph</div>
                <div class="fuse-graph-svg">{svg}</div>
            </div>
            """
        else:
            # Fallback: show ops as chips
            ops_chips = "".join(
                f'<span class="fuse-op-chip">{_html.escape(op)}<span class="fuse-op-count">{count}</span></span>'
                for op, count in sorted(meta["op_counts"].items())
            )
            graph_html = f"""
            <div class="fuse-graph-container">
                <div class="fuse-graph-title">🔧 Operations</div>
                <div class="fuse-ops-list">{ops_chips if ops_chips else '<em>No operations</em>'}</div>
            </div>
            """
        
        # Details section (collapsed)
        meta_json = json.dumps(meta["meta_props"], indent=2) if meta["meta_props"] else "{}"
        init_list = ", ".join(meta["initializers"][:10])
        if len(meta["initializers"]) > 10:
            init_list += f", ... (+{len(meta['initializers']) - 10} more)"
        
        details = f"""
        <details class="fuse-details">
            <summary>Show Details</summary>
            <div class="fuse-details-content">
                <strong>Graph:</strong> {_html.escape(meta['graph_name'])}<br>
                <strong>Domain:</strong> {_html.escape(str(meta['domain']))}<br>
                <strong>Initializers:</strong> {_html.escape(init_list) or 'None'}<br>
                <strong>Metadata:</strong><br><pre>{_html.escape(meta_json)}</pre>
            </div>
        </details>
        """
        
        return f"""
        {style}
        <div class="fuse-model-card">
            {header}
            {toolbar}
            <div class="fuse-model-body">
                {stats}
                {io_section}
                {graph_html}
                {details}
            </div>
        </div>
        """

    def __repr__(self) -> str:
        """Plain text representation."""
        meta = self._get_metadata()
        return f"<Model '{meta['name']}' nodes={meta['nodes']} inputs={len(meta['inputs'])} outputs={len(meta['outputs'])}>"


@magics_class
class FuseMagics(Magics):
    def __init__(self, shell):
        super().__init__(shell)
        self.registry: Dict[str, Model] = {}

    # --- Helper parsers -------------------------------------------------
    def _coerce_val(self, v: str):
        # convert common boolean/number strings
        if isinstance(v, bool):
            return v
        if v is True:
            return True
        if isinstance(v, str):
            lv = v.lower()
            if lv in ("true", "yes", "1"):
                return True
            if lv in ("false", "no", "0"):
                return False
            # numbers
            try:
                if "." in v:
                    return float(v)
                return int(v)
            except Exception:
                return v
        return v

    def _normalize_kwargs(self, kw: Dict[str, Any]):
        out = {}
        for k, v in kw.items():
            nk = k.replace("-", "_")
            out[nk] = self._coerce_val(v)
        return out

    def _parse_kv_args(self, line: str):
        # returns (positional list, kwargs dict)
        toks = shlex.split(line)
        pos = []
        kw = {}
        i = 0
        while i < len(toks):
            t = toks[i]
            if t.startswith("--"):
                k = t.lstrip("-")
                # lookahead for value
                v = True
                if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                    v = toks[i + 1]
                    i += 1
                kw[k] = v
            else:
                pos.append(t)
            i += 1
        return pos, self._normalize_kwargs(kw)

    @cell_magic
    def fuse(self, line: str, cell: str = ""):
        """Compile the cell Fuse source into an in-memory model.

        Usage:
          %%fuse [name]
          <fuse source...>
        Returns the compiled Model object and registers it under `name`.
        """
        name = line.strip() or "model"
        # If the user didn't provide a top-level @fuse declaration, inject a
        # sensible default derived from the installed package version so
        # interactive cells remain ergonomic.
        src_text = cell
        if not any(l.strip().startswith("@fuse") for l in cell.splitlines()):
            try:
                from src import __version__ as _fv
                ver = ".".join(_fv.split(".")[:2])
            except Exception:
                ver = "0.0"
            src_text = f"@fuse {ver}\n" + cell

        try:
            ast = fuse_parser.parse(src_text, filename="<memory>")
            fl = FuseLowerer()
            onnx_model = fl.lower(ast)
            m = Model(name, onnx_model)
            self.registry[name] = m
            
            # Also expose the ONNX model directly to the user namespace for convenience
            # This allows users to reference _fuse_model immediately after %%fuse
            self.shell.push({"_fuse_model": onnx_model})
            
            return m
        except Exception as e:
            # Gracefully handle parse or lowering errors without raising.
            # This keeps interactive cells responsive even with syntax errors.
            print(f"[fuse] error: {e}")
            return None

    @line_magic("fuse")
    def fuse_line(self, line: str):
        """Line magic: simple commands for Fuse models.

        Usage: %fuse export <path> - Save _fuse_model to ONNX file
        """
        args = line.split()
        if not args:
            print("[fuse] usage: %fuse export <path>")
            return
        cmd = args[0]
        if cmd == "export":
            if len(args) < 2:
                print("[fuse] export requires <path>")
                return
            path = args[1]
            model = self.shell.user_ns.get("_fuse_model")
            if model is None:
                print("[fuse] no model found in _fuse_model; run %%fuse first")
                return
            try:
                import onnx
                onnx.save_model(model, path)
                print(f"[fuse] saved model to {path}")
            except Exception as e:
                print(f"[fuse] failed to save model: {e}")
            return
        print(f"[fuse] unknown command: {cmd}")

    @line_magic
    def fuse_compile(self, line: str):
        """Compile a named file or a source by path.

        Usage: %fuse.compile <path.fuse> [--name <model-name>] [--out-dir DIR]
        """
        pos, kw = self._parse_kv_args(line)
        if not pos:
            raise ValueError("expected path or name")
        path = pos[0]
        # Delegate to existing helpers when possible; for now, support file compile
        from src.cli.cli_helpers import parse_fuse_file

        ast = parse_fuse_file(path)
        fl = FuseLowerer()
        model = fl.lower(ast, source_file=path)
        name = kw.get("name") or (model.graph.name or path)
        m = Model(name, model)
        self.registry[name] = m
        return m

    def _display_paths(self, paths: List[str]):
        # Show clickable links in notebook when possible
        if not paths:
            return paths
        html = "<ul>"
        for p in paths:
            html += f"<li><a href=\"{p}\">{p}</a></li>"
        html += "</ul>"
        display(HTML(html))
        return paths

    @line_magic
    @needs_local_scope
    def fuse_run(self, line: str, local_ns=None):
        """Run a compiled model.

        Usage: %fuse.run <name> [--input '<py-expr-or-json>'] [--provider <runtime>]
        Example: %fuse.run model --input "{'x':[1.0]}" --provider onnxruntime
        """
        pos, kw = self._parse_kv_args(line)
        if not pos:
            raise ValueError("expected model name or path")
        name = pos[0]
        provider = kw.get("provider", "reference")
        inp = kw.get("input")
        if inp:
            # Try JSON then Python eval in user namespace
            try:
                inputs = json.loads(inp)
            except Exception:
                # Evaluate as Python expression in the IPython user namespace
                try:
                    inputs = eval(inp, self.shell.user_ns, local_ns or {})
                except Exception as e:
                    raise ValueError(f"could not parse input expression: {e}")
        else:
            inputs = {}
        if name not in self.registry:
            # try compiling a file path provided
            from pathlib import Path

            if Path(name).exists():
                # compile and run temporary
                ast = fuse_parser.parse(Path(name).read_text(), filename=name)
                model = FuseLowerer().lower(ast, source_file=name)
                tmp = Model(name, model)
                return tmp.run(inputs, provider=provider)
            raise KeyError(f"unknown model: {name}")
        m = self.registry[name]
        return m.run(inputs, provider=provider)

    @line_magic
    def fuse_show(self, line: str):
        pos, kw = self._parse_kv_args(line)
        name = pos[0] if pos else None
        if not name:
            return list(self.registry.keys())
        m = self.registry.get(name)
        if not m:
            raise KeyError(f"model not found: {name}")
        return m.show()

    @line_magic
    def fuse_clear(self, line: str):
        pos, kw = self._parse_kv_args(line)
        name = pos[0] if pos else None
        if not name:
            self.registry.clear()
            return True
        if name in self.registry:
            del self.registry[name]
            return True
        return False

    # --- CLI wrappers (expanded) ---------------------------------------
    @line_magic
    def fuse_lint(self, line: str):
        pos, kw = self._parse_kv_args(line)
        from src.cli.commands import cmd_lint

        paths = pos or []
        res = cmd_lint(paths, fail_on_warn=bool(kw.get("fail_on_warn", False)))
        return res

    @line_magic
    def fuse_verify(self, line: str):
        pos, kw = self._parse_kv_args(line)
        from src.cli.commands import cmd_verify

        paths = pos or []
        return cmd_verify(paths)

    @line_magic
    def fuse_onnx(self, line: str):
        pos, kw = self._parse_kv_args(line)
        from src.cli.commands import cmd_compile

        files = pos or []
        opts = {}
        # copy common opts and coerce types
        for k in ("out-dir", "training", "bake", "seal"):
            if k.replace('-', '_') in kw:
                opts[k.replace("-", "_")] = kw.get(k.replace('-', '_'))
        res = cmd_compile(files, **opts)
        # cmd_compile returns list of tuples (src, outpath, error)
        paths = [r[1] for r in res if r[1]]
        return self._display_paths(paths)

    @line_magic
    def fuse_graphviz(self, line: str):
        pos, kw = self._parse_kv_args(line)
        from src.cli.commands import cmd_graphviz

        files = pos or []
        res = cmd_graphviz(files, dot_dir=kw.get('dot_dir'), render=kw.get('render', False), out_dir=kw.get('out_dir'))
        paths = []
        for _, outs, _ in res:
            if outs:
                paths.extend(outs)
        return self._display_paths(paths)

    @line_magic
    def fuse_inspect(self, line: str):
        pos, kw = self._parse_kv_args(line)
        from src.cli.commands import cmd_inspect

        files = pos or []
        res = cmd_inspect(files, out_dir=kw.get('out_dir'), dot=kw.get('dot', True), render=kw.get('render', False), plots=kw.get('plots', False))
        paths = []
        for _, outs, _ in res:
            if outs:
                paths.extend(outs)
        return self._display_paths(paths)

    @line_magic
    def fuse_docs(self, line: str):
        pos, kw = self._parse_kv_args(line)
        from src.cli.commands import cmd_docs

        files = pos or []
        res = cmd_docs(files, out_dir=kw.get('out_dir'), md=kw.get('md', False), dot=kw.get('dot', False), render=kw.get('render', False))
        paths = []
        for _, outs, _ in res:
            if outs:
                paths.extend(outs)
        return self._display_paths(paths)

    @line_magic
    def fuse_metrics(self, line: str):
        pos, kw = self._parse_kv_args(line)
        from src.cli.commands import cmd_metrics

        files = pos or []
        res = cmd_metrics(files)
        # return structured metrics
        return res

    @line_magic
    def fuse_models(self, line: str):
        pos, kw = self._parse_kv_args(line)
        from src.cli.commands import cmd_models

        files = pos or []
        res = cmd_models(files, root=kw.get('root'), manifest_only=kw.get('manifest_only', False), manifest_dir=kw.get('manifest_dir'), overwrite=kw.get('overwrite', False), variant=kw.get('variant'))
        return res

    @line_magic
    def fuse_golden(self, line: str):
        pos, kw = self._parse_kv_args(line)
        from src.cli.commands import cmd_golden

        files = pos or []
        res = cmd_golden(files, quiet=kw.get('quiet', False), fail_fast=kw.get('fail_fast', False))
        return res

    @line_magic
    def fuse_zoo(self, line: str):
        pos, kw = self._parse_kv_args(line)
        from src.cli.commands import cmd_zoo

        # simple passthrough; args are in kw/pos
        class Args:
            pass

        a = Args()
        for k, v in kw.items():
            setattr(a, k, v)
        if pos:
            a.op = pos[0]
            if len(pos) > 1:
                a.id = pos[1]
        return cmd_zoo(a)

    @line_magic
    def fuse_ebnf(self, line: str):
        pos, kw = self._parse_kv_args(line)
        from src.cli.commands import cmd_ebnf

        class Args:
            pass

        a = Args()
        a.out = kw.get('out')
        a.asts = kw.get('asts')
        return cmd_ebnf(a)

    @line_magic
    def fuse_sandbox(self, line: str):
        pos, kw = self._parse_kv_args(line)
        from src.cli.commands import cmd_sandbox

        class Args:
            pass

        a = Args()
        a.op = kw.get('op', 'run')
        a.model = kw.get('model') or (pos[0] if pos else None)
        a.input = kw.get('input')
        a.runtime = kw.get('runtime', 'reference')
        a.zoo_root = kw.get('zoo_root')
        a.timeout = kw.get('timeout')
        return cmd_sandbox(a)

    @line_magic
    def fuse_run_cli(self, line: str):
        pos, kw = self._parse_kv_args(line)
        from src.cli.commands import cmd_run

        files = pos or []
        inp = kw.get('input')
        return cmd_run(files, input_path=inp, provider=kw.get('provider'), entry=kw.get('entry'))


def load_ipython_extension(ipython):
    ipython.register_magics(FuseMagics)


def unload_ipython_extension(ipython):
    # no-op for now
    pass
