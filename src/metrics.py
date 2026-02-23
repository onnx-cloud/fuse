"""Compute simple YAML-like metrics for Fuse source files and emitted ONNX models.

Exports:
- compute_metrics_for_file(path) -> dict
- format_metrics(metrics) -> str

Metrics include:
- ops: mapping op_type -> count
- total_nodes: int
- graphs: list of {name, nodes}
- weights: list of {name, trainable, dtype, shape, elements, bytes}
- total_bytes: int

This module intentionally avoids external deps and emits a small readable
YAML-like string for CLI display.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any
from math import prod

from src.parser import fuse_parser
from src.graph_context import as_tensor_type

# Defer importing CLI helpers to runtime to avoid import-time cycles.

# Approximate byte sizes per scalar dtype
BYTES_PER_DTYPE = {
    "f32": 4,
    "f64": 8,
    "f16": 2,
    "bf16": 2,
    "i8": 1,
    "i16": 2,
    "i32": 4,
    "i64": 8,
    "u8": 1,
    "u16": 2,
    "u32": 4,
    "u64": 8,
    "bool": 1,
    "complex64": 8,
    "complex128": 16,
}


import ast as _ast


def _eval_int_expr(s: str) -> Optional[int]:
    """Safely evaluate a simple integer arithmetic expression string.

    Supports integer literals, unary +/- and binary +, -, *, //, / operators.
    Returns integer result or None if expression contains unknown constructs or
    non-integer/unsafe results.
    """
    try:
        node = _ast.parse(s, mode="eval")
    except Exception:
        return None

    def _eval(n):
        if isinstance(n, _ast.Expression):
            return _eval(n.body)
        if isinstance(n, _ast.Constant):
            v = n.value
            if isinstance(v, int):
                return v
            return None
        # Compatibility for older AST node types: avoid referencing
        # `_ast.Num` directly (it is deprecated on Python >=3.12) to
        # prevent DeprecationWarning. Check by node class name instead.
        if type(n).__name__ == "Num":
            return int(getattr(n, "n", None))
        if isinstance(n, _ast.UnaryOp) and isinstance(n.op, (_ast.UAdd, _ast.USub)):
            val = _eval(n.operand)
            if val is None:
                return None
            return +val if isinstance(n.op, _ast.UAdd) else -val
        if isinstance(n, _ast.BinOp) and isinstance(n.op, (_ast.Add, _ast.Sub, _ast.Mult, _ast.Div, _ast.FloorDiv)):
            l = _eval(n.left)
            r = _eval(n.right)
            if l is None or r is None:
                return None
            if isinstance(n.op, _ast.Add):
                return l + r
            if isinstance(n.op, _ast.Sub):
                return l - r
            if isinstance(n.op, _ast.Mult):
                return l * r
            if isinstance(n.op, _ast.Div) or isinstance(n.op, _ast.FloorDiv):
                # require integer division
                if r == 0:
                    return None
                if l % r != 0:
                    return None
                return l // r
        return None

    try:
        res = _eval(node)
        if isinstance(res, int):
            return res
        return None
    except Exception:
        return None


def _num_elements(dims: List[Any]) -> Optional[int]:
    if not dims:
        return 1
    el = 1
    for d in dims:
        # allow integer literals
        if isinstance(d, int):
            el *= d
            continue
        # allow simple arithmetic expressions as strings
        if isinstance(d, str):
            v = _eval_int_expr(d)
            if v is None or v <= 0:
                return None
            el *= v
            continue
        return None
    return el


def _format_hr_bytes(n: int) -> str:
    # simple human-readable bytes
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n}{unit}"
        n = n // 1024
    return f"{n}TB"


def _elt_count_from_shape(shape: List[Optional[int]]) -> Optional[int]:
    if not shape:
        return 1
    out = 1
    for d in shape:
        if d is None:
            return None
        out *= d
    return out


def _broadcast_element_count(shapes: List[Optional[List[Optional[int]]]]) -> Optional[int]:
    """Compute element count of broadcasted shape from input shapes if possible.

    Rules (conservative):
    - If any dimension is unknown (None) and any other operand has a known >1 dim
      in that axis, we cannot resolve and return None.
    - Scalars (empty shapes) broadcast to other shapes.
    - If all axes are resolvable to concrete ints (or implied 1), return product.
    """
    if not shapes:
        return None
    # Normalize shapes: treat None list as unknown shape
    ranks = [len(s) for s in shapes if s is not None]
    max_rank = max(ranks) if ranks else 0
    total = 1
    for axis in range(max_rank):
        dims = []
        for s in shapes:
            if s is None:
                dims.append(None)
            else:
                # align right
                idx = len(s) - max_rank + axis
                if idx < 0:
                    dims.append(1)
                else:
                    dims.append(s[idx])
        # resolve this axis
        # if any dim is None but others have >1, we cannot resolve
        known_non1 = [d for d in dims if d is not None and d != 1]
        if any(d is None for d in dims):
            if known_non1:
                return None
            # all unknowns or ones -> treat as 1
            axis_size = 1
        else:
            # all known
            # pick the max (broadcast rule) but ensure compatibility
            axis_size = max(dims)
            # validate that dims are either 1 or equal to axis_size
            for d in dims:
                if d not in (1, axis_size):
                    # incompatible shapes
                    return None
        total *= axis_size
    return total if total > 0 else None


def _bytes_for_shape_and_dtype(shape: List[Optional[int]], onnx_dtype: Optional[int]) -> Optional[int]:
    if shape is None or onnx_dtype is None:
        return None
    # convert onnx dtype int -> fuse dtype string
    from src.graph_context import onnx_dtype_to_fuse

    fuse_dtype = onnx_dtype_to_fuse(onnx_dtype)
    bpe = BYTES_PER_DTYPE.get(fuse_dtype, 4)
    el = _elt_count_from_shape(shape)
    if el is None:
        return None
    return el * bpe


def _value_info_map(model) -> Dict[str, Tuple[List[Optional[int]], Optional[int]]]:
    """Return mapping name -> (shape-list (ints or None), onnx_dtype-int)"""
    out: Dict[str, Tuple[List[Optional[int]], Optional[int]]] = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        name = vi.name
        try:
            tt = vi.type.tensor_type
            elem = tt.elem_type if tt.HasField("elem_type") else None
            dims = []
            if tt.HasField("shape"):
                for d in tt.shape.dim:
                    if d.HasField("dim_value"):
                        dims.append(int(d.dim_value))
                    else:
                        dims.append(None)
            out[name] = (dims, elem)
        except Exception:
            # Skip non-tensor value_info
            continue
    # Also ensure initializers appear (they may not have value_info entries)
    for init in model.graph.initializer:
        if init.name not in out:
            dims = list(init.dims)
            out[init.name] = (dims, init.data_type)
    return out


def _init_map(model):
    return {init.name: init for init in model.graph.initializer}


def _node_identifier(n, idx: int) -> str:
    # Prefer explicit node name, else op_type with index for deterministic id
    if getattr(n, "name", ""):
        return n.name
    return f"{n.op_type}_{idx}"


def _safe_int(v: Optional[int]) -> Optional[int]:
    return int(v) if v is not None else None


def compute_metrics_for_file(path: str) -> Dict[str, Any]:
    """Compute extended metrics (FLOPs, bytes moved, compute/memory ratio).

    Deterministic and explainable: relies on lowering + ONNX shape inference.
    """
    from src import cli_helpers
    ast = cli_helpers.parse_fuse_file(path)

    # Collect AST-declared weights (unchanged behavior)
    weights: List[Dict[str, Any]] = []
    for d in ast or []:
        if not isinstance(d, dict):
            continue
        if d.get("type") == "param":
            name = d.get("name")
            typ = d.get("type_decl")
            t = as_tensor_type(typ)
            dtype = t.get("scalar") or "f32"
            dims = t.get("dims") or []
            elements = _num_elements(dims)
            bpe = BYTES_PER_DTYPE.get(dtype, 4)
            size_bytes = elements * bpe if elements is not None else None
            weights.append(
                {
                    "name": name,
                    "trainable": d.get("trainable") is True,
                    "dtype": dtype,
                    "shape": dims,
                    "elements": elements,
                    "bytes": size_bytes,
                    "bytes_hr": _format_hr_bytes(size_bytes) if size_bytes is not None else None,
                }
            )

    # Lower to ONNX
    from src.lowering import FuseLowerer
    import onnx

    fl = FuseLowerer(emit_training=False)
    try:
        # Do not provide source_file here to avoid triggering namespacing checks
        # for small ad-hoc inputs used by tests and tooling (they may not
        # include a top-level @module). When callers need strict namespacing
        # validation, invoke lowering explicitly with a source file.
        model = fl.lower(ast)
    except Exception as e:
        return {
            "ops": {},
            "total_nodes": 0,
            "graphs": [],
            "weights": weights,
            "total_bytes": sum(w.get("bytes") or 0 for w in weights),
            "error": str(e),
        }

    # Deterministically serialize model to compute a hash
    import hashlib

    try:
        data = model.SerializeToString(deterministic=True)
        model_hash = hashlib.sha256(data).hexdigest()
    except Exception:
        model_hash = ""

    # Attempt shape inference (best-effort). Run in a subprocess to avoid
    # native crashes in the onnx C++ extension (which can cause SIGSEGV and
    # cannot be caught by Python). If subprocess inference fails, fall back to
    # the original model.
    def _safe_infer_shapes(model):
        import tempfile
        import subprocess
        import sys
        import os

        inferred = None
        in_name = None
        out_name = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as inf:
                in_name = inf.name
                inf.write(model.SerializeToString(deterministic=True))
            out_name = in_name + ".inferred.onnx"
            # Run a short-lived subprocess that performs shape inference and
            # writes the inferred model to out_name. Failures / crashes result
            # in non-zero exit codes or missing output file.
            cmd = [
                sys.executable,
                "-c",
                (
                    "import onnx,sys; m=onnx.load(sys.argv[1]); "
                    "m2=onnx.shape_inference.infer_shapes(m); "
                    "open(sys.argv[2],'wb').write(m2.SerializeToString())"
                ),
                in_name,
                out_name,
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=15)
            if proc.returncode == 0 and out_name and os.path.exists(out_name):
                inferred = onnx.load(out_name)
        except Exception:
            inferred = None
        finally:
            try:
                if in_name and os.path.exists(in_name):
                    os.remove(in_name)
            except Exception:
                pass
            try:
                if out_name and os.path.exists(out_name):
                    os.remove(out_name)
            except Exception:
                pass
        return inferred or model

    inferred = _safe_infer_shapes(model)

    vmap = _value_info_map(inferred)
    inits = _init_map(inferred)

    ops: Dict[str, int] = {}
    per_node: List[Dict[str, Any]] = []
    total_flops = 0
    total_bytes_moved = 0

    # Compute total parameters (sum of elements for declared weights)
    total_parameters = 0
    for w in weights:
        if w.get("elements") is not None:
            total_parameters += int(w.get("elements"))

    # iterate nodes deterministically by position
    for idx, n in enumerate(list(inferred.graph.node)):
        ops[n.op_type] = ops.get(n.op_type, 0) + 1

        node_id = _node_identifier(n, idx)
        inputs = []
        input_shapes = {}
        input_bytes = 0
        for nm in n.input:
            if nm in vmap:
                shape, dtype = vmap[nm]
                input_shapes[nm] = {"shape": shape, "dtype": dtype}
                b = _bytes_for_shape_and_dtype(shape, dtype)
                if b is not None:
                    input_bytes += b
            elif nm in inits:
                init = inits[nm]
                shape = list(init.dims)
                dtype = init.data_type
                input_shapes[nm] = {"shape": shape, "dtype": dtype}
                b = _bytes_for_shape_and_dtype(shape, dtype)
                if b is not None:
                    input_bytes += b
            else:
                input_shapes[nm] = {"shape": None, "dtype": None}

        output_shapes = {}
        output_bytes = 0
        for nm in n.output:
            if nm in vmap:
                shape, dtype = vmap[nm]
                output_shapes[nm] = {"shape": shape, "dtype": dtype}
                b = _bytes_for_shape_and_dtype(shape, dtype)
                if b is not None:
                    output_bytes += b
            else:
                output_shapes[nm] = {"shape": None, "dtype": None}

        # Compute FLOPs using supported op formulas
        flops = None
        formula = None
        try:
            if n.op_type in ("MatMul", "Gemm"):
                # A:(M,K), B:(K,N) -> 2*M*N*K
                a = n.input[0]
                b = n.input[1]
                ashape = vmap.get(a, (None, None))[0]
                bshape = vmap.get(b, (None, None))[0]
                if ashape and bshape and len(ashape) >= 2 and len(bshape) >= 2:
                    M = ashape[-2]
                    K = ashape[-1]
                    N = bshape[-1]
                    if None not in (M, K, N):
                        flops = 2 * M * N * K
                        formula = f"2*{M}*{N}*{K}"
            elif n.op_type == "Conv":
                # X: N,Cin,Hin,Win ; W: Cout,Cin/group,KH,KW ; Y: N,Cout,Hout,Wout
                x = n.input[0]
                wnm = n.input[1]
                xshape = vmap.get(x, (None, None))[0]
                wshape = vmap.get(wnm, (None, None))[0]
                y = n.output[0] if n.output else None
                yshape = vmap.get(y, (None, None))[0] if y else None
                # group attr
                group = 1
                for a in n.attribute:
                    if a.name == "group":
                        group = int(a.i)
                if xshape and wshape and yshape:
                    N = xshape[0]
                    Cout = wshape[0]
                    Cin_per_group = wshape[1]
                    KH = wshape[2]
                    KW = wshape[3]
                    Hout = yshape[2]
                    Wout = yshape[3]
                    if None not in (N, Cout, Cin_per_group, KH, KW, Hout, Wout):
                        flops = 2 * N * Cout * Hout * Wout * Cin_per_group * KH * KW
                        formula = f"2*{N}*{Cout}*{Hout}*{Wout}*{Cin_per_group}*{KH}*{KW}"
            elif n.op_type in ("Add", "Sub", "Mul", "Div", "Relu", "Sigmoid", "Tanh"):
                # elementwise unary/binary: cost ~ output elements
                # prefer output shape, else try to compute broadcasted shape from inputs
                el = None
                if n.output:
                    out0 = n.output[0]
                    shape = vmap.get(out0, (None, None))[0]
                    if shape is not None:
                        el = _elt_count_from_shape(shape)
                if el is None:
                    # attempt from inputs
                    in_shapes = []
                    for inp in n.input:
                        s = vmap.get(inp, (None, None))[0]
                        if s is None and inp in inits:
                            # initializer dims are concrete
                            init = inits.get(inp)
                            s = list(init.dims) if init is not None else None
                        in_shapes.append(s)
                    el = _broadcast_element_count(in_shapes)
                if el is not None:
                    flops = el
                    formula = f"1*{el}"
            else:
                # fallback: try using output elements as approximate cost
                if n.output:
                    out0 = n.output[0]
                    shape = vmap.get(out0, (None, None))[0]
                    el = _elt_count_from_shape(shape) if shape is not None else None
                    if el is not None:
                        flops = el
                        formula = f"approx1*{el}"
        except Exception:
            flops = None
            formula = None

        # bytes moved: sum inputs + outputs + weight reads (weights are in initializers)
        bytes_moved = None
        if input_bytes is not None and output_bytes is not None:
            bytes_moved = input_bytes + output_bytes

        cm_ratio = None
        if flops is not None and bytes_moved and bytes_moved > 0:
            cm_ratio = flops / bytes_moved

        per_node.append(
            {
                "id": node_id,
                "op": n.op_type,
                "inputs": input_shapes,
                "outputs": output_shapes,
                "flops": flops,
                "flops_formula": formula,
                "bytes_moved": bytes_moved,
                "compute_to_memory": cm_ratio,
            }
        )

        if flops is not None:
            total_flops += flops
        if bytes_moved is not None:
            total_bytes_moved += bytes_moved

    total_nodes = len(list(inferred.graph.node))

    total_bytes = sum(w.get("bytes") or 0 for w in weights)
    return {
        "method": "lowering+shape_inference",
        "model_hash": model_hash,
        "ops": ops,
        "total_nodes": total_nodes,
        "graphs": [{"name": inferred.graph.name or "model", "nodes": total_nodes}],
        "weights": weights,
        "flops": {"total": total_flops},
        "bytes_moved": {"total": total_bytes_moved},
        "total_bytes_moved": total_bytes_moved,
        "per_node": per_node,
        "total_bytes": total_bytes,
        "total_bytes_hr": _format_hr_bytes(total_bytes),
        "total_parameters": total_parameters,
        "total_parameters_hr": f"{total_parameters:,}",
    }


def format_metrics(m: Dict[str, Any]) -> str:
    """Return a YAML-like human-readable string for extended metrics dict."""
    lines: List[str] = []
    # 'source' intentionally omitted to avoid leaking local filesystem paths
    lines.append("method: " + str(m.get("method", "")))
    if m.get("model_hash"):
        lines.append(f"model_hash: {m.get('model_hash')}")
    if m.get("model_metadata") is not None:
        lines.append("model_metadata:")
        for k, v in sorted(m.get("model_metadata", {}).items()):
            lines.append(f"  {k}: {v}")
    lines.append("ops:")
    if m.get("ops"):
        for k, v in sorted(m.get("ops", {}).items(), key=lambda x: x[0]):
            lines.append(f"  {k}: {v}")
    else:
        lines.append("  {}")
    # Group numeric totals together for easy scanning
    lines.append("metrics:")
    lines.append(f"  total_nodes: {m.get('total_nodes', 0)}")
    lines.append(f"  total_parameters: {m.get('total_parameters', 0)}")
    if m.get("total_parameters_hr") is not None:
        lines.append(f"  total_parameters_hr: {m.get('total_parameters_hr')}")
    lines.append(f"  total_bytes_moved: {m.get('total_bytes_moved', 0)}")
    lines.append(f"  total_bytes: {m.get('total_bytes', 0)}")
    if m.get("total_bytes_hr"):
        lines.append(f"  total_bytes_hr: {m.get('total_bytes_hr')}")
    lines.append(f"  total_flops: {m.get('flops', {}).get('total', 0)}")

    lines.append("graphs:")
    for g in m.get("graphs", []):
        lines.append(f"  - name: {g.get('name')}")
        lines.append(f"    nodes: {g.get('nodes')}")
    lines.append("weights:")
    for w in m.get("weights", []):
        lines.append(f"  - name: {w.get('name')}")
        lines.append(f"    trainable: {str(w.get('trainable')).lower()}")
        lines.append(f"    dtype: {w.get('dtype')}")
        lines.append(f"    shape: {w.get('shape')}")
        lines.append(f"    elements: {w.get('elements')}")
        lines.append(f"    bytes: {w.get('bytes')}")
        if w.get("bytes_hr"):
            lines.append(f"    bytes_hr: {w.get('bytes_hr')}")
    lines.append("flops:")
    lines.append(f"  total: {m.get('flops', {}).get('total', 0)}")
    lines.append("per_node:")
    for n in m.get("per_node", []):
        lines.append(f"  - id: {n.get('id')}")
        lines.append(f"    op: {n.get('op')}")
        lines.append(f"    flops: {n.get('flops')}")
        if n.get("flops_formula"):
            lines.append(f"    flops_formula: {n.get('flops_formula')}")
        lines.append(f"    bytes_moved: {n.get('bytes_moved')}")
        if n.get("compute_to_memory") is not None:
            lines.append(f"    compute_to_memory: {n.get('compute_to_memory')}")
    lines.append(f"total_bytes: {m.get('total_bytes', 0)}")
    if m.get("total_bytes_hr"):
        lines.append(f"total_bytes_hr: {m.get('total_bytes_hr')}")
    return "\n".join(lines) + "\n"