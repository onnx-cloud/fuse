"""ONNX -> Fuse conversion.

Today this is intentionally conservative: it generates a Fuse wrapper that
imports the ONNX model as a variant and exposes it as a Fuse `model`.

This avoids having to fully decompile every ONNX operator into Fuse syntax.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import onnx
from src.graph_context import onnx_dtype_to_fuse
from src.onnx_opset import get_model_default_opset


@dataclass(frozen=True)
class FuseSignature:
    name: str
    inputs: List[Tuple[str, str, List[str]]]
    outputs: List[Tuple[str, str, List[str]]]
    opset: int


def _dim_to_fuse(d: onnx.TensorShapeProto.Dimension, fallback: str) -> str:
    if d.HasField("dim_value"):
        return str(int(d.dim_value))
    if d.HasField("dim_param") and d.dim_param:
        # Grammar IDENT allows letters/underscore then alnum/_/.
        # ONNX dim_param is typically a valid identifier.
        return str(d.dim_param)
    return fallback


def _tensor_type_to_fuse(
    vi: onnx.ValueInfoProto, unknown_dim: str = "_"
) -> Tuple[str, List[str]]:
    tt = vi.type.tensor_type
    scalar = onnx_dtype_to_fuse(int(tt.elem_type))
    dims: List[str] = []
    if tt.HasField("shape"):
        for idx, d in enumerate(tt.shape.dim):
            dims.append(_dim_to_fuse(d, unknown_dim))
    return scalar, dims


def get_fuse_signature(
    model: onnx.ModelProto, name: Optional[str] = None
) -> FuseSignature:
    graph = model.graph
    opset = get_model_default_opset(model) or 18

    sig_name = name or (graph.name if graph.name else "imported")

    inputs: List[Tuple[str, str, List[str]]] = []
    for vi in graph.input:
        scalar, dims = _tensor_type_to_fuse(vi)
        inputs.append((vi.name, scalar, dims))

    # Support single- and multi-output models. For multi-output models
    # return the complete list of outputs and let the wrapper use a tuple
    # return type (e.g., `-> (t1, t2)`) so existing inspect flows can handle
    # them uniformly.
    outs: List[Tuple[str, str, List[str]]] = []
    for out_vi in graph.output:
        out_scalar, out_dims = _tensor_type_to_fuse(out_vi)
        outs.append((out_vi.name, out_scalar, out_dims))

    return FuseSignature(
        name=sig_name,
        inputs=inputs,
        outputs=outs,
        opset=int(opset),
    )


def onnx_to_fuse(
    onnx_path: str | Path,
    *,
    module: str = "external.onnx",
    import_name: Optional[str] = None,
    import_version: float = 1.0,
    alias: str = "Imported",
    wrapper_name: Optional[str] = None,
    embed_absolute_path: bool = True,
) -> str:
    path = Path(onnx_path)
    model = onnx.load(str(path))
    sig = get_fuse_signature(model, name=wrapper_name or path.stem)

    imp_name = import_name or f"local.{path.stem}"
    file_path = str(path.resolve() if embed_absolute_path else path)

    def fmt_tensor(scalar: str, dims: List[str]) -> str:
        if dims:
            return f"<{scalar}>[{', '.join(dims)}]"
        # Omit empty `[]` for scalar tensors to match Fuse type syntax
        return f"<{scalar}>"

    # Use simplified local parameter names (take the last qualified segment)
    # to produce concise and parser-friendly Fuse signatures. Ensure uniqueness
    # by appending numeric suffixes when necessary.
    local_names = {}
    used = set()
    params_parts = []
    args_parts = []
    for (orig, scalar, dims) in sig.inputs:
        base = orig.split(".")[-1]
        candidate = base
        i = 1
        while candidate in used:
            i += 1
            candidate = f"{base}_{i}"
        used.add(candidate)
        local_names[orig] = candidate
        params_parts.append(f"{candidate}: {fmt_tensor(scalar, dims)}")
        args_parts.append(candidate)
    params_src = ", ".join(params_parts)

    # Support either a single output or multiple outputs via a tuple return
    if len(sig.outputs) == 1:
        ret_src = fmt_tensor(sig.outputs[0][1], sig.outputs[0][2])
    else:
        ret_src = "(" + ", ".join(fmt_tensor(s, d) for (_, s, d) in sig.outputs) + ")"

    # Call the imported variant using positional args in the source order
    args_src = ", ".join(args_parts)

    # We use a single local variant so `fuse import` can fuse it and calls can be wired.
    return (
        f"@fuse 0.7\n"
        f"@opset onnx {sig.opset}\n"
        f"@domain {module}\n\n"
        f"@import {imp_name} @{import_version} as {alias} {{\n"
        f'  @variant default file="{file_path}" default\n'
        f"}}\n\n"
        f"model {sig.name}({params_src}) -> {ret_src} {{\n"
        f"  {alias}({args_src})\n"
        f"}}\n"
    )
