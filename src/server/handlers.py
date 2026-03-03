import base64
import time
from typing import List

from .models import (
    LintRequest,
    LintResponse,
    CompileRequest,
    CompileResponse,
    DecompileRequest,
    DecompileResponse,
)

# Local imports from project
from src import parser


def _collect_symbols(ast) -> List[str]:
    out = []

    def _walk(x):
        if isinstance(x, dict):
            if x.get("type") in ("node", "model", "export") and x.get("name"):
                out.append(x.get("name"))
            for v in x.values():
                _walk(v)
        elif isinstance(x, list):
            for i in x:
                _walk(i)

    _walk(ast)
    return out


def lint_handler(req: LintRequest) -> LintResponse:
    start = time.time()
    try:
        ast = parser.fuse_parser.parse(req.source)
        valid = True
        errors = []
    except Exception as e:  # Parser/ParseError
        valid = False
        ast = None
        errors = [
            {
                "severity": "error",
                "message": str(e),
            }
        ]

    diag = {
        "parse_time_ms": (time.time() - start) * 1000,
        "symbols": _collect_symbols(ast) if ast is not None else [],
    }

    return LintResponse(valid=valid, warnings=[], errors=errors, diagnostics=diag)


def compile_handler(req: CompileRequest) -> CompileResponse:
    start = time.time()
    try:
        ast = parser.fuse_parser.parse(req.source)
    except Exception as e:
        return CompileResponse(
            success=False,
            errors=[{"phase": "parsing", "message": str(e)}],
            diagnostics={"parse_time_ms": (time.time() - start) * 1000},
        )

    try:
        # Lazy import lowering and onnx to avoid hard dependency at module import.
        from src.lowering.main import FuseLowerer

        fl = FuseLowerer()
        model = fl.lower(ast)
        # Serialize model
        try:
            raw = model.SerializeToString()
        except Exception:
            # best-effort: if model has no SerializeToString, return success but no onnx
            raw = None

        import base64

        onnx_b64 = base64.b64encode(raw).decode("ascii") if raw else None
        metadata = {
            "opset": getattr(model, "opset_import", None) or None,
            "producer": getattr(model, "producer_name", "fuse"),
            "nodes": len(model.graph.node) if hasattr(model, "graph") else None,
        }
        diagnostics = {
            "compile_time_ms": (time.time() - start) * 1000,
        }
        return CompileResponse(
            success=True,
            onnx=onnx_b64,
            metadata=metadata,
            diagnostics=diagnostics,
            warnings=[],
        )
    except Exception as e:
        return CompileResponse(
            success=False,
            errors=[{"phase": "lowering", "message": str(e)}],
            diagnostics={"compile_time_ms": (time.time() - start) * 1000},
        )


def decompile_handler(req: DecompileRequest) -> DecompileResponse:
    if not req.onnx:
        # validation error
        return DecompileResponse(success=False, errors=[{"message": "missing onnx"}])

    try:
        raw = base64.b64decode(req.onnx)
        # Try to import onnx and build a small fuse wrapper using decompile utilities
        import onnx
        from src.decompile import get_fuse_signature

        model = onnx.load_model_from_string(raw)
        sig = get_fuse_signature(model, name=None)

        # Build a minimal Fuse wrapper source (like onnx_to_fuse but in-memory)
        params_src = ", ".join(
            f"{name}: <{scalar}>[{', '.join(dims)}]" if dims else f"{name}: <{scalar}>"
            for (name, scalar, dims) in sig.inputs
        )
        ret_scalar = sig.output[1]
        ret_dims = sig.output[2]
        ret_src = f"<{ret_scalar}>[{', '.join(ret_dims)}]" if ret_dims else f"<{ret_scalar}>"
        ", ".join(name for (name, _, _) in sig.inputs)

        src = f"@fuse 0.7\n@opset onnx {sig.opset}\nmodel {sig.name}({params_src}) -> {ret_src} {{\n  /* imported model */\n}}\n"

        return DecompileResponse(success=True, source=src, metadata={"opset": sig.opset}, warnings=[])
    except Exception as e:
        return DecompileResponse(success=False, errors=[{"message": str(e)}])
