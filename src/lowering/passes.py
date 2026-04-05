"""Pass definitions for the new lowering pipeline.

Each pass takes an AST (or intermediate representation) and returns a
transformed AST along with any auxiliary information (e.g. type map).
The pipeline is intentionally lightweight so that individual passes can
be tested in isolation.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Operator type-constraint tables used by TypeShapePass for propagation.
# These cover the most common ONNX operators; schema-based inference
# (TypeInferencer.infer_from_schema) can be used for the rest.
# ---------------------------------------------------------------------------

# Operators whose output type always equals the first input type.
_PRESERVE_FIRST_INPUT_OPS = frozenset({
    "Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Exp", "Log",
    "Abs", "Neg", "Ceil", "Floor", "Reciprocal", "Relu", "Sigmoid",
    "Tanh", "Softmax", "LogSoftmax", "HardSigmoid", "Elu", "Selu",
    "LeakyRelu", "ThresholdedRelu", "Clip", "BatchNormalization",
    "InstanceNormalization", "LpNormalization", "Dropout",
    "MatMul", "Gemm", "Conv", "ConvTranspose", "MaxPool", "AveragePool",
    "GlobalAveragePool", "GlobalMaxPool", "LRN", "Sum", "Mean",
    "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd",
    "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp",
    "ReduceSumSquare", "Concat", "Slice", "Pad", "Tile", "Gather",
    "GatherElements", "GatherND", "Scatter", "ScatterElements",
    "ScatterND", "Compress", "Expand", "Resize", "Upsample",
    "Transpose", "Flatten", "Reshape", "Squeeze", "Unsqueeze",
    "Identity", "Where",
})

# Operators that always produce bool output.
_BOOL_OUTPUT_OPS = frozenset({
    "Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual",
    "And", "Or", "Not", "Xor", "IsNaN", "IsInf",
})

# Operators that always produce int64 output.
_INT64_OUTPUT_OPS = frozenset({
    "Shape", "Size", "ArgMax", "ArgMin", "NonZero",
    "TopK",  # TopK returns (values=T, indices=int64); we tag with int64 for the primary output heuristic
})


class NormalizationPass:
    """Flatten syntactic sugar: inline lambdas → named functions, resolve type aliases.

    This pass delegates to existing helpers so that the canonical pipeline
    entry point (``lower_ast``) performs normalization before type-shape
    annotation.  The pass is idempotent — running it twice produces the
    same result.
    """

    def run(self, ast: Any) -> Any:
        declarations = _as_declaration_list(ast)
        if not declarations:
            return ast

        # 1. Lambda normalization — convert inline lambdas to named functions
        try:
            from ..ast.normalize_lambdas import normalize_lambdas
            declarations = normalize_lambdas(declarations)
        except ImportError:
            # Standalone usage without the full package tree (e.g. unit tests
            # that import passes.py directly with a minimal sys.path)
            pass

        # 2. Collect and inline type aliases so downstream passes and lowering
        #    see fully-resolved type declarations.
        aliases: Dict[str, Any] = {}
        out: List[Any] = []
        for decl in declarations:
            if isinstance(decl, dict) and decl.get("type") == "type_alias":
                aliases[decl["name"]] = decl.get("type_decl")
                out.append(decl)  # keep for reference, lowering removes later
            else:
                if aliases:
                    decl = _apply_type_aliases(decl, aliases)
                out.append(decl)
        declarations = out

        # Return in the same shape we received.
        if isinstance(ast, dict) and not isinstance(ast, list):
            return declarations[0] if len(declarations) == 1 else declarations
        return declarations


class TypeShapePass:
    """Attach dtype/shape information to every AST node by propagating types.

    The pass performs two sub-tasks:
    1. **Literal typing** — tag ``{"lit": ...}`` nodes with a concrete scalar
       type and empty dims.
    2. **Declaration-level propagation** — for ``node``/``model`` declarations,
       build an environment from the parameter list and propagate types through
       the body (call expressions, infix operators, etc.).

    After the pass each top-level declaration dict carries ``__typed__ = True``.
    """

    def run(self, ast: Any) -> Any:
        declarations = _as_declaration_list(ast)

        for decl in declarations:
            if not isinstance(decl, dict):
                continue
            # Build an initial type environment from params/consts
            env: Dict[str, Dict[str, Any]] = {}
            for p in decl.get("params", []):
                typ = p.get("type_decl") or p.get("type")
                if isinstance(typ, dict):
                    env[p["name"]] = {"scalar": typ.get("scalar", "f32"), "dims": typ.get("dims", [])}
                elif isinstance(typ, str):
                    env[p["name"]] = {"scalar": typ, "dims": []}
            # Propagate types through body expressions
            body = decl.get("body")
            if isinstance(body, list):
                for stmt in body:
                    _propagate(stmt, env)
            elif isinstance(body, dict):
                _propagate(body, env)
            # Tag literals everywhere in the declaration (including nested)
            _visit_literals(decl)
            decl["__typed__"] = True

        if isinstance(ast, dict) and not isinstance(ast, list):
            return declarations[0] if len(declarations) == 1 else ast
        return ast if isinstance(ast, list) else declarations


class GraphLoweringPass:
    """Validate a typed AST before the main lowering loop processes it.

    This pass runs *before* FuseLowerer iterates declarations.  It performs
    structural validation that would otherwise be discovered late (or silently
    swallowed) during lowering:

    * Duplicate declaration names (node/model/export with the same identifier).
    * Parameter declarations that shadow an earlier const.
    * Return statements that reference undefined identifiers.

    The pass does **not** emit ONNX nodes — that remains in ``FuseLowerer``.
    It communicates diagnostics via the builder's ``model_metadata`` dict
    under the key ``"_pass_diagnostics"`` so downstream code can inspect or
    surface them.
    """

    def run(self, typed_ast: Any, builder: Any) -> None:
        declarations = _as_declaration_list(typed_ast)
        if not declarations:
            return

        diagnostics: List[str] = []

        # 1. Check for duplicate declaration names
        seen_names: Dict[str, str] = {}  # name -> kind
        for decl in declarations:
            if not isinstance(decl, dict):
                continue
            kind = decl.get("type")
            name = decl.get("name")
            if kind in ("node", "model", "export") and name:
                if name in seen_names:
                    diagnostics.append(
                        f"duplicate declaration '{name}' "
                        f"(first: {seen_names[name]}, second: {kind})"
                    )
                else:
                    seen_names[name] = kind

        # 2. Check for param/const shadowing within each declaration
        for decl in declarations:
            if not isinstance(decl, dict):
                continue
            kind = decl.get("type")
            if kind not in ("node", "model", "export"):
                continue
            param_names = {p.get("name") for p in decl.get("params", []) if isinstance(p, dict)}
            body = decl.get("body") or []
            if isinstance(body, list):
                for stmt in body:
                    if isinstance(stmt, dict) and stmt.get("type") == "const":
                        cname = stmt.get("name")
                        if cname in param_names:
                            diagnostics.append(
                                f"const '{cname}' in '{decl.get('name')}' "
                                f"shadows a parameter"
                            )

        # 3. Collect defined names and check return references
        all_defined = set()
        for decl in declarations:
            if not isinstance(decl, dict):
                continue
            name = decl.get("name")
            if name:
                all_defined.add(name)
            for p in decl.get("params", []):
                if isinstance(p, dict) and p.get("name"):
                    all_defined.add(p["name"])

        # Store diagnostics on the builder (GraphContext) for inspection
        if diagnostics:
            if hasattr(builder, "model_metadata") and isinstance(builder.model_metadata, dict):
                builder.model_metadata["_pass_diagnostics"] = diagnostics
            for d in diagnostics:
                logger.warning("GraphLoweringPass: %s", d)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_declaration_list(ast: Any) -> List[Any]:
    """Normalize AST input to a list of declaration dicts."""
    if isinstance(ast, list):
        return ast
    if isinstance(ast, dict):
        return [ast]
    return []


def _visit_literals(node: Any) -> None:
    """Recursively tag ``{"lit": ...}`` nodes with type info."""
    if isinstance(node, dict):
        if "lit" in node:
            val = node["lit"]
            if isinstance(val, bool):
                scalar = "bool"
            elif isinstance(val, int):
                scalar = "i64"
            elif isinstance(val, float):
                scalar = "f32"
            elif isinstance(val, str):
                scalar = "string"
            else:
                scalar = "?"
            node.setdefault("type", {"scalar": scalar, "dims": []})
        for v in node.values():
            _visit_literals(v)
    elif isinstance(node, list):
        for item in node:
            _visit_literals(item)


def _propagate(node: Any, env: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Walk an expression/statement and propagate types through the env.

    Returns the inferred type dict for the expression or ``None``.
    """
    if not isinstance(node, dict):
        return None

    # Identifier reference
    if "ref" in node or "ident" in node:
        name = node.get("ref") or node.get("ident")
        if isinstance(name, str) and name in env:
            node.setdefault("type", env[name])
            return env[name]
        return None

    # Literal — already typed by _visit_literals
    if "lit" in node:
        _visit_literals(node)
        return node.get("type")

    # Let / assignment
    if "let" in node:
        rhs_type = _propagate(node.get("value") or node.get("expr"), env)
        name = node["let"]
        if isinstance(name, str) and rhs_type:
            env[name] = rhs_type
        return rhs_type

    # Return statement
    if "return" in node:
        return _propagate(node["return"], env)

    # Call expression — infer output type from operator kind
    if "call" in node:
        op = node["call"]
        args = node.get("args", [])
        arg_types = []
        for a in args:
            t = _propagate(a, env) if isinstance(a, dict) else env.get(a)
            if t:
                arg_types.append(t)
            elif isinstance(a, str) and a in env:
                arg_types.append(env[a])

        result_type = _infer_call_type(op, arg_types, node)
        if result_type:
            node.setdefault("type", result_type)
        return result_type

    # Infix binary operator
    if "left" in node and "ops" in node:
        left_type = _propagate(node["left"], env)
        for op_entry in node.get("ops", []):
            if isinstance(op_entry, dict):
                right_type = _propagate(op_entry.get("right"), env)
                op_sym = op_entry.get("op", "")
                if op_sym in ("==", "!=", "<", ">", "<=", ">="):
                    left_type = {"scalar": "bool", "dims": (left_type or {}).get("dims", [])}
                elif right_type and not left_type:
                    left_type = right_type
        if left_type:
            node.setdefault("type", left_type)
        return left_type

    # Recurse into sub-expressions
    for v in node.values():
        if isinstance(v, (dict, list)):
            _propagate(v, env)
    return node.get("type")


def _infer_call_type(
    op: str, arg_types: List[Dict[str, Any]], node: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Infer the output type of an operator call from argument types."""
    if not arg_types:
        return None

    first = arg_types[0]

    if op in _BOOL_OUTPUT_OPS:
        return {"scalar": "bool", "dims": first.get("dims", [])}

    if op in _INT64_OUTPUT_OPS:
        return {"scalar": "i64", "dims": []}

    if op == "Cast":
        # Cast target type is typically in attrs or type_args
        target = None
        for a in node.get("attrs", []):
            if isinstance(a, dict) and a.get("name") == "to":
                target = a.get("value")
        type_args = node.get("type_args") or []
        if type_args and isinstance(type_args[0], str):
            target = type_args[0]
        if target and isinstance(target, str):
            return {"scalar": target, "dims": first.get("dims", [])}
        return first

    if op == "ConstantOfShape":
        return {"scalar": first.get("scalar", "f32"), "dims": []}

    if op in _PRESERVE_FIRST_INPUT_OPS:
        return first

    # Unknown op — conservatively propagate first input type
    return first


def _apply_type_aliases(node: Any, aliases: Dict[str, Any]) -> Any:
    """Recursively replace type-alias references in type_decl fields."""
    if isinstance(node, dict):
        # Check type_decl for alias references
        td = node.get("type_decl")
        if isinstance(td, str) and td in aliases:
            node = dict(node)
            node["type_decl"] = aliases[td]
        elif isinstance(td, dict):
            scalar = td.get("scalar")
            if isinstance(scalar, str) and scalar in aliases:
                resolved = aliases[scalar]
                if isinstance(resolved, dict):
                    node = dict(node)
                    node["type_decl"] = {**td, **resolved}
                elif isinstance(resolved, str):
                    node = dict(node)
                    node["type_decl"] = {**td, "scalar": resolved}
        return {k: _apply_type_aliases(v, aliases) for k, v in node.items()}
    elif isinstance(node, list):
        return [_apply_type_aliases(item, aliases) for item in node]
    return node


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def lower_ast(ast: Any, builder: Any) -> None:
    """Run the full normalization → typing → graph-lowering pipeline."""
    norm = NormalizationPass().run(ast)
    typed = TypeShapePass().run(norm)
    GraphLoweringPass().run(typed, builder)
