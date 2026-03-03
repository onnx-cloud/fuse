"""Pass definitions for the new lowering pipeline.

Each pass takes an AST (or intermediate representation) and returns a
transformed AST along with any auxiliary information (e.g. type map).
The pipeline is intentionally lightweight so that individual passes can
be tested in isolation.
"""
from typing import Any, Dict


class NormalizationPass:
    """Flatten syntactic sugar such as lambdas, imports, and type aliases.

    Future passes may normalize lambdas, expand imports, etc.
    """

    def run(self, ast: Any) -> Any:
        return ast


class TypeShapePass:
    """Attach dtype/shape information to every node by propagating types.

    For now this pass simply annotates the AST with a ``__typed__`` flag so
    downstream code knows the pass has executed.  A real implementation will
    traverse expressions and compute value info.
    """

    def run(self, ast: Any) -> Any:
        # simple propagation: assign type info to any literal expressions
        def visit(node: Any):
            if isinstance(node, dict):
                # literal forms used by parser
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
                    node["type"] = {"scalar": scalar, "dims": []}
                for v in node.values():
                    visit(v)
            elif isinstance(node, list):
                for item in node:
                    visit(item)
        visit(ast)

        if isinstance(ast, dict):
            ast["__typed__"] = True
        elif isinstance(ast, list):
            for entry in ast:
                if isinstance(entry, dict):
                    entry["__typed__"] = True
        return ast


class GraphLoweringPass:
    """Consume a typed AST and emit ONNX graph elements via a builder.

    This pass is where the registry and context stack come into play; the
    builder should expose only methods for creating nodes, values, and
    functions rather than raw ONNX manipulation.
    """

    def run(self, typed_ast: Dict[str, Any], builder: Any) -> None:
        # TODO: implement
        pass


# helper to run full pipeline

def lower_ast(ast: Dict[str, Any], builder: Any) -> None:
    norm = NormalizationPass().run(ast)
    typed = TypeShapePass().run(norm)
    GraphLoweringPass().run(typed, builder)
