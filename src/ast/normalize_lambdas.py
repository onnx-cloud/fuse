"""AST normalization helpers for converting inline `lambda` AST nodes
into top-level function declarations and replacing lambda occurrences
with references to the generated function name.

This transformation is intentionally simple and deterministic: each lambda
occurrence yields a new function named `__lambda_node_{n}`. The function
body is a single `return` statement returning the lambda `body`.

This keeps lowering simple: callers can refer to the generated function
by name (e.g., `body=__lambda_node_0`) and lowering of functions will
lower the lambda as a standard function.
"""

from typing import Any, Dict, List


def _walk_and_replace(node: Any, make_node) -> Any:
    """Recursively walk AST `node`, replacing lambda nodes with function
    name strings using `make_node(lambda_node)` which returns (node_name, node_decl).
    Returns the transformed node (does not mutate input).
    """
    if isinstance(node, dict):
        if "lambda" in node:
            node_name, node_decl = make_node(node["lambda"])
            return node_name, node_decl
        out = {}
        acc_nodes = []
        for k, v in node.items():
            new_v = _walk_and_replace(v, make_node)
            if isinstance(new_v, tuple) and isinstance(new_v[0], str):
                # Replaced a lambda: collect the generated node_decl and use its name
                node_name, node_decl = new_v
                out[k] = node_name
                acc_nodes.append(node_decl)
            else:
                out[k] = new_v
        if acc_nodes:
            # attach generated nodes using a special key to the caller to allow
            # the caller to place them in the declaration list.
            out["__generated_nodes__"] = acc_nodes
        return out
    elif isinstance(node, list):
        new_list = []
        acc_nodes = []
        for item in node:
            new_item = _walk_and_replace(item, make_node)
            if isinstance(new_item, tuple) and isinstance(new_item[0], str):
                node_name, node_decl = new_item
                new_list.append(node_name)
                acc_nodes.append(node_decl)
            else:
                new_list.append(new_item)
                # If new_item is a dict with generated nodes attached, pull them up
                if (
                    isinstance(new_item, dict)
                    and "__generated_nodes__" in new_item
                ):
                    acc_nodes.extend(new_item.pop("__generated_nodes__"))
        if acc_nodes:
            return (
                new_list + []
            )  # caller will pick up generated nodes via dict wrapping
        return new_list
    else:
        return node


def normalize_lambdas(
    declarations: List[Any], prefix: str = "__lambda_node_"
) -> List[Dict[str, Any]]:
    """Walk the top-level declarations and replace inline `lambda` nodes
    with generated function names. Generated function declarations are
    inserted immediately before the declaration that referenced them.

    Returns a new list of declarations.
    """
    out: List[Dict[str, Any]] = []
    counter = 0

    def make_node(lambda_node: Dict[str, Any]):
        nonlocal counter
        name = f"{prefix}{counter}"
        counter += 1
        # lambda_node: {"args": [...], "body": <expr_or_list>}
        args = lambda_node.get("args", [])
        body = lambda_node.get("body")
        # Build params list: each param is {'name': <ident>}
        params = [{"name": a, "type": None} for a in args]
        # Function body: single return statement
        node_body = [{"return": body}]
        node_decl = {
            "type": "node",
            "name": name,
            "params": params,
            "ret_type": None,
            "body": node_body,
        }
        # Mark helper functions as inline-only so they are not emitted as
        # top-level graphs; they are lowered on-demand when embedded as
        # GraphProto attributes (e.g., Loop/If bodies).
        node_decl["_inline_only"] = True
        return name, node_decl

    # Deduplicate identical lambdas so multiple occurrences reuse the
    # same generated helper function name (keeps lowering deterministic).
    seen: Dict[str, str] = {}
    for decl in declarations:
        if not isinstance(decl, dict):
            out.append(decl)
            continue
        # Walk the declaration and collect generated functions
        generated_nodes: List[Dict[str, Any]] = []

        def walker(node: Any) -> Any:
            # We reuse _walk_and_replace but adapt to collect nodes
            if isinstance(node, dict):
                if "lambda" in node:
                    # Use a lightweight text key to detect identical lambda shapes
                    key = repr(node["lambda"])
                    if key in seen:
                        return seen[key]
                    node_name, node_decl = make_node(node["lambda"])
                    seen[key] = node_name
                    generated_nodes.append(node_decl)
                    return node_name
                new_d = {}
                for k, v in node.items():
                    new_v = walker(v)
                    new_d[k] = new_v
                return new_d
            elif isinstance(node, list):
                return [walker(i) for i in node]
            else:
                return node

        new_decl = walker(decl)
        # Insert any generated functions immediately before this decl
        for node in generated_nodes:
            out.append(node)
        out.append(new_decl)
    return out
