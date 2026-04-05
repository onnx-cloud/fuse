from typing import Dict, Optional

import onnx
from src.graph_context import GraphContext


def _get_domain_from_meta(meta: dict) -> str | None:
    if not isinstance(meta, dict):
        return None
    return meta.get("domain")


class InMemoryImportManager:
    """A simple in-memory import manager for tests.

    Provide a mapping `name` -> `onnx.ModelProto` (or bytes). The `fuse_import`
    method mirrors the side-effects of the disk-based ImportManager but avoids
    any filesystem or network access.
    """

    def __init__(self, models: Optional[Dict[str, onnx.ModelProto]] = None):
        self.models: Dict[str, onnx.ModelProto] = models or {}
        self.fused_signatures: Dict[str, Dict[str, list]] = {}
        self.loaded: Dict[str, onnx.ModelProto] = {}

    def add_model(self, name: str, model: onnx.ModelProto):
        self.models[name] = model

    def fuse_import(
        self,
        ctx: GraphContext,
        import_decl: Dict[str, any],
        variant_name: Optional[str] = None,
        refresh: bool = False,
    ):
        name = str(import_decl.get("name"))
        if name not in self.models:
            raise ValueError(f"InMemoryImportManager: unknown import '{name}'")
        imported_model = self.models[name]

        # Note: we do not perform variant selection or caching; callers
        # should prepare the import_decl accordingly in tests.

        # Compose prefix based on ctx.scope_prefix or module/domain metadata
        module_prefix = _get_domain_from_meta(ctx.model_metadata)
        prefix = (
            f"{ctx.scope_prefix}_"
            if getattr(ctx, "scope_prefix", None)
            else (f"{module_prefix}_" if module_prefix else "")
        )
        alias = import_decl.get("alias") or name
        aliased = f"{prefix}{alias}"

        # Build fused_signature and record
        input_infos = []
        for vi in imported_model.graph.input:
            ttype = vi.type.tensor_type
            dims = []
            for d in ttype.shape.dim:
                if d.HasField("dim_value"):
                    dims.append(int(d.dim_value))
                else:
                    dims.append(0)
            input_infos.append(
                {
                    "name": f"{aliased}_{vi.name}",
                    "elem_type": int(ttype.elem_type),
                    "dims": dims,
                }
            )
        outputs = [
            f"{aliased}_{vi.name}" for vi in imported_model.graph.output
        ]
        if not outputs and imported_model.graph.node:
            last = imported_model.graph.node[-1]
            if last.output:
                outputs = [f"{aliased}_{last.output[0]}"]
        self.fused_signatures[aliased] = {
            "inputs": [i["name"] for i in input_infos],
            "outputs": outputs,
            "input_infos": input_infos,
        }
        self.fused_signatures[alias] = self.fused_signatures[aliased]

        # Record nodes and initializers into ctx
        ctx.import_node_start[aliased] = len(ctx.nodes)
        ctx.import_node_start[alias] = ctx.import_node_start[aliased]

        # Prefix names in graph (simple: rename inputs/outputs/initializers and node names)
        for vi in (
            list(imported_model.graph.input)
            + list(getattr(imported_model.graph, "value_info", []))
            + list(imported_model.graph.output)
        ):
            if getattr(vi, "name", None):
                vi.name = f"{aliased}_{vi.name}"
        for init in list(getattr(imported_model.graph, "initializer", [])):
            if getattr(init, "name", None):
                init.name = f"{aliased}_{init.name}"
        for node in imported_model.graph.node:
            node.name = (
                f"{aliased}_{node.name}"
                if node.name
                else ctx._next_node_name(aliased)
            )
            node.input[:] = [f"{aliased}_{i}" for i in node.input]
            node.output[:] = [f"{aliased}_{o}" for o in node.output]

        self.loaded[aliased] = imported_model
        self.loaded[alias] = imported_model

        ctx.nodes.extend(imported_model.graph.node)
        ctx.initializers.update(
            {init.name: init for init in imported_model.graph.initializer}
        )
        for init in imported_model.graph.initializer:
            ctx.defined_values.add(init.name)
        for node in imported_model.graph.node:
            for o in node.output:
                ctx.defined_values.add(o)
