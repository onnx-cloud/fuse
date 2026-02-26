"""Helpers to assemble TrainingInfoProto entries from emitted optimizer nodes.

This module builds minimal algorithm GraphProto(s) containing optimizer nodes
and sets `update_binding` mappings so models lowered with --emit-training can
serialize ONNX TrainingInfo metadata compatible with validation tooling.
"""
from typing import Dict, List

# Use a simple id()-based dict for storing loss bindings for proto
# instances as TrainingInfoProto instances are not weakref-able.
import onnx
from onnx import helper
from src.graph_context import fuse_dtype_to_onnx, DEFAULT_SCALAR

TRAINING_DOMAIN = "ai.onnx.preview.training"

# Compatibility shim: some onnx versions' TrainingInfoProto do not include
# a `loss_binding` field. Provide a read-only `loss_binding` property on the
# message type that exposes an external per-instance list via a WeakKeyDict
# so we can persist bindings without mutating the proto instance.
_loss_bindings_map = {}
try:
    _pb = onnx.onnx_ml_pb2.TrainingInfoProto
    if not hasattr(_pb, "loss_binding"):
        def _get_loss_binding(self):
            try:
                return _loss_bindings_map.setdefault(id(self), [])
            except Exception:
                return []

        _pb.loss_binding = property(_get_loss_binding)
except Exception:
    # be tolerant if onnx internals are not available
    pass


def _strip_scope(name: str, ctx=None) -> str:
    # Use last dotted component to create compact local names for algorithm
    # graphs (e.g., "module.W.opt" -> "W.opt"). Also strip import aliases
    # that are prefixed with an underscore (e.g., "Imported_W.opt" -> "W.opt").
    if not isinstance(name, str):
        return name
    # Dotted scope has priority: strip scope from the left-hand component but
    # preserve trailing dotted qualifiers (e.g., "Imported_W.opt" -> "W.opt").
    if "." in name:
        parts = name.split(".")
        head = parts[0]
        tail = parts[1:]
        stripped_head = _strip_scope(head, ctx)
        return stripped_head + ("." + ".".join(tail) if tail else "")
    # If context is available, try to strip known import aliases like "Alias_..."
    if ctx is not None:
        # try to find an alias prefix present in the name
        for alias in list(getattr(ctx, "fused_signatures", {}).keys()):
            prefix = f"{alias}_"
            if name.startswith(prefix):
                return name[len(prefix) :]
    # Fallback: if the name contains a single underscore and looks like a prefixed name
    if "_" in name:
        parts = name.split("_", 1)
        # Heuristic: if the left part looks like an alias (alphanumeric), strip it
        if parts[0].isalnum():
            return parts[1]
    return name


def emit_training_info(ctx, grad_summary: Dict[str, List]) -> None:
    """Construct and append a single TrainingInfoProto when optimizer nodes
    were emitted.

    - `grad_summary` is expected to contain keys:
      - 'opt_updates': mapping param -> optimizer_output
      - 'optimizer_nodes': list of node names (as returned by ctx.add_node)
    """
    opt_updates: Dict[str, str] = grad_summary.get("opt_updates", {}) or {}
    opt_nodes: List[str] = grad_summary.get("optimizer_nodes", []) or []
    # If no optimizer nodes/updates were emitted but an explicit algorithm
    # graph was supplied by the lowering pass, still emit a TrainingInfoProto
    # so downstream tooling can find the declared algorithm.
    explicit_alg = None
    try:
        explicit_alg = ctx.model_metadata.get("training", {}).get("algorithm_graph")
    except Exception:
        explicit_alg = None
    if not opt_updates and not explicit_alg:
        return

    # Helper to resolve initializer names robustly (exact, stripped, qualified,
    # and imported-prefixed forms). Placed early so it can be used while
    # assembling algorithm initializers.
    def _find_initializer_key(ctx, p):
        if p in ctx.initializers:
            return p
        s = _strip_scope(p, ctx)
        if s in ctx.initializers:
            return s
        q = ctx.qualify_name(p)
        if q in ctx.initializers:
            return q
        for k in ctx.initializers:
            if k.endswith("_%s" % p) or k.endswith(".%s" % p) or k == p:
                return k
        return p

    # Collect NodeProtos in the original emission order (preserve ordering
    # determinism by building a set once)
    opt_set = set(opt_nodes)
    nodes = [n for n in ctx.nodes if n.name in opt_set]
    # If there are no optimizer nodes and no explicit algorithm graph
    # provided, there's nothing to do.
    if not nodes and explicit_alg is None:
        return

    # Build local name mapping and algorithm inputs/outputs
    alg_nodes = []
    alg_inputs = {}
    alg_outputs = {}
    alg_initializers = {}

    # trainable map from model metadata to avoid copying model weights
    trainables = ctx.model_metadata.get("trainables", {}) or {}

    # Prepare quick lookup sets for optimizer outputs
    raw_opt_outs = set(opt_updates.values())
    stripped_opt_outs = set(_strip_scope(v, ctx) for v in raw_opt_outs)

    for n in nodes:
        # remap inputs/outputs to stripped local names (pass ctx so import
        # prefixes are handled)
        local_inputs = [_strip_scope(i, ctx) for i in n.input]
        local_outputs = [_strip_scope(o, ctx) for o in n.output]
        # copy node attributes and create new node; use stripped node name
        # for readability but preserve original node.name when possible.
        node_name = _strip_scope(n.name, ctx) or n.name
        new_node = helper.make_node(n.op_type, local_inputs, local_outputs, name=node_name)
        # ensure training domain for algorithm nodes
        try:
            new_node.domain = TRAINING_DOMAIN
        except Exception:
            pass
        # preserve attributes when possible
        try:
            for a in n.attribute:
                new_attr = onnx.AttributeProto()
                new_attr.CopyFrom(a)
                new_node.attribute.extend([new_attr])
        except Exception:
            pass
        alg_nodes.append(new_node)
        for inp in n.input:
            # If the input is a model initializer, include it as an initializer
            init_key = _find_initializer_key(ctx, inp) if hasattr(ctx, 'initializers') else inp
            if init_key in ctx.initializers:
                # If this initializer is a trainable model weight, do NOT copy it
                short = _strip_scope(init_key, ctx)
                if trainables.get(init_key) or trainables.get(short):
                    # skip copying model weight into algorithm initializers
                    continue
                # Special-case: if training config requested `lr_input` or
                # `step_input`, expose the matching literal as an algorithm
                # input instead of copying it into algorithm initializers.
                tm = ctx.model_metadata.get("training")
                training_meta = tm if isinstance(tm, dict) else {}
                training_cfg = ctx.model_metadata.get("training_config") or {}
                lr_lit = training_meta.get("lr_literal")
                step_lit = training_meta.get("step_literal")
                if (training_cfg.get("lr_input") and init_key == lr_lit) or (
                    training_cfg.get("step_input") and init_key == step_lit
                ):
                    # create a value-info for this input
                    try:
                        t = ctx.value_types.get(init_key) or {"scalar": "f32", "dims": []}
                        vi = helper.make_tensor_value_info(
                            _strip_scope(init_key, ctx),
                            fuse_dtype_to_onnx(t.get("scalar") or DEFAULT_SCALAR),
                            t.get("dims") or [],
                        )
                        alg_inputs[_strip_scope(init_key, ctx)] = vi
                    except Exception:
                        pass
                    continue
                init = ctx.initializers[init_key]
                new_init = onnx.TensorProto()
                new_init.CopyFrom(init)
                new_init.name = _strip_scope(new_init.name)
                alg_initializers[new_init.name] = new_init
            else:
                # otherwise record type info for inputs
                if inp not in alg_inputs:
                    t = ctx.value_types.get(inp) or ctx.value_types.get(_strip_scope(inp)) or {"scalar": "f32", "dims": []}
                    # Make a TensorValueInfoProto
                    try:
                        vi = helper.make_tensor_value_info(
                            _strip_scope(inp),
                            fuse_dtype_to_onnx(t.get("scalar") or DEFAULT_SCALAR),
                            t.get("dims") or [],
                        )
                        alg_inputs[_strip_scope(inp)] = vi
                    except Exception:
                        pass

    # Expose configured lr/step as algorithm inputs when requested even if the
    # corresponding initializer was not produced during gradient generation.
    try:
        training_cfg = ctx.model_metadata.get("training_config") or {}
        tm = ctx.model_metadata.get("training")
        training_meta = tm if isinstance(tm, dict) else {}
        if training_cfg.get("lr_input"):
            lr_lit = training_meta.get("lr_literal") if isinstance(training_meta, dict) else None
            lr_name = _strip_scope(lr_lit or ((ctx.scope_display or ctx.name) + ".lr"), ctx)
            if lr_name not in alg_inputs:
                # default to a 1-element float tensor
                try:
                    vi = helper.make_tensor_value_info(lr_name, fuse_dtype_to_onnx("f32"), [1])
                    alg_inputs[lr_name] = vi
                except Exception:
                    pass
        if training_cfg.get("step_input"):
            step_lit = training_meta.get("step_literal") if isinstance(training_meta, dict) else None
            step_name = _strip_scope(step_lit or ((ctx.scope_display or ctx.name) + ".step"), ctx)
            if step_name not in alg_inputs:
                try:
                    vi = helper.make_tensor_value_info(step_name, fuse_dtype_to_onnx("i64"), [])
                    alg_inputs[step_name] = vi
                except Exception:
                    pass
    except Exception:
        pass
        # Per-node outputs: add optimizer update outputs and any 'loss' outputs
        for out in n.output:
            if out in raw_opt_outs or _strip_scope(out, ctx) in stripped_opt_outs:
                # output is an optimizer update target; create a value-info
                base = _strip_scope(out, ctx)
                # Prefer explicit type mappings; fall back to the underlying
                # parameter's type when the output name is a qualified variant
                # (e.g., 'x.opt' -> 'x').
                t = ctx.value_types.get(out) or ctx.value_types.get(base)
                if not t:
                    if "." in base:
                        candidate = base.rsplit(".", 1)[0]
                        t = ctx.value_types.get(candidate)
                if not t:
                    t = {"scalar": "f32", "dims": []}
                try:
                    vi = helper.make_tensor_value_info(
                        base,
                        fuse_dtype_to_onnx(t.get("scalar") or DEFAULT_SCALAR),
                        t.get("dims") or [],
                    )
                    alg_outputs[base] = vi
                except Exception:
                    pass
            else:
                # include any outputs that look like a loss symbol so they can
                # be bound to model loss (e.g., 'loss' outputs from optimizers)
                stripped_out = _strip_scope(out, ctx)
                if "loss" in str(out).lower() or "loss" in str(stripped_out).lower():
                    t = ctx.value_types.get(out) or ctx.value_types.get(stripped_out) or {"scalar": "f32", "dims": []}
                    try:
                        vi = helper.make_tensor_value_info(
                            stripped_out,
                            fuse_dtype_to_onnx(t.get("scalar") or DEFAULT_SCALAR),
                            t.get("dims") or [],
                        )
                        alg_outputs[stripped_out] = vi
                    except Exception:
                        pass

    # Ensure any 'loss' outputs that were missed are included
    try:
        for n in nodes:
            for out in n.output:
                if "loss" in str(out).lower() or "loss" in str(_strip_scope(out, ctx)).lower():
                    base = _strip_scope(out, ctx)
                    if base not in alg_outputs:
                        t = ctx.value_types.get(out) or ctx.value_types.get(base) or {"scalar": "f32", "dims": []}
                        try:
                            vi = helper.make_tensor_value_info(
                                base,
                                fuse_dtype_to_onnx(t.get("scalar") or DEFAULT_SCALAR),
                                t.get("dims") or [],
                            )
                            alg_outputs[base] = vi
                        except Exception:
                            pass
    except Exception:
        pass

    # Fallback: if no outputs were captured due to transient ordering or
    # other issues, create outputs for known optimizer targets from
    # the opt_updates mapping so algorithm graphs include outputs.
    try:
        if not alg_outputs and raw_opt_outs:
            for out in sorted(raw_opt_outs):
                base = _strip_scope(out, ctx)
                t = ctx.value_types.get(out) or ctx.value_types.get(base)
                if not t and "." in base:
                    candidate = base.rsplit(".", 1)[0]
                    t = ctx.value_types.get(candidate)
                if not t:
                    t = {"scalar": "f32", "dims": []}
                try:
                    vi = helper.make_tensor_value_info(
                        base,
                        fuse_dtype_to_onnx(t.get("scalar") or DEFAULT_SCALAR),
                        t.get("dims") or [],
                    )
                    alg_outputs[base] = vi
                except Exception:
                    pass
    except Exception:
        pass
    # `@training { algorithm: name }`), prefer it over the synthesized
    # optimizer-node-based algorithm.
    explicit_alg = None
    try:
        explicit_alg = ctx.model_metadata.get("training", {}).get("algorithm_graph")
    except Exception:
        explicit_alg = None

    if explicit_alg is not None:
        # Ensure opset imports are present and node domains are set for
        # training ops if needed.
        try:
            core_opset = getattr(ctx, "opset", None) or getattr(ctx, "opset", 18)
            training_version = int(ctx.extra_opsets.get(TRAINING_DOMAIN, 1))
            # We do not attach opset_import into the GraphProto directly
            # (not all onnx versions support it). The final ModelProto
            # generated by `GraphContext.build_model` will contain the needed
            # opset imports (core and training domain) in `model.opset_import`.
            pass
        except Exception:
            pass
        # Ensure nodes are in training domain
        try:
            for n in explicit_alg.node:
                n.domain = TRAINING_DOMAIN
        except Exception:
            pass
        ti = onnx.onnx_ml_pb2.TrainingInfoProto()
        ti.algorithm.CopyFrom(explicit_alg)
    else:
        # Build graph proto with deterministic ordering for inputs/outputs
        # Debug: inspect inputs/outputs we're about to emit for algorithm
        try:
            print('DEBUG emit_training_info alg_inputs:', sorted(list(alg_inputs.keys())))
            print('DEBUG emit_training_info alg_outputs:', sorted(list(alg_outputs.keys())))
        except Exception:
            pass
        alg_graph = helper.make_graph(
            nodes=alg_nodes,
            name=(ctx.scope_display or ctx.name) + ".training_alg",
            inputs=[alg_inputs[k] for k in sorted(alg_inputs)],
            outputs=[alg_outputs[k] for k in sorted(alg_outputs)],
            initializer=[alg_initializers[k] for k in sorted(alg_initializers)],
        )

        # Use the current context opset for core ops and the recorded training
        # domain opset (default to 1) for training ops
        core_opset = getattr(ctx, "opset", None) or getattr(ctx, "opset", 18)
        training_version = int(ctx.extra_opsets.get(TRAINING_DOMAIN, 1))
        # We do not attach opset_import into the GraphProto directly
        # (not all onnx versions support it). The final ModelProto built by
        # `ctx.build_model()` will include the core and training opset
        # imports based on `ctx.opset` and `ctx.extra_opsets`.
        pass

        ti = onnx.onnx_ml_pb2.TrainingInfoProto()
        ti.algorithm.CopyFrom(alg_graph)

    # Populate update_binding mapping: map model initializer name -> algorithm output name
    for p, out in opt_updates.items():
        init_key = _find_initializer_key(ctx, p)
        out_key = _strip_scope(out, ctx)
        try:
            ti.update_binding[init_key] = out_key
        except Exception:
            e = ti.update_binding.add()
            e.key = init_key
            e.value = out_key

    # Attach an (empty) initialization graph for compatibility
    try:
        init_graph = helper.make_graph([], (ctx.scope_display or ctx.name) + ".training_init", [], [])
        ti.initialization.CopyFrom(init_graph)
    except Exception:
        pass

    # Determine loss binding:
    #  - prefer explicit training.loss metadata (a function name)
    #  - else, fall back to any in-graph 'loss' value
    tm = ctx.model_metadata.get("training")
    training_meta = tm if isinstance(tm, dict) else {}
    training_cfg = ctx.model_metadata.get("training_config") or {}
    loss_candidate = (
        training_meta.get("loss")
        or (training_cfg.get("loss") if isinstance(training_cfg, dict) else None)
        or ("loss" if "loss" in ctx.value_types else None)
    )
    if loss_candidate:
        found = None
        # match against algorithm outputs (stripped names) or explicit
        # algorithm outputs when provided.
        candidates = list(alg_outputs.keys()) or []
        if explicit_alg is not None:
            try:
                candidates = [o.name for o in explicit_alg.output]
            except Exception:
                pass
        for k in sorted(list(candidates)):
            if str(k).endswith(str(loss_candidate)) or str(loss_candidate) in str(k) or str(k).endswith("loss"):
                found = k
                break
        # fallback: if algorithm exposes a 'loss' output, use that
        if not found:
            for k in sorted(list(candidates)):
                if "loss" in str(k):
                    found = k
                    break
        if found:
            # Prefer the protobuf map-like API when available (newer onnx)
            if hasattr(ti, "loss_binding"):
                # the attribute may be a map-like object (newer onnx), a
                # protobuf repeated field with .add() or our shimmed list.
                lb = getattr(ti, "loss_binding")
                try:
                    # Prefer map-like assignment
                    lb[str(loss_candidate)] = found
                except Exception:
                    try:
                        # Try protobuf repeated field add()
                        e = lb.add()
                        e.key = str(loss_candidate)
                        e.value = found
                    except Exception:
                        try:
                            # If lb is a list (shim), append a simple KV
                            class _KV:
                                def __init__(self, key, value):
                                    self.key = key
                                    self.value = value

                            lb.append(_KV(str(loss_candidate), found))
                        except Exception:
                            # fallthrough to instance-level storage
                            pass
            else:
                # Older onnx versions may not have a loss_binding field; store
                # bindings in a per-instance list named '_loss_bindings'. Use
                # object.__setattr__ to avoid protobuf's attribute protection.
                class _KV:
                    def __init__(self, key, value):
                        self.key = key
                        self.value = value

                try:
                    if not hasattr(ti, "_loss_bindings"):
                        object.__setattr__(ti, "_loss_bindings", [])
                    ti._loss_bindings.append(_KV(str(loss_candidate), found))
                except Exception:
                    # last-resort: silently ignore if we can't attach
                    pass

    # Avoid emitting duplicate training info entries for the same
    # model parameters when multiple passes or node variants are present.
    try:
        existing = getattr(ctx, "_training_info", []) or []
        existing_keys = set()
        for e in existing:
            try:
                for b in e.update_binding:
                    existing_keys.add(b.key)
            except Exception:
                pass
        # If any of the update keys are already present in existing
        # TrainingInfo, merge the new information into the first matching
        # entry instead of emitting a duplicate TrainingInfoProto.
        for e in existing:
            try:
                e_keys = set(b.key for b in e.update_binding)
            except Exception:
                e_keys = set()
            if any(p in e_keys for p in opt_updates.keys()):
                # Merge initialization graph if missing
                try:
                    if getattr(e, "initialization", None) is None and getattr(ti, "initialization", None) is not None:
                        e.initialization.CopyFrom(ti.initialization)
                except Exception:
                    pass
                # Merge algorithm inputs/outputs
                try:
                    exist_inp = {vi.name for vi in e.algorithm.input}
                    for vi in ti.algorithm.input:
                        if vi.name not in exist_inp:
                            new_vi = helper.make_tensor_value_info(vi.name, vi.type.tensor_type.elem_type, [d.dim_value for d in vi.type.tensor_type.shape.dim])
                            e.algorithm.input.extend([new_vi])
                    exist_out = {vi.name for vi in e.algorithm.output}
                    for vi in ti.algorithm.output:
                        if vi.name not in exist_out:
                            new_vo = helper.make_tensor_value_info(vi.name, vi.type.tensor_type.elem_type, [d.dim_value for d in vi.type.tensor_type.shape.dim])
                            e.algorithm.output.extend([new_vo])
                except Exception:
                    pass
                # Merge loss_binding entries
                try:
                    for lb in getattr(ti, "loss_binding", []):
                        # reuse existing binding insertion logic adapted for both
                        # proto maps and shimmed lists
                        try:
                            if hasattr(e, "loss_binding"):
                                lb_obj = getattr(e, "loss_binding")
                                try:
                                    lb_obj[str(lb.key)] = lb.value
                                except Exception:
                                    try:
                                        sub = lb_obj.add()
                                        sub.key = lb.key
                                        sub.value = lb.value
                                    except Exception:
                                        try:
                                            lb_obj.append(type(lb)(lb.key, lb.value))
                                        except Exception:
                                            pass
                            else:
                                # fallback: attach to per-id mapping (handled by property shim)
                                pass
                        except Exception:
                            pass
                except Exception:
                    pass
                return
    except Exception:
        pass

    # Add the training info to context so build_model will append it
    ctx.add_training_info(ti)
