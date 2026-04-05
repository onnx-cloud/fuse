from typing import Dict, Any

from src.graph_context import as_tensor_type


def generate_gradients(ctx) -> Dict[str, Any]:
    """Generate training-oriented outputs for gradients and (optional) loss.

    This is a lightweight scaffold: it creates graph outputs named
    `<param>.grad` for every entry in `ctx.model_metadata.get('trainables')`
    that is truthy and whose type is known in `ctx.value_types`. It will also
    expose an existing `loss` value as a graph output if present in
    `ctx.value_types`.

    Returns a summary dict of outputs added.
    """
    added = {"grads": [], "loss": None, "opt_updates": {}, "optimizer_nodes": []}
    trainables = ctx.model_metadata.get("trainables", {}) or {}

    # If an explicit training opset was not provided, ensure we have a
    # reasonable default so training ops can be declared in the model.
    training_domain = "ai.onnx.preview.training"
    if training_domain not in ctx.extra_opsets:
        ctx.extra_opsets[training_domain] = max(int(ctx.extra_opsets.get(training_domain, 0)), 1)

    # Build a mapping of value -> producing node for the forward graph
    out_to_node = {}
    node_inputs = {}
    for n in ctx.nodes:
        for out in list(n.output):
            out_to_node[out] = n
        node_inputs[n.name] = list(n.input)

    # Determine which trainable params we may need gradients for
    grad_names = []
    for pname, enabled in trainables.items():
        if enabled and (ctx.value_types.get(pname) or ctx.value_types.get(pname.split(".")[-1])):
            grad_names.append(f"{pname}.grad")

    # If a loss is present, perform a limited reverse-mode autodiff over
    # supported ops (MatMul, Add) to compute per-parameter gradients.
    loss_present = "loss" in ctx.value_types
    seed_grad = None

    # Emit a GenerateGradients node as a compatibility marker if a loss
    # and trainable params exist. This mirrors common ONNX training
    # conventions and lets tools that expect a GenerateGradients op find it
    # even when we also emit per-op autodiff nodes.
    if loss_present and grad_names:
        try:
            import json

            params_list = [p for p, e in trainables.items() if e]
            ctx.add_node("GenerateGradients", ["loss"], grad_names, attrs={"params": json.dumps(params_list)})
            try:
                ctx.nodes[-1].domain = training_domain
            except Exception:
                pass
        except Exception:
            pass

    if loss_present:
        # Create a scalar '1.0' literal to seed dloss
        seed_grad = ctx.add_tensor_literal([1.0], {"scalar": "f32", "dims": []})

        # If requested, create a step counter literal so algorithms may accept
        # a dynamic step/iteration input (and be exposed as an algorithm input
        # when `step_input` is set in training config).
        training_cfg = ctx.model_metadata.get("training_config") or {}
        try:
            if training_cfg.get("step_input"):
                qname = (ctx.scope_display or ctx.name) + ".step"
                step_name = ctx.add_literal(0, {"scalar": "i64", "dims": []}, name=qname)
                ctx.model_metadata.setdefault("training", {})["step_literal"] = step_name
        except Exception:
            pass
        # Map any graph-visible outputs that look like 'loss' to the seed
        grads = {}
        for out_name in list(ctx.outputs.keys()):
            if str(out_name).endswith("loss"):
                grads[out_name] = seed_grad

        # Process nodes in reverse order of emission for a simple reverse topological
        for n in reversed(list(ctx.nodes)):
            # Determine if node contributes to any current gradient target
            produces = list(n.output)
            contributes = [p for p in produces if p in grads]
            if not contributes:
                continue
            # For each supported op attempt to backprop
            op = n.op_type
            if op == "MatMul":
                # Assume outputs: [C]; inputs: [A, B]
                A = n.input[0]
                B = n.input[1]
                C = n.output[0]
                dC = grads.get(C)
                if not dC:
                    continue
                # dA = MatMul(dC, Transpose(B))
                tb = ctx._next_const_name()
                try:
                    ctx.add_node("Transpose", [B], [tb], attrs={"perm": [1, 0]})
                    try:
                        ctx.nodes[-1].domain = training_domain
                    except Exception:
                        pass
                except Exception:
                    ctx.add_node("Transpose", [B], [tb])
                dA = ctx._next_const_name()
                ctx.add_node("MatMul", [dC, tb], [dA])
                try:
                    ctx.nodes[-1].domain = training_domain
                except Exception:
                    pass
                # accumulate into grads[A]
                if A in grads:
                    s = ctx._next_const_name()
                    ctx.add_node("Add", [grads[A], dA], [s])
                    try:
                        ctx.nodes[-1].domain = training_domain
                    except Exception:
                        pass
                    grads[A] = s
                else:
                    grads[A] = dA

                # dB = MatMul(Transpose(A), dC)
                ta = ctx._next_const_name()
                try:
                    ctx.add_node("Transpose", [A], [ta], attrs={"perm": [1, 0]})
                    try:
                        ctx.nodes[-1].domain = training_domain
                    except Exception:
                        pass
                except Exception:
                    ctx.add_node("Transpose", [A], [ta])
                dB = ctx._next_const_name()
                ctx.add_node("MatMul", [ta, dC], [dB])
                try:
                    ctx.nodes[-1].domain = training_domain
                except Exception:
                    pass
                if B in grads:
                    s = ctx._next_const_name()
                    ctx.add_node("Add", [grads[B], dB], [s])
                    try:
                        ctx.nodes[-1].domain = training_domain
                    except Exception:
                        pass
                    grads[B] = s
                else:
                    grads[B] = dB

            elif op == "Add":
                # C = Add(A,B) -> dA += dC, dB += dC
                A = n.input[0]
                B = n.input[1]
                C = n.output[0]
                dC = grads.get(C)
                if not dC:
                    continue
                for X in (A, B):
                    if X in grads:
                        s = ctx._next_const_name()
                        ctx.add_node("Add", [grads[X], dC], [s])
                        grads[X] = s
                    else:
                        grads[X] = dC

            else:
                # Unsupported ops: do not propagate
                continue

        # Once we have grads for parameters, materialize them as outputs
        for pname, enabled in trainables.items():
            if not enabled:
                continue
            # The parameter may be present as qualified name in ctx.inputs/initializers
            param_name = pname
            if param_name not in grads:
                # maybe stored under qualified name variants
                candidates = [param_name, f"{ctx.scope_display}.{param_name}" if getattr(ctx, 'scope_display', None) else param_name]
                found = None
                for c in candidates:
                    if c in grads:
                        found = c
                        break
                if not found:
                    continue
                grad_src = grads[found]
            else:
                grad_src = grads[param_name]
            grad_name = f"{pname}.grad"
            # Prefer to avoid emitting a synthetic Identity node: if the
            # underlying producer value can be renamed to the canonical
            # `<param>.grad` name, do so so the graph output references the
            # original producer directly (ergonomic for Netron/human). Fall
            # back to emitting an Identity when renaming is not possible.
            t = ctx.value_types.get(pname) or {}
            try:
                # Attempt to rename the producing value to the canonical
                # gradient name so no extra Identity node is necessary.
                ctx.rename_value(grad_src, grad_name)
                # Ensure type info is recorded for both the internal and
                # graph-visible forms.
                ctx.value_types[grad_name] = as_tensor_type(t)
                ctx.add_output(grad_name, as_tensor_type(t))
            except Exception:
                try:
                    ctx.add_node("Identity", [grad_src], [grad_name])
                    try:
                        ctx.nodes[-1].domain = training_domain
                    except Exception:
                        pass
                    ctx.value_types[grad_name] = as_tensor_type(t)
                    try:
                        ctx.add_output(grad_name, as_tensor_type(t))
                    except Exception:
                        pass
                except Exception:
                    # Best-effort: if we cannot create an identity, skip.
                    continue
            added["grads"].append(grad_name)

    # Per-param Gradient fallback (used when no loss is present or
    # GenerateGradients failed)
    if not added["grads"]:
        for pname, enabled in trainables.items():
            if not enabled:
                continue
            t = ctx.value_types.get(pname) or ctx.value_types.get(pname.split(".")[-1])
            if not t:
                continue
            grad_name = f"{pname}.grad"
            try:
                ctx.add_node("Gradient", [pname], [grad_name], attrs={"for": pname})
                try:
                    ctx.nodes[-1].domain = training_domain
                except Exception:
                    pass
            except Exception:
                ctx.add_node("Identity", [pname], [grad_name])
            ctx.value_types[grad_name] = as_tensor_type(t)
            try:
                ctx.add_output(grad_name, as_tensor_type(t))
            except Exception:
                pass
            added["grads"].append(grad_name)
    # If an optimizer is specified in the training config, wire a simple
    # optimizer node per parameter that consumes the param and its gradient.
    cfg = ctx.model_metadata.get("training_config") or {}
    if isinstance(cfg.get("optimizer"), dict):
        opt_spec = cfg.get("optimizer")
    elif isinstance(cfg.get("optimizer"), str):
        opt_spec = {"type": cfg.get("optimizer")}
    else:
        opt_spec = None

    if opt_spec and isinstance(opt_spec, dict):
        opt_type = str(opt_spec.get("type") or opt_spec.get("optimizer") or "").lower()
        lr = opt_spec.get("lr") or opt_spec.get("learning_rate")
        mapping = {
            "adam": "Adam",
            "adamw": "AdamW",
            "adagrad": "Adagrad",
            "momentum": "Momentum",
            "sgd": "SGD",
        }
        op_name = mapping.get(opt_type, (opt_type.title() if opt_type else None))
        if op_name:
            for g in added["grads"]:
                param = g[:-5]
                opt_out = f"{param}.opt"
                attrs = {}
                if lr is not None:
                    try:
                        attrs["lr"] = float(lr)
                    except Exception:
                        attrs["lr"] = lr
                # Create optimizer state initializers when required by the
                # optimizer (e.g., Adam has 'm' and 'v'). Initializers are
                # best-effort: only created for concrete dims.
                state_inputs = []
                if op_name in ("Adam", "AdamW"):
                    for sname in ("m", "v"):
                        t = ctx.value_types.get(param) or {}
                        dims = t.get("dims", []) or []
                        # Try to materialize zeros if shape is concrete
                        try:
                            if dims and all(isinstance(d, int) and d > 0 for d in dims):
                                total = 1
                                for d in dims:
                                    total *= int(d)
                                zeros = [0.0] * total
                                name = ctx.add_tensor_literal(zeros, {"scalar": t.get("scalar"), "dims": dims})
                                state_inputs.append(name)
                            else:
                                # fallback: create a named initializer with no data
                                name = f"{param}.{sname}"
                                try:
                                    ctx.add_const({"name": name, "type_decl": t, "value": {"lit_list": []}})
                                except Exception:
                                    pass
                                state_inputs.append(name)
                        except Exception:
                            pass
                # For ONNX training optimizer ops, canonical inputs are
                # often [var, m, v, lr, grad] and outputs [var, m, v]
                inputs = [param] + state_inputs
                if g:
                    inputs.append(g)
                if lr is not None:
                    # Use lr as an initializer literal so it's available as input.
                    # If training config requests a dynamic learning-rate (`lr_input`),
                    # create a named literal 'lr' so training emission can expose it
                    # as an algorithm graph input instead of copying it as an initializer.
                    training_cfg = ctx.model_metadata.get("training_config") or {}
                    try:
                        if training_cfg.get("lr_input"):
                            # use a qualified, human-friendly name to avoid collisions
                            qname = (ctx.scope_display or ctx.name) + ".lr"
                            lr_name = ctx.add_tensor_literal([float(lr)], {"scalar": "f32", "dims": [1]}, name=qname)
                            # record for emit_training_info to detect and expose
                            ctx.model_metadata.setdefault("training", {})["lr_literal"] = lr_name
                        else:
                            lr_name = ctx.add_tensor_literal([float(lr)], {"scalar": "f32", "dims": [1]})
                        inputs.append(lr_name)
                    except Exception:
                        pass

                try:
                    # Some optimizers return updated var and state tensors
                    out_count = 3 if op_name in ("Adam", "AdamW") else 1
                    outputs = [opt_out] if out_count == 1 else [opt_out, f"{param}.m", f"{param}.v"]
                    node_name = ctx.add_node(op_name, inputs, outputs, attrs=attrs)
                    try:
                        ctx.nodes[-1].domain = training_domain
                    except Exception:
                        pass
                    # Record optimizer node and mapping for later TrainingInfo
                    added["opt_updates"][param] = outputs[0]
                    added["optimizer_nodes"].append(node_name)
                except Exception:
                    pass
                # Build optimizer inputs: [param, grad, <state...>]
                inputs = [param, g] + state_inputs
                try:
                    # second attempt to add the same op with canonical inputs
                    node_name = ctx.add_node(op_name, inputs, [opt_out], attrs=attrs)
                    try:
                        ctx.nodes[-1].domain = training_domain
                    except Exception:
                        pass
                    added["opt_updates"][param] = opt_out
                    added["optimizer_nodes"].append(node_name)
                except Exception:
                    pass

    # Expose existing loss value as a graph output if it exists
    if "loss" in ctx.value_types:
        try:
            ctx.add_output("loss", as_tensor_type(ctx.value_types["loss"]))
            added["loss"] = "loss"
        except Exception:
            added["loss"] = None

    return added
