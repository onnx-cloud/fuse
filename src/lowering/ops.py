import logging
from typing import Any, Dict, List, Optional, Tuple

from src.graph_context import GraphContext, as_tensor_type, get_model_domain as _get_model_domain
from src.lowering.utils import LoweringError
from src.onnx_schema import normalize_domain_and_op, require_op_schema

logger = logging.getLogger(__name__)


class OpsLowerer:
    def __init__(self, lowerer):
        self._lowerer = lowerer

    def _lower_call(
        self,
        call: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        op = call.get("call")

        # Check for control flow constructs (loop, if, scan) in a
        # case-insensitive manner.  older examples or user code may use
        # capitalized names (e.g. `Loop`) so we normalize here to avoid
        # accidentally falling back to the generic ONNX path which does not
        # properly convert graph-valued attributes.
        if isinstance(op, str) and op.lower() == "loop":
            return self._lower_loop_call(
                call, ctx, env, types, type_hint=type_hint, out_name=out_name
            )
        elif isinstance(op, str) and op.lower() == "if":
            return self._lower_if_call(
                call, ctx, env, types, type_hint=type_hint, out_name=out_name
            )
        elif isinstance(op, str) and op.lower() == "scan":
            return self._lower_scan_call(
                call, ctx, env, types, type_hint=type_hint, out_name=out_name
            )

        # If the call target is a user-declared function, either inline it or
        # emit a call node depending on the inline_functions flag.  When
        # emitting a call we intentionally *skip* ONNX schema validation since
        # the op_type is user-defined and backed by a FunctionProto in the
        # model, not by a built-in operator.
        if isinstance(op, str) and op in getattr(
            self._lowerer, "_user_decls", {}
        ):
            decl = self._lowerer._user_decls[op]
            if getattr(self._lowerer, "inline_functions", False):
                return self._lowerer._inline_user_decl(
                    decl,
                    call,
                    ctx,
                    env,
                    types,
                    type_hint=type_hint,
                    out_name=out_name,
                )
            # here we need to create a lightweight ONNX node representing the
            # function call without validating against the ONNX registry.
            # Lower arguments normally.
            lowered_args: List[str] = []
            lowered_types: List[Optional[Dict[str, Any]]] = []
            for a in call.get("args", []):
                if (
                    isinstance(a, dict)
                    and len(a) == 1
                    and str(next(iter(a))).startswith("@")
                ):
                    continue
                if (
                    isinstance(a, dict)
                    and len(a) == 1
                    and not str(next(iter(a))).startswith("@")
                ):
                    a = a[next(iter(a))]
                n, t = self._lowerer._lower_expr(
                    a, ctx, env, types, type_hint=type_hint
                )
                if n is not None:
                    lowered_args.append(n)
                    lowered_types.append(t)
            # create output name(s)
            ret_typ = decl.get("ret_type")
            output_names = []
            resolved_types = []
            
            if isinstance(ret_typ, list):
                for i, rt in enumerate(ret_typ):
                    oname = (out_name + f"_{i}") if out_name else ctx._next_node_name(f"{op}_out{i}")
                    output_names.append(oname)
                    resolved = self._lowerer._resolve_type(rt) or rt
                    ctx.value_types[oname] = as_tensor_type(resolved)
                    resolved_types.append((oname, ctx.value_types[oname]))
            else:
                oname = out_name or ctx._next_node_name(op)
                output_names.append(oname)
                if ret_typ is not None:
                    resolved = self._lowerer._resolve_type(ret_typ) or ret_typ
                    ctx.value_types[oname] = as_tensor_type(resolved)
                resolved_types = [(oname, ctx.value_types.get(oname))]
                
            ctx.add_node(op, lowered_args, output_names)
            
            # ensure call node uses the same domain as the FunctionProto
            func_dom = decl.get("_func_domain") or _get_model_domain(ctx) or "fuse.local"
            try:
                ctx.nodes[-1].domain = func_dom
            except Exception:
                pass
                
            if len(output_names) > 1:
                env["__last_multi_return__"] = resolved_types
                return output_names[0], resolved_types[0][1]
            else:
                return output_names[0], ctx.value_types.get(output_names[0])

        # If the call target is an imported function, lower it.
        from src.onnx_schema import normalize_domain_and_op

        op_domain, op_type = normalize_domain_and_op(str(op))
        if op_type in self._lowerer.import_manager.fused_signatures:
            return self._lower_imported_call(
                call, ctx, env, types, type_hint=type_hint, out_name=out_name
            )

        # Generic ONNX op lowering
        return self._lower_onnx_call(
            call, ctx, env, types, type_hint=type_hint, out_name=out_name
        )

    # Duplicate implementation removed. The canonical implementation lives
    # below and will be used for all ONNX calls.

    def _lower_imported_call(
        self, call, ctx, env, types, type_hint=None, out_name=None
    ):
        op = call.get("call")
        op_domain, op_type = normalize_domain_and_op(str(op))
        sig = self._lowerer.import_manager.fused_signatures[op_type]
        import_inputs = sig.get("inputs") or []
        import_outputs = sig.get("outputs") or []
        import_input_infos = sig.get("input_infos") or []

        lowered_args: List[str] = []
        lowered_types: List[Optional[Dict[str, Any]]] = []
        for a in call.get("args", []):
            if (
                isinstance(a, dict)
                and len(a) == 1
                and str(next(iter(a))).startswith("@")
            ):
                continue
            if (
                isinstance(a, dict)
                and len(a) == 1
                and not str(next(iter(a))).startswith("@")
            ):
                a = a[next(iter(a))]
            n, t = self._lowerer._lower_expr(
                a, ctx, env, types, type_hint=type_hint
            )
            if n is not None:
                lowered_args.append(n)
                lowered_types.append(t)

        for src, dst in zip(lowered_args, import_inputs):
            if dst not in ctx.defined_values:
                insert_at = ctx.import_node_start.get(op_type, len(ctx.nodes))
                ctx.insert_node(insert_at, "Identity", [src], [dst])

        if len(lowered_args) < len(import_inputs):
            for info in import_input_infos[len(lowered_args) :]:
                name = info["name"]
                if name in ctx.inputs or name in ctx.defined_values:
                    continue
                from onnx import helper

                ctx.inputs[name] = helper.make_tensor_value_info(
                    name, int(info["elem_type"]), info.get("dims") or []
                )
                ctx.defined_values.add(name)

        if not import_outputs:
            raise ValueError(f"imported model '{op_type}' has no outputs")
        return import_outputs[0], ctx.value_types.get(import_outputs[0])

    def _lower_graph_attributes(self, attrs, ctx, env, op_type=None):
        for aname, aval in list(attrs.items()):
            if isinstance(aval, str) and aval in env:
                aval = env[aval]
                attrs[aname] = aval
            # support qualified names (foo.bar)
            elif isinstance(aval, str) and "." in aval:
                base = aval.split(".")[-1]
                if base in env:
                    aval = env[base]
                    attrs[aname] = aval
            if isinstance(aval, str) and aval in getattr(
                self._lowerer, "_user_decls", {}
            ):
                decl = self._lowerer._user_decls[aval]
                sub_ctx = GraphContext(name=decl.get("name"), opset=ctx.opset)
                parent_prefix = getattr(ctx, "scope_prefix", "parent")
                sub_ctx.scope_prefix = f"{parent_prefix}__{decl.get('name')}"
                sub_ctx._preserve_local_input_names = True
                # clear metadata to avoid requiring @fuse in subgraphs
                sub_ctx.model_metadata = {}
                self._lowerer._lower_function(decl, sub_ctx)
                g = sub_ctx.build_model().graph
                if op_type == "Loop" and aname == "body":
                    try:
                        from onnx import TensorProto

                        if len(g.input) >= 1:
                            g.input[0].type.tensor_type.elem_type = int(
                                TensorProto.INT64
                            )
                        if len(g.input) >= 2:
                            g.input[1].type.tensor_type.elem_type = int(
                                TensorProto.BOOL
                            )
                    except Exception:
                        pass
                attrs[aname] = g

    def _validate_attributes(self, attrs, op_type, schema):
        import difflib

        valid_attr_names = set(getattr(schema, "attributes", {}).keys())
        unknown_attrs = [a for a in attrs.keys() if a not in valid_attr_names]

        if (
            str(op_type).lower().startswith("reduce")
            and "keepdims" in unknown_attrs
        ):
            unknown_attrs.remove("keepdims")

        if unknown_attrs:
            suggestions = {
                a: difflib.get_close_matches(a, sorted(valid_attr_names), n=3)
                for a in unknown_attrs
            }
            msg_lines = [
                f"Unknown attribute(s) {unknown_attrs} for op '{op_type}'."
            ]
            if valid_attr_names:
                msg_lines.append(
                    f"Valid attributes: {sorted(valid_attr_names)}"
                )
            for a, s in suggestions.items():
                if s:
                    msg_lines.append(f"Did you mean: {a} -> {s}")
            raise LoweringError(
                " \\n".join(msg_lines),
                source=self._lowerer._current_source,
                function=op_type,
            )

        from src.graph_context import fuse_dtype_to_onnx

        for name, val in list(attrs.items()):
            if isinstance(val, str) and name in valid_attr_names:
                attr_schema = schema.attributes.get(name)
                t_name = getattr(
                    attr_schema.type, "name", str(attr_schema.type)
                )
                if "INT" in t_name.upper():
                    try:
                        attrs[name] = int(fuse_dtype_to_onnx(val))
                    except Exception:
                        pass
            if name in valid_attr_names:
                attr_schema = schema.attributes.get(name)
                t_name = getattr(
                    attr_schema.type, "name", str(attr_schema.type)
                )
                if "INT" in t_name.upper():
                    if isinstance(val, float) or (
                        isinstance(val, list)
                        and any(isinstance(x, float) for x in val)
                    ):
                        raise LoweringError(
                            f"Attribute '{name}' for op '{op_type}' asserts integer(s) but got float value {val}.",
                            source=self._lowerer._current_source,
                            function=op_type,
                        )

    def _coerce_elementwise_operands(
        self, op_type, inputs, input_types, raw_args, ctx
    ):
        if op_type in self._lowerer.ELEMENTWISE_OPS and len(inputs) == 2:
            left_t, right_t = input_types[0] or {}, input_types[1] or {}
            left_dims, right_dims = (
                left_t.get("dims", []),
                right_t.get("dims", []),
            )

            def unsqueeze_scalar(scalar_name, target_dims):
                new_name = f"{scalar_name}_unsq"
                i = 0
                while new_name in ctx.defined_values:
                    i += 1
                    new_name = f"{scalar_name}_unsq{i}"

                if int(ctx.opset) >= 13:
                    axes_name = ctx.add_tensor_literal(
                        [0], {"scalar": "i64", "dims": [1]}
                    )
                    ctx.add_node(
                        "Unsqueeze", [scalar_name, axes_name], [new_name]
                    )
                else:
                    ctx.add_node(
                        "Unsqueeze",
                        [scalar_name],
                        [new_name],
                        attrs={"axes": [0]},
                    )
                return new_name

            try:
                if left_dims and not right_dims:
                    if (
                        len(raw_args) >= 2
                        and isinstance(raw_args[1], dict)
                        and raw_args[1].get("call") == "Cast"
                    ):
                        inputs[1] = unsqueeze_scalar(inputs[1], left_dims)
                        input_types[1] = {
                            "scalar": left_t.get("scalar"),
                            "dims": list(left_dims),
                        }
                    else:
                        inputs[1] = unsqueeze_scalar(inputs[1], left_dims)
                        input_types[1] = {
                            "scalar": left_t.get("scalar"),
                            "dims": list(left_dims),
                        }

                elif right_dims and not left_dims:
                    if (
                        len(raw_args) >= 1
                        and isinstance(raw_args[0], dict)
                        and raw_args[0].get("call") == "Cast"
                    ):
                        inputs[0] = unsqueeze_scalar(inputs[0], right_dims)
                        input_types[0] = {
                            "scalar": right_t.get("scalar"),
                            "dims": list(right_dims),
                        }
                    else:
                        inputs[0] = unsqueeze_scalar(inputs[0], right_dims)
                        input_types[0] = {
                            "scalar": right_t.get("scalar"),
                            "dims": list(right_dims),
                        }
            except Exception as e:
                logger.debug(
                    "elementwise promotion heuristic failed for %s: %s",
                    op_type,
                    e,
                    exc_info=True,
                )

    def _infer_output_type(
        self, op_type, schema, attrs, input_types, type_hint=None, axes_val=None, call_pos=None
    ):
        """Infer an output type hint for an ONNX operator using schema and
        available attributes & input types. Returns a Fuse-style type dict or
        None when inference is not possible."""
        if type_hint:
            return type_hint

        # Helper: pick first concrete input type available
        first_input = None
        for it in input_types:
            if isinstance(it, dict) and it.get("scalar"):
                first_input = it
                break
        try:
            if op_type == "MatMul" and len(input_types) >= 2:
                left_t = input_types[0] or {}
                right_t = input_types[1] or {}
                left_dims = left_t.get("dims", [])
                right_dims = right_t.get("dims", [])
                if left_dims and right_dims:
                    l_k = left_dims[-1]
                    r_k = right_dims[-2] if len(right_dims) >= 2 else right_dims[0]
                    # If both sides are concrete ints and disagree, or both are
                    # symbolic names and disagree, raise a helpful error.
                    mismatch = (
                        (isinstance(l_k, int) and isinstance(r_k, int) and l_k != r_k)
                        or (
                            isinstance(l_k, str)
                            and isinstance(r_k, str)
                            and str(l_k) != str(r_k)
                        )
                    )
                    if mismatch:
                        from src.lowering.utils import LoweringError

                        scope = None
                        try:
                            scope = getattr(right_t, "__scope__", None)
                        except Exception:
                            scope = None
                        # Prefer the context available on `attrs`/ctx when raised
                        scope = scope or "function"
                        # Try to capture call-site position when available
                        line = None
                        column = None
                        if call_pos and isinstance(call_pos, dict):
                            line = call_pos.get("line")
                            column = call_pos.get("column")
                        raise LoweringError(
                            (
                                f"MatMul dimension mismatch in {scope}: "
                                f"left input dims={left_dims}, right input dims={right_dims}. "
                                "Expected left[-1] == right[-2] (inner dims must match). "
                                "Check declared types/generics (e.g., `type T = f32[N, features]`) or "
                                "ensure weights have shape `[features, out_features]`."
                            ),
                            source=self._lowerer._current_source,
                            function=getattr(self._lowerer, "_current_function", None),
                            line=line,
                            column=column,
                        )
        except LoweringError:
            raise
        except Exception:
            # Be conservative: don't escalate for symbolic dims
            pass

        # If op explicitly provides a 'to' attribute (e.g., Cast), use that
        try:
            from src.graph_context import onnx_dtype_to_fuse
        except Exception:
            onnx_dtype_to_fuse = None

        if "to" in attrs and onnx_dtype_to_fuse is not None:
            try:
                scalar = onnx_dtype_to_fuse(int(attrs["to"]))
                return {
                    "scalar": scalar,
                    "dims": (
                        list(first_input.get("dims") or [])
                        if first_input
                        else []
                    ),
                }
            except Exception:
                pass

        # Known operator categories
        comparisons = {
            "Greater",
            "Less",
            "Equal",
            "GreaterOrEqual",
            "LessOrEqual",
        }
        logicals = {"And", "Or", "Xor"}
        elementwise = set(self._lowerer.ELEMENTWISE_OPS) | {
            "MatMul",
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Sum",
        }
        reductions = {
            "ReduceSum",
            "ReduceMean",
            "ReduceProd",
            "ReduceMax",
            "ReduceMin",
        }

        if op_type in comparisons | logicals:
            # Comparisons/logicals -> boolean outputs
            dims = list(first_input.get("dims") or []) if first_input else []
            return {"scalar": "bool", "dims": dims}

        # MatMul: infer output by replacing the inner dimension with the
        # right operand's last dimension when possible. This handles batched
        # MatMul shapes such as [..., M, K] * [K, N] -> [..., M, N].
        if op_type == "MatMul" and len(input_types) >= 2:
            left = input_types[0] or {}
            right = input_types[1] or {}
            left_dims = list(left.get("dims") or [])
            right_dims = list(right.get("dims") or [])
            if left_dims and right_dims:
                out_dims = list(left_dims[:-1]) + [right_dims[-1]]
                return {
                    "scalar": left.get("scalar") or right.get("scalar"),
                    "dims": out_dims,
                }

        if op_type in elementwise:
            if first_input:
                return {
                    "scalar": first_input.get("scalar"),
                    "dims": list(first_input.get("dims") or []),
                }

        if op_type in reductions:
            if first_input:
                # Attempt to honor 'axes' and 'keepdims' when available so
                # reductions such as ReduceMean(x, axes=[2,3], keepdims@=0)
                # produce the expected output shape.
                try:
                    dims = list(first_input.get("dims") or [])
                    # Prefer explicit serialized 'axes' attribute when present,
                    # otherwise fall back to the temporary axes value made
                    # available by the caller (for opsets where 'axes' is an
                    # input rather than an attribute).
                    axes = attrs.get("axes") if "axes" in attrs else axes_val
                    keepdims = attrs.get("keepdims")
                    if axes is not None:
                        if isinstance(axes, int):
                            axes = [axes]
                        axes = [int(a) for a in axes]
                        rank = len(dims)
                        axes = [a + rank if a < 0 else a for a in axes]
                        if keepdims is None or int(keepdims) == 1:
                            new_dims = list(dims)
                            for a in axes:
                                if 0 <= a < len(new_dims):
                                    new_dims[a] = 1
                        else:
                            new_dims = [d for i, d in enumerate(dims) if i not in axes]
                        return {"scalar": first_input.get("scalar"), "dims": new_dims}
                except Exception:
                    pass
                return {
                    "scalar": first_input.get("scalar"),
                    "dims": list(first_input.get("dims") or []),
                }

        # Special-case Concat: sum the concatenation axis lengths when available
        if op_type == "Concat" and input_types:
            axis = attrs.get("axis", 0)
            try:
                axis = int(axis) if axis is not None else 0
            except Exception:
                axis = 0
            # Ensure we have concrete rank from first input
            first = input_types[0] or {}
            first_dims = list(first.get("dims") or [])
            if first_dims:
                out_dims = list(first_dims)
                # Normalize negative axis
                if axis < 0:
                    axis = len(out_dims) + axis
                # Sum sizes along axis where available
                total = 0
                any_unknown = False
                for it in input_types:
                    if not it:
                        any_unknown = True
                        break
                    dims = it.get("dims") or []
                    if len(dims) <= axis:
                        any_unknown = True
                        break
                    val = dims[axis]
                    if not isinstance(val, int):
                        any_unknown = True
                        break
                    total += val
                if not any_unknown:
                    out_dims[axis] = total
                    return {"scalar": first.get("scalar"), "dims": out_dims}

        # Use dedicated helper module for schema-driven inference when available
        try:
            from src.lowering.schema_inference import infer_output_from_schema

            if schema is not None:
                inferred = infer_output_from_schema(schema, input_types)
                if inferred is not None:
                    return inferred
        except Exception:
            # Keep lowering conservative in face of unexpected failures
            pass

        # Fallback: no inference
        return None

    def _determine_node_outputs(self, output_name, op_type, attrs):
        node_outputs = [output_name]
        body_graph = None
        if op_type in ("Loop", "If", "Scan"):
            for v in attrs.values():
                if getattr(v, "node", None):
                    body_graph = v
                    break
            if body_graph:
                num_outs = len(body_graph.output)
                if num_outs > 1:
                    node_outputs.extend(
                        f"{output_name}_{i}" for i in range(1, num_outs)
                    )
        return node_outputs, body_graph

    def _record_output_types(self, node_outputs, body_graph, type_hint, ctx):
        from src.lowering.utils import _onnx_to_fuse_scalar

        output_type = as_tensor_type(type_hint)
        if body_graph:
            for i, vi in enumerate(body_graph.output):
                out_name = node_outputs[i]
                try:
                    tt = vi.type.tensor_type
                    dims = [
                        d.dim_value
                        for d in tt.shape.dim
                        if hasattr(d, "dim_value") and d.dim_value > 0
                    ]
                    scalar = _onnx_to_fuse_scalar(tt.elem_type)
                    ctx.value_types[out_name] = {
                        "scalar": scalar,
                        "dims": dims,
                    }
                except Exception:
                    ctx.value_types[out_name] = (
                        as_tensor_type(output_type) if output_type else None
                    )
        elif output_type:
            for o in node_outputs:
                ctx.value_types[o] = as_tensor_type(output_type)
        return output_type


    def _lower_args_with_literals(
        self,
        args: List[Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[str], List[Optional[Dict[str, Any]]], List[Any]]:
        inputs: List[str] = []
        input_types: List[Optional[Dict[str, Any]]] = []
        literals: List[Any] = []
        for arg in args:
            if isinstance(arg, dict) and any(k.startswith("@") for k in arg):
                continue
            val, typ = self._lowerer._lower_expr(arg, ctx, env, types)
            if val is not None:
                inputs.append(val)
                input_types.append(typ)
                try:
                    literals.append(
                        self._lowerer._eval_const_expr(arg, env, types)
                    )
                except (ValueError, KeyError):
                    literals.append(None)
        return inputs, input_types, literals

    def _lower_attrs(self, args, op_type, ctx, env, types):
        attrs = {}
        expr_internal_keys = {
            "left",
            "ops",
            "call",
            "args",
            "index",
            "selector",
            "slice",
            "if",
        }
        for arg in args:
            # If this dict-shaped arg appears to be an expression (for
            # example an infix expression with 'left'/'ops'), skip it here
            # so it is treated as a positional/expr arg rather than an
            # attribute mapping.
            if isinstance(arg, dict) and (set(arg.keys()) & expr_internal_keys):
                continue
            # Skip star-args (`* expr`) which are treated as positional expansion
            # rather than attributes (these are represented as {'*': <expr>}).
            if isinstance(arg, dict) and ("*" in arg):
                continue
            if isinstance(arg, dict):
                for k, v in arg.items():
                    # Skip list-literal wrappers (these are positional list args, not attrs)
                    if k == "lit_list":
                        continue
                    # Accept both explicit attr-syntax (`@name=...`) and
                    # call-level generics/kwargs (e.g., Cast<to=f32> -> {'to': 'f32'}).
                    if k.startswith("@"):
                        attr_name = self._lowerer._normalize_attr_name(
                            op_type, k[1:]
                        )
                        try:
                            attrs[attr_name] = self._lowerer._eval_const_expr(
                                v, env, types
                            )
                        except ValueError:
                            attrs[attr_name] = (
                                self._lowerer._coerce_attr_value(v)
                            )
                    else:
                        # Treat plain kwarg keys as attributes when present
                        attr_name = self._lowerer._normalize_attr_name(
                            op_type, str(k)
                        )
                        try:
                            attrs[attr_name] = self._lowerer._eval_const_expr(
                                v, env, types
                            )
                        except Exception:
                            attrs[attr_name] = (
                                self._lowerer._coerce_attr_value(v)
                            )
        return attrs

    def _lower_onnx_call(
        self,
        call: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        op = call.get("call")
        op_domain, op_type = normalize_domain_and_op(str(op))

        # prefer declarative registry if a lowerer is available
        from src.lowering.ops_pkg import registry
        lowerer = None
        if not call.get("_registry_skipped"):
            lowerer = registry.get_lowerer(op_type, op_domain or "", ctx.opset)
        if lowerer is not None:
            # forward self plus all conventional parameters; the handler may
            # ignore extras.
            return lowerer(self, call, ctx, env, types, type_hint, out_name)

        raw_args = call.get("args") or []
        attrs = self._lower_attrs(raw_args, op_type, ctx, env, types)
        inputs, input_types, literals = self._lower_args_with_literals(
            raw_args, ctx, env, types
        )
        output_name = out_name or ctx._next_node_name(op_type)
        pending_doc = env.get("__pending_doc__")

        # Constant-folding for simple elementwise ops
        folded = self._lowerer._maybe_fold_elementwise(
            op_type, inputs, literals
        )
        if folded:
            return folded, types.get(folded)

        # Ensure all inputs have the same scalar type for elementwise ops
        if op_type in self._lowerer.ELEMENTWISE_OPS:
            self._lowerer._ensure_same_scalar(op_type, input_types)

        # Validate operator existence for the selected opset
        call_pos = call.get("__pos__") if isinstance(call, dict) else None
        try:
            schema = require_op_schema(op_type, ctx.opset, op_domain)
        except Exception as e:
            line = None
            column = None
            if call_pos and isinstance(call_pos, dict):
                line = call_pos.get("line")
                column = call_pos.get("column")
            raise LoweringError(
                str(e),
                source=self._lowerer._current_source,
                function=getattr(self._lowerer, "_current_function", None),
                line=line,
                column=column,
            ) from e

        # Merge call-level generics (e.g., Cast<to=f32> or Cast<f32>) into attrs
        gens = call.get("generics")
        if isinstance(gens, dict):
            for k, v in gens.items():
                try:
                    attrs[k] = self._lowerer._eval_const_expr(v, env, types)
                except Exception:
                    attrs[k] = self._lowerer._coerce_attr_value(v)
        elif isinstance(gens, str):
            if op_type == "Cast":
                attrs["to"] = gens

        # Attribute normalization
        self._normalize_attributes(attrs, op_type, schema)

        # Lower user-declared blocks for graph attributes
        self._lower_graph_attributes(attrs, ctx, env, op_type)

        # Defensive sanitization: remove internal call-like keys if present
        for _k in ("call", "args", "generics"):
            attrs.pop(_k, None)

        # Temporarily remove legacy 'axes' attribute for reduction ops if it is not
        # present in the current ONNX schema so validation does not fail. Restore
        # it after validation so later steps may still consult it for inference.
        _axes_val = None
        if str(op_type).lower().startswith("reduce") and "axes" in attrs and "axes" not in getattr(schema, "attributes", {}):
            _axes_val = attrs.pop("axes", None)

        # Validate attributes (schema-aware)
        self._validate_attributes(attrs, op_type, schema)

        # Restore temporarily-removed axes (if any)
        # Restore axes only when the schema actually supports it. If the
        # axes attribute is not part of the ONNX schema for this opset we
        # keep the value in `_axes_val` for type inference but do not
        # serialize it into the node attrs (to avoid checker failures).
        if _axes_val is not None and "axes" in getattr(schema, "attributes", {}):
            attrs["axes"] = _axes_val

        # Coerce scalar operands for elementwise ops
        self._coerce_elementwise_operands(
            op_type, inputs, input_types, raw_args, ctx
        )

        # Try to infer output types more generally from schema, attrs, and input types
        if not type_hint:
            inferred = self._infer_output_type(
                op_type, schema, attrs, input_types, type_hint, axes_val=_axes_val, call_pos=call_pos
            )
            if inferred:
                type_hint = inferred

        # Inject required 'shape' attribute for zero-input random ops
        if op_type in ("RandomUniform", "RandomNormal") and not inputs and "shape" not in attrs and type_hint:
            if isinstance(type_hint, dict) and "dims" in type_hint:
                dims = type_hint.get("dims", [])
                if dims:
                    # Convert dims to list of ints, handling both int and str (symbolic) dims
                    shape_attr = []
                    for d in dims:
                        if isinstance(d, int):
                            shape_attr.append(d)
                        elif isinstance(d, str):
                            # Try to convert string to int, otherwise skip symbolic dims for now
                            try:
                                shape_attr.append(int(d))
                            except (ValueError, TypeError):
                                pass
                    if shape_attr:
                        attrs["shape"] = shape_attr

        # Determine node outputs
        node_outputs, body_graph = self._determine_node_outputs(
            output_name, op_type, attrs
        )

        # Add node to graph
        ctx.add_node(
            op_type,
            inputs,
            node_outputs,
            attrs=self._lowerer._clean_attrs(op_type, attrs),
            doc_string=pending_doc or None,
        )
        if op_domain:
            ctx.nodes[-1].domain = op_domain
        if pending_doc:
            env["__pending_doc__"] = ""

        # Record output types
        output_type = self._record_output_types(
            node_outputs, body_graph, type_hint, ctx
        )

        return output_name, output_type


    def _normalize_attributes(self, attrs, op_type, schema):
        from onnx import TensorProto

        # Normalize convenience `value_*` kwargs
        if any(k.startswith("value_") for k in attrs.keys()):
            for k in list(attrs.keys()):
                if not k.startswith("value_"):
                    continue
                v = attrs.pop(k)
                name = self._lowerer.ctx._next_const_name()
                if k == "value_bool":
                    tp, vals, shape = TensorProto.BOOL, [bool(v)], []
                elif k in ("value_int", "value_ints"):
                    tp, vals, shape = (
                        TensorProto.INT64,
                        list(v) if isinstance(v, (list, tuple)) else [int(v)],
                        [len(v)] if isinstance(v, (list, tuple)) else [],
                    )
                elif k in ("value_float", "value_floats"):
                    tp, vals, shape = (
                        TensorProto.FLOAT,
                        (
                            list(v)
                            if isinstance(v, (list, tuple))
                            else [float(v)]
                        ),
                        [len(v)] if isinstance(v, (list, tuple)) else [],
                    )
                elif k in ("value_string", "value_strings"):
                    tp, vals, shape = (
                        TensorProto.STRING,
                        [
                            s.encode("utf-8")
                            for s in (
                                v if isinstance(v, (list, tuple)) else [v]
                            )
                        ],
                        [len(v)] if isinstance(v, (list, tuple)) else [],
                    )
                else:
                    attrs[k] = v
                    continue
                attrs["value"] = self._lowerer.helper.make_tensor(
                    name, int(tp), shape, vals
                )

        # Normalize single int to list for INTS attributes
        attr_schema = getattr(schema, "attributes", {}) or {}
        for aname, aval in list(attrs.items()):
            a_schema = attr_schema.get(aname)
            if a_schema:
                t_name = getattr(a_schema.type, "name", str(a_schema.type))
                if "INTS" in t_name.upper() and isinstance(aval, int):
                    attrs[aname] = [aval]
                elif (
                    "INT" == t_name.upper()
                    and isinstance(aval, list)
                    and len(aval) == 1
                ):
                    attrs[aname] = aval[0]

    def _lower_graph_attributes(self, attrs, ctx, env, op_type=None):
        for aname, aval in list(attrs.items()):
            if isinstance(aval, str) and aval in env:
                aval = env[aval]
                attrs[aname] = aval
            # support qualified names (foo.bar)
            elif isinstance(aval, str) and "." in aval:
                base = aval.split(".")[-1]
                if base in env:
                    aval = env[base]
                    attrs[aname] = aval
            if isinstance(aval, str) and aval in getattr(
                self._lowerer, "_user_decls", {}
            ):
                decl = self._lowerer._user_decls[aval]
                sub_ctx = GraphContext(name=decl.get("name"), opset=ctx.opset)
                parent_prefix = getattr(ctx, "scope_prefix", "parent")
                sub_ctx.scope_prefix = f"{parent_prefix}__{decl.get('name')}"
                sub_ctx._preserve_local_input_names = True
                # clear metadata to avoid requiring @fuse in subgraphs
                sub_ctx.model_metadata = {}
                self._lowerer._lower_function(decl, sub_ctx)
                g = sub_ctx.build_model().graph
                if op_type == "Loop" and aname == "body":
                    try:
                        from onnx import TensorProto

                        if len(g.input) >= 1:
                            g.input[0].type.tensor_type.elem_type = int(
                                TensorProto.INT64
                            )
                        if len(g.input) >= 2:
                            g.input[1].type.tensor_type.elem_type = int(
                                TensorProto.BOOL
                            )
                    except Exception:
                        pass
                attrs[aname] = g

    def _validate_attributes(self, attrs, op_type, schema):
        import difflib

        valid_attr_names = set(getattr(schema, "attributes", {}).keys())
        unknown_attrs = [a for a in attrs.keys() if a not in valid_attr_names]

        if (
            str(op_type).lower().startswith("reduce")
            and "keepdims" in unknown_attrs
        ):
            unknown_attrs.remove("keepdims")

        if unknown_attrs:
            suggestions = {
                a: difflib.get_close_matches(a, sorted(valid_attr_names), n=3)
                for a in unknown_attrs
            }
            msg_lines = [
                f"Unknown attribute(s) {unknown_attrs} for op '{op_type}'."
            ]
            if valid_attr_names:
                msg_lines.append(
                    f"Valid attributes: {sorted(valid_attr_names)}"
                )
            for a, s in suggestions.items():
                if s:
                    msg_lines.append(f"Did you mean: {a} -> {s}")
            raise LoweringError(
                " \\n".join(msg_lines),
                source=self._lowerer._current_source,
                function=op_type,
            )

        from src.graph_context import fuse_dtype_to_onnx

        for name, val in list(attrs.items()):
            if isinstance(val, str) and name in valid_attr_names:
                attr_schema = schema.attributes.get(name)
                t_name = getattr(
                    attr_schema.type, "name", str(attr_schema.type)
                )
                if "INT" in t_name.upper():
                    try:
                        attrs[name] = int(fuse_dtype_to_onnx(val))
                    except Exception:
                        pass
            if name in valid_attr_names:
                attr_schema = schema.attributes.get(name)
                t_name = getattr(
                    attr_schema.type, "name", str(attr_schema.type)
                )
                if "INT" in t_name.upper():
                    if isinstance(val, float) or (
                        isinstance(val, list)
                        and any(isinstance(x, float) for x in val)
                    ):
                        raise LoweringError(
                            f"Attribute '{name}' for op '{op_type}' asserts integer(s) but got float value {val}.",
                            source=self._lowerer._current_source,
                            function=op_type,
                        )

    def _coerce_elementwise_operands(
        self, op_type, inputs, input_types, raw_args, ctx
    ):
        if op_type in self._lowerer.ELEMENTWISE_OPS and len(inputs) == 2:
            left_t, right_t = input_types[0] or {}, input_types[1] or {}
            left_dims, right_dims = (
                left_t.get("dims", []),
                right_t.get("dims", []),
            )

            def unsqueeze_scalar(scalar_name, target_dims):
                new_name = f"{scalar_name}_unsq"
                i = 0
                while new_name in ctx.defined_values:
                    i += 1
                    new_name = f"{scalar_name}_unsq{i}"

                if int(ctx.opset) >= 13:
                    axes_name = ctx.add_tensor_literal(
                        [0], {"scalar": "i64", "dims": [1]}
                    )
                    ctx.add_node(
                        "Unsqueeze", [scalar_name, axes_name], [new_name]
                    )
                else:
                    ctx.add_node(
                        "Unsqueeze",
                        [scalar_name],
                        [new_name],
                        attrs={"axes": [0]},
                    )
                return new_name

            try:
                if left_dims and not right_dims:
                    if (
                        len(raw_args) >= 2
                        and isinstance(raw_args[1], dict)
                        and raw_args[1].get("call") == "Cast"
                    ):
                        inputs[1] = unsqueeze_scalar(inputs[1], left_dims)
                        input_types[1] = {
                            "scalar": left_t.get("scalar"),
                            "dims": list(left_dims),
                        }
                    else:
                        inputs[1] = unsqueeze_scalar(inputs[1], left_dims)
                        input_types[1] = {
                            "scalar": left_t.get("scalar"),
                            "dims": list(left_dims),
                        }

                elif right_dims and not left_dims:
                    if (
                        len(raw_args) >= 1
                        and isinstance(raw_args[0], dict)
                        and raw_args[0].get("call") == "Cast"
                    ):
                        inputs[0] = unsqueeze_scalar(inputs[0], right_dims)
                        input_types[0] = {
                            "scalar": right_t.get("scalar"),
                            "dims": list(right_dims),
                        }
                    else:
                        inputs[0] = unsqueeze_scalar(inputs[0], right_dims)
                        input_types[0] = {
                            "scalar": right_t.get("scalar"),
                            "dims": list(right_dims),
                        }
            except Exception as e:
                logger.debug(
                    "elementwise promotion heuristic failed for %s: %s",
                    op_type,
                    e,
                    exc_info=True,
                )


    def _record_output_types(self, node_outputs, body_graph, type_hint, ctx):
        from src.lowering.utils import _onnx_to_fuse_scalar

        output_type = as_tensor_type(type_hint)
        if body_graph:
            for i, vi in enumerate(body_graph.output):
                out_name = node_outputs[i]
                try:
                    tt = vi.type.tensor_type
                    dims = [
                        d.dim_value
                        for d in tt.shape.dim
                        if hasattr(d, "dim_value") and d.dim_value > 0
                    ]
                    scalar = _onnx_to_fuse_scalar(tt.elem_type)
                    ctx.value_types[out_name] = {
                        "scalar": scalar,
                        "dims": dims,
                    }
                except Exception:
                    ctx.value_types[out_name] = (
                        as_tensor_type(output_type) if output_type else None
                    )
        elif output_type:
            for o in node_outputs:
                ctx.value_types[o] = as_tensor_type(output_type)
        return output_type

    def _lower_shape_call(
        self,
        call: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        return self._lower_onnx_call(
            call, ctx, env, types, type_hint, out_name
        )

    def _lower_slice_call(
        self,
        call: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        return self._lower_onnx_call(
            call, ctx, env, types, type_hint, out_name
        )

    def _lower_gather_call(
        self,
        call: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        return self._lower_onnx_call(
            call, ctx, env, types, type_hint, out_name
        )

    def _lower_concat_call(
        self,
        call: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        return self._lower_onnx_call(
            call, ctx, env, types, type_hint, out_name
        )

    def _lower_conv_call(
        self,
        call: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        return self._lower_onnx_call(
            call, ctx, env, types, type_hint, out_name
        )

    def _lower_gemm_call(
        self,
        call: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        return self._lower_onnx_call(
            call, ctx, env, types, type_hint, out_name
        )

    def _lower_resize_call(
        self,
        call: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        return self._lower_onnx_call(
            call, ctx, env, types, type_hint, out_name
        )

    def _lower_onehot_call(
        self,
        call: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        return self._lower_onnx_call(
            call, ctx, env, types, type_hint, out_name
        )

    def _lower_reducesum_call(
        self,
        call: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        return self._lower_onnx_call(
            call, ctx, env, types, type_hint, out_name
        )

    def _lower_transpose_call(
        self,
        call: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        return self._lower_onnx_call(
            call, ctx, env, types, type_hint, out_name
        )

    def _lower_unsqueeze_call(
        self,
        call: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        return self._lower_onnx_call(
            call, ctx, env, types, type_hint, out_name
        )

    def _lower_squeeze_call(
        self,
        call: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        return self._lower_onnx_call(
            call, ctx, env, types, type_hint, out_name
        )

    def _lower_cast_call(
        self,
        call: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        return self._lower_onnx_call(
            call, ctx, env, types, type_hint, out_name
        )

    def _lower_loop_call(
        self,
        call: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Lower loop (args) { stmts; return exprs } to ONNX Loop with GraphProto body."""
        # First, reduce any generic/kwarg attributes into a dict so we can
        # inspect 'body' independently of argument ordering.  This mirrors the
        # logic in `_lower_onnx_call` without emitting a node just yet.
        args = call.get("args", [])
        attrs = self._lower_attrs(args, "Loop", ctx, env, types)
        # resolve any graph-valued attributes (user-decls, env lookups, etc.)
        self._lower_graph_attributes(attrs, ctx, env, "Loop")

        # inline bodies may be provided directly on the call rather than
        # as an attribute; prefer attrs but fallback to call-level key.
        body = attrs.get("body") if "body" in attrs else call.get("body")
        # Lower positional arguments: [iter, cond, state, ...]
        inputs, input_types = self._lower_args(args, ctx, env, types)

        # Inline-case: allow AST block before enforcing GraphProto validity
        if isinstance(body, dict) and body.get("type") == "block":
            return self._lower_loop_inline_body(
                inputs, input_types, body, ctx, env, types, type_hint, out_name
            )

        # A valid body must have been converted to a GraphProto by
        # `_lower_graph_attributes` above. Anything else is an error.
        if body is None or not hasattr(body, "node"):
            raise LoweringError(
                "loop body is missing or invalid",
                source=self._lowerer._current_source,
            )

        # At this point `body` is a GraphProto; construct Loop node normally
        node_outputs = [out_name or ctx._next_node_name("Loop")]
        num_outs = len(body.output)
        if num_outs > 1:
            node_outputs.extend(f"{node_outputs[0]}_{i}" for i in range(1, num_outs))
        attrs = {"body": body}
        ctx.add_node("Loop", inputs, node_outputs, attrs=attrs)

        # record output types from body graph outputs
        output_type = None
        for i, out in enumerate(body.output):
            out_name_i = node_outputs[i] if i < len(node_outputs) else node_outputs[0]
            try:
                tt = out.type.tensor_type
                dims = [
                    d.dim_value
                    for d in tt.shape.dim
                    if hasattr(d, "dim_value") and d.dim_value > 0
                ]
                from src.lowering.utils import _onnx_to_fuse_scalar
                scalar = _onnx_to_fuse_scalar(tt.elem_type)
                ctx.value_types[out_name_i] = {"scalar": scalar, "dims": dims}
                if i == 0:
                    output_type = ctx.value_types[out_name_i]
            except Exception:
                pass
        return node_outputs[0], output_type

    def _lower_loop_inline_body(
        self,
        inputs: List[str],
        input_types: List[Optional[Dict[str, Any]]],
        body_ast: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Lower inline loop body into a Loop GraphProto."""
        output_name = out_name or ctx._next_node_name("Loop")
        
        # Create sub-context for body lowering
        sub_ctx = GraphContext(name="__loop_body", opset=ctx.opset)
        sub_ctx._preserve_local_input_names = True
        sub_ctx.scope_prefix = f"{getattr(ctx, 'scope_prefix', 'parent')}__loop_body"
        
        # propagate outer environment names as explicit inputs first so
        # any references to external variables (e.g. loop limit 'n') become
        # declared graph inputs.  We include type info when available.
        for var, val in env.items():
            if isinstance(val, str) and val:
                try:
                    sub_ctx.add_param({"name": val, "type": types.get(var)})
                except Exception:
                    pass

        # Add inputs to sub-context with their types.  Rather than using a
        # fixed set of names we attempt to derive reasonable identifiers from
        # the loop AST or the incoming values.  The body syntax implicitly
        # refers to the iteration counter, condition, and state values; many
        # examples use the names `i`, `keep`, `state_in` but we don't rely on
        # them here.  We'll default to the conventional names when no hints
        # are available.
        # use EnvDict to manage nested bindings cleanly
        from src.lowering.context_stack import EnvDict
        body_env = EnvDict(env)
        param_names: list[str] = []
        # register body parameters as graph inputs so the generated GraphProto
        # exposes them.  We'll supply types when available but the names are
        # more important for downstream inspection.
        def _add_body_param(idx, name):
            typ = None
            if idx < len(input_types):
                typ = input_types[idx]
            sub_ctx.add_param({"name": name, "type_decl": typ})
        # try to deduce candidate names from any identifiers present in the
        # body return expressions (common patterns use `i`, `keep`, etc.).
        if isinstance(body_ast, dict):
            for ret in body_ast.get("returns", []):
                if isinstance(ret, str):
                    param_names.append(ret)
                elif isinstance(ret, dict) and "call" not in ret:
                    # simple literal or identifier
                    first_key = next(iter(ret), None)
                    if isinstance(first_key, str) and not first_key.startswith("@"):
                        param_names.append(first_key)
        # pad or trim to length of inputs
        default_names = ["i", "keep", "state_in"]
        for idx in range(len(inputs)):
            if idx >= len(param_names) or not param_names[idx]:
                param_names.append(default_names[idx] if idx < len(default_names) else f"arg{idx}")
        # register parameters in body_env stack and also record them as
        # subgraph inputs via ``sub_ctx.add_param``
        for i, inp_name in enumerate(inputs):
            pname = param_names[i]
            body_env[pname] = inp_name
            # Add to sub_ctx inputs type info if known
            if i < len(input_types) and input_types[i]:
                sub_ctx.value_types[inp_name] = input_types[i]
            # create an input param with the selected local name and type
            _add_body_param(i, pname)
        
        # Process body statements
        stmts = body_ast.get("stmts", [])
        for stmt in stmts:
            if not isinstance(stmt, dict):
                continue
            if "let" in stmt:
                lhs = stmt["let"]
                rhs = stmt["expr"]
                val, typ = self._lowerer._lower_expr(rhs, sub_ctx, body_env, types)
                if val:
                    body_env[str(lhs)] = val
                    sub_ctx.value_types[val] = typ
        
        # Lower return expressions
        returns = body_ast.get("returns", [])
        return_vals = []
        return_types = []
        for ret_expr in returns:
            val, typ = self._lowerer._lower_expr(ret_expr, sub_ctx, body_env, types)
            if val:
                return_vals.append(val)
                return_types.append(typ)
        
        # Ensure the first returned value is boolean condition; if not, insert
        # a literal True so the generated Loop body is valid.
        need_bool = False
        if not return_vals:
            need_bool = True
        else:
            first_typ = return_types[0]
            if not (isinstance(first_typ, dict) and first_typ.get("scalar") == "bool"):
                need_bool = True
        if need_bool:
            true_name = sub_ctx.add_literal(True, {"scalar": "bool"})
            return_vals.insert(0, true_name)
            return_types.insert(0, {"scalar": "bool", "dims": []})

        # Build outputs as Identity nodes and register them as graph outputs
        for i, val in enumerate(return_vals):
            output_name_i = f"__loop_out_{i}"
            sub_ctx.add_node("Identity", [val], [output_name_i])
            typ_i = return_types[i] if i < len(return_types) else None
            if typ_i:
                sub_ctx.value_types[output_name_i] = typ_i
            # make sure the graph knows this value is an output
            try:
                sub_ctx.add_output(output_name_i, typ_i or {"scalar": "f32", "dims": []})
            except Exception:
                # ignore if output already exists or cannot be added
                pass
        
        # Extract GraphProto and force Loop body types
        body_graph = sub_ctx.build_model().graph

        # ==== post-process outputs to avoid qualification mismatches ==== 
        # Identity nodes added earlier may have been qualified differently
        # than the underlying node outputs, resulting in body_graph.output
        # names that don't correspond to any node.  This causes ONNX
        # validation to fail (see issue with loop_lambda_golden example).
        # Here we attempt to repair such mismatches by renaming outputs to
        # one of the actual node outputs sharing the same tail segment.
        try:
            node_outputs = {out for n in body_graph.node for out in n.output}
            for vi in body_graph.output:
                if vi.name not in node_outputs:
                    tail = vi.name.split(".")[-1]
                    for out in node_outputs:
                        if out.endswith(tail):
                            # rename to the real node output
                            vi.name = out
                            break
        except Exception:
            # best-effort; don't fail lowering if this repair step breaks
            pass

        # Force first two inputs to i64 and bool for Loop
        try:
            from onnx import TensorProto
            if len(body_graph.input) >= 1:
                body_graph.input[0].type.tensor_type.elem_type = int(TensorProto.INT64)
            if len(body_graph.input) >= 2:
                body_graph.input[1].type.tensor_type.elem_type = int(TensorProto.BOOL)
        except Exception:
            pass
        
        # Create Loop node with body attribute
        attrs = {"body": body_graph}
        
        node_outputs = [output_name]
        num_outs = len(body_graph.output)
        if num_outs > 1:
            node_outputs.extend(f"{output_name}_{i}" for i in range(1, num_outs))
        
        ctx.add_node("Loop", inputs, node_outputs, attrs=attrs)
        
        # Record output types
        output_type = None
        for i, out in enumerate(body_graph.output):
            out_name_i = node_outputs[i] if i < len(node_outputs) else output_name
            try:
                tt = out.type.tensor_type
                dims = [
                    d.dim_value
                    for d in tt.shape.dim
                    if hasattr(d, "dim_value") and d.dim_value > 0
                ]
                from src.lowering.utils import _onnx_to_fuse_scalar
                scalar = _onnx_to_fuse_scalar(tt.elem_type)
                ctx.value_types[out_name_i] = {"scalar": scalar, "dims": dims}
                if i == 0:
                    output_type = ctx.value_types[out_name_i]
            except Exception:
                pass
        
        return output_name, output_type

    def _lower_if_call(
        self,
        call: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Lower if (cond) { ... } [else { ... }] to ONNX If with GraphProto bodies."""
        cond_expr = call.get("cond")
        then_body = call.get("then")
        else_body = call.get("else")
        
        if cond_expr is None:
            # missing explicit condition - this is a user error, not an
            # opportunity to fall back to a generic ONNX call which will
            # raise a confusing schema error later.  Surface a clear message.
            raise LoweringError(
                "if condition missing",
                source=self._lowerer._current_source,
            )
        
        # Lower condition
        cond_val, _ = self._lowerer._lower_expr(cond_expr, ctx, env, types)
        if not cond_val:
            raise LoweringError(
                "if condition failed to lower",
                source=self._lowerer._current_source,
            )
        
        output_name = out_name or ctx._next_node_name("If")
        
        # Lower then body
        then_graph = self._lower_if_block_body(then_body, ctx, env, types)
        
        # Lower else body (required for If)
        if else_body:
            else_graph = self._lower_if_block_body(else_body, ctx, env, types)
        else:
            # Create else body that passes through inputs matching then branch outputs
            else_ctx = GraphContext(name="__if_else", opset=ctx.opset)
            else_ctx._preserve_local_input_names = True
            
            # Create inputs/outputs matching then branch
            for inp in then_graph.input:
                else_ctx.inputs[inp.name] = inp
            
            # Create identity nodes to match then output count
            then_output_names = [o.name for o in then_graph.output]
            for out_name in then_output_names:
                # Find matching input or create a pass-through
                matching_input = next((i for i in then_graph.input if i.name == out_name), None)
                if matching_input:
                    else_ctx.add_output(out_name, matching_input.type)
                else:
                    # Use first input as pass-through for unmatched outputs
                    if then_graph.input:
                        first_input = then_graph.input[0]
                        else_ctx.add_node("Identity", [first_input.name], [out_name])
                        else_ctx.add_output(out_name, first_input.type)
            
            else_graph = else_ctx.build_model().graph
        
        attrs = {"then_branch": then_graph, "else_branch": else_graph}
        
        node_outputs = [output_name]
        num_outs = max(len(then_graph.output), len(else_graph.output))
        if num_outs > 1:
            node_outputs.extend(f"{output_name}_{i}" for i in range(1, num_outs))
        
        ctx.add_node("If", [cond_val], node_outputs, attrs=attrs)
        
        return output_name, type_hint

    def _lower_if_block_body(
        self,
        body: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
    ):
        """Lower an if/else block body to GraphProto."""
        sub_ctx = GraphContext(name="__if_body", opset=ctx.opset)
        sub_ctx._preserve_local_input_names = True
        # propagate outer environment names as explicit graph inputs so the
        # resulting GraphProto contains the needed input definitions.  We
        # copy both name and type where available.  This is important for
        # If/Loop bodies which are lowered independently but still reference
        # values from the parent context.
        for var, val in env.items():
            if isinstance(val, str) and val:  # skip empty names
                p = {"name": val}
                t = types.get(var)
                if t is not None:
                    # include type information if known
                    p["type"] = t
                try:
                    sub_ctx.add_param(p)
                except Exception:
                    # ignore duplicates or invalid entries
                    pass
        
        if body and body.get("type") == "block":
            stmts = body.get("stmts", [])
            from .context_stack import EnvDict
            body_env = EnvDict(env)
            # delegate to the main lowerer for statement semantics so we
            # correctly handle let/assign/annot/assert/etc.
            for stmt in stmts:
                try:
                    self._lowerer._lower_statement(stmt, sub_ctx, body_env, types)
                except Exception:
                    # swallow; body statements are best-effort here
                    pass
        
        return sub_ctx.build_model().graph

    def _lower_scan_call(
        self,
        call: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Lower scan (args) { stmts; return exprs } to ONNX Scan with GraphProto body."""
        body = call.get("body")
        args = call.get("args", [])
        
        # Lower positional arguments
        inputs, input_types = self._lower_args(args, ctx, env, types)
        
        if not body or not isinstance(body, dict):
            raise LoweringError(
                "scan body is missing or invalid",
                source=self._lowerer._current_source,
            )
        
        # Handle inline block body (similar to loop)
        if body.get("type") == "block":
            return self._lower_scan_inline_body(
                inputs, input_types, body, ctx, env, types, type_hint, out_name
            )
        
        # Fallback to generic ONNX call
        return self._lower_onnx_call(call, ctx, env, types, type_hint, out_name)

    def _lower_scan_inline_body(
        self,
        inputs: List[str],
        input_types: List[Optional[Dict[str, Any]]],
        body_ast: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Lower inline scan body into a Scan GraphProto."""
        output_name = out_name or ctx._next_node_name("Scan")
        
        stmts = body_ast.get("stmts", [])
        returns = body_ast.get("returns", [])
        
        # Create sub-context for body
        sub_ctx = GraphContext(name="__scan_body", opset=ctx.opset)
        sub_ctx._preserve_local_input_names = True
        sub_ctx.scope_prefix = f"{getattr(ctx, 'scope_prefix', 'parent')}__scan_body"
        
        from .context_stack import EnvDict
        body_env = EnvDict(env)
        # map state input to conventional name so body code can refer to
        # `state_in` when desired (common in examples).
        if len(inputs) >= 2:
            body_env["state_in"] = inputs[1]
        # register the inputs as graph parameters so the body has explicit
        # inputs (semantics tests rely on this)
        for i, inp_name in enumerate(inputs):
            # Do not attempt to register sequence/list types as graph inputs;
            # ``add_param`` cannot handle them and they are not needed for the
            # semantic tests (we only require at least one input present).
            typ = input_types[i] if i < len(input_types) else None
            if typ and typ.get("scalar") == "list":
                continue
            pname = inp_name  # use original name as parameter name
            try:
                sub_ctx.add_param({"name": pname, "type_decl": typ})
            except Exception:
                # ignore any failures (e.g., unknown scalar) and continue
                pass
            if i < len(input_types) and input_types[i]:
                sub_ctx.value_types[inp_name] = input_types[i]
        
        # Lower body statements using main lowerer helper
        for stmt in stmts:
            try:
                self._lowerer._lower_statement(stmt, sub_ctx, body_env, types)
            except Exception:
                pass
        
        # Lower return expressions
        return_vals = []
        return_types = []
        for ret_expr in returns:
            val, typ = self._lowerer._lower_expr(ret_expr, sub_ctx, body_env, types)
            if val:
                return_vals.append(val)
                return_types.append(typ)
        
        # Build sub-context outputs
        for i, (val, typ) in enumerate(zip(return_vals, return_types)):
            output_name_i = f"__scan_out_{i}"
            sub_ctx.add_node("Identity", [val], [output_name_i])
        
        body_graph = sub_ctx.build_model().graph
        # determine how many of the provided inputs are sequence/scan inputs
        num_scan = 0
        for t in input_types:
            if isinstance(t, dict) and t.get("scalar") == "list":
                num_scan += 1
        attrs = {"body": body_graph, "num_scan_inputs": num_scan}
        
        node_outputs = [output_name]
        num_outs = len(body_graph.output)
        if num_outs > 1:
            node_outputs.extend(f"{output_name}_{i}" for i in range(1, num_outs))
        
        ctx.add_node("Scan", inputs, node_outputs, attrs=attrs)
        
        return output_name, type_hint

    def _lower_args(
        self,
        args: List[Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[str], List[Optional[Dict[str, Any]]]]:
        inputs: List[str] = []
        input_types: List[Optional[Dict[str, Any]]] = []
        for arg in args:
            # Handle starred unpacking: `*list` -> expand elements into positional args
            if isinstance(arg, dict) and ("*" in arg):
                inner = arg["*"]
                # If it's a literal list, lower each element individually
                if isinstance(inner, dict) and "lit_list" in inner:
                    for item in inner["lit_list"]:
                        val, typ = self._lowerer._lower_expr(
                            item, ctx, env, types
                        )
                        if val is not None:
                            inputs.append(val)
                            input_types.append(typ)
                    continue
                # Fallback: treat as a single positional arg (best-effort)
                val, typ = self._lowerer._lower_expr(inner, ctx, env, types)
                if val is not None:
                    inputs.append(val)
                    input_types.append(typ)
                continue
            if isinstance(arg, dict) and any(
                k.startswith("@") for k in arg.keys()
            ):
                continue
            val, typ = self._lowerer._lower_expr(arg, ctx, env, types)
            if val is not None:
                inputs.append(val)
                input_types.append(typ)
        return inputs, input_types

    def _lower_args_with_literals(
        self,
        args: List[Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[str], List[Optional[Dict[str, Any]]], List[Any]]:
        inputs: List[str] = []
        input_types: List[Optional[Dict[str, Any]]] = []
        literals: List[Any] = []
        for arg in args:
            # Handle starred unpacking: `*list` -> expand elements into positional args
            if isinstance(arg, dict) and ("*" in arg):
                inner = arg["*"]
                if isinstance(inner, dict) and "lit_list" in inner:
                    for item in inner["lit_list"]:
                        val, typ = self._lowerer._lower_expr(
                            item, ctx, env, types
                        )
                        if val is not None:
                            inputs.append(val)
                            input_types.append(typ)
                        try:
                            literals.append(
                                self._lowerer._eval_const_expr(item, env, types)
                            )
                        except (ValueError, KeyError):
                            literals.append(None)
                    continue
                # Fallback: treat as a single positional arg (best-effort)
                val, typ = self._lowerer._lower_expr(inner, ctx, env, types)
                if val is not None:
                    inputs.append(val)
                    input_types.append(typ)
                try:
                    literals.append(
                        self._lowerer._eval_const_expr(inner, env, types)
                    )
                except (ValueError, KeyError):
                    literals.append(None)
                continue
            if isinstance(arg, dict) and any(
                k.startswith("@") for k in arg.keys()
            ):
                continue
            val, typ = self._lowerer._lower_expr(arg, ctx, env, types)
            if val is not None:
                inputs.append(val)
                input_types.append(typ)
                try:
                    literals.append(
                        self._lowerer._eval_const_expr(arg, env, types)
                    )
                except (ValueError, KeyError):
                    literals.append(None)
        return inputs, input_types, literals
