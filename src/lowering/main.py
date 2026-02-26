import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

from ..graph_context import GraphContext, as_tensor_type, fuse_dtype_to_onnx
from ..import_fusion import ImportManager
from ..onnx_opset import validate_opset_version
from ..onnx_schema import normalize_domain_and_op
from ..errors import E030_LoweringError
from .ops import OpsLowerer
from .utils import LoweringError
from .training_info_emit import TRAINING_DOMAIN
# helper for domain lookup moved to graph_context to avoid circular imports
from src.graph_context import get_model_domain as _get_model_domain
from .type_inference import TypeInferencer
from .graph_qualifier import GraphQualifier

logger = logging.getLogger(__name__)


class FuseLowerer:
    ELEMENTWISE_OPS = {"Add", "Sub", "Mul", "Div"}

    def __init__(
        self,
        imports: Optional[Dict[str, Any]] = None,
        refresh_cache: bool = False,
        refresh_imports: Optional[List[str]] = None,
        max_pow_folds: int = 8,
        fold_externalize_mb: int = 0,
        fold_externalize_dir: Optional[str] = None,
        emit_training: bool = False,
        import_manager: Optional[ImportManager] = None,
        embed_external_data: bool = False,
        strict: bool = False,
        inline_functions: bool = True,
    ): 
        self.import_manager = (
            import_manager if import_manager is not None else ImportManager()
        )
        self.inline_functions = bool(inline_functions)
        self.imports = imports or {}
        self._current_source: Optional[str] = None
        self._current_module: Optional[str] = None
        self.refresh_cache: bool = bool(refresh_cache)
        self.refresh_imports = set(refresh_imports or [])
        self.max_pow_folds: int = int(max_pow_folds or 0)
        self.fold_externalize_mb = int(fold_externalize_mb or 0)
        self.fold_externalize_dir = fold_externalize_dir
        self.emit_training: bool = bool(emit_training)
        self.embed_external_data: bool = bool(embed_external_data)
        self.strict: bool = bool(strict)
        from onnx import helper as onnx_helper
        self.helper = onnx_helper
        self.ops_lowerer = OpsLowerer(self)
        self.ctx: Optional[GraphContext] = None
        self.type_inferencer: Optional[TypeInferencer] = None

    def lower(
        self,
        ast,
        ctx: Optional[GraphContext] = None,
        source_file: Optional[str] = None,
        name_allocator=None,
        compact: bool = False,
        target: Optional[str] = None,
    ):
        """Lower AST into an ONNX model or into an existing GraphContext.

        Optional `name_allocator` may be provided to control emitted node/const
        naming for deterministic tests and external tooling.

        When `target` is provided, only the top-level `model` declaration whose
        name matches `target` will be emitted; other `model` declarations are
        skipped. This supports emitting one ONNX per declared graph in a file.
        """
        # run early normalization/type‑shape passes
        from .passes import NormalizationPass, TypeShapePass
        ast = NormalizationPass().run(ast)
        ast = TypeShapePass().run(ast)

        own_ctx = ctx is None
        # Compact mode suppresses emitting an initial Identity 'entry' node
        self._compact = bool(compact)
        # Store optional per-call target model name for use during lowering
        self._target_model = str(target) if target is not None else None
        self.ctx, declarations = self._prepare_for_lowering(
            ast, ctx, source_file, name_allocator=name_allocator
        )

        model_inputs, model_outputs, has_explicit_model = (
            self._lower_declarations(self.ctx, declarations)
        )

        # If requested, run the gradient/TrainingInfo generation pass once.
        if getattr(self, "emit_training", False):
            try:
                from .gradients import generate_gradients
                from .training_info_emit import emit_training_info

                grad_summary = generate_gradients(self.ctx)
                emit_training_info(self.ctx, grad_summary)
            except Exception as e:
                if getattr(self, "strict", False):
                    raise
                logger.warning("training emission failed: %s", e)
        
        self._finalize_model(
            self.ctx, has_explicit_model, model_inputs, model_outputs
        )

        # After finalization, ensure a simple flag exists so tooling can
        # detect a training variant even if structured metadata was omitted.
        if own_ctx and getattr(self, "emit_training", False):
            try:
                if "training" not in self.ctx.model_metadata:
                    self.ctx.model_metadata["training"] = True
            except Exception:
                pass

        # Final augmentation: if a training.loss reference was supplied in
        # metadata but no loss_binding was emitted, look for a suitable
        # algorithm output and attach the binding to an existing TrainingInfo.
        try:
            tm = self.ctx.model_metadata.get("training")
            training_meta = tm if isinstance(tm, dict) else {}
            training_cfg = self.ctx.model_metadata.get("training_config") or {}
            loss_candidate = training_meta.get("loss") or (
                training_cfg.get("loss") if isinstance(training_cfg, dict) else None
            )
            if loss_candidate:
                for ti in getattr(self.ctx, "_training_info", []) or []:
                    try:
                        existing = list(ti.loss_binding)
                        if existing:
                            continue
                    except Exception:
                        # attribute might be a shim or missing; treat as empty
                        pass
                    # inspect algorithm outputs
                    try:
                        outs = [o.name for o in ti.algorithm.output]
                    except Exception:
                        outs = []
                    found = None
                    for k in outs:
                        if str(k).endswith(str(loss_candidate)) or str(loss_candidate) in str(k) or str(k).endswith("loss"):
                            found = k
                            break
                    if not found:
                        for k in outs:
                            if "loss" in str(k):
                                found = k
                                break
                    if found:
                        try:
                            ti.loss_binding[str(loss_candidate)] = found
                        except Exception:
                            try:
                                e = ti.loss_binding.add()
                                e.key = str(loss_candidate)
                                e.value = found
                            except Exception:
                                try:
                                    lb = getattr(ti, "loss_binding")
                                    class _KV:
                                        def __init__(self, key, value):
                                            self.key = key
                                            self.value = value

                                    lb.append(_KV(str(loss_candidate), found))
                                except Exception:
                                    pass
        except Exception:
            pass

            # Ensure training_info exists when an explicit algorithm graph is
            # present even if gradient generation did not emit optimizer nodes.
            try:
                if not getattr(self.ctx, "_training_info", None):
                    tm = self.ctx.model_metadata.get("training")
                    training_meta = tm if isinstance(tm, dict) else {}
                    if training_meta.get("algorithm_graph") is not None:
                        from .training_info_emit import emit_training_info

                        emit_training_info(self.ctx, {"opt_updates": {}, "optimizer_nodes": []})
            except Exception:
                pass

        if own_ctx:
            model = self.ctx.build_model()
            # Post-process model.training_info to ensure explicit training loss
            # references are propagated into loss_binding when possible.
            try:
                tm = self.ctx.model_metadata.get("training")
                training_meta = tm if isinstance(tm, dict) else {}
                training_cfg = self.ctx.model_metadata.get("training_config") or {}
                loss_candidate = training_meta.get("loss") or (
                    training_cfg.get("loss") if isinstance(training_cfg, dict) else None
                )
                if loss_candidate:
                    for ti in model.training_info:
                        try:
                            # skip if binding already present
                            has = bool(list(ti.loss_binding))
                            if has:
                                continue
                        except Exception:
                            pass
                        # inspect algorithm outputs
                        try:
                            outs = [o.name for o in ti.algorithm.output]
                        except Exception:
                            outs = []
                        found = None
                        for k in outs:
                            if str(k).endswith(str(loss_candidate)) or str(loss_candidate) in str(k) or str(k).endswith("loss"):
                                found = k
                                break
                        if not found:
                            for k in outs:
                                if "loss" in str(k):
                                    found = k
                                    break
                        if found:
                            try:
                                ti.loss_binding[str(loss_candidate)] = found
                            except Exception:
                                try:
                                    e = ti.loss_binding.add()
                                    e.key = str(loss_candidate)
                                    e.value = found
                                except Exception:
                                    try:
                                        lb = getattr(ti, "loss_binding")
                                        class _KV:
                                            def __init__(self, key, value):
                                                self.key = key
                                                self.value = value

                                        lb.append(_KV(str(loss_candidate), found))
                                    except Exception:
                                        pass
            except Exception:
                pass
            return model
        return None

    def _prepare_for_lowering(
        self,
        ast,
        ctx: Optional[GraphContext],
        source_file: Optional[str],
        name_allocator=None,
    ):
        self._current_source = source_file
        incoming_ctx = ctx

        # Normalize inline lambdas into named functions first so lowering
        # can treat them as ordinary declarations (deterministic function names
        # are allocated by the normalizer).
        from ..ast.normalize_lambdas import normalize_lambdas

        declarations = self._flatten(ast)
        declarations = normalize_lambdas(declarations)
        self.type_aliases = {}
        self._user_decls = {
            d.get("name"): d
            for d in declarations
            if isinstance(d, dict)
            and d.get("type") in ("node", "model", "export")
            and d.get("name")
        }

        called: set[str] = set()
        from collections import defaultdict

        callers: dict[str, set[str]] = defaultdict(set)

        def _collect_calls(x: Any, current_kind: Optional[str] = None):
            if isinstance(x, dict):
                if "call" in x and isinstance(x.get("call"), str):
                    called.add(x.get("call"))
                    if current_kind:
                        callers[x.get("call")].add(current_kind)
                for v in x.values():
                    _collect_calls(v, current_kind)
            elif isinstance(x, list):
                for i in x:
                    _collect_calls(i, current_kind)

        for d in declarations:
            kind = d.get("type") if isinstance(d, dict) else None
            _collect_calls(d, kind)

        self._called_user_decls = {c for c in called if c in self._user_decls}
        self._callers = {k: set(v) for k, v in callers.items()}

        node_names = [n for n in list(self._user_decls.keys()) if n]
        ctx = ctx or GraphContext(
            name=node_names[0] if len(node_names) == 1 else "fused_model",
            name_allocator=name_allocator,
            embed_external_data=self.embed_external_data,
        )
        # Mark root context so nested subgraphs can be differentiated from
        # top-level models; top-level lowering should still require explicit
        # @fuse metadata while nested graphs may default to the runtime version.
        if incoming_ctx is None:
            setattr(ctx, "_is_root", True)
        
        # Initialize type inferencer for this context
        self.type_inferencer = TypeInferencer(ctx)

        # Namespacing is enabled by default: when a source file is provided
        # require an explicit @domain (formerly @module) declaration. If
        # consumers want to opt out they should use the CLI flag `--no-ns`.
        has_domain_decl = any(
            isinstance(d, dict)
            and d.get("type") == "meta"
            and d.get("name") in ("domain", "module")
            for d in declarations
        )
        if (
            incoming_ctx is None
            and self._current_source
            and not has_domain_decl
            and node_names
        ):
            raise LoweringError(
                "Namespacing requires a domain (@domain) declaration (use --no-ns to disable)"
            )


        # Post-pass: ensure any separately-declared algorithm `node`s are
        # captured as algorithm GraphProto entries in model metadata even when
        # their configuration isn't discoverable earlier. This aids cases
        # where `@training` metadata may not have been attached as a meta
        # declaration (parser subtlety) but an explicit `@algorithm` node is
        # present.
        try:
            for decl in declarations:
                if isinstance(decl, dict) and decl.get("type") == "fn" and decl.get("algorithm"):
                    # Skip if already recorded
                    if isinstance(ctx.model_metadata.get("training"), dict) and "algorithm_graph" in ctx.model_metadata.get("training", {}):
                        continue
                    try:
                        sub_ctx = GraphContext(name=decl.get("name"), opset=ctx.opset)
                        sub_ctx.scope_prefix = f"{ctx.scope_prefix}__{decl.get('name')}" if getattr(ctx, "scope_prefix", None) else decl.get("name")
                        sub_ctx._preserve_local_input_names = True
                        # Use a temporary lowerer to avoid polluting current lowering
                        from src.lowering import FuseLowerer as _FuseLowerer

                        tmp_lowerer = _FuseLowerer(emit_training=self.emit_training, embed_external_data=self.embed_external_data, strict=self.strict, import_manager=self.import_manager)
                        tmp_lowerer._lower_function(decl, sub_ctx)
                        g = sub_ctx.build_model().graph
                        try:
                            for n in g.node:
                                n.domain = TRAINING_DOMAIN
                        except Exception:
                            pass
                        ctx.model_metadata.setdefault("training", {})["algorithm_graph"] = g
                    except Exception:
                        # best-effort: continue
                        pass
        except Exception:
            pass

        return ctx, declarations

    def _lower_declarations(
        self, ctx: GraphContext, declarations: List[Dict[str, Any]]
    ):
        # run GraphLoweringPass as part of the new pipeline; currently a no-op
        from .passes import GraphLoweringPass
        GraphLoweringPass().run(declarations, ctx)

        model_inputs: set[str] = set()
        model_outputs: set[str] = set()
        has_explicit_model = False

        for decl in declarations:
            if not isinstance(decl, dict):
                continue
            if decl.get("_inline_only"):
                continue
            kind = decl.get("type")
            if kind == "meta":
                self._apply_meta(ctx, decl)
            elif kind == "param":
                # If a parameter has a default value (common for `weight`),
                # treat it as a constant initializer instead of a runtime input.
                if decl.get("value") is not None:
                    c = dict(decl)
                    resolved = self._resolve_type(c.get("type_decl"))
                    if resolved is not None:
                        c["type_decl"] = resolved
                    val = c.get("value")
                    if (
                        isinstance(val, dict)
                        and "imported_tensors" in val
                        and self._current_source
                    ):
                        from pathlib import Path

                        nodeame = val["imported_tensors"].get("file")
                        src_path = (
                            Path(self._current_source).parent / Path(nodeame)
                        ).resolve()
                        c["value"]["imported_tensors"]["src"] = str(src_path)
                    name = ctx.add_const(c)
                    # propagate trainable into model metadata when present
                    if decl.get("trainable") is not None:
                        ctx.model_metadata.setdefault("trainables", {})[
                            name
                        ] = bool(decl.get("trainable"))
                else:
                    graph_name = ctx.add_param(decl)
                    if decl.get("trainable") is not None:
                        ctx.model_metadata.setdefault("trainables", {})[
                            graph_name
                        ] = bool(decl.get("trainable"))
            elif kind == "const":
                c = dict(decl)
                resolved = self._resolve_type(c.get("type_decl"))
                if resolved is not None:
                    c["type_decl"] = resolved
                val = c.get("value")
                if (
                    isinstance(val, dict)
                    and "imported_tensors" in val
                    and self._current_source
                ):
                    from pathlib import Path

                    nodeame = val["imported_tensors"].get("file")
                    src_path = (
                        Path(self._current_source).parent / Path(nodeame)
                    ).resolve()
                    c["value"]["imported_tensors"]["src"] = str(src_path)
                name = ctx.add_const(c)
                # propagate trainable for standalone consts too
                if decl.get("trainable") is not None:
                    ctx.model_metadata.setdefault("trainables", {})[name] = (
                        bool(decl.get("trainable"))
                    )
            elif kind == "import":
                refresh = (
                    self.refresh_cache
                    or (decl.get("name") in self.refresh_imports)
                    or (decl.get("alias") in self.refresh_imports)
                )
                self.import_manager.fuse_import(ctx, decl, refresh=refresh)
            elif kind == "type_alias":
                self.type_aliases[decl["name"]] = decl.get("type_decl")
            elif kind == "proof":
                continue
            elif kind in ("node", "model", "export"):
                # Special-case user-defined nodes/functions: when the caller
                # requested function inlining we lower their bodies inline as
                # before.  When `inline_functions` is False we instead convert
                # the declaration into a reusable FunctionProto and skip
                # lowering here.  Models and exports are always lowered normally
                # (they form the graph entrypoints).
                if kind == "node" and not getattr(self, "inline_functions", False):
                    # skip lowering to graph; emit a FunctionProto instead
                    try:
                        self._emit_function_proto(decl, ctx)
                        # record that this function has been seen so calls can
                        # be mapped to a function op later
                        self._user_decls.setdefault(decl.get("name"), decl)
                    except Exception:
                        # fallback: inline if protos cannot be emitted
                        self._lower_function(decl, ctx)
                    continue

                # If a per-call target model was provided, skip unrelated
                # top-level `model` declarations to support emitting one
                # ONNX per declared graph inside the source file.
                if getattr(self, "_target_model", None) is not None and decl.get("type") == "model" and decl.get("name") != self._target_model:
                    logger.debug("skipping model %s due to target filter %s", decl.get("name"), self._target_model)
                    continue

                name = decl.get("name")
                try:
                    any_subgraph = any(
                        (
                            isinstance(
                                self._resolve_type(
                                    p.get("type") or p.get("type_decl")
                                ),
                                dict,
                            )
                            and self._resolve_type(
                                p.get("type") or p.get("type_decl")
                            ).get("scalar")
                            == "subgraph"
                        )
                        for p in decl.get("params", [])
                    )
                except Exception:
                    any_subgraph = False

                if (
                    name
                    and name in getattr(self, "_called_user_decls", set())
                    and decl.get("type") == "node"
                ):
                    callers = getattr(self, "_callers", {})
                    called_only_by_tests = name in callers and callers[
                        name
                    ] <= {"proof"}
                    if any_subgraph and not called_only_by_tests:
                        logger.debug(
                            "skipping emission of function %s due to subgraph param",
                            name,
                        )
                        continue
                    if not called_only_by_tests:
                        continue

                    has_subgraph_param = False
                    for p in decl.get("params", []):
                        try:
                            resolved_p = self._resolve_type(
                                p.get("type") or p.get("type_decl")
                            )
                        except Exception:
                            resolved_p = None
                        if (
                            isinstance(resolved_p, dict)
                            and resolved_p.get("scalar") == "subgraph"
                        ):
                            has_subgraph_param = True
                            break
                    logger.debug(
                        "decl=%s, has_subgraph_param=%s, in_called=%s",
                        name,
                        has_subgraph_param,
                        name in getattr(self, "_called_user_decls", set()),
                    )
                    if has_subgraph_param:
                        logger.debug(
                            "skipping emission of function %s due to subgraph param",
                            name,
                        )
                        continue

                    callers = getattr(self, "_callers", {})
                    called_only_by_tests = name in callers and callers[
                        name
                    ] <= {"proof"}
                    if not called_only_by_tests:
                        continue
                try:
                    inputs_before = set(ctx.inputs.keys())
                    outputs_before = set(ctx.outputs.keys())
                    self._lower_function(decl, ctx)
                    inputs_after = set(ctx.inputs.keys())
                    outputs_after = set(ctx.outputs.keys())
                    added_inputs = inputs_after - inputs_before
                    added_outputs = outputs_after - outputs_before
                    if decl.get("type") == "model":
                        has_explicit_model = True
                        model_inputs.update(added_inputs)
                        model_outputs.update(added_outputs)
                except Exception as e:
                    name = decl.get("name") if isinstance(decl, dict) else None
                    msg = f"{name or '<unknown>'}: {e}"
                    raise LoweringError(
                        msg, source=self._current_source, function=name
                    ) from e
            else:
                raise TypeError(f"Unasserted AST declaration kind: {kind}")
        return model_inputs, model_outputs, has_explicit_model

    def _finalize_model(
        self,
        ctx: GraphContext,
        has_explicit_model: bool,
        model_inputs: set[str],
        model_outputs: set[str],
    ):
        if has_explicit_model:
            ctx.inputs = {
                k: v for k, v in ctx.inputs.items() if k in model_inputs
            }
            # Preserve gradient outputs added by generate_gradients (training mode)
            ctx.outputs = {
                k: v for k, v in ctx.outputs.items()
                if k in model_outputs or k.endswith(".grad")
            }

        # Ensure module-level declared parameters that are referenced by
        # lowered nodes (but not explicitly listed in the model signature)
        # are preserved as graph inputs so ONNX validation accepts them.
        # This can happen for top-level `@train weight` declarations that are
        # used inside a particular model but were not declared as its params.
        try:
            from onnx import helper as onnx_helper

            referenced = set()
            for n in ctx.nodes:
                for inp in n.input:
                    referenced.add(inp)
            # Try direct matches first, then fallback to matching by suffix
            for ref in sorted(referenced):
                if ref in ctx.inputs or ref in ctx.initializers:
                    continue
                if ref in ctx.value_types:
                    t = ctx.value_types[ref]
                    dims = list(t.get("dims") or [])
                    vi = onnx_helper.make_tensor_value_info(ref, fuse_dtype_to_onnx(t.get("scalar") or "f32"), dims)
                    ctx.inputs[ref] = vi
                    continue
                # fallback: match keys whose tail segment matches the reference
                for name, t in list(ctx.value_types.items()):
                    tail = name.split(".")[-1]
                    if tail == ref and name not in ctx.inputs and name not in ctx.initializers:
                        dims = list(t.get("dims") or [])
                        vi = onnx_helper.make_tensor_value_info(name, fuse_dtype_to_onnx(t.get("scalar") or "f32"), dims)
                        ctx.inputs[name] = vi
                        break
        except Exception:
            # Best-effort: do not fail lowering on inference errors here
            pass

        node_names = [n for n in list(self._user_decls.keys()) if n]
        if node_names and len(node_names) == 1:
            pass

        # If lowering to a training variant is requested, do not eagerly
        # mutate model_metadata here (it may be populated by the gradient
        # pass later). We will annotate the metadata after gradient
        # emission to avoid clobbering structured training dicts.

    def _flatten(self, ast):
        flat: List[Any] = []
        if isinstance(ast, list):
            for item in ast:
                flat.extend(self._flatten(item))
        else:
            flat.append(ast)
        return flat

    def _apply_meta(self, ctx: GraphContext, decl: Dict[str, Any]):
        if (
            decl.get("name") == "opset"
            and isinstance(decl.get("value"), list)
            and len(decl["value"]) >= 2
        ):
            domain = str(decl["value"][0])
            version = (
                validate_opset_version(int(decl["value"][1]))
                if domain == "onnx"
                else int(decl["value"][1])
            )
            if domain == "onnx":
                ctx.opset = version
            else:
                ctx.extra_opsets[domain] = max(
                    int(ctx.extra_opsets.get(domain, 0)), int(version)
                )

        if decl.get("name") in ("domain", "module"):
            val = str(decl.get("value"))
            self._current_module = val
            # always store under the canonical key
            ctx.model_metadata["domain"] = val
            # keep legacy key too for a short transition period
            if decl.get("name") == "module":
                ctx.model_metadata.setdefault("module", val)
                warnings.warn(
                    "@module metadata is deprecated; use @domain instead", DeprecationWarning
                )

        if decl.get("name") == "id":
            self._current_module = str(decl.get("value"))
            ctx.model_metadata["@id"] = str(decl.get("value"))

        if decl.get("name") == "type":
            self._current_module = str(decl.get("value"))
            ctx.model_metadata["@type"] = str(decl.get("value"))

        if decl.get("name") == "meta" and isinstance(decl.get("value"), dict):
            for k, v in decl["value"].items():
                ctx.model_metadata[str(k)] = v

        # Support top-level @version X.Y.Z metadata (map to `version` key)
        if decl.get("name") == "version":
            ver = str(decl.get("value"))
            import re
            is_semver = bool(re.match(r"^\d+\.\d+\.\d+$", ver))
            # If author supplied a full semantic version, we prefer to use
            # the Fuse package version as the canonical emitted 'version'
            # metadata. For shorthand/incomplete versions, non-strict mode
            # will record the author's value; strict mode rejects it.
            if is_semver:
                if getattr(self, "strict", False) and not is_semver:
                    # This branch is unreachable but kept for clarity
                    raise LoweringError(
                        f"invalid @version '{ver}': expected semantic version 'MAJOR.MINOR.PATCH'",
                        source=self._current_source,
                    )
                # Valid semver: do not override package 'version' metadata.
                pass
            else:
                # Non-semver values: accept in non-strict mode, reject in strict
                if getattr(self, "strict", False):
                    raise LoweringError(
                        f"invalid @version '{ver}': expected semantic version 'MAJOR.MINOR.PATCH'",
                        source=self._current_source,
                    )
                ctx.model_metadata["version"] = ver

        # Support top-level @fuse VERSION metadata to pin runtime fuse
        if decl.get("name") == "fuse":
            try:
                ctx.model_metadata["fuse"] = str(decl.get("value"))
            except Exception:
                pass
        # Module-level training configuration produced by `@training { ... }`.
        # Store as `training_config` to be consumed by gradient/optimizer passes.
        if decl.get("name") == "fuse.training" and isinstance(decl.get("value"), dict):
            ctx.model_metadata["training_config"] = decl.get("value")
        # Backwards-friendly: accept plain `@training { ... }` as well (tests and
        # some user sources use the shorthand form).
        if decl.get("name") == "training" and isinstance(decl.get("value"), dict):
            ctx.model_metadata["training_config"] = decl.get("value")


    def _resolve_type(self, typ: Optional[Any]) -> Optional[Dict[str, Any]]:
        if typ is None:
            return None
        type_aliases = getattr(self, "type_aliases", {})
        if isinstance(typ, str):
            if typ in type_aliases:
                return type_aliases[typ]
            return {"scalar": typ, "dims": []}
        if isinstance(typ, dict):
            scalar = typ.get("scalar")
            if isinstance(scalar, str) and scalar in type_aliases:
                return type_aliases[scalar]
            return typ
        return None

    def _emit_function_proto(self, decl: Dict[str, Any], ctx: GraphContext):
        """Convert a user-declared function/`node` into an ONNX FunctionProto.

        The body is lowered into a temporary GraphContext and then turned into a
        FunctionProto which is appended to ``ctx.functions``.
        """
        from onnx import FunctionProto

        # Detect subgraph-typed parameters; FunctionProto cannot express
        # runtime subgraph arguments (they are static attributes in ONNX).
        any_subgraph = False
        for p in decl.get("params", []):
            try:
                resolved = self._resolve_type(p.get("type") or p.get("type_decl"))
            except Exception:
                resolved = None
            if isinstance(resolved, dict) and resolved.get("scalar") == "subgraph":
                any_subgraph = True
                break
        if any_subgraph:
            # fall back to inline lowering so the subgraph parameter is handled
            self._lower_function(decl, ctx)
            return

        sub_ctx = GraphContext(name=decl.get("name"), opset=ctx.opset)
        sub_ctx.scope_prefix = (
            f"{ctx.scope_prefix}__{decl.get('name')}"
            if getattr(ctx, "scope_prefix", None)
            else decl.get("name")
        )
        sub_ctx._preserve_local_input_names = True
        # Lower the declaration as though it were a normal function; this
        # populates sub_ctx.nodes/inputs/outputs appropriately.
        try:
            self._lower_function(decl, sub_ctx)
        except Exception as e:
            # if lowering fails, we fall back to inlining as a best-effort
            if getattr(self, "inline_functions", False):
                # inline into parent context rather than emitting a proto
                self._lower_function(decl, ctx)
                return
            # otherwise propagate the original error so callers can see it
            raise
        # build a full ModelProto so we can access opset_import and graph
        model_proto = sub_ctx.build_model()
        g = model_proto.graph
        # compute a domain for this function.  Use the declared model
        # domain (fallback to deprecated module) or a stable sentinel so
        # custom functions never live in the empty/builtin domain.
        func_domain = _get_model_domain(ctx) or "fuse.local"

        func = FunctionProto()
        if decl.get("name"):
            func.name = decl.get("name")
        func.domain = func_domain
        # copy inputs/outputs
        for inp in g.input:
            func.input.append(inp.name)
        for out in g.output:
            func.output.append(out.name)
        # copy nodes and other relevant fields
        for n in g.node:
            func.node.append(n)
        # include shape/type info for all values: inputs, outputs, and others
        for vi in g.input:
            func.value_info.append(vi)
        for vi in g.output:
            func.value_info.append(vi)
        for vi in g.value_info:
            func.value_info.append(vi)
        # copy opset imports from model_proto
        for o in model_proto.opset_import:
            func.opset_import.append(o)
        func.doc_string = g.doc_string
        # record domain back on declaration so callers can reference it
        decl["_func_domain"] = func_domain
        ctx.add_function(func)

    def _lower_function(self, decl: Dict[str, Any], ctx: GraphContext):
        # use EnvDict for nested bindings and shadowing
        from .context_stack import EnvDict
        env: EnvDict = EnvDict({})
        types: Dict[str, Dict[str, Any]] = dict(ctx.value_types)
        env["__pending_doc__"] = ""
        # Track original local return names (e.g., `loss`) so they can be
        # exposed as graph outputs in training variants.
        returned_locals: list[str] = []

        old_prefix = getattr(ctx, "scope_prefix", None)
        old_scope_display = getattr(ctx, "scope_display", None)
        scope = self._make_scope(decl.get("name"))
        if scope:
            ctx.scope_prefix = scope
            ctx.scope_display = scope

        if isinstance(decl, dict) and decl.get("type") == "model":
            # Only set model '@id' metadata when explicitly provided by the
            # author via an `@id` pragma. Do not auto-inject a scope-derived
            # id (e.g., module.name) as this should be a user-provided stable
            # identifier suitable for publication (IRI or CURIE).
            if decl.get("@id") is not None:
                ctx.model_metadata["@id"] = decl.get("@id")

        # Track current function name for downstream diagnostics to
        # attach more precise locations for lowering errors.
        old_current_function = getattr(self, "_current_function", None)
        self._current_function = decl.get("name")

        for p in decl.get("params", []):
            typ_decl = p.get("type") or p.get("type_decl")
            resolved = self._resolve_type(typ_decl)
            p2 = dict(p)
            if resolved is not None:
                p2["type_decl"] = resolved
            graph_name = ctx.add_param(p2)
            env[p["name"]] = graph_name
            types[p["name"]] = as_tensor_type(
                resolved or p.get("type") or p.get("type_decl")
            )

        # Unless compact mode is requested, emit a first Identity node to act
        # as an explicit, named entry point for the graph. This helps
        # visualization tools (e.g. Netron) display a clear starting node.
        try:
            # Only emit the optional entry Identity for model (graph) declarations
            # to avoid changing node-level lowering and breaking tests that rely
            # on deterministic node ordering and naming.
            if (
                not getattr(self, "_compact", False)
                and decl.get("params")
                and decl.get("type") == "model"
            ):
                first_p = decl.get("params")[0]
                pname = first_p.get("name")
                # original graph-visible name for the parameter
                input_name = env.get(pname, pname)
                # Entry output name (qualify to avoid collisions)
                entry_name = f"{decl.get('name')}_{pname}_entry"
                q_entry = ctx.qualify_name(entry_name)
                # Decide a friendly node name: prefer `module.<func>` when a
                # module/namespace is declared; fall back to the previous
                # `<func>.entry` name when no module is available.
                module_name = _get_model_domain(ctx) or self._current_module
                if module_name:
                    node_name_str = f"{module_name}.{decl.get('name')}"
                else:
                    node_name_str = f"{decl.get('name')}.entry"
                # Insert Identity node at start (index 0) and wire subsequent
                # lowering to use the entry output instead of the raw input.
                node_name = ctx.insert_node(0, "Identity", [input_name], [q_entry], name=node_name_str)
                # Advance the name allocator to account for the manually
                # inserted 'first' node so subsequent auto-generated names do
                # not collide (StableNameAllocator would otherwise reuse the
                # scope_display as the first generated node name).
                try:
                    ctx._next_node_name("Identity")
                except Exception:
                    pass
                # Redirect lowering to use the entry Identity output so
                # subsequent nodes are wired to the inserted Identity. This
                # mirrors the behavior used for annotated inputs and ensures
                # the entry node actually acts as the graph's input source.
                try:
                    if pname in ctx.value_types:
                        ctx.value_types[q_entry] = ctx.value_types[pname]
                    # Temporarily redirect lowering to use the entry Identity
                    # output for subsequent node emissions in this function.
                    # We'll restore the original env mapping after lowering
                    # the function body to preserve canonical external names.
                    # Use output rename mapping to ensure subsequent nodes
                    # consume the entry Identity without modifying `env`.
                    _old_output_rename = ctx._output_renames.get(input_name)
                    ctx._output_renames[input_name] = q_entry
                    # Track whether the entry name was already a graph input
                    _entry_was_input = q_entry in ctx.inputs
                except Exception:
                    _old_output_rename = None
                    _entry_was_input = False
                    pass
        except Exception:
            # Best-effort: do not make lowering fail for minor visualization
            # enhancements.
            pass

        # Handle input annotations: validate keys, record metadata, and insert
        # an Identity node that will be convenient for dev workflows (named
        # `{module}.{arg}`) and may be removed during optimization passes.
        inputs_meta = decl.get("input") or {}
        if inputs_meta:
            for k, v in inputs_meta.items():
                if k not in env:
                    raise LoweringError(
                        f"@input annotation targets unknown parameter '{k}'",
                        source=self._current_source,
                    )
                # make unique temp output name for the identity node
                base = f"{k}_annot"
                tmp = base
                i = 0
                while tmp in ctx.defined_values:
                    i += 1
                    tmp = f"{base}{i}"
                module_name = (
                    _get_model_domain(ctx) or self._current_module
                )
                ident_name = f"{module_name}.{k}" if module_name else f"{k}"
                ctx.add_node("Identity", [env[k]], [tmp], name=ident_name)
                # redirect parameter binding to identity output
                env[k] = tmp
                # propagate type and record metadata
                ctx.value_types[tmp] = types.get(k)
                ctx.model_metadata.setdefault("fuse.inputs", {})[k] = v

        quant_meta = decl.get("quantize")
        quant_scale_name = None
        quant_zp_name = None
        if quant_meta:
            target = str(quant_meta.get("target") or "").lower()
            if target == "auto":
                ret = decl.get("ret_type")
                scalar = None
                if isinstance(ret, dict):
                    scalar = ret.get("scalar")
                elif isinstance(ret, str):
                    scalar = ret
                if scalar and str(scalar).startswith("i"):
                    target = "int8"
                elif scalar in ("f16", "bf16"):
                    target = "fp16"
                else:
                    target = "int8"

            scale_val = float(quant_meta.get("scale", 1.0))
            zero_point = int(quant_meta.get("zero_point", 0))
            quant_scale_name = ctx.add_tensor_literal(
                [scale_val], {"scalar": "f32", "dims": [1]}
            )
            quant_zp_name = ctx.add_tensor_literal(
                [zero_point], {"scalar": "i8", "dims": [1]}
            )
            from onnx import TensorProto

            for p in decl.get("params", []):
                pname = p["name"]
                ptype = types.get(pname) or ctx.value_types.get(pname)
                if not ptype or ptype.get("scalar") != "f32":
                    continue
                qname = ctx._next_const_name()
                pname_graph = env.get(pname, ctx.qualify_name(pname))
                if target in ("int8", "i8"):
                    ctx.add_node(
                        "QuantizeLinear",
                        [pname_graph, quant_scale_name, quant_zp_name],
                        [qname],
                        doc_string=None,
                    )
                    ctx.value_types[qname] = {
                        "scalar": "i8",
                        "dims": ptype.get("dims", []),
                    }
                elif target in ("fp16", "f16"):
                    ctx.add_node(
                        "Cast",
                        [pname_graph],
                        [qname],
                        attrs={"to": int(TensorProto.FLOAT16)},
                        doc_string=None,
                    )
                    ctx.value_types[qname] = {
                        "scalar": "f16",
                        "dims": ptype.get("dims", []),
                    }
                else:
                    continue
                env[pname] = qname
                types[pname] = ctx.value_types[qname]

        last_value: Optional[str] = None
        last_type: Optional[Dict[str, Any]] = None
        body = decl.get("body") or []
        # Normalize parser edge-cases where a bare call may be emitted as two
        # adjacent items: an IDENT followed by an args-list (e.g., `Add` and
        # `['y','B']`), which can happen for bare expression statements.
        # Combine such fragments into a single call node so lowering can
        # treat them uniformly.
        if isinstance(body, list) and body:
            normalized_body = []
            i = 0
            while i < len(body):
                cur = body[i]
                if (
                    isinstance(cur, str)
                    and i + 1 < len(body)
                    and isinstance(body[i + 1], list)
                    and all(isinstance(a, (str, int, float, dict)) for a in body[i + 1])
                ):
                    normalized_body.append({"call": cur, "args": body[i + 1]})
                    i += 2
                    continue
                normalized_body.append(cur)
                i += 1
            body = normalized_body

        try:
            for stmt in body:
                try:
                    val, typ = self._lower_statement(stmt, ctx, env, types)
                except TypeError as e:
                    # Provide a succint helpful error with context to aid debugging
                    raise LoweringError(
                        f"Error in {stmt!r} in function {decl.get('name')}: {e}",
                        source=self._current_source,
                        function=decl.get("name"),
                    ) from e
                if val is not None:
                    last_value, last_type = val, typ
        finally:
            # Restore any temporary output renames we added for the entry
            try:
                if '_old_output_rename' in locals():
                    if _old_output_rename is None:
                        ctx._output_renames.pop(input_name, None)
                    else:
                        ctx._output_renames[input_name] = _old_output_rename
                # If we inserted an entry Identity and it was not originally
                # present as a graph input, ensure we don't leave its output
                # mistakenly registered as a graph input (violates SSA).
                if '_entry_was_input' in locals() and not _entry_was_input:
                    ctx.inputs.pop(q_entry, None)
            except Exception:
                pass

        multi = env.get("__last_multi_return__")
        if multi:
            outputs_meta = decl.get("output") or {}
            for idx, (nm, typ) in enumerate(multi):
                internal = (
                    nm
                    if isinstance(nm, str)
                    else f"{decl.get('name')}_out_{idx}"
                )
                out_type = as_tensor_type(
                    (
                        decl.get("ret_type")[idx]
                        if isinstance(decl.get("ret_type"), list)
                        and idx < len(decl.get("ret_type"))
                        else typ
                    )
                    or typ
                )
                # If there's an output annotation for this named return, insert
                # an Identity node named `{module}.{name}` and export its output
                if isinstance(nm, str) and nm in outputs_meta:
                    base = f"{internal}_annot"
                    tmp = base
                    i = 0
                    while tmp in ctx.defined_values:
                        i += 1
                        tmp = f"{base}{i}"
                    module_name = (
                        _get_model_domain(ctx) or self._current_module
                    )
                    ident_name = f"{module_name}.{nm}" if module_name else f"{nm}"
                    ctx.add_node("Identity", [internal], [tmp], name=ident_name)
                    internal = tmp
                    ctx.model_metadata.setdefault("fuse.outputs", {})[nm] = outputs_meta[nm]

                ctx.add_output(internal, out_type)
            return (
                multi[0][0]
                if isinstance(multi[0][0], str)
                else f"{decl.get('name')}_out_0"
            )

        if last_value is None and decl.get("params"):
            first = decl["params"][0]
            last_value = first["name"]
            last_type = types.get(last_value) or as_tensor_type(
                first.get("type") or first.get("type_decl")
            )
            if "dequantize" in decl:
                logger.debug(
                    "inserting dequant for function %s param %s",
                    decl.get("name"),
                    last_value,
                )
                dequant_meta = decl.get("dequantize") or {}
                scale_val = float(dequant_meta.get("scale", 1.0))
                zero_point = int(dequant_meta.get("zero_point", 0))
                quant_scale_name = ctx.add_tensor_literal(
                    [scale_val], {"scalar": "f32", "dims": [1]}
                )
                quant_zp_name = ctx.add_tensor_literal(
                    [zero_point], {"scalar": "i8", "dims": [1]}
                )
                dq_name = f"{last_value}_dq"
                input_name = env.get(last_value, last_value)
                ctx.add_node(
                    "DequantizeLinear",
                    [input_name, quant_scale_name, quant_zp_name],
                    [dq_name],
                )
                last_value = dq_name
                last_type = {
                    "scalar": "f32",
                    "dims": last_type.get("dims") if last_type else [],
                }

        if scope and (old_prefix is not None or old_scope_display is not None):
            ctx.scope_prefix = old_prefix
            ctx.scope_display = old_scope_display

        # If the declaration provided output annotations without named returns,
        # fail early: output annotations require named returns (e.g., `return {name: val}`)
        if decl.get("output") and not env.get("__last_multi_return__"):
            raise LoweringError(
                "@output annotations require named returns (use `return { name: val }`).",
                source=self._current_source,
            )

        # If this function was annotated as the canonical loss, record it
        # in the model metadata for later binding by training emission.
        if decl.get("loss"):
            try:
                ctx.model_metadata.setdefault("training", {})["loss"] = decl.get("name")
            except Exception:
                ctx.model_metadata["training"] = {"loss": decl.get("name")}

        # If this function was annotated as a custom algorithm, lower it
        # into a standalone GraphProto and stash it in model metadata so
        # TrainingInfo emission can prefer it instead of synthesized ops.
        training_cfg = ctx.model_metadata.get("training_config") or {}
        requested_alg = training_cfg.get("algorithm") if isinstance(training_cfg, dict) else None
        if (
            decl.get("algorithm")
            or (isinstance(requested_alg, str) and requested_alg == decl.get("name"))
            # Treat an explicit top-level `model` declaration as an algorithm
            # when training configuration is present (explicit algorithm
            # graphs may be provided via `graph`/`model`). This lets authors
            # declare an algorithm signature that will be preferred over a
            # synthesized optimizer-node graph.
            or (decl.get("type") == "model" and isinstance(training_cfg, dict) and training_cfg)
        ):
            # Lower the function into an isolated sub-GraphContext
            try:
                sub_ctx = GraphContext(name=decl.get("name"), opset=ctx.opset)
                sub_ctx.scope_prefix = f"{ctx.scope_prefix}__{decl.get('name')}" if getattr(ctx, "scope_prefix", None) else decl.get("name")
                sub_ctx._preserve_local_input_names = True
                self._lower_function(decl, sub_ctx)
                g = sub_ctx.build_model().graph
                # Ensure nodes in the algorithm graph are in the training domain
                try:
                    for n in g.node:
                        n.domain = TRAINING_DOMAIN
                except Exception:
                    pass
                ctx.model_metadata.setdefault("training", {})["algorithm_graph"] = g
            except Exception:
                # Best-effort: do not fail lowering if algorithm graph extraction fails
                pass

        ret_type = decl.get("ret_type") or last_type
        # Resolve named type aliases for return types (e.g., `-> Latent`)
        resolved_ret = self._resolve_type(ret_type) or ret_type
        output_type = as_tensor_type(resolved_ret or last_type)

        if "dequantize" in decl and last_value is not None:
            dequant_meta = decl.get("dequantize") or {}
            from onnx import TensorProto

            target = None
            if quant_meta:
                target = str(quant_meta.get("target") or "").lower()
            if not quant_scale_name:
                scale_val = float(dequant_meta.get("scale", 1.0))
                zero_point = int(dequant_meta.get("zero_point", 0))
                quant_scale_name = ctx.add_tensor_literal(
                    [scale_val], {"scalar": "f32", "dims": [1]}
                )
                quant_zp_name = ctx.add_tensor_literal(
                    [zero_point], {"scalar": "i8", "dims": [1]}
                )
            dq_name = ctx._next_const_name()
            if types.get(last_value, {}).get("scalar") in ("f16",):
                input_name = env.get(last_value, last_value)
                ctx.add_node(
                    "Cast",
                    [input_name],
                    [dq_name],
                    attrs={"to": int(TensorProto.FLOAT)},
                )
                last_value = dq_name
                last_type = {
                    "scalar": "f32",
                    "dims": last_type.get("dims") if last_type else [],
                }
            else:
                input_name = env.get(last_value, last_value)
                ctx.add_node(
                    "DequantizeLinear",
                    [input_name, quant_scale_name, quant_zp_name],
                    [dq_name],
                )
                last_value = dq_name
                last_type = {
                    "scalar": "f32",
                    "dims": last_type.get("dims") if last_type else [],
                }

        if isinstance(last_value, str):
            last_value = env.get(last_value, last_value)

        # Prefer exposing the actual internal value as the graph output. This
        # avoids emitting a synthetic Identity node that merely renames the
        # internal result to the declaration name (e.g., `model`). Using the
        # internal name lets `add_output` qualify it deterministically.
        if decl.get("type") == "model":
            if last_value:
                graph_output_internal = last_value
            elif decl.get("params"):
                # no body: the first parameter is the implicit return
                param_one = decl.get("params")[0].get("name")
                graph_output_internal = env.get(param_one, param_one)
            else:
                graph_output_internal = decl.get("name")
        else:
            if last_value:
                graph_output_internal = last_value
            else:
                graph_output_internal = f"{decl['name']}_out"

        graph_out = ctx.add_output(graph_output_internal, output_type)

        # If the function explicitly returned a local variable name (e.g.,
        # `return loss`), expose that original local name as an additional
        # graph output so downstream training tooling can reference it by
        # name. Use the same `output_type` as the function's return.
        for rn in returned_locals:
            try:
                # Use the internal value bound to the returned local (if any)
                # when creating the graph-output so downstream passes that
                # inspect producer nodes can locate the correct value.
                internal = env.get(rn, rn)
                if internal not in ctx.outputs:
                    ctx.add_output(internal, output_type)
            except Exception:
                # Best-effort: don't fail lowering due to naming clashes
                pass

        # Restore previous current function context
        try:
            self._current_function = old_current_function
        except Exception:
            try:
                delattr(self, "_current_function")
            except Exception:
                pass

        return graph_out

    def _lower_statement(
        self,
        stmt: Any,
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        if isinstance(stmt, dict):
            if "let" in stmt:
                if isinstance(stmt["let"], (list, tuple)):
                    targets = stmt["let"]
                    expr = stmt["expr"]
                    if isinstance(expr, dict) and expr.get("call"):
                        for i, tgt in enumerate(targets):
                            sel_expr = {
                                "__call_select__": {"call": expr, "idx": i}
                            }
                            val, typ = self._lower_expr(
                                sel_expr, ctx, env, types
                            )
                            if val is not None:
                                env[str(tgt)] = val
                                if typ:
                                    types[str(tgt)] = typ
                        return None, None
                    tmp = ctx._next_const_name()
                    val, typ = self._lower_expr(
                        expr, ctx, env, types, out_name=tmp
                    )
                    for i, tgt in enumerate(targets):
                        idx_expr = {"call": "__index__", "args": [tmp, i]}
                        v, t = self._lower_expr(idx_expr, ctx, env, types)
                        if v is not None:
                            env[str(tgt)] = v
                            if t:
                                types[str(tgt)] = t
                    return None, None

                val, typ = self._lower_expr(
                    stmt["expr"], ctx, env, types, out_name=stmt["let"]
                )
                if val is not None:
                    env[stmt["let"]] = val
                    if typ:
                        types[stmt["let"]] = typ
                return val, typ
            if "assign" in stmt:
                typ_hint = self._resolve_type(stmt.get("type"))
                val, typ = self._lower_expr(
                    stmt["expr"],
                    ctx,
                    env,
                    types,
                    type_hint=typ_hint,
                    out_name=stmt["assign"],
                )
                if val is not None:
                    env[stmt["assign"]] = val
                    types[stmt["assign"]] = as_tensor_type(typ_hint or typ)
                return val, types.get(stmt["assign"])
            if "assert" in stmt:
                expr = stmt["assert"]
                try:
                    val = self._eval_const_expr(expr, env, types)
                    if bool(val):
                        return None, None
                    raise LoweringError(
                        f"Static assertion failed: {self._expr_to_str(expr)}",
                        source=self._current_source,
                    )
                except ValueError:
                    s = self._expr_to_str(expr)
                    asserts = ctx.model_metadata.get("fuse.asserts", [])
                    asserts.append(s)
                    ctx.model_metadata["fuse.asserts"] = asserts
                    return None, None
            if "note" in stmt:
                if not ctx.graph_doc_string:
                    ctx.graph_doc_string = stmt["note"]
                env["__pending_doc__"] = stmt["note"]
                return None, None
            if "annot" in stmt:
                return None, None
            if "return" in stmt:
                if isinstance(stmt.get("return"), (list, tuple)):
                    lowered = []
                    for e in stmt.get("return"):
                        n, t = self._lower_expr(e, ctx, env, types)
                        lowered.append((n, t))
                    env["__last_multi_return__"] = lowered
                    return (lowered[0][0] if lowered else None), (
                        lowered[0][1] if lowered else None
                    )

                # Support named return mapping syntax: `return { name: expr }`.
                # Named return mapping: `return { name: expr }`.
                # Avoid misclassifying call nodes (which are also dicts with
                # string keys such as 'call'/'args') by ensuring the mapping
                # does not contain an operator call entry.
                if (
                    isinstance(stmt.get("return"), dict)
                    and all(isinstance(k, str) for k in stmt.get("return"))
                ):
                    # Disambiguate between an expression-like dict (e.g.,
                    # infix call: {'left':..., 'ops': [...]}) and an actual
                    # named return mapping `{ name: expr }`. If the dict
                    # contains expression-internal keys, do not treat it as
                    # a mapping. This ensures forms like `return x + y` or
                    # `return Add(x, y)` are handled as expressions, not
                    # as name-to-expr mappings.
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
                    if not (set(stmt.get("return").keys()) & expr_internal_keys):
                        lowered = []
                        for k, v in stmt.get("return").items():
                            n, t = self._lower_expr(v, ctx, env, types)
                            lowered.append((k, t))
                        env["__last_multi_return__"] = lowered
                        return (lowered[0][0] if lowered else None), (
                            lowered[0][1] if lowered else None
                        )

                if (
                    isinstance(stmt.get("return"), dict)
                    and stmt.get("return").get("index")
                    and isinstance(stmt.get("return").get("selector"), dict)
                    and stmt.get("return").get("selector").get("slice")
                ):
                    base = stmt.get("return").get("index")
                    sl = stmt.get("return").get("selector").get("slice")
                    start, stop = sl
                    if (
                        start == 1
                        and stop is None
                        and isinstance(base, dict)
                        and base.get("call")
                    ):
                        self._lower_call(base, ctx, env, types)
                        op = base.get("call")
                        op_domain, op_type = normalize_domain_and_op(str(op))
                        if op_type in self.import_manager.fused_signatures:
                            sig = self.import_manager.fused_signatures[op_type]
                            import_outputs = sig.get("outputs") or []
                            lowered = []
                            for nm in import_outputs[1:]:
                                lowered.append((nm, ctx.value_types.get(nm)))
                            env["__last_multi_return__"] = lowered
                            return lowered[0][0] if lowered else None, (
                                lowered[0][1] if lowered else None
                            )
                val, typ = self._lower_expr(stmt["return"], ctx, env, types)
                return val, typ
        return self._lower_expr(stmt, ctx, env, types)

    def _lower_expr(
        self,
        expr: Any,
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        # Support a special selection wrapper used by tuple-destructuring
        # assignment lowering: {"__call_select__": {"call": <call>, "idx": i}}
        # This lowers the inner call and returns the i-th output name/type.
        if isinstance(expr, dict) and "__call_select__" in expr:
            sel = expr["__call_select__"]
            call_expr = sel.get("call")
            idx = int(sel.get("idx"))
            # Clear any stale multi-return marker before lowering to avoid
            # accidental reuse of a previous call's marker.
            prev_multi = env.pop("__last_multi_return__", None)
            # Lower the inner call directly (ensures imports / inlining run)
            val, typ = self.ops_lowerer._lower_call(
                call_expr, ctx, env, types, type_hint=type_hint, out_name=out_name
            )
            multi = env.get("__last_multi_return__")
            if multi:
                if idx < len(multi):
                    nm, t = multi[idx]
                    return nm, t
                from .utils import LoweringError
                raise LoweringError(
                    f"call returned {len(multi)} values; selection index {idx} out of range",
                    source=self._current_source,
                )
            # Single-output call: only valid to select index 0
            if idx == 0:
                return val, typ
            from .utils import LoweringError
            raise LoweringError(
                "attempt to select non-zero index from single-output call",
                source=self._current_source,
            )

        if isinstance(expr, dict):
            if "call" in expr:
                return self.ops_lowerer._lower_call(
                    expr,
                    ctx,
                    env,
                    types,
                    type_hint=type_hint,
                    out_name=out_name,
                )
            if "left" in expr and "ops" in expr:
                return self._lower_infix(
                    expr, ctx, env, types, type_hint=type_hint
                )
            if "if" in expr:
                # `if` expressions are represented in the AST as a tuple of
                # parts: (cond, then_node[, else_node]).  Earlier versions
                # simply lowered the true branch and ignored the condition,
                # resulting in missing environment bindings (see smoke tests)
                # and stray string literals.  Convert the tuple into a
                # pseudo-ONNX call and delegate to the existing
                # `_lower_if_call` implementation which handles subgraph
                # generation, condition lowering, and output wiring.
                parts = expr.get("if") or ()
                if not parts:
                    return None, None

                cond = parts[0]
                then_node = parts[1] if len(parts) > 1 else None
                else_node = parts[2] if len(parts) > 2 else None

                def _wrap(node):
                    # the if call helper expects a body dict with
                    # ``type: 'block'``; the parser returns a bare list of
                    # statements for a `{ ... }` node, so normalize it here.
                    if node is None:
                        return None
                    if isinstance(node, list):
                        return {"type": "block", "stmts": node, "returns": []}
                    return node

                call = {
                    "call": "if",
                    "cond": cond,
                    "then": _wrap(then_node),
                    "else": _wrap(else_node) if else_node is not None else None,
                }
                return self.ops_lowerer._lower_if_call(
                    call,
                    ctx,
                    env,
                    types,
                    type_hint=type_hint,
                    out_name=out_name,
                )

        if isinstance(expr, str):
            # Normalize boolean literal strings to actual booleans
            if expr in ("true", "false"):
                lit_type = {"scalar": "bool", "dims": []}
                name = ctx.add_literal(expr == "true", lit_type)
                return name, lit_type
            if expr in getattr(self, "_user_decls", {}):
                return expr, {"scalar": "subgraph", "dims": []}
            if expr in env:
                return env[expr], types.get(expr)
            if (
                expr in ctx.value_types
                or expr in ctx.inputs
                or expr in ctx.initializers
            ):
                return expr, types.get(expr) or ctx.value_types.get(expr)
            # DEBUG: detect cases where operator names are being treated as literals
            if expr in ("MatMul", "Cast", "Greater"):
                # debugging helper: unexpected operator names treated as literals
                logger.debug("stray operator string treated as literal: %s", expr)
                logger.debug("env keys: %s", list(env.keys()))
                logger.debug("ctx.value_types keys: %s", list(ctx.value_types.keys())[:20])
                try:
                    import traceback

                    traceback.print_stack(file=None)
                except Exception:
                    pass
            name = ctx.add_literal(expr, as_tensor_type(type_hint))
            return name, as_tensor_type(type_hint)

        if isinstance(expr, bool):
            lit_type = {"scalar": "bool", "dims": []}
            name = ctx.add_literal(expr, lit_type)
            return name, lit_type

        if isinstance(expr, (int, float)):
            lit_type = as_tensor_type(type_hint)
            name = ctx.add_literal(expr, lit_type)
            return name, lit_type

        if isinstance(expr, dict) and "lit_list" in expr:
            lit_type = as_tensor_type(type_hint)
            name = ctx.add_tensor_literal(expr["lit_list"], lit_type)
            return name, lit_type

        return None, None

    def _lower_infix(
        self,
        expr: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
    ):
        """Lower infix expressions (unchanged)"""
        left_name, left_type = self._lower_expr(
            expr["left"], ctx, env, types, type_hint=type_hint
        )
        current_name, current_type = left_name, left_type
        for op_entry in expr.get("ops", []):
            raw_op = op_entry["op"]
            if not (
                isinstance(raw_op, str)
                and len(raw_op) <= 2
                and raw_op in "+-*/@⊕"
            ):
                raw_op = "+"
            op_type = self._map_operator(raw_op)
            if current_name is None:
                break
            env[current_name] = current_name
            types[current_name] = current_type or as_tensor_type(type_hint)
            call = {"call": op_type, "args": [current_name, op_entry["right"]]}
            current_name, current_type = self.ops_lowerer._lower_call(
                call, ctx, env, types, type_hint=current_type
            )
        return current_name, current_type

    def _map_operator(self, op_token: str) -> str:
        return {
            "+": "Add",
            "-": "Sub",
            "*": "Mul",
            "/": "Div",
            "@": "MatMul",
            "⊕": "Add",
        }.get(op_token, op_token)

    def _inline_user_decl(
        self,
        decl: Dict[str, Any],
        call: Dict[str, Any],
        ctx: GraphContext,
        env: Dict[str, str],
        types: Dict[str, Dict[str, Any]],
        type_hint: Optional[Dict[str, Any]] = None,
        out_name: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        logger.debug("inlining decl %s at call %s", decl.get("name"), call)
        args = call.get("args") or []
        lowered_args: List[str] = []
        lowered_types: List[Optional[Dict[str, Any]]] = []
        fail_msg = "Failed to lower argument when inlining user-declared block"
        for a in args:
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
                k = next(iter(a))
                if k == "ident_list":
                    for ident in a[k]:
                        n, t = self._lower_expr(
                            ident, ctx, env, types, type_hint=type_hint
                        )
                        if n is None:
                            # Debug: include failing ident info
                            raise LoweringError(
                                f"{fail_msg}: failed to lower ident {ident!r}",
                                source=self._current_source,
                                function=decl.get("name"),
                            )
                        lowered_args.append(n)
                        lowered_types.append(t)
                    continue
                if k in ("*", "star"):
                    inner = a[k]
                    if isinstance(inner, dict) and "lit_list" in inner:
                        for el in inner["lit_list"]:
                            n, t = self._lower_expr(
                                el, ctx, env, types, type_hint=type_hint
                            )
                            if n is None:
                                raise LoweringError(
                                    fail_msg,
                                    source=self._current_source,
                                    function=decl.get("name"),
                                )
                            lowered_args.append(n)
                            lowered_types.append(t)
                        continue
                # Handle keyword-style args represented as single-key dicts,
                # e.g., {'target_mode': False} — lower the inner expr and
                # append its value so inlining supports kwarg passing.
                inner = a[k]
                # Special-case a literal list (e.g., {'lit_list': [...]}) which
                # appears as a top-level dict key at call sites like `f([1,2])`.
                if k == "lit_list":
                    n, t = self._lower_expr({"lit_list": inner}, ctx, env, types, type_hint=type_hint)
                    if n is None:
                        raise LoweringError(
                            fail_msg,
                            source=self._current_source,
                            function=decl.get("name"),
                        )
                    lowered_args.append(n)
                    lowered_types.append(t)
                    continue
                n, t = self._lower_expr(inner, ctx, env, types, type_hint=type_hint)
                if n is None:
                    raise LoweringError(
                        fail_msg,
                        source=self._current_source,
                        function=decl.get("name"),
                    )
                lowered_args.append(n)
                lowered_types.append(t)
                continue
            n, t = self._lower_expr(a, ctx, env, types, type_hint=type_hint)
            if n is None:
                # Debug: include failing arg string for easier diagnosis
                raise LoweringError(
                    f"{fail_msg}: failed to lower arg {a!r}",
                    source=self._current_source,
                    function=decl.get("name"),
                )
            lowered_args.append(n)
            lowered_types.append(t)

        param_names = [p.get("name") for p in decl.get("params", [])]
        if len(lowered_args) < len(param_names):
            raise LoweringError(
                f"Not enough arguments provided to '{decl.get('name')}'",
                source=self._current_source,
                function=decl.get("name"),
            )
        param_map = {n: v for n, v in zip(param_names, lowered_args)}

        local_names = set()
        returned_locals: list[str] = []
        for stmt in decl.get("body", []) or []:
            if isinstance(stmt, dict):
                if "let" in stmt:
                    local_names.add(stmt["let"])
                if "assign" in stmt:
                    local_names.add(stmt["assign"])
                if "return" in stmt and isinstance(stmt["return"], str):
                    # remember the original local name used in a return so
                    # we can expose it as a separate graph output (e.g.,
                    # `loss`) even when the lowered/internal value maps to
                    # an input or alias.
                    returned_locals.append(str(stmt["return"]))

        rename_map: Dict[str, str] = {}
        for ln in sorted(local_names):
            rename_map[ln] = f"{decl.get('name')}_{ln}_{ctx._node_id}"
            ctx._node_id += 1

        def _rename_ast(x: Any) -> Any:
            if isinstance(x, str):
                if x in param_map:
                    logger.debug(
                        "renaming string '%s' -> '%s'", x, param_map[x]
                    )
                    return param_map[x]
                if x in rename_map:
                    logger.debug(
                        "renaming local '%s' -> '%s'", x, rename_map[x]
                    )
                    return rename_map[x]
                return x
            if isinstance(x, dict):
                out = {}
                for k, v in x.items():
                    if (
                        k in ("let", "assign")
                        and isinstance(v, str)
                        and v in rename_map
                    ):
                        out[k] = rename_map[v]
                        continue
                    if k == "left" and isinstance(v, (str, dict)):
                        out[k] = _rename_ast(v)
                        continue
                    if k == "ops" and isinstance(v, list):
                        out[k] = [
                            {
                                "op": op.get("op"),
                                "right": _rename_ast(op.get("right")),
                            }
                            for op in v
                        ]
                        continue
                    if k == "args" and isinstance(v, list):
                        out[k] = [_rename_ast(a) for a in v]
                        continue
                    if k == "call" and isinstance(v, str):
                        out[k] = _rename_ast(v)
                        continue
                    out[k] = _rename_ast(v)
                return out
            if isinstance(x, list):
                return [_rename_ast(i) for i in x]
            return x

        inline_env = dict(env)  # EnvDict supports dict() conversion
        for pname, gname in param_map.items():
            inline_env[pname] = gname

        inline_types = dict(types)
        for pname, t in zip(param_names, lowered_types):
            if t:
                inline_types[pname] = t

        last_value = None
        last_type = None
        for stmt in decl.get("body", []) or []:
            stmt2 = _rename_ast(stmt)
            logger.debug("original stmt: %s", stmt)
            logger.debug("renamed stmt: %s", stmt2)
            val, typ = self._lower_statement(
                stmt2, ctx, inline_env, inline_types
            )
            if val is not None:
                last_value, last_type = val, typ

        if last_value is None:
            final_name = out_name or f"{decl.get('name')}_out"
            final_type = None
        else:
            final_name = last_value
            final_type = last_type

        # Prefer any multi-return information that the inlined body may have
        # set on the *inline* environment (inline_env). When inlining we used
        # a shallow copy of the caller's env (`inline_env`) for lowering; if
        # the inlined block emitted a multi-return it will be recorded there.
        # Propagate it back into the outer `env` so callers and selection
        # helpers can observe it.
        multi = inline_env.get("__last_multi_return__") or env.get("__last_multi_return__")
        if multi:
            env["__last_multi_return__"] = multi
        if multi and isinstance(decl.get("ret_type"), (list, tuple)):
            for idx, (nm, typ) in enumerate(multi):
                internal = (
                    nm
                    if isinstance(nm, str)
                    else f"{decl.get('name')}_out_{idx}"
                )
                out_type = as_tensor_type(
                    (
                        decl.get("ret_type")[idx]
                        if isinstance(decl.get("ret_type"), list)
                        and idx < len(decl.get("ret_type"))
                        else typ
                    )
                    or typ
                )
                ctx.add_output(internal, out_type)
            return (
                multi[0][0]
                if multi and isinstance(multi[0][0], str)
                else final_name
            ), (multi[0][1] if multi else final_type)

        if out_name and final_name != out_name:
            # Ensure we reference the graph-visible (possibly qualified) name
            # for the source when emitting an Identity. Some inlined nodes
            # may have their outputs qualified by scope; prefer that name so
            # the Identity input refers to an existing value.
            src = ctx._output_renames.get(final_name, final_name)
            ctx.add_node("Identity", [src], [out_name])
            final_name = out_name

        # Register per-function tensor-bus names to aid external binding. This
        # creates deterministic fully-qualified names under the module/function
        # namespace (e.g., `examples.golden.clip.clip_demo.P_txt`) which map to
        # the actual graph-visible identifier (qualified by the context). This
        # helps external tooling bind external tensors to local initializers
        # and inputs by a stable global name.
        try:
            module = _get_model_domain(ctx) or self._current_module
            func = decl.get("name")
            tb = ctx.model_metadata.setdefault("tensor_bus", {})
            # Iterate over local types recorded during lowering and export
            # a concise tensor descriptor for known tensor-like values.
            for local_name, t in list(types.items()):
                if not isinstance(t, dict) or not t.get("scalar"):
                    continue
                # Build the fully-qualified external name and locate the
                # actual graph-visible name used during lowering.
                fq = f"{module}.{func}.{local_name}" if module and func else f"{func}.{local_name}"
                graph_name = env.get(local_name) or ctx.qualify_name(local_name)
                scalar = t.get("scalar")
                dims = t.get("dims") or []
                # Format tensor type string like 'f32[1,768]'
                dim_str = ",".join(str(d) for d in dims) if dims else ""
                tensor_desc = f"{scalar}[{dim_str}]" if dim_str else f"{scalar}[]"
                tb[fq] = {"name": graph_name, "tensor": tensor_desc}
        except Exception:
            # Best-effort: do not fail lowering if tensor bus registration fails
            pass

        # Ensure we always return the final inlined value and its type. Some
        # code paths above computed `final_name`/`final_type` but did not
        # explicitly return them, causing implicit None returns upstream.
        return final_name, final_type


    def _expr_to_str(self, expr: Any) -> str:
        """Convert a small subset of expression ASTs back to a human readable
        string. This is intentionally lightweight — it's intended for use in
        diagnostic metadata rather than round-tripping arbitrary source.
        """
        if isinstance(expr, dict):
            if "call" in expr:
                args = expr.get("args") or []
                args_s = ", ".join(self._expr_to_str(a) for a in args)
                return f"{expr['call']}({args_s})"
            if "left" in expr and "ops" in expr:
                s = self._expr_to_str(expr["left"])
                for op in expr.get("ops", []):
                    s = f"{s} {op['op']} {self._expr_to_str(op['right'])}"
                return s
            # Support legacy/optional equality AST shape: {left: X, right: Y}
            if "left" in expr and "right" in expr and "ops" not in expr:
                if expr["right"] is None:
                    return self._expr_to_str(expr["left"])
                return (
                    f"{self._expr_to_str(expr['left'])} == "
                    f"{self._expr_to_str(expr['right'])}"
                )
            if "lit_list" in expr:
                return (
                    "["
                    + ", ".join(self._expr_to_str(i) for i in expr["lit_list"])
                    + "]"
                )
            if "if" in expr:
                return "if(...)"
        if isinstance(expr, str):
            return expr
        if isinstance(expr, bool):
            return "true" if expr else "false"
        if isinstance(expr, (int, float)):
            return str(expr)
        return str(expr)

    def _eval_const_expr(
        self, expr: Any, env: Dict[str, str], types: Dict[str, Dict[str, Any]]
    ):
        """Attempt to fully evaluate simple constant expressions composed of
        literals and arithmetic ops. Returns the evaluated value or raises
        ValueError when evaluation isn't possible.
        """
        if isinstance(expr, bool) or isinstance(expr, (int, float)):
            return expr
        if isinstance(expr, dict):
            if "left" in expr and "ops" in expr:
                v = self._eval_const_expr(expr["left"], env, types)
                for op in expr.get("ops", []):
                    op_token = op["op"]
                    r = self._eval_const_expr(op["right"], env, types)
                    if op_token == "+":
                        v = v + r
                    elif op_token == "-":
                        v = v - r
                    elif op_token == "*":
                        v = v * r
                    elif op_token == "/":
                        v = v / r
                    elif op_token == "//":
                        v = v // r
                    else:
                        raise ValueError(
                            "unsupported operator for static eval"
                        )
                return v
            # support optional equality AST shape {left: X, right: Y}
            if "left" in expr and "right" in expr and "ops" not in expr:
                if expr["right"] is None:
                    return self._eval_const_expr(expr["left"], env, types)
                # equality check may be static-evaluable for literals
                left_val = self._eval_const_expr(expr["left"], env, types)
                right_val = self._eval_const_expr(expr["right"], env, types)
                return left_val == right_val
            if "lit_list" in expr:
                return [
                    self._eval_const_expr(i, env, types)
                    for i in expr["lit_list"]
                ]
            # calls, ifs, and other constructs are not supported for static eval
            raise ValueError("non-constant expression")
        # names / identifiers not supported for static eval
        raise ValueError("non-constant expression")

    def _ensure_same_scalar(
        self, op: str, input_types: List[Optional[Dict[str, Any]]]
    ):
        scalars = [
            t.get("scalar")
            for t in input_types
            if isinstance(t, dict) and t.get("scalar")
        ]
        if scalars and len(set(scalars)) > 1:
            raise TypeError(
                f"{op} requires matching scalar types, got {scalars}"
            )

    def _maybe_fold_elementwise(
        self, op: str, inputs: List[str], literals: List[Optional[Any]]
    ):
        if op == "Add" and len(inputs) == 2:
            for idx, lit in enumerate(literals):
                if lit == 0:
                    return inputs[1 - idx]
        if op == "Mul" and len(inputs) == 2:
            for idx, lit in enumerate(literals):
                if lit == 1:
                    return inputs[1 - idx]
        return None

    def _clean_attrs(self, op: str, attrs: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for k, v in attrs.items():
            if op == "Conv" and k == "stride":
                cleaned["strides"] = self._as_list(v)
            else:
                cleaned[k] = v
        return cleaned

    def _normalize_attr_name(self, op: str, name: str) -> str:
        # Accept legacy singular 'stride' as alias for 'strides' on Conv
        if op.lower() == "conv" and name == "stride":
            return "strides"
        return name

    def _as_list(self, v: Any) -> List[Any]:
        if isinstance(v, list):
            return v
        return [v]

    def _coerce_attr_value(self, v: Any) -> Any:
        if isinstance(v, (int, float, bool, str)):
            return v
        if isinstance(v, list):
            return [self._coerce_attr_value(x) for x in v]
        return v

    def _pick_type(
        self, candidates: List[Optional[Dict[str, Any]]]
    ) -> Optional[Dict[str, Any]]:
        for c in candidates:
            if c:
                return as_tensor_type(c)
        return None

    def _sanitize_scope(self, s: str) -> str:
        import re

        # keep alphanumeric and underscore; replace others with '_'
        s = re.sub(r"[^0-9A-Za-z_]+", "_", s)
        # trim leading/trailing separators
        s = s.strip("_")
        # limit length to keep names reasonable
        return s[:64]

    def _make_scope(self, name: Optional[str]) -> Optional[str]:
        if self._current_module and name:
            return f"{self._current_module}.{name}"
        return name
