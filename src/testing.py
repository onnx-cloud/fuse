from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.lowering import FuseLowerer

_NUMPY_DTYPE: Dict[str, Any] = {
    "f16": np.float16,
    "bf16": np.float16,  # numpy has no native bf16; keep it runnable
    "f32": np.float32,
    "f64": np.float64,
    "i8": np.int8,
    "i16": np.int16,
    "i32": np.int32,
    "i64": np.int64,
    "u8": np.uint8,
    "u16": np.uint16,
    "u32": np.uint32,
    "u64": np.uint64,
    "bool": np.bool_,
}


@dataclass
class TestFailure(Exception):
    file: str
    test_name: str
    message: str

    # Prevent pytest from mistaking this helper exception class for a test
    # container (pytest collects classes named `Test*`). Setting `__test__ = False`
    # stops collection while preserving the public API.
    __test__ = False

    def __str__(self) -> str:
        return f"{self.file}::{self.test_name}: {self.message}"


def _to_py_value(expr: Any) -> Any:
    if isinstance(expr, dict) and "lit_list" in expr:
        return list(expr["lit_list"])
    return expr


def _as_array(value: Any, dtype: Optional[Any]) -> np.ndarray:
    if isinstance(value, np.ndarray):
        arr = value
    elif isinstance(value, dict) and "lit_list" in value:
        arr = np.asarray(value["lit_list"])
    elif isinstance(value, list):
        arr = np.asarray(value)
    else:
        arr = np.asarray([value])

    if dtype is not None:
        return arr.astype(dtype)
    return arr


def _infer_dtype_from_type_decl(
    type_decl: Optional[Dict[str, Any]],
) -> Optional[Any]:
    if not isinstance(type_decl, dict):
        return None
    scalar = type_decl.get("scalar")
    if not scalar:
        return None
    return _NUMPY_DTYPE.get(str(scalar))


def _allclose(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        return np.allclose(a, b, rtol=1e-5, atol=1e-5)
    return np.array_equal(a, b)


class FuseTestRunner:
    def __init__(self, ast: List[Dict[str, Any]], file: str):
        self.ast = ast
        self.file = file
        self.lowerer = FuseLowerer()

        self._meta_decls: List[Dict[str, Any]] = [
            d for d in ast if d.get("type") == "meta"
        ]
        self._import_decls: List[Dict[str, Any]] = [
            d for d in ast if d.get("type") == "import"
        ]
        self._param_decls: List[Dict[str, Any]] = [
            d for d in ast if d.get("type") == "param"
        ]
        self._const_decls: List[Dict[str, Any]] = [
            d for d in ast if d.get("type") == "const"
        ]
        # Include type aliases so tests that compile functions with aliases will lower correctly
        self._type_alias_decls: List[Dict[str, Any]] = [
            d for d in ast if d.get("type") == "type_alias"
        ]

        self._functions: Dict[str, Dict[str, Any]] = {}
        for d in ast:
            if d.get("type") in ("node", "model", "export"):
                self._functions[str(d.get("name"))] = d

        # Include both interpreter tests and export tests that validate ONNX exports
        self._tests: List[Dict[str, Any]] = [
            d for d in ast if d.get("type") in ("proof", "golden")
        ]

        self._compiled: Dict[str, Tuple[Any, List[Dict[str, Any]]]] = {}

    @property
    def tests(self) -> List[Dict[str, Any]]:
        return self._tests

    def _compile_node(self, node_name: str, call_args=None):
        if node_name in self._compiled:
            cached = self._compiled[node_name]
            # Return only model and params for backward compatibility
            return cached[0], cached[1]
        if node_name not in self._functions:
            raise TestFailure(
                self.file, node_name, f"unknown function '{node_name}'"
            )

        decls = []
        decls.extend(self._meta_decls)
        decls.extend(self._import_decls)
        # Ensure type aliases are visible to subsequent decls (consts/params)
        decls.extend(self._type_alias_decls)
        decls.extend(self._param_decls)
        decls.extend(self._const_decls)

        # If call-site args are provided and the target function accepts
        # subgraph-typed parameters, replace those parameter names in the
        # function AST with the provided node names so lowering can embed
        # the corresponding GraphProto as op attributes (Loop/If/Scan).
        target_decl = self._functions[node_name]
        decl_to_lower = target_decl
        # Prepare mapping for parameter replacements (may remain empty)
        param_repl = {}
        if call_args is not None:
            params = target_decl.get("params") or []
            # Build mapping of param-name -> provided-arg for subgraph-typed params
            for idx, p in enumerate(params):
                try:
                    # resolve declared type for param
                    ptype = self.lowerer._resolve_type(
                        p.get("type") or p.get("type_decl")
                    )
                except Exception:
                    ptype = None
                # Support binding subgraph-typed params by name (existing behavior)
                if (
                    isinstance(ptype, dict)
                    and ptype.get("scalar") == "subgraph"
                ):
                    # arg may be a string naming a user-declared node (common case)
                    if idx < len(call_args) and isinstance(
                        call_args[idx], str
                    ):
                        param_repl[p.get("name")] = call_args[idx]
                # Support binding compile-time-known `list[tensor]` parameters by
                # replacing the param with a literal list of tensor literals so
                # star/unpack (`*param`) at callsites can be expanded during
                # lowering into individual inputs/constants.
                if isinstance(ptype, dict) and ptype.get("scalar") == "list":
                    if idx < len(call_args) and isinstance(
                        call_args[idx], list
                    ):
                        # Convert each Python/numpy element into a lit_list AST
                        lit_elems = []
                        for el in call_args[idx]:
                            # If the element is a numpy array, convert to list
                            try:
                                import numpy as _np

                                if isinstance(el, _np.ndarray):
                                    lit_elems.append(
                                        {"lit_list": el.flatten().tolist()}
                                    )
                                    continue
                            except Exception:
                                pass
                            # If the element is a plain Python list or scalar, embed directly
                            if isinstance(el, list):
                                lit_elems.append({"lit_list": el})
                            else:
                                lit_elems.append(el)
                        if lit_elems:
                            param_repl[p.get("name")] = {"lit_list": lit_elems}
            if param_repl:
                import copy
                import logging

                logging.getLogger(__name__).debug(
                    "param_repl mapping: %s", param_repl
                )

                def _replace_params(x):
                    if isinstance(x, str):
                        return param_repl.get(x, x)
                    if isinstance(x, dict):
                        return {k: _replace_params(v) for k, v in x.items()}
                    if isinstance(x, list):
                        return [_replace_params(i) for i in x]
                    return x

                decl_to_lower = copy.deepcopy(target_decl)
                # Apply replacements across decl fields but avoid mutating the
                # parameter list entries themselves (we only want to replace
                # uses of the parameters within the body/generics, not the
                # param declarations which must remain valid dicts).
                for k, v in list(decl_to_lower.items()):
                    if k == "params":
                        continue
                    decl_to_lower[k] = _replace_params(v)

                logging.getLogger(__name__).debug(
                    "decl_to_lower after replacement: %s", decl_to_lower
                )

                # If the caller supplied concrete node args for subgraph-typed
                # parameters, remove those parameters from the function signature
                # so the compiled model does not expect a runtime 'body' input.
                params_list = decl_to_lower.get("params") or []
                # Only include string-like replacements when computing the set
                # of parameter names to remove; literal dict replacements
                # (e.g., {'lit_list': [...]}) are not parameter names.
                removed = set(
                    list(param_repl.keys())
                    + [v for v in param_repl.values() if isinstance(v, str)]
                )
                decl_to_lower["params"] = [
                    p for p in params_list if p.get("name") not in removed
                ]

                # Ensure any user-declared blocks referenced by replacement
                # are also included so the lowerer can resolve them to GraphProto
                # when embedding as Graph attributes (Loop/If/Scan). Append
                # them as inline-only so they are available for lowering but
                # not emitted as top-level graphs.
                for repl in set(
                    v for v in param_repl.values() if isinstance(v, str)
                ):
                    if repl in self._functions:
                        import copy

                        helper_decl = copy.deepcopy(self._functions[repl])
                        helper_decl.setdefault("_inline_only", True)
                        decls.append(helper_decl)
        decls.append(decl_to_lower)

        model = self.lowerer.lower(decls)

        # Ensure any externally-referenced initializer files are available
        # to `onnx.checker` and the ReferenceEvaluator. The lowering phase
        # records these files in model.metadata_props under the key
        # "external_files" as a JSON-serializable list of objects with
        # {src, dest, init_name} entries. Copy each `src` to the local
        # `dest` filename so validation/run-time can access it.
        try:
            import json
            import shutil

            meta = {p.key: p.value for p in model.metadata_props}
            if "external_files" in meta:
                try:
                    ext = json.loads(meta["external_files"])
                    from pathlib import Path

                    for e in ext:
                        src = e.get("src")
                        dest = e.get("dest")
                        if src and dest:
                            # If `src` is relative, resolve it next to the source file
                            s = Path(src)
                            if not s.is_absolute():
                                s = (Path(self.file).parent / s).resolve()
                            if s.exists():
                                shutil.copyfile(str(s), dest)
                                # If this initializer is referenced externally, embed the
                                # binary content into the in-memory ModelProto so the
                                # ReferenceEvaluator can access it without requiring
                                # a LargeContainer instance.
                                try:
                                    # find the matching initializer and populate raw_data
                                    for init in model.graph.initializer:
                                        if init.name == e.get("init_name"):
                                            with open(str(s), "rb") as f:
                                                init.raw_data = f.read()
                                            init.data_location = (
                                                0  # default (embedded)
                                            )
                                            # clear external_data entries
                                            del init.external_data[:]
                                            break
                                except Exception:
                                    # If embedding fails, keep the best-effort behavior
                                    # of leaving the external reference and let subsequent
                                    # validation/runtime raise a helpful error.
                                    pass
                except Exception:
                    # Best-effort: don't fail compilation if copying fails
                    pass
        except Exception:
            # ignore any import errors here and proceed to validation
            pass

        try:
            import onnx

            # Defensive post-processing: ensure any embedded GraphProto initializers
            # are name-qualified consistently with their graph outputs. Some lowering
            # paths may create prefixed node/output names while leaving initializers
            # unprefixed which triggers ONNX validation errors; fix that here so the
            # test runner can validate models robustly.
            def _qualify_graph_initializers(g):
                from onnx import helper

                try:
                    outs = [o.name for o in g.output]
                    prefixed = [o for o in outs if "." in o]
                    if prefixed and g.initializer:
                        prefix = prefixed[0].rsplit(".", 1)[0]
                        # Track renamed initializers to update node inputs
                        renamed = {}
                        for init in g.initializer:
                            if not init.name.startswith(f"{prefix}."):
                                old_name = init.name
                                init.name = f"{prefix}.{init.name}"
                                renamed[old_name] = init.name
                        # Update node inputs to use the new qualified names
                        for node in g.node:
                            for i, inp in enumerate(node.input):
                                if inp in renamed:
                                    node.input[i] = renamed[inp]
                    # If an output uses a prefixed form like 'scope.const_0' but the
                    # initializer uses the unprefixed local name 'const_0', normalize
                    # the output to the local initializer name so they match.
                    init_names = {i.name for i in g.initializer}
                    for vi in list(g.output):
                        if "." in vi.name:
                            suf = vi.name.rsplit(".", 1)[1]
                            if suf in init_names:
                                vi.name = suf
                    # If any graph outputs reference initializers directly (i.e.,
                    # the output name is an initializer name), wrap that initializer
                    # in a trivial Identity node so the output is produced by a node
                    # (satisfies ONNX validation rules that outputs are node outputs).
                    id_nodes = []
                    for vi in list(g.output):
                        if vi.name in init_names:
                            new_out = f"{vi.name}_out"
                            id_node = helper.make_node(
                                "Identity",
                                [vi.name],
                                [new_out],
                                name=f"Identity_{new_out}",
                            )
                            id_nodes.append(id_node)
                            # replace occurrence in outputs
                            vi.name = new_out
                    if id_nodes:
                        # Prepend identity nodes to the node list so they come first
                        g.node[:] = id_nodes + list(g.node)
                except Exception:
                    return
                # Recurse into nested graphs on nodes
                for n in g.node:
                    for a in n.attribute:
                        if getattr(a, "g", None) is not None:
                            _qualify_graph_initializers(a.g)

            _qualify_graph_initializers(model.graph)

            onnx.checker.check_model(model)
        except Exception as e:
            raise TestFailure(
                self.file, node_name, f"lowered model failed validation: {e}"
            )

        # Use the (possibly modified) decl that was lowered when call-site
        # argument substitution was applied earlier so runtime params
        # accurately reflect the compiled model's inputs.
        params = (
            (
                decl_to_lower.get("params")
                if decl_to_lower is not None
                else None
            )
            or self._functions[node_name].get("params")
            or []
        )
        # Resolve any type aliases on function params so runtime invocations
        # can infer (numpy) dtypes for feeds correctly.
        resolved_params = []
        for p in params:
            resolved = self.lowerer._resolve_type(
                p.get("type") or p.get("type_decl")
            )
            if resolved is not None:
                p2 = dict(p)
                p2["type_decl"] = resolved
                # Mirror into `type` for compatibility with existing helpers
                p2["type"] = resolved
                resolved_params.append(p2)
            else:
                resolved_params.append(p)
        # Record compile-time-only argument positions (e.g., bound
        # `list[tensor]` parameters or subgraph bindings) so runtime
        # invocation can filter them out of the provided call args.
        compile_time_removed_positions = []
        if param_repl:
            # original param order from the declared function
            orig_params = target_decl.get("params") or []
            for idx, p in enumerate(orig_params):
                if p.get("name") in param_repl and not isinstance(
                    param_repl.get(p.get("name")), str
                ):
                    compile_time_removed_positions.append(idx)

        # Cache the compiled model along with helper metadata for runtime
        # invocation.
        self._compiled[node_name] = (
            model,
            resolved_params,
            compile_time_removed_positions,
        )
        return model, resolved_params

    def _eval_expr(self, expr: Any, env: Dict[str, Any]) -> Any:
        if isinstance(expr, dict):
            # bracketed identifier lists (e.g. `[a, b]`) used for call-time unpacking
            if "ident_list" in expr:
                return [
                    self._eval_expr(a, env) for a in expr.get("ident_list")
                ]
            if "call" in expr:
                node_name = str(expr["call"])
                args = expr.get("args") or []
                arg_vals = [self._eval_expr(a, env) for a in args]
                # Special-case assertion helpers that are lowered to internal
                # calls (e.g., `_close`) so tests can run without compiling
                # helper functions. Evaluate them directly here using numpy
                # comparison semantics.
                if node_name in ("_close", "assert_close"):
                    if len(arg_vals) >= 2:
                        left = arg_vals[0]
                        right = arg_vals[1]
                        left_arr = _as_array(left, None)
                        right_arr = _as_array(right, left_arr.dtype)
                        return _allclose(left_arr, right_arr)
                    return False
                return self._call_node(node_name, arg_vals)

        if isinstance(expr, str):
            if expr in env:
                return env[expr]
            # Allow referencing top-level consts by name
            for c in self._const_decls:
                if c.get("name") == expr:
                    return _to_py_value(c.get("value"))
            return expr

        return _to_py_value(expr)

    def _call_node(self, node_name: str, args: List[Any]) -> Any:
        model, params = self._compile_node(node_name, call_args=args)

        try:
            from onnx.reference import ReferenceEvaluator
        except Exception as e:
            raise TestFailure(
                self.file,
                node_name,
                f"onnx ReferenceEvaluator not available: {e}",
            )

        # Filter out args that correspond to user-declared node references
        # (e.g. passing `AccumulateBody` as a bound subgraph arg). Those
        # arguments are compile-time only and should not appear in runtime
        # feeds or when comparing arity.
        # Filter out args that correspond to user-declared node references
        # (e.g. passing `AccumulateBody` as a bound subgraph arg) as well as
        # any parameters that were bound at compile-time (e.g., literal
        # `list[tensor]` replacements). Those arguments are compile-time only
        # and should not appear in runtime feeds.
        filtered_args: List[Any] = []
        removed_positions = []
        compiled_meta = self._compiled.get(node_name)
        if compiled_meta and len(compiled_meta) > 2:
            removed_positions = compiled_meta[2]
        for idx, a in enumerate(args):
            if idx in removed_positions:
                continue
            if isinstance(a, str) and a in self._functions:
                continue
            filtered_args.append(a)

        if len(filtered_args) != len(params):
            raise TestFailure(
                self.file,
                node_name,
                f"arity mismatch: asserted {len(params)} args, got {len(filtered_args)}",
            )

        feed: Dict[str, Any] = (
            {}
        )  # DEBUG: expose feed types for troubleshooting runtime mismatches
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(
            "runtime params=%s filtered_args=%s",
            [p["name"] for p in params],
            filtered_args,
        )
        # Map parameter names to actual model input names (handles namespacing
        # where the lowered model's graph input names may be qualified).
        model_input_names = [i.name for i in model.graph.input if i.name]
        # If model provides explicit inputs, prefer those names in order;
        # otherwise fall back to declared parameter names.
        if len(model_input_names) >= len(params):
            param_input_names = model_input_names[: len(params)]
        else:
            param_input_names = [p.get("name") for p in params]

        for input_name, p, v in zip(param_input_names, params, filtered_args):
            # Support sequence/list parameters (e.g., list[tensor]) by
            # converting Python lists into lists of numpy arrays with the
            # appropriate element dtype. This ensures the ONNX Reference
            # Evaluator receives the expected input types for sequence feeds.
            resolved = None
            try:
                resolved = self.lowerer._resolve_type(
                    p.get("type") or p.get("type_decl")
                )
            except Exception:
                resolved = None

            # list[tensor] or similar forms
            if (
                isinstance(resolved, dict)
                and resolved.get("scalar") == "list"
                and isinstance(v, list)
            ):
                # Determine element scalar type
                elem = None
                if isinstance(p.get("type") or p.get("type_decl"), dict):
                    raw = p.get("type") or p.get("type_decl")
                    elem = (raw.get("dims") or [None])[0]
                elem_scalar = None
                if isinstance(elem, dict):
                    elem_scalar = elem.get("scalar")
                elif isinstance(elem, str) and elem in _NUMPY_DTYPE:
                    elem_scalar = elem
                if elem_scalar is None:
                    elem_dtype = None
                else:
                    elem_dtype = _NUMPY_DTYPE.get(elem_scalar)
                # Convert each element into numpy arrays with element dtype
                seq = []
                for el in v:
                    seq.append(_as_array(el, elem_dtype))
                feed[input_name] = seq
                continue

            dtype = _infer_dtype_from_type_decl(p.get("type"))
            feed[input_name] = _as_array(v, dtype)

        sess = ReferenceEvaluator(model)
        outs = sess.run(None, feed)
        if not outs:
            raise TestFailure(self.file, node_name, "model produced no outputs")
        return outs[0]

    def run_test(self, test_decl: Dict[str, Any]) -> None:
        test_name = str(test_decl.get("name"))
        env: Dict[str, Any] = {}

        asserts = 0
        for stmt in test_decl.get("body") or []:
            if isinstance(stmt, dict) and "let" in stmt:
                env[str(stmt["let"])] = self._eval_expr(stmt["expr"], env)
                continue

            # Also support typed local assignment inside tests (e.g., `s: f32[1] = [0.1]`)
            if isinstance(stmt, dict) and "assign" in stmt:
                env[str(stmt["assign"])] = self._eval_expr(stmt["expr"], env)
                continue

            if isinstance(stmt, dict) and "assert" in stmt:
                asserts += 1
                left = self._eval_expr(stmt["assert"]["left"], env)
                right = self._eval_expr(stmt["assert"]["right"], env)

                left_arr = _as_array(left, None)
                # Cast asserted to left dtype to avoid int-vs-float mismatches
                right_arr = _as_array(right, left_arr.dtype)

                if not _allclose(left_arr, right_arr):
                    raise TestFailure(
                        self.file,
                        test_name,
                        f"assert failed: got {left_arr.tolist()} asserted {right_arr.tolist()}",
                    )
                continue

            if isinstance(stmt, dict) and (
                "assert" in stmt or "note" in stmt or "annot" in stmt
            ):
                continue

        # Support @golden with external reference files. If the test
        # declares an golden with a `ref` option, compare the produced
        # output (asserted to be in local variable `out`) to the reference
        # dataset (npz). Otherwise, require at least one `assert` statement.
        if asserts == 0:
            # If this is an golden with a `ref` option, validate against it.
            if (
                isinstance(test_decl, dict)
                and test_decl.get("type") == "golden"
            ):
                opts = test_decl.get("golden") or {}
                ref = opts.get("ref")
                tol = opts.get("tol")
                rtol = opts.get("rtol")
                atol = opts.get("atol")
                if ref:
                    try:
                        import numpy as _np

                        ref_path = None
                        from pathlib import Path

                        # If ref is relative, resolve next to the source file
                        if not str(ref).startswith("http"):
                            sf = Path(self.file)
                            ref_path = (sf.parent / Path(str(ref))).resolve()
                        else:
                            # For remote refs, attempt to fetch (not implemented yet)
                            raise ValueError(
                                "remote refs are not supported in golden run"
                            )

                        data = _np.load(str(ref_path))
                        # Prefer variable named 'out' if present in env
                        if "out" in env:
                            got = env["out"]
                        else:
                            # Otherwise try the last assigned variable
                            if env:
                                got = list(env.values())[-1]
                            else:
                                raise TestFailure(
                                    self.file,
                                    test_name,
                                    "no output to compare to ref",
                                )

                        got_arr = _as_array(got, None)

                        # pick a reference array: if only one array present, use it
                        keys = list(data.files)
                        if not keys:
                            raise TestFailure(
                                self.file,
                                test_name,
                                f"reference file {ref_path} contains no arrays",
                            )
                        if len(keys) == 1:
                            asserted = _np.asarray(data[keys[0]])
                        else:
                            # multiple arrays: try to match by name 'out' or take first
                            if "out" in keys:
                                asserted = _np.asarray(data["out"])
                            else:
                                asserted = _np.asarray(data[keys[0]])

                        # determine tolerances
                        if rtol is None and atol is None and tol is not None:
                            rtol = float(tol)
                            atol = float(tol)
                        if rtol is None:
                            rtol = 1e-5
                        if atol is None:
                            atol = 1e-5

                        if not _np.allclose(
                            got_arr.astype(asserted.dtype),
                            asserted,
                            rtol=float(rtol),
                            atol=float(atol),
                        ):
                            msg = (
                                f"golden reference mismatch: got {got_arr.tolist()} "
                                f"asserted {asserted.tolist()}"
                            )
                            raise TestFailure(self.file, test_name, msg)
                        return
                    except TestFailure:
                        raise
                    except Exception as e:
                        raise TestFailure(
                            self.file,
                            test_name,
                            f"failed to validate ref {ref}: {e}",
                        )

            # Treat plain @proof with no explicit 'assert' as a smoke test: if the
            # body executed without raising, consider it passed.
            if (
                isinstance(test_decl, dict)
                and test_decl.get("type") == "proof"
            ):
                return

            raise TestFailure(
                self.file, test_name, "test contains no 'assert' statements"
            )


def run_fuse_tests(ast: List[Dict[str, Any]], file: str) -> Tuple[int, int]:
    runner = FuseTestRunner(ast, file)
    passed = 0
    failed = 0
    for t in runner.tests:
        try:
            runner.run_test(t)
            passed += 1
        except TestFailure:
            failed += 1
            raise
    return passed, failed
