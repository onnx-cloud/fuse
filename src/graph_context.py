"""GraphContext and dtype helpers.

Kept separate from lowering logic to keep modules small and make opset/dtype
updates straightforward.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import onnx
from onnx import TensorProto, helper
import numpy as np
from src.onnx_opset import latest_onnx_opset, validate_opset_version


def _get_project_version_from_pyproject() -> str | None:
    """Attempt to read the authoritative project version from a local pyproject.toml.

    Behavior:
    - If environment variable `FUSE_PROJECT_VERSION` is set, return it (testing override).
    - Else, search parents for `pyproject.toml` and return `project.version` if found.

    Falls back to None when the file is missing or cannot be parsed.
    """
    # Allow explicit override for test environments or CI without monkeypatching
    try:
        import os
        v = os.environ.get("FUSE_PROJECT_VERSION")
        if v:
            return str(v)
    except (OSError, ImportError, KeyError):
        pass

    try:
        import tomllib
        p = Path(__file__).resolve()
        for parent in p.parents:
            py = parent / "pyproject.toml"
            if py.exists():
                try:
                    data = tomllib.loads(py.read_text(encoding="utf-8"))
                    ver = data.get("project", {}).get("version")
                    if ver:
                        return str(ver)
                except ValueError:
                    return None
    except ImportError:
        # tomllib or parsing not available — best-effort only
        return None


# -----
# defaults / constants
# -----

DEFAULT_SCALAR = "f32"
DEFAULT_OPSET = latest_onnx_opset()

# -----
# dtype mapping
# -----

DTYPE_MAP: Dict[str, int] = {
    "f32": TensorProto.FLOAT,
    "f64": TensorProto.DOUBLE,
    "f16": TensorProto.FLOAT16,
    "bf16": TensorProto.BFLOAT16,
    "i8": TensorProto.INT8,
    "i16": TensorProto.INT16,
    "i32": TensorProto.INT32,
    "i64": TensorProto.INT64,
    "int64": TensorProto.INT64,
    "u8": TensorProto.UINT8,
    "u16": TensorProto.UINT16,
    "u32": TensorProto.UINT32,
    "u64": TensorProto.UINT64,
    "uint64": TensorProto.UINT64,
    "bool": TensorProto.BOOL,
    "str": TensorProto.STRING,
    "string": TensorProto.STRING,
    "complex64": TensorProto.COMPLEX64,
    "complex128": TensorProto.COMPLEX128,
}

ONNX_TO_FUSE: Dict[int, str] = {v: k for k, v in DTYPE_MAP.items()}


def fuse_dtype_to_onnx(dtype: str) -> int:
    return DTYPE_MAP[dtype]


# helper moved from lowering.main to avoid circular imports

def get_model_domain(ctx: "GraphContext") -> str | None:
    """Return the model's declared domain key.

    Useful for any code needing a stable domain lookup without depending on
    lowering.main which imports ops.
    """
    if not ctx or not isinstance(ctx, GraphContext):
        return None
    return ctx.model_metadata.get("domain")


def onnx_dtype_to_fuse(dtype: int) -> str:
    return ONNX_TO_FUSE.get(dtype, DEFAULT_SCALAR)


def as_tensor_type(
    typ: Optional[Dict[str, Any]] = None,
    fallback_scalar: str = DEFAULT_SCALAR,
    fallback_dims: Optional[Iterable[int]] = None,
) -> Dict[str, Any]:
    if typ is None:
        return {"scalar": fallback_scalar, "dims": list(fallback_dims or [])}
    if isinstance(typ, str):
        return {"scalar": typ, "dims": list(fallback_dims or [])}
    if isinstance(typ, dict):
        scalar = typ.get("scalar", fallback_scalar) or fallback_scalar
        if scalar == "None":
            scalar = fallback_scalar
        dims = typ.get("dims", fallback_dims or [])
        return {
            "scalar": scalar,
            "dims": list(dims),
            **({"meta": typ["meta"]} if "meta" in typ else {}),
        }
    return {"scalar": fallback_scalar, "dims": list(fallback_dims or [])}


from src.name_allocator import NameAllocator, StableNameAllocator  # noqa: E402


class GraphContext:
    def __init__(
        self,
        name: str = "fuse",
        opset: int = DEFAULT_OPSET,
        scope_prefix: Optional[str] = None,
        name_allocator: Optional[NameAllocator] = None,
        embed_external_data: bool = False,
    ):
        """GraphContext manages per-graph naming, nodes, and value info.

        Parameters:
        - name: graph name
        - opset: ONNX opset to target
        - scope_prefix/scope_display: optional scoping for qualified names
        - name_allocator: Optional[NameAllocator] to provide deterministic, testable naming.
          Prefer passing a fresh allocator instance (e.g., `StableNameAllocator`);
          avoid using global singletons as they may introduce cross-graph state.
        """
        self.name = name
        self.opset = validate_opset_version(opset)
        self.extra_opsets: Dict[str, int] = {}
        self.nodes: List[onnx.NodeProto] = []
        self.inputs: Dict[str, onnx.ValueInfoProto] = {}
        self.outputs: Dict[str, onnx.ValueInfoProto] = {}
        self.initializers: Dict[str, onnx.TensorProto] = {}
        self.value_types: Dict[str, Dict[str, Any]] = {}
        self.defined_values: set[str] = set()
        self.import_node_start: Dict[str, int] = {}
        self.model_metadata: Dict[str, Any] = {}
        self.graph_doc_string: str = ""
        # Optional training_info entries to append to the emitted ModelProto
        self._training_info: list = []
        # Local FunctionProto definitions collected during lowering
        self.functions: List[onnx.FunctionProto] = []
        # Backwards compatible counters used when no allocator is provided
        self._node_id = 0
        self._const_id = 0
        self.scope_prefix: Optional[str] = scope_prefix
        self._output_renames: Dict[str, str] = {}
        self.scope_display: Optional[str] = None
        # Name allocator for deterministic naming and test injection
        if name_allocator is None:
            self._name_allocator: NameAllocator = StableNameAllocator(
                scope_prefix, self.scope_display
            )
        else:
            self._name_allocator = name_allocator
        # When True, embed external/imported tensor bytes directly into
        # initializer.raw_data rather than creating external_data entries
        self.embed_external_data: bool = bool(embed_external_data)
        # flag indicating whether we should add graph inputs for constant
        # initializers. ONNX opset 9+ no longer requires this; removing them
        # avoids spurious inputs in generated models.
        self._emit_inputs_for_consts: bool = self.opset < 9

    def _next_node_name(self, op_type: str) -> str:
        # Prefer injected allocator when present; fall back to legacy counter
        try:
            return self._name_allocator.next_node_name(op_type)
        except (TypeError, ValueError, AttributeError, KeyError):
            # Legacy behavior
            if self._node_id == 0 and getattr(self, "scope_display", None):
                name = self.scope_display
                self._node_id += 1
                return name
            prefix = f"{self.scope_prefix}__" if self.scope_prefix else ""
            name = f"{prefix}{op_type}_{self._node_id}"
            self._node_id += 1
            return name

    def _next_const_name(self) -> str:
        try:
            return self._name_allocator.next_const_name()
        except (TypeError, ValueError, AttributeError, KeyError):
            prefix = f"{self.scope_prefix}__" if self.scope_prefix else ""
            name = f"{prefix}const_{self._const_id}"
            self._const_id += 1
            return name

    def qualify_name(self, name: str) -> str:
        """Return the graph-visible (qualified) name for an internal identifier.

        - If `scope_display` is present use dot-separated display (e.g. "mod.node.x").
        - Otherwise fall back to converting `scope_prefix` into a reasonable
          dotted display and append the identifier.
        - If no scope is set, return the original name unchanged.
        """
        # If the name already appears to be qualified for the current
        # scope, return it unchanged (idempotent qualification).
        sd = getattr(self, "scope_display", None)
        sp = getattr(self, "scope_prefix", None)
        # If the name already appears dotted, treat it as fully-qualified and
        # return it unchanged to avoid double-qualification.
        if isinstance(name, str) and "." in name:
            return name
        if sd and name.startswith(f"{sd}."):
            return name
        # Accept both '_' and '.' prefix separators as already-qualified
        if sp and (name.startswith(f"{sp}_") or name.startswith(f"{sp}.")):
            return name
        if sd:
            return f"{sd}.{name}"
        # If the scope prefix already contains a dot, treat it as dotted and
        # append the identifier directly (preserves canonical module.name forms)
        if sp and "." in sp:
            return f"{sp}.{name}"
        # Backwards-compatibility: split on first '_' into two path components
        if sp and "_" in sp:
            parts = sp.split("_", 1)
            return f"{parts[0]}.{parts[1]}.{name}"
        if sp:
            return f"{sp}_{name}"
        return name

    def add_function(self, func: "onnx.FunctionProto"):
        """Record a FunctionProto for later inclusion in the emitted ModelProto.

        The caller is responsible for ensuring any names inside *func* are
        already prefixed/qualified appropriately for the current graph context.
        """
        # Validate name presence
        if not func.name:
            raise ValueError("FunctionProto must have a non-empty name")
        dom = getattr(func, "domain", "") or ""
        key = (dom, func.name)
        # deduplicate identical domain/name pairs
        for existing in self.functions:
            if (getattr(existing, "domain", "") or "", existing.name) == key:
                # duplicate – log and ignore
                import logging

                logging.getLogger(__name__).warning(
                    "skipping duplicate FunctionProto %s.%s",
                    dom or "<core>",
                    func.name,
                )
                return
        try:
            self.functions.append(func)
        except (TypeError, ValueError, AttributeError, KeyError) as e:
            # best-effort: log and respect strict mode
            import logging

            logging.getLogger(__name__).warning(
                "failed to add FunctionProto %s.%s: %s",
                dom or "<core>",
                func.name,
                e,
            )
            if getattr(self, "strict", False):
                raise

    def add_param(self, param: Dict[str, Any]) -> str:
        # Accept both resolved type dicts and raw parser-produced forms. Some
        # parameter kinds (e.g. `subgraph`) are not ONNX graph inputs and must
        # be handled specially.
        raw_typ = param.get("type_decl") or param.get("type")
        # Special-case subgraph-typed params: they are not serialized as
        # runtime graph inputs (ONNX subgraphs are static attributes). Record
        # their type info but do not create a Tensor/Sequence ValueInfo.
        if raw_typ == "subgraph" or (
            isinstance(raw_typ, dict) and raw_typ.get("scalar") == "subgraph"
        ):
            internal_name = param["name"]
            # Allow subgraph contexts to preserve local parameter/input names
            # (Loop/If/Scan bodies rely on positionally-bound local names).
            if getattr(self, "_preserve_local_input_names", False):
                graph_name = internal_name
            else:
                graph_name = self.qualify_name(internal_name)
            self.value_types[internal_name] = {
                "scalar": "subgraph",
                "dims": (
                    raw_typ.get("dims") if isinstance(raw_typ, dict) else []
                ),
            }
            self.defined_values.add(internal_name)
            self.defined_values.add(graph_name)
            return graph_name

        typ = as_tensor_type(raw_typ)
        internal_name = param["name"]
        # Allow subgraph-lowered contexts to keep parameter names local
        if getattr(self, "_preserve_local_input_names", False):
            graph_name = internal_name
        else:
            graph_name = self.qualify_name(internal_name)
        # If the graph already exposes an input under this graph-visible name,
        # return it (idempotent); otherwise register a new TensorValueInfo.
        if graph_name in self.inputs:
            # ensure internal type info exists for the short name as well
            self.value_types.setdefault(
                internal_name,
                {"scalar": typ["scalar"], "dims": typ.get("dims") or []},
            )
            self.defined_values.add(internal_name)
            self.defined_values.add(graph_name)
            return graph_name

        # Support sequence/list parameters (e.g. list[tensor]) by emitting a
        # Sequence value-info when the parser produced a container-like type
        # for tensors. Fall back to a tensor ValueInfo for ordinary tensor types.
        if (
            isinstance(raw_typ, dict)
            and typ.get("scalar") == "list"
            and raw_typ.get("type") == "tensor"
        ):
            # element type may be specified as nested dict or omitted -> use default
            elem = (raw_typ.get("dims") or [None])[0]
            elem_scalar = None
            elem_shape = []
            if isinstance(elem, dict):
                elem_scalar = elem.get("scalar")
                elem_shape = elem.get("dims") or []
            elif isinstance(elem, str) and elem in DTYPE_MAP:
                elem_scalar = elem
            if not elem_scalar:
                elem_scalar = DEFAULT_SCALAR
            elem_type = fuse_dtype_to_onnx(elem_scalar)
            # convert shape placeholders -> ints (0 for dynamic)
            shape = []
            for d in elem_shape or []:
                try:
                    shape.append(int(d))
                except (ValueError, TypeError):
                    shape.append(0)
            vi = helper.make_tensor_sequence_value_info(
                graph_name, elem_type, shape
            )
            self.inputs[graph_name] = vi
            self.value_types[internal_name] = {
                "scalar": typ["scalar"],
                "dims": typ.get("dims") or [],
            }
            self.defined_values.add(internal_name)
            self.defined_values.add(graph_name)
            return graph_name

        # Preserve original dimension tokens so we can map symbolic
        # dimension identifiers (e.g., 'N', 'features') into ONNX's
        # `dim_param` fields while emitting numeric dims into the value-info.
        orig_dims = typ.get("dims") or []
        dims = []
        for d in orig_dims:
            try:
                dims.append(int(d))
            except (ValueError, TypeError):
                dims.append(0)
        vi = helper.make_tensor_value_info(
            graph_name,
            fuse_dtype_to_onnx(typ["scalar"]),
            dims,
        )
        # If the original dimensions contained symbolic identifiers, set the
        # corresponding `dim_param` fields so ONNX preserves param names.
        try:
            for i, d in enumerate(orig_dims):
                if i < len(vi.type.tensor_type.shape.dim):
                    if not isinstance(d, int):
                        # Convert tokens or other types to string
                        try:
                            vi.type.tensor_type.shape.dim[i].dim_param = str(d)
                        except AttributeError:
                            pass
        except AttributeError:
            pass
        meta = None
        if isinstance(param.get("type"), dict) and isinstance(
            param["type"].get("meta"), dict
        ):
            meta = param["type"]["meta"]
        if isinstance(typ, dict) and isinstance(typ.get("meta"), dict):
            meta = typ.get("meta")
        if meta:
            vi.doc_string = json.dumps(meta, sort_keys=True)
        # Register both the graph-visible input and the internal type info so
        # lowering (which keys types by the internal name) continues to work.
        self.inputs[graph_name] = vi
        self.value_types[internal_name] = {
            "scalar": typ["scalar"],
            "dims": typ.get("dims") or [],
        }
        self.defined_values.add(internal_name)
        self.defined_values.add(graph_name)
        return graph_name

    def add_const(self, const: Dict[str, Any]) -> str:
        typ = as_tensor_type(const.get("type_decl") or const.get("type"))
        name = const["name"]
        dims = typ.get("dims") or []
        value = const.get("value")

        # Some parser edge-cases can produce an ``infix`` wrapper around
        # a literal value such as
        #   {'left': {'lit_list': [...]}, 'ops': [{'op': '@', 'right': 'frozen'}]}
        # which happens when a following declaration's leading annotation
        # (e.g., `@frozen`) is lexed as an infix operator and attached to the
        # previous default expression. Normalize these cases by unwrapping the
        # underlying literal so downstream code can treat it normally.
        if isinstance(value, dict) and "left" in value and isinstance(value["left"], dict) and "lit_list" in value["left"]:
            value = value["left"]
            const["value"] = value

        # Handle explicit @import: create a TensorProto that references
        # an external file rather than embedding raw_data
        if isinstance(value, dict) and "imported_tensors" in value:
            info = value["imported_tensors"]
            src = info.get("src") or info.get("file")
            dest_name = (
                (Path(src).name)
                if src
                else (
                    Path(info.get("file")).name
                    if info.get("file")
                    else f"{name}.bin"
                )
            )
            from onnx import TensorProto

            # If embedding requested, try to read and inline the bytes from
            # the source file (.bin/.npz/.onnx supported). Otherwise fall back
            # to the previous external_data path.
            if self.embed_external_data:
                if not src:
                    raise Exception("imported tensor missing source path")
                p = Path(src)
                if not p.exists():
                    raise FileNotFoundError(f"import source not found: {src}")
                ext = p.suffix.lower()
                # Normalize gzipped .npz detection (Path.suffix returns only last suffix
                # so top-level '.gz' would miss the underlying '.npz' extension). Treat
                # '.npz.gz' specially so callers see a clear error message about gzipped .npz.
                if ext == ".gz" and any(s == ".npz" for s in p.suffixes):
                    raise Exception("gzipped .npz archives are not supported; use plain .npz")

                # Helper: map Fuse scalar types to numpy dtypes
                NP_MAP = {
                    "f32": np.float32,
                    "f64": np.float64,
                    "f16": np.float16,
                    "bf16": np.float32,  # bf16 may need special handling; use float32 here
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
                from onnx import numpy_helper

                if ext == ".bin":
                    # Read raw bytes and interpret as the declared dtype
                    if typ.get("scalar") == "bf16":
                        # BF16 requires specialized conversion; not implemented yet
                        raise Exception("bf16 embedding is not yet supported")
                    np_dtype = NP_MAP.get(typ["scalar"])
                    if np_dtype is None:
                        raise Exception(f"unsupported dtype for .bin import: {typ.get('scalar')}")
                    count = 1
                    for d in dims:
                        count *= int(d)
                    offset = int(info.get("offset", 0) or 0)
                    itemsize = np.dtype(np_dtype).itemsize
                    # Validate offset is aligned to itemsize
                    if offset % itemsize != 0:
                        raise Exception("offset misalignment for .bin import")
                    need = count * itemsize
                    with open(src, "rb") as f:
                        f.seek(offset)
                        raw = f.read(need)
                    if len(raw) < need:
                        raise Exception("not enough bytes in source file for declared shape")
                    # Support explicit endianness kwarg (default little)
                    endian = (info.get("endian") or "little").lower()
                    if endian not in ("little", "big"):
                        raise Exception("invalid endian specifier; expected 'little' or 'big'")
                    dtype = np.dtype(np_dtype)
                    if (endian == "big") != (dtype.byteorder == ">"):
                        # read with native byte order, then byte-swap if needed
                        arr = np.frombuffer(raw, dtype=np_dtype, count=count).copy()
                        arr.byteswap(inplace=True)
                        arr = arr.astype(np_dtype)
                    else:
                        # byte order matches, read directly
                        arr = np.frombuffer(raw, dtype=np_dtype, count=count)
                    if dims:
                        arr = arr.reshape(tuple(dims))
                    tensor = numpy_helper.from_array(arr, name)
                elif ext == ".npz":
                    # Disallow gzipped npz variants (requires decompression support)
                    if any(s == ".gz" for s in p.suffixes):
                        raise Exception("gzipped .npz archives are not supported; use plain .npz")
                    with np.load(src) as npz:
                        key = info.get("key")
                        if key is None:
                            files = list(npz.files)
                            if len(files) == 1:
                                key = files[0]
                            else:
                                raise Exception("multiple arrays in .npz; specify key=\"name\"")
                        arr = npz[key]
                    tensor = numpy_helper.from_array(arr, name)
                elif ext == ".onnx":
                    import onnx as _onnx

                    m = _onnx.load(src)
                    inits = {i.name: i for i in m.graph.initializer}
                    sel_name = info.get("name")
                    if sel_name:
                        if sel_name not in inits:
                            raise Exception(f"initializer '{sel_name}' not found in source onnx")
                        src_init = inits[sel_name]
                    else:
                        if len(inits) == 1:
                            src_init = next(iter(inits.values()))
                        else:
                            raise Exception("multiple initializers in source .onnx; specify name=\"...\"")
                    # make a copy of the tensor proto
                    src_init_bytes = src_init.SerializeToString()
                    tensor = TensorProto()
                    tensor.ParseFromString(src_init_bytes)
                else:
                    raise Exception(f"unsupported import extension for embedding: {ext}")

                # Qualify the tensor for graph visibility and register it
                graph_name = self.qualify_name(name)
                tensor.name = graph_name
                self.initializers[graph_name] = tensor
                self.value_types[name] = {"scalar": typ["scalar"], "dims": dims}
                self.value_types[graph_name] = self.value_types[name]
                self.defined_values.add(name)
                self.defined_values.add(graph_name)
                try:
                    if (
                        self._emit_inputs_for_consts
                        and graph_name not in self.inputs
                        and not getattr(self, "_preserve_local_input_names", False)
                    ):
                        vi = helper.make_tensor_value_info(
                            graph_name, fuse_dtype_to_onnx(typ["scalar"]), dims
                        )
                        self.inputs[graph_name] = vi
                except (TypeError, ValueError, AttributeError, KeyError):
                    pass
                return name

            # Fallback: preserve legacy externalized behavior
            tensor = helper.make_tensor(
                name, fuse_dtype_to_onnx(typ["scalar"]), dims, []
            )
            # mark as external and add location entry
            tensor.data_location = TensorProto.EXTERNAL
            entry = tensor.external_data.add()
            entry.key = "location"
            entry.value = dest_name
            # optional metadata: offset/length
            if "offset" in info:
                entry2 = tensor.external_data.add()
                entry2.key = "offset"
                entry2.value = str(info.get("offset"))
            # register initializer and record external file to copy during export
            self.initializers[name] = tensor
            self.value_types[name] = {"scalar": typ["scalar"], "dims": dims}
            self.defined_values.add(name)
            # Register a graph-visible input for this const initializer so
            # validation and runtime mapping can locate it as a known input.
            # For nested subgraph contexts where local names are preserved
            # (`_preserve_local_input_names`), skip registering inputs so
            # subgraph initializers do not become extra positional inputs.
            try:
                if name not in self.inputs and not getattr(
                    self, "_preserve_local_input_names", False
                ):
                    vi = helper.make_tensor_value_info(
                        name, fuse_dtype_to_onnx(typ["scalar"]), dims
                    )
                    self.inputs[name] = vi
            except (TypeError, ValueError, AttributeError, KeyError):
                pass
            external_files = self.model_metadata.get("external_files", [])
            if src:
                external_files.append(
                    {"src": str(src), "dest": dest_name, "init_name": name}
                )
                self.model_metadata["external_files"] = external_files
            return name

        # Normalize literal list forms produced by the parser which use
        # {'lit_list': [...]} wrappers, potentially nested. Convert them to
        # plain Python lists so numpy can construct ndarrays from them.
        def _unwrap(v):
            if isinstance(v, dict) and "lit_list" in v:
                return [_unwrap(x) for x in v["lit_list"]]
            return v

        if isinstance(value, dict) and "lit_list" in value:
            value = _unwrap(value)

        if isinstance(value, list):
            values = value
            if not dims:
                # infer a nested shape for nested lists (e.g. [[1,2],[3,4]] -> [2,2])
                def _shape(v):
                    if isinstance(v, list) and v:
                        return [len(v)] + _shape(v[0])
                    return []

                dims = _shape(values)
        else:
            values = [value]
            if not dims:
                dims = []
        tensor = helper.make_tensor(
            name, fuse_dtype_to_onnx(typ["scalar"]), dims, values
        )
        # Qualify the tensor name for graph visibility so the initializer's
        # name matches any qualified output names that reference it (e.g.
        # "module.fn.const"). This ensures ONNX graph outputs may refer to
        # initializers without causing validation errors.
        graph_name = self.qualify_name(name)
        tensor.name = graph_name
        self.initializers[graph_name] = tensor
        # keep internal (unqualified) and graph-qualified type info
        self.value_types[name] = {"scalar": typ["scalar"], "dims": dims}
        self.value_types[graph_name] = self.value_types[name]
        self.defined_values.add(name)
        self.defined_values.add(graph_name)
        # Register a graph-visible ValueInfo for the qualified name so that
        # graph outputs referencing the qualified identifier (e.g.
        # "module.fn.const") can be resolved to an initializer. This mirrors
        # the behaviour used for external imported tensors.
        try:
            if (
                self._emit_inputs_for_consts
                and not getattr(self, "_preserve_local_input_names", False)
            ):
                if graph_name not in self.inputs:
                    vi = helper.make_tensor_value_info(
                        graph_name, fuse_dtype_to_onnx(typ["scalar"]), dims
                    )
                    self.inputs[graph_name] = vi
        except (TypeError, ValueError, AttributeError, KeyError):
            pass
        return name

    def add_tensor_literal(
        self,
        values: List[Any],
        typ: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> str:
        tinfo = as_tensor_type(typ)
        const_name = name or self._next_const_name()
        dims = tinfo.get("dims") or [len(values)]
        tensor = helper.make_tensor(
            const_name, fuse_dtype_to_onnx(tinfo["scalar"]), dims, values
        )
        # Qualify the tensor name for graph visibility so nested scopes and
        # runtime naming conventions remain consistent with node outputs.
        qname = self.qualify_name(const_name)
        tensor.name = qname
        self.initializers[qname] = tensor
        self.value_types[const_name] = {
            "scalar": tinfo["scalar"],
            "dims": dims,
        }
        self.defined_values.add(const_name)
        self.defined_values.add(qname)
        # Also register a graph-visible input for this initializer so nodes
        # that consume constant literals are satisfied by ONNX validation
        # which expects non-produced inputs to be listed as graph inputs.
        try:
            if (
                self._emit_inputs_for_consts
                and qname not in self.inputs
                and not getattr(self, "_preserve_local_input_names", False)
            ):
                vi = helper.make_tensor_value_info(
                    qname, fuse_dtype_to_onnx(tinfo["scalar"]), dims
                )
                self.inputs[qname] = vi
        except (TypeError, ValueError, AttributeError, KeyError):
            pass
        return qname

    def add_literal(
        self,
        value: Any,
        typ: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> str:
        tinfo = as_tensor_type(typ)
        const_name = name or self._next_const_name()
        dims = tinfo.get("dims") or []
        tensor = helper.make_tensor(
            const_name,
            fuse_dtype_to_onnx(tinfo["scalar"]),
            dims,
            [value],
        )
        qname = self.qualify_name(const_name)
        tensor.name = qname
        self.initializers[qname] = tensor
        self.value_types[const_name] = {
            "scalar": tinfo["scalar"],
            "dims": dims,
        }
        self.defined_values.add(const_name)
        self.defined_values.add(qname)
        # Also register a graph-visible input for this initializer so nodes
        # that consume constant literals are satisfied by ONNX validation
        # which expects non-produced inputs to be listed as graph inputs.
        try:
            if (
                self._emit_inputs_for_consts
                and qname not in self.inputs
                and not getattr(self, "_preserve_local_input_names", False)
            ):
                vi = helper.make_tensor_value_info(
                    qname, fuse_dtype_to_onnx(tinfo["scalar"]), dims
                )
                self.inputs[qname] = vi
        except (TypeError, ValueError, AttributeError, KeyError):
            pass
        return qname

    def add_node(
        self,
        op_type: str,
        inputs: List[str],
        outputs: List[str],
        attrs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        doc_string: Optional[str] = None,
    ) -> str:
        node_name = name or self._next_node_name(op_type)

        def _sanitize(x):
            if isinstance(x, str):
                return x.replace("\n", "").strip()
            return x

        outputs = [_sanitize(o) for o in outputs]
        inputs = [_sanitize(i) for i in inputs]

        # Qualify output names when a scope prefix is present to avoid
        # collisions between nodes from different nested graphs. Also ensure
        # that inputs referencing previously-emitted (unqualified) outputs
        # are rewritten to the qualified names to preserve internal DAG order.
        # Rewrite inputs using any existing output renames so nodes may
        # consume the outputs of previously-inserted helpers (e.g., entry
        # Identity nodes) without mutating the lowering env mapping.
        inputs = [self._output_renames.get(i, i) for i in inputs]

        if getattr(self, "scope_prefix", None):
            # compute qualified outputs and register renames
            qualified_outputs = []
            for o in outputs:
                q = self.qualify_name(o) if isinstance(o, str) else o
                qualified_outputs.append(q)
                if isinstance(o, str) and q != o:
                    self._output_renames[o] = q
        else:
            qualified_outputs = outputs

        node = helper.make_node(
            op_type, inputs, qualified_outputs, name=node_name, **(attrs or {})
        )
        if doc_string:
            node.doc_string = doc_string
        self.nodes.append(node)
        for o in qualified_outputs:
            self.defined_values.add(o)
        return node_name

    def rename_value(self, old_name: str, new_name: str) -> None:
        """Rename a value throughout the graph (nodes, inputs, outputs, initializers, and type maps).

        This is used to avoid emitting synthetic Identity nodes by renaming an
        existing producer output to a desired canonical name (e.g.,
        `<param>.grad`). The operation is best-effort and will raise if the
        target name is already present in the graph or when the old name is
        not present.
        """
        if old_name == new_name:
            return
        q_old = self.qualify_name(old_name) if isinstance(old_name, str) else old_name
        q_new = self.qualify_name(new_name) if isinstance(new_name, str) else new_name
        # Ensure the old value exists and the new name is not already used
        if q_old not in self.defined_values and old_name not in self.defined_values:
            raise KeyError(f"value to rename not found: {old_name}")
        if q_new in self.defined_values:
            raise KeyError(f"target name already exists: {new_name}")
        # Replace node inputs and outputs
        for n in self.nodes:
            n.input[:] = [q_new if (i == q_old or i == old_name) else i for i in n.input]
            n.output[:] = [q_new if (o == q_old or o == old_name) else o for o in n.output]
        # Initializers
        if q_old in self.initializers:
            init = self.initializers.pop(q_old)
            new_init = onnx.TensorProto()
            new_init.CopyFrom(init)
            new_init.name = q_new
            self.initializers[q_new] = new_init
        # Inputs and outputs
        if q_old in self.inputs:
            vi = self.inputs.pop(q_old)
            vi.name = q_new
            self.inputs[q_new] = vi
        if q_old in self.outputs:
            vi = self.outputs.pop(q_old)
            vi.name = q_new
            self.outputs[q_new] = vi
        # Value types
        if old_name in self.value_types:
            self.value_types[new_name] = self.value_types.pop(old_name)
        if q_old in self.value_types:
            self.value_types[q_new] = self.value_types.pop(q_old)
        # Update defined values sets
        if old_name in self.defined_values:
            self.defined_values.remove(old_name)
        if q_old in self.defined_values:
            self.defined_values.remove(q_old)
        self.defined_values.add(new_name)
        self.defined_values.add(q_new)

    def add_training_info(self, training_info_proto) -> None:
        """Store a TrainingInfoProto to be appended during model emission."""
        self._training_info.append(training_info_proto)

    def insert_node(
        self,
        index: int,
        op_type: str,
        inputs: List[str],
        outputs: List[str],
        attrs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> str:
        node_name = name or self._next_node_name(op_type)
        node = helper.make_node(
            op_type, inputs, outputs, name=node_name, **(attrs or {})
        )
        self.nodes.insert(index, node)
        for o in outputs:
            self.defined_values.add(o)
        for k, v in list(self.import_node_start.items()):
            if v >= index:
                self.import_node_start[k] = v + 1
        return node_name

    def add_output(self, name: str, typ: Dict[str, Any]) -> str:
        # `name` may be an internal identifier or already graph-visible; ensure
        # the exported graph uses the qualified name when a scope is active.
        tinfo = as_tensor_type(typ)
        internal_name = name
        graph_name = self.qualify_name(internal_name)

        # Support sequence/list outputs (e.g. `-> list[tensor]`) by emitting a
        # Sequence value-info when appropriate; otherwise fall back to tensor.
        if tinfo.get("scalar") == "list":
            # Prefer explicit nested element description when present, else
            # fall back to the first entry in the dims/type information.
            elem = None
            if isinstance(typ, dict):
                elem = (
                    (typ.get("dims") or [None])[0] if typ.get("dims") else None
                )
                if isinstance(elem, dict) and elem.get("scalar"):
                    elem_scalar = elem.get("scalar")
                    elem_shape = elem.get("dims") or []
                else:
                    elem_scalar = None
                    elem_shape = []
            else:
                elem_scalar = None
                elem_shape = []

            # If dims were emitted as shorthand (e.g. ['tensor']), prefer that
            if not elem_scalar and tinfo.get("dims"):
                first = tinfo.get("dims")[0]
                if isinstance(first, str) and first in DTYPE_MAP:
                    elem_scalar = first
            if not elem_scalar:
                elem_scalar = DEFAULT_SCALAR
            elem_type = fuse_dtype_to_onnx(elem_scalar)
            shape = []
            for d in elem_shape or []:
                try:
                    shape.append(int(d))
                except (ValueError, TypeError):
                    shape.append(0)
            vi = helper.make_tensor_sequence_value_info(
                graph_name, elem_type, shape
            )
            if isinstance(typ, dict) and isinstance(typ.get("meta"), dict):
                vi.doc_string = json.dumps(typ["meta"], sort_keys=True)
            self.outputs[graph_name] = vi
            self.value_types[internal_name] = {
                "scalar": tinfo["scalar"],
                "dims": tinfo.get("dims") or [],
            }
            if graph_name != internal_name:
                self.value_types[graph_name] = self.value_types[internal_name]
            self.defined_values.add(internal_name)
            self.defined_values.add(graph_name)
            return graph_name

        dims = []
        for d in tinfo.get("dims") or []:
            try:
                dims.append(int(d))
            except (ValueError, TypeError):
                dims.append(0)
        vi = helper.make_tensor_value_info(
            graph_name,
            fuse_dtype_to_onnx(tinfo["scalar"]),
            dims,
        )
        if isinstance(typ, dict) and isinstance(typ.get("meta"), dict):
            vi.doc_string = json.dumps(typ["meta"], sort_keys=True)
        self.outputs[graph_name] = vi
        # Keep type info accessible by both the internal name used during
        # lowering and the graph-visible name used in serialized models.
        self.value_types[internal_name] = {
            "scalar": tinfo["scalar"],
            "dims": tinfo.get("dims") or [],
        }
        if graph_name != internal_name:
            self.value_types[graph_name] = self.value_types[internal_name]
        self.defined_values.add(internal_name)
        self.defined_values.add(graph_name)
        # If an unqualified initializer exists for this internal name, copy it
        # to the qualified graph name so the serialized `initializer` name
        # matches the qualified output and ONNX validation succeeds.
        try:
            if (internal_name in self.initializers) and (
                graph_name not in self.initializers
            ):
                new_init = onnx.TensorProto()
                new_init.CopyFrom(self.initializers[internal_name])
                new_init.name = graph_name
                self.initializers[graph_name] = new_init
        except (TypeError, ValueError, AttributeError, KeyError):
            pass
        return graph_name

    def build_model(self, opset: Optional[int] = None) -> onnx.ModelProto:
        default_opset = validate_opset_version(opset or self.opset)
        # Compute deterministic opset imports via helper (caps to SAFE_MAX_OPSET)
        from src.util.opset_utils import compute_opset_imports

        tuples = compute_opset_imports(default_opset, self.extra_opsets)
        # Convert tuples into onnx opset id protos lazily to avoid circular imports
        opset_imports = []
        for domain, ver in tuples:
            opset_imports.append(helper.make_opsetid(domain, int(ver)))

        graph_name = self.scope_display or self.name
        # preserve SSA form. This is conservative and primarily prevents
        # accidental inclusion of intermediate identity outputs as inputs.
        node_outputs = set(o for n in self.nodes for o in n.output)
        for k in list(self.inputs.keys()):
            if k in node_outputs:
                self.inputs.pop(k, None)

        graph = helper.make_graph(
            nodes=self.nodes,
            name=graph_name,
            inputs=[self.inputs[k] for k in self.inputs],
            # Preserve the declared output insertion order (do not sort).
            outputs=[self.outputs[k] for k in self.outputs],
            initializer=[
                self.initializers[k] for k in sorted(self.initializers)
            ],
        )
        if self.graph_doc_string:
            graph.doc_string = self.graph_doc_string

        model = helper.make_model(graph, opset_imports=opset_imports)
        # determine minimal IR version required for the target opset.  ONNX
        # teams provide a mapping in opset_utils; we default to 8 if unknown.
        try:
            from src.util.opset_utils import compute_opset_imports
            # compute_opset_imports already caps to SAFE_MAX_OPSET; use the
            # first entry (core opset) to infer IR requirement
            core_opset = int(tuples[0][1]) if tuples else int(default_opset)
            # minimal IR versions derived from ONNX changelog/spec; keep
            # this table simple and update as ONNX evolves.
            OPSET_TO_IR = {
                1: 1,
                6: 3,
                7: 3,
                8: 3,
                9: 4,
                10: 5,
                11: 6,
                12: 7,
                13: 8,
                14: 9,
                15: 10,
                16: 11,
                17: 12,
                18: 13,
                19: 14,
                20: 15,
            }
            model.ir_version = OPSET_TO_IR.get(core_opset, 8)
        except (TypeError, ValueError, AttributeError, KeyError):
            # best-effort: ignore if import or mapping fails
            pass

        # Build emitted metadata using shared helper (centralized validation & merging)
        from src.util.graph_metadata import build_emitted_metadata

        # If model_metadata is empty and this is not a root context, treat as
        # None (allow nested/subgraph lowering without explicit `@fuse`). For
        # top-level/root contexts preserve the explicit (possibly-empty) dict so
        # `build_emitted_metadata` can validate and raise when appropriate.
        if not self.model_metadata and not getattr(self, "_is_root", False):
            emitted = build_emitted_metadata(None)
        else:
            emitted = build_emitted_metadata(self.model_metadata)

        # Append deduplicated metadata_props in deterministic order
        for k in sorted(emitted):
            v = emitted[k]
            if v is None:
                continue
            if isinstance(v, (dict, list)):
                v = json.dumps(v, sort_keys=True)
            model.metadata_props.append(
                onnx.StringStringEntryProto(key=str(k), value=str(v))
            )

        # Append any TrainingInfoProto entries accumulated during lowering
        try:
            if getattr(self, "_training_info", None):
                for ti in self._training_info:
                    model.training_info.append(ti)
        except (TypeError, ValueError, AttributeError, KeyError):
            pass

        # Append any user-defined FunctionProtos collected during lowering
        try:
            for fn in getattr(self, "functions", []):
                # copy to avoid accidental shared-state modifications
                model.functions.append(fn)
        except (TypeError, ValueError, AttributeError, KeyError):
            pass

        return model
