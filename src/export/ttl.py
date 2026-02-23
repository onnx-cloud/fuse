"""RDF/Turtle (TTL) export for ONNX models.

This module provides deterministic export of ONNX ModelProto to RDF/Turtle
format, following the specification in RDFS.md.

Key features:
- Deterministic output (same input → identical bytes)
- Uses stable URIs based on model hash or provided namespace
- Supports ONNX vocabulary with `onnx:` prefix
- Configurable user namespace prefix

Example usage:
    from src.export.ttl import model_to_ttl, save_ttl
    import onnx

    model = onnx.load("model.onnx")
    ttl_str = model_to_ttl(model, user_ns="my:")
    save_ttl(model, "model.ttl")
"""

from __future__ import annotations

import hashlib
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import onnx
from onnx import TensorProto

# Constants for ONNX vocabulary namespace
ONNX_NS = "https://ns.onnx.cloud/onnx#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
SKOS_NS = "http://www.w3.org/2004/02/skos/core#"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"


# ONNX dtype int -> string name mapping
DTYPE_NAMES: Dict[int, str] = {
    TensorProto.FLOAT: "float32",
    TensorProto.DOUBLE: "float64",
}

from typing import Sequence, Optional

def _ensure_absolute_iri(val: str, *, strict: bool = False, allowed_prefixes: Optional[Sequence[str]] = None) -> str:
    """Accept either an absolute IRI (http/https) or a CURIE (prefix:local).

    - If an absolute IRI is provided, return it enclosed in angle brackets
      (e.g., '<https://example.org/Type>').
    - If a CURIE is provided (e.g., 'foaf:Person'), return it as-is; the
      caller is responsible for ensuring a matching prefix is declared in
      the TTL output. If a CURIE uses an unknown prefix, a UserWarning is
      emitted to inform the caller that the CURIE may be unresolved. When
      `strict=True`, an unknown CURIE prefix raises ValueError instead.

    Raises ValueError for other invalid forms (non-strings or strings that are
    neither a http(s) IRI nor a CURIE of the form 'prefix:local').
    """
    if not isinstance(val, str):
        raise ValueError("@id/@type values must be strings and either absolute IRIs or CURIEs (prefix:local)")
    v = val.strip()
    # Absolute IRI
    if v.startswith("http://") or v.startswith("https://"):
        return f"<{v}>"
    # CURIE (prefix:localname)
    # Accept reasonably permissive localname characters (no whitespace)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\-]*:[^\s]+", v):
        prefix = v.split(":", 1)[0]
        known = set(("onnx", "rdf", "skos", "xsd"))
        if allowed_prefixes:
            known.update([p.rstrip(":") for p in allowed_prefixes if p])
        if prefix not in known:
            if strict:
                raise ValueError(f"CURIE '{v}' uses unknown prefix '{prefix}' and strict mode is enabled")
            warnings.warn(f"CURIE '{v}' uses unknown prefix '{prefix}'. Declare a matching prefix via 'user_ns' when calling ttl export to ensure resolvability.")
        return v
    raise ValueError(f"value '{val}' is not an absolute IRI nor a CURIE (expected 'http(s)://...' or 'prefix:local')")


def _dtype_name(dtype: int) -> str:
    """Convert ONNX dtype int to string name."""
    return DTYPE_NAMES.get(dtype, f"unknown_{dtype}")


def _escape_ttl_string(s: str) -> str:
    """Escape a string for TTL literal format."""
    # Escape backslashes, quotes, and newlines
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    s = s.replace("\r", "\\r")
    s = s.replace("\t", "\\t")
    return s


def _safe_local_name(name: str) -> str:
    """Convert a name to a safe TTL local name (after the colon in prefix:local).

    Replaces characters that are not valid in TTL local names.
    """
    # Replace problematic characters with underscores
    safe = ""
    for c in name:
        if c.isalnum() or c in ("_", "-", "."):
            safe += c
        else:
            safe += "_"
    # Ensure it doesn't start with a digit
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe or "_unnamed"


def _model_hash(model: onnx.ModelProto) -> str:
    """Compute a stable SHA256 hash for the model (for URI generation)."""
    data = model.SerializeToString()
    return hashlib.sha256(data).hexdigest()[:16]


def _shape_to_string(shape: Optional[onnx.TensorShapeProto]) -> str:
    """Convert ONNX TensorShapeProto to a string representation."""
    if shape is None:
        return "[]"
    dims = []
    for d in shape.dim:
        if d.HasField("dim_value"):
            dims.append(str(d.dim_value))
        elif d.HasField("dim_param") and d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append("?")
    return f"[{','.join(dims)}]"


def _get_tensor_type_info(vi: onnx.ValueInfoProto) -> Tuple[str, str]:
    """Extract dtype and shape from ValueInfoProto.

    Returns (dtype_name, shape_string).
    """
    if not vi.type.HasField("tensor_type"):
        return ("unknown", "[]")
    tt = vi.type.tensor_type
    dtype = _dtype_name(tt.elem_type)
    shape = _shape_to_string(tt.shape if tt.HasField("shape") else None)
    return dtype, shape


@dataclass
class Triple:
    """Represents an RDF triple for deterministic serialization."""

    subject: str
    predicate: str
    obj: str
    is_literal: bool = False
    datatype: Optional[str] = None

    def __lt__(self, other: "Triple") -> bool:
        """Lexicographic ordering for deterministic output."""
        return (self.subject, self.predicate, self.obj) < (
            other.subject,
            other.predicate,
            other.obj,
        )


@dataclass
class TTLBuilder:
    """Builder for deterministic TTL output."""

    user_ns: str = ""
    user_ns_uri: str = ""
    model_id: str = ""
    triples: List[Triple] = field(default_factory=list)

    def add(
        self,
        subject: str,
        predicate: str,
        obj: str,
        is_literal: bool = False,
        datatype: Optional[str] = None,
    ) -> None:
        """Add a triple to the builder."""
        self.triples.append(Triple(subject, predicate, obj, is_literal, datatype))

    def add_literal(
        self, subject: str, predicate: str, value: Any, datatype: str = "xsd:string"
    ) -> None:
        """Add a triple with a literal object."""
        if isinstance(value, bool):
            self.add(
                subject,
                predicate,
                "true" if value else "false",
                is_literal=True,
                datatype="xsd:boolean",
            )
        elif isinstance(value, int):
            self.add(
                subject,
                predicate,
                str(value),
                is_literal=True,
                datatype="xsd:integer",
            )
        elif isinstance(value, float):
            self.add(
                subject,
                predicate,
                str(value),
                is_literal=True,
                datatype="xsd:decimal",
            )
        else:
            self.add(
                subject,
                predicate,
                _escape_ttl_string(str(value)),
                is_literal=True,
                datatype=datatype,
            )

    def _format_object(self, triple: Triple) -> str:
        """Format the object part of a triple."""
        if triple.is_literal:
            if triple.datatype and triple.datatype != "xsd:string":
                return f'"{triple.obj}"^^{triple.datatype}'
            return f'"{triple.obj}"'
        return triple.obj

    def serialize(self) -> str:
        """Serialize all triples to deterministic TTL format."""
        lines: List[str] = []

        # Emit prefix declarations (sorted alphabetically)
        prefixes = [
            ("onnx", ONNX_NS),
            ("rdf", RDF_NS),
            ("skos", SKOS_NS),
            ("xsd", XSD_NS),
        ]
        if self.user_ns and self.user_ns_uri:
            prefixes.append((self.user_ns.rstrip(":"), self.user_ns_uri))

        prefixes.sort(key=lambda x: x[0])
        for prefix, uri in prefixes:
            lines.append(f"@prefix {prefix}: <{uri}> .")
        lines.append("")  # blank line after prefixes

        # Sort triples for deterministic output
        sorted_triples = sorted(self.triples)

        # Group triples by subject for more readable output
        current_subject: Optional[str] = None
        subject_predicates: List[Tuple[str, str]] = []

        def flush_subject():
            if current_subject is None or not subject_predicates:
                return
            lines.append(f"{current_subject}")
            for i, (pred, obj_str) in enumerate(subject_predicates):
                sep = " ;" if i < len(subject_predicates) - 1 else " ."
                lines.append(f"  {pred} {obj_str}{sep}")

        for triple in sorted_triples:
            if triple.subject != current_subject:
                flush_subject()
                current_subject = triple.subject
                subject_predicates = []
            subject_predicates.append((triple.predicate, self._format_object(triple)))

        flush_subject()

        return "\n".join(lines) + "\n"


def model_to_ttl(
    model: onnx.ModelProto,
    *,
    user_ns: str = "",
    user_ns_uri: str = "",
    include_initializers: bool = True,
    include_metadata: bool = True,
    strict: bool = False,
) -> str:
    """Convert an ONNX ModelProto to RDF/Turtle format.

    Args:
        model: The ONNX model to convert.
        user_ns: User namespace prefix (e.g., "my:"). If empty, uses model hash.
        user_ns_uri: User namespace URI (e.g., "https://example.org/#").
        include_initializers: Whether to include initializer details.
        include_metadata: Whether to include model metadata properties.
        strict: If True, treat unknown CURIE prefixes as errors (raise ValueError).

    Returns:
        Deterministic TTL string representation.
    """
    model_hash = _model_hash(model)
    graph = model.graph

    # Determine namespace prefix for model entities
    if user_ns:
        ns = user_ns if user_ns.endswith(":") else f"{user_ns}:"
        ns_uri = user_ns_uri or f"https://fuse.ai/models/{model_hash}#"
    else:
        ns = ""
        ns_uri = ""

    builder = TTLBuilder(user_ns=ns, user_ns_uri=ns_uri, model_id=model_hash)

    # Model URI
    model_uri = f"{ns}model/{model_hash}" if ns else f"onnx:model/{model_hash}"

    # Add Model triples
    builder.add(model_uri, "a", "onnx:Model")

    # Model opset versions
    for opset in model.opset_import:
        domain = opset.domain or "ai.onnx"
        version = opset.version
        builder.add_literal(model_uri, "onnx:opset", f"{domain}:{version}")

    # Model metadata
    # Emit special mappings for @type -> rdf:type and @id -> skos:exactMatch (author intent must be IRIs or resolvable CURIEs)
    meta_map = {prop.key: prop.value for prop in model.metadata_props}
    allowed_prefixes = []
    if ns:
        allowed_prefixes.append(ns.rstrip(":"))
    type_val = meta_map.get("@type") or meta_map.get("type")
    if type_val:
        try:
            iri = _ensure_absolute_iri(type_val, strict=strict, allowed_prefixes=allowed_prefixes)
            builder.add(model_uri, "rdf:type", iri)
        except ValueError as ve:
            raise ValueError(f"invalid @type value for TTL export: {ve}")

    id_val = meta_map.get("@id") or meta_map.get("id")
    if id_val:
        try:
            iri = _ensure_absolute_iri(id_val, strict=strict, allowed_prefixes=allowed_prefixes)
            builder.add(model_uri, "skos:exactMatch", iri)
        except ValueError as ve:
            raise ValueError(f"invalid @id value for TTL export: {ve}")

    if include_metadata:
        if model.producer_name:
            builder.add_literal(model_uri, "onnx:producerName", model.producer_name)
        if model.producer_version:
            builder.add_literal(
                model_uri, "onnx:producerVersion", model.producer_version
            )
        if model.domain:
            builder.add_literal(model_uri, "onnx:domain", model.domain)
        if model.model_version:
            builder.add_literal(
                model_uri,
                "onnx:modelVersion",
                model.model_version,
                datatype="xsd:integer",
            )
        if model.doc_string:
            builder.add_literal(model_uri, "onnx:docString", model.doc_string)

        # Metadata properties
        for prop in model.metadata_props:
            builder.add_literal(
                model_uri, f"onnx:meta/{_safe_local_name(prop.key)}", prop.value
            )

    # Graph
    graph_name = graph.name or "main"
    graph_uri = f"{ns}graph/{_safe_local_name(graph_name)}" if ns else f"onnx:{_safe_local_name(graph_name)}"
    builder.add(model_uri, "onnx:hasGraph", graph_uri)
    builder.add(graph_uri, "a", "onnx:Graph")

    if graph.name:
        builder.add_literal(graph_uri, "onnx:name", graph.name)
    if graph.doc_string:
        builder.add_literal(graph_uri, "onnx:docString", graph.doc_string)

    # Graph inputs
    # Determine trainables metadata; we only expose initializers that are
    # explicitly marked trainable. All other initializers are internal
    # constants and should not be part of the public GraphInput surface.
    import json

    trainables_map = {}
    try:
        tm = meta_map.get("trainables")
        if tm:
            trainables_map = json.loads(tm)
    except Exception:
        trainables_map = {}

    # All initializer names (internal constants)
    all_initializer_names = {init.name for init in graph.initializer}

    # Initializers that should be emitted (only those explicitly marked trainable)
    if include_initializers and trainables_map:
        trainable_initializer_names = {init.name for init in graph.initializer if trainables_map.get(init.name)}
    else:
        trainable_initializer_names = set()

    for vi in graph.input:
        # Skip inputs that are actual initializers (constants). These are internal
        # implementation details and should not be part of the public surface.
        if vi.name in all_initializer_names:
            continue
        input_uri = f"{graph_uri}#input/{_safe_local_name(vi.name)}"
        builder.add(graph_uri, "onnx:hasInput", input_uri)
        builder.add(input_uri, "a", "onnx:GraphInput")
        builder.add_literal(input_uri, "onnx:name", vi.name)
        dtype, shape = _get_tensor_type_info(vi)
        builder.add_literal(input_uri, "onnx:dtype", dtype)
        builder.add_literal(input_uri, "onnx:shape", shape)

    # Graph outputs
    for vi in graph.output:
        output_uri = f"{graph_uri}#output/{_safe_local_name(vi.name)}"
        builder.add(graph_uri, "onnx:hasOutput", output_uri)
        builder.add(output_uri, "a", "onnx:GraphOutput")
        builder.add_literal(output_uri, "onnx:name", vi.name)
        dtype, shape = _get_tensor_type_info(vi)
        builder.add_literal(output_uri, "onnx:dtype", dtype)
        builder.add_literal(output_uri, "onnx:shape", shape)

    # Initializers
    # Emit only those initializers that are explicitly marked trainable. Non-
    # trainable internal initializers are intentionally omitted from TTL to
    # avoid leaking implementation details.
    if include_initializers and trainable_initializer_names:
        for init in graph.initializer:
            if init.name not in trainable_initializer_names:
                continue
            init_uri = f"{graph_uri}#init/{_safe_local_name(init.name)}"
            builder.add(graph_uri, "onnx:hasInitializer", init_uri)
            builder.add(init_uri, "a", "onnx:Initializer")
            builder.add_literal(init_uri, "onnx:name", init.name)
            builder.add_literal(init_uri, "onnx:dtype", _dtype_name(init.data_type))
            builder.add_literal(
                init_uri, "onnx:shape", f"[{','.join(str(d) for d in init.dims)}]"
            )

            # External data reference
            if init.data_location == TensorProto.EXTERNAL:
                for ext in init.external_data:
                    if ext.key == "location":
                        builder.add_literal(init_uri, "onnx:externalLocation", ext.value)
                    elif ext.key == "offset":
                        builder.add_literal(
                            init_uri,
                            "onnx:externalOffset",
                            int(ext.value),
                            datatype="xsd:integer",
                        )
                    elif ext.key == "length":
                        builder.add_literal(
                            init_uri,
                            "onnx:externalLength",
                            int(ext.value),
                            datatype="xsd:integer",
                        )

    # Emit a compact summary of nodes instead of emitting every operator
    # (traversing the AST and only exposing top-level Graph and Function resources).
    node_count = len(graph.node)
    builder.add_literal(graph_uri, "onnx:nodeCount", node_count, datatype="xsd:integer")

    # Emit functions defined in the model (if any) as first-class resources.
    # This follows the request to include function nodes but not individual ops.
    funcs = getattr(model, "functions", None)
    if funcs:
        for f in funcs:
            func_name = f.name or "function"
            func_uri = f"{ns}function/{_safe_local_name(func_name)}" if ns else f"onnx:function/{_safe_local_name(func_name)}"
            builder.add(model_uri, "onnx:hasFunction", func_uri)
            builder.add(func_uri, "a", "onnx:Function")
            if f.name:
                builder.add_literal(func_uri, "onnx:name", f.name)
            if getattr(f, "input", None):
                builder.add_literal(func_uri, "onnx:inputs", f"[{','.join(f.input)}]")
            if getattr(f, "output", None):
                builder.add_literal(func_uri, "onnx:outputs", f"[{','.join(f.output)}]")

    return builder.serialize()


def save_ttl(
    model: onnx.ModelProto,
    path: Union[str, Path],
    *,
    user_ns: str = "",
    user_ns_uri: str = "",
    include_initializers: bool = True,
    include_metadata: bool = True,
    strict: bool = False,
) -> Path:
    """Save an ONNX model to TTL format.

    Args:
        model: The ONNX model to convert.
        path: Output file path.
        user_ns: User namespace prefix (e.g., "my:").
        user_ns_uri: User namespace URI (e.g., "https://example.org/#").
        include_initializers: Whether to include initializer details.
        include_metadata: Whether to include model metadata properties.
        strict: When True, treat unknown CURIE prefixes as errors (ValueError) during TTL conversion.

    Returns:
        Path to the saved TTL file.
    """
    ttl = model_to_ttl(
        model,
        user_ns=user_ns,
        user_ns_uri=user_ns_uri,
        include_initializers=include_initializers,
        include_metadata=include_metadata,
        strict=strict,
    )
    out_path = Path(path)
    out_path.write_text(ttl, encoding="utf-8")
    return out_path


def onnx_file_to_ttl(
    onnx_path: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
    *,
    user_ns: str = "",
    user_ns_uri: str = "",
    include_initializers: bool = True,
    include_metadata: bool = True,
    strict: bool = False,
) -> str:
    """Convert an ONNX file to TTL format.

    Args:
        onnx_path: Path to the ONNX model file.
        out_path: Optional output file path. If None, returns TTL string only.
        user_ns: User namespace prefix (e.g., "my:").
        user_ns_uri: User namespace URI (e.g., "https://example.org/#").
        include_initializers: Whether to include initializer details.
        include_metadata: Whether to include model metadata properties.
        strict: When True, treat unknown CURIE prefixes as errors (ValueError) during TTL conversion.

    Returns:
        TTL string representation.
    """
    model = onnx.load(str(onnx_path))
    ttl = model_to_ttl(
        model,
        user_ns=user_ns,
        user_ns_uri=user_ns_uri,
        include_initializers=include_initializers,
        include_metadata=include_metadata,
        strict=strict,
    )

    if out_path:
        Path(out_path).write_text(ttl, encoding="utf-8")

    return ttl
