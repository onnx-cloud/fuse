"""Sealing helpers: compute deterministic graph and initializer hashes and embed as metadata."""
from __future__ import annotations

import json
from typing import Dict, List, Optional

try:
    import blake3
except Exception:  # pragma: no cover - optional fast path
    blake3 = None

import hashlib

import onnx
from onnx import helper
from onnx import numpy_helper


def _hash_bytes(data: bytes, algorithm: str = "blake3") -> str:
    algo = algorithm.lower()
    if algo.startswith("blake3") and blake3 is not None:
        return blake3.blake3(data).hexdigest()
    if algo.startswith("sha256"):
        return hashlib.sha256(data).hexdigest()
    # fallback: prefer blake3 if available else sha256
    if blake3 is not None:
        return blake3.blake3(data).hexdigest()
    return hashlib.sha256(data).hexdigest()


def canonicalize_graph_bytes(model: onnx.ModelProto) -> bytes:
    """Return a deterministic byte representation of the model graph.

    This is intentionally compact and human-readable: it walks nodes in their
    declared order and serializes essential fields (name/op_type/inputs/outputs/
    attributes) in a stable way.
    """
    g = model.graph
    parts: List[str] = []
    parts.append(f"ir_version:{getattr(model, 'ir_version', '')}")
    # opset imports
    opset_lines = []
    for oi in model.opset_import:
        opset_lines.append(f"{oi.domain or 'ai.onnx'}:{oi.version}")
    parts.append("opsets:" + ",".join(sorted(opset_lines)))

    # inputs
    inp_names = [i.name for i in g.input]
    parts.append("inputs:" + ",".join(inp_names))
    out_names = [o.name for o in g.output]
    parts.append("outputs:" + ",".join(out_names))

    # initializers summary
    init_lines = []
    for it in sorted(g.initializer, key=lambda x: x.name):
        init_lines.append(f"{it.name}:{it.data_type}:{','.join(str(d) for d in it.dims)}")
    parts.append("initializers:" + ",".join(init_lines))

    # nodes
    for n in g.node:
        header = f"node:{n.name or ''}:{n.op_type}"
        parts.append(header)
        if n.input:
            parts.append("  in:" + ",".join(n.input))
        if n.output:
            parts.append("  out:" + ",".join(n.output))
        # attributes sorted by name for determinism
        for a in sorted(n.attribute, key=lambda x: x.name):
            # use attribute SerializeToString to capture typed value consistently
            try:
                av = a.SerializeToString(deterministic=True)
            except TypeError:
                av = a.SerializeToString()
            parts.append(f"  attr:{a.name}:{av.hex()}")
    return "\n".join(parts).encode("utf-8")


def _hash_initializer(it: onnx.TensorProto, algorithm: str = "blake3") -> str:
    # Compose bytes: name \x00 dtype \x00 shape \x00 raw_bytes
    name_b = it.name.encode("utf-8")
    dtype_b = str(it.data_type).encode("utf-8")
    shape_b = ",".join(str(d) for d in it.dims).encode("utf-8")
    raw = b""
    # Use raw_data when available, else infer from data_field
    if it.raw_data:
        raw = it.raw_data
    else:
        try:
            arr = numpy_helper.to_array(it)
            raw = arr.tobytes()
        except Exception:
            raw = b""
    payload = name_b + b"\x00" + dtype_b + b"\x00" + shape_b + b"\x00" + raw
    return _hash_bytes(payload, algorithm=algorithm)


def _merkle_root(hex_leaves: List[str], algorithm: str = "blake3") -> Optional[str]:
    if not hex_leaves:
        return None
    # convert to bytes
    nodes = [bytes.fromhex(h) for h in hex_leaves]
    # Build binary merkle tree (left||right)
    while len(nodes) > 1:
        next_level: List[bytes] = []
        for i in range(0, len(nodes), 2):
            left = nodes[i]
            right = nodes[i + 1] if i + 1 < len(nodes) else nodes[i]
            next_level.append(bytes.fromhex(_hash_bytes(left + right, algorithm=algorithm)))
        nodes = next_level
    return nodes[0].hex()


def compute_seal(
    model: onnx.ModelProto,
    algorithm: str = "blake3",
    inits: str = "merkle",
    include_external: bool = False,
    force: bool = False,
) -> Dict:
    """Compute and return a seal JSON-compatible dict and embed into the ModelProto if requested.

    Returns the seal blob as dict (without modifying model). Caller may embed into
    model.metadata_props as desired.
    """
    algo_tag = f"{algorithm}-v1"
    blob: Dict = {"algorithm": algo_tag}
    graph_bytes = canonicalize_graph_bytes(model)
    blob["graph_hash"] = _hash_bytes(graph_bytes, algorithm=algorithm)

    # initializers
    inits_list = sorted(list(model.graph.initializer), key=lambda x: x.name)
    per_init = {}
    for it in inits_list:
        per_init[it.name] = _hash_initializer(it, algorithm=algorithm)
    if inits == "none":
        blob["inits_merkle"] = None
        blob["init_count"] = len(inits_list)
    elif inits == "per-init":
        blob["inits_merkle"] = None
        blob["init_count"] = len(inits_list)
        blob["per_init"] = per_init
    else:
        # merkle or full
        leaves = [per_init[n] for n in sorted(per_init.keys())]
        blob["inits_merkle"] = _merkle_root(leaves, algorithm=algorithm)
        blob["init_count"] = len(inits_list)
        if inits == "full":
            blob["per_init"] = per_init

    return blob


def embed_seal(model: onnx.ModelProto, blob: Dict, force: bool = False) -> None:
    # If already sealed, honor force flag
    existing = None
    for e in model.metadata_props:
        if e.key == "fuse.seal":
            existing = e.value
            break
    if existing and not force:
        raise ValueError("model already sealed; use --seal-force to overwrite")
    # remove existing
    if existing:
        # remove existing entry
        new_props = [p for p in model.metadata_props if p.key != "fuse.seal"]
        del model.metadata_props[:]
        for p in new_props:
            model.metadata_props.append(p)
    # add normalized JSON blob
    model.metadata_props.add(key="fuse.seal", value=json.dumps(blob, sort_keys=True))


# Small helper to verify seals quickly
def verify_seal(model: onnx.ModelProto) -> Dict:
    """Recompute and compare embedded seal, returning mismatch detail if any.

    Returns a dict {ok: bool, reason: str (if mismatch), expected: blob, found: blob}
    """
    # find seal
    seal = None
    for e in model.metadata_props:
        if e.key == "fuse.seal":
            try:
                seal = json.loads(e.value)
            except Exception:
                return {"ok": False, "reason": "invalid_seal_value"}
            break
    if not seal:
        return {"ok": False, "reason": "no_seal"}
    algo = seal.get("algorithm", "blake3-v1").split("-")[0]
    recomputed = compute_seal(model, algorithm=algo, inits=("per-init" if seal.get("per_init") else "merkle"))
    # Normalize compare: only keys present in seal
    for k in seal.keys():
        if seal.get(k) != recomputed.get(k):
            return {"ok": False, "reason": f"mismatch_{k}", "expected": recomputed, "found": seal}
    return {"ok": True}
