# Sealing models with deterministic hashes

## Design principles 💡
- Determinism: The same logical source must yield identical hashes. We must canonicalize ordering and serialization before hashing.
- Efficiency: Use a fast cryptographic hash (BLAKE3 preferred) for speed; support SHA-256 (--sha256) for interoperability.
- Granularity: Offer both a compact global seal and optional per-initializer hashes and a Merkle root for fast partial verification.
- Non-destructive: Add only metadata properties to ModelProto (no schema changes), leaving ONNX internals untouched.
- Extensible: Version and name algorithm fields so future improvements remain compatible.

## What to hash 🔧
1. Graph fingerprint
   - Canonical representation of the lowered graph (deterministic ordering of nodes, attributes, and value names) rather than raw ModelProto bytes which can vary.
   - Include: op names, op types, attribute key/value pairs, node inputs/outputs, and stable metadata required to reinstantiate the graph deterministically.
   - Exclude: non-deterministic or environment-dependent fields (timestamps, build machine metadata).

2. Initializers
   - Each initializer should be hashed as a leaf: hash(name, dtype, shape, raw bytes).
   - Leaves should be sorted deterministically (e.g., by name) before computing aggregate (Merkle) hash.
   - For large initializers we still compute BLAKE3 on the bytes (efficient streaming API).

3. External data and references
   - For external tensors, include resolved reference (relative path normalized, or an explicit external hash field if available) or make it opt-in to include external file contents in the seal.

## Choice of hash algorithms 🧾
- Default: BLAKE3 (fast, streaming-friendly, secure for integrity checks). Emit hex lowercase.
- Fallback/compat: SHA-256 (hex) if needed for compatibility or regulatory reasons.
- Metadata must store algorithm id and version (e.g. `blake3-v1`, `sha256-v1`).

## Metadata layout and keys 🗂️
Store a small JSON blob under a single metadata key, and optionally store per-initializer keys for quick lookup.

- Primary key: `fuse.seal` (value = JSON string)

Example JSON structure:

{
  "algorithm": "blake3-v1",
  "graph_hash": "<hex>",
  "inits_merkle": "<hex>",    // Merkle root of initializer leaves (or null if inits omitted)
  "init_count": 42,
  "per_init": {
    "weight1": "<hex>",
    "bias1": "<hex>"
  }
}

Notes:
- `per_init` is optional (can be omitted to keep metadata small).
- If per-initializer hashes are included, they MUST be computed with the same algorithm as the aggregate.

Alternative: add flat keys for very small/legacy use (`fuse.seal.algorithm`, `fuse.seal.graph_hash`) — but prefer the single JSON blob for forward compatibility.

## CLI UX & flags 🧭
- `fuse --seal` or `fuse -S` (default algorithm = blake3)
- Options:
  - `--seal-algo <blake3|sha256>` (override default)
  - `--seal-inits [none|merkle|per-init|full]` (default: `merkle`)
  - `--seal-include-external [yes|no]` (default: `no`)
  - `fuse verify --seal <model.onnx>` to recompute and compare
  - `fuse --seal --sign <keyfile>` (optional, produce or attach signature; out of scope for initial PR but noted as follow-up)

## Verification flow ✔️
- `fuse verify --seal model.onnx`:
  1. Load model, extract `fuse.seal` metadata and algorithm.
  2. Recompute canonical graph hash and initializer hashes using deterministic ordering and encoding.
  3. Compare recomputed values with metadata. Return success if identical, else fail with details.
- For per-initializer mismatches, report names and mismatched hashes.

## Implementation notes 🔧
1. Canonical graph serialization
   - Add a deterministic serializer (function e.g. `canonicalize_graph_for_seal(graph_ctx, model_proto) -> bytes`). Use existing deterministic name allocation and ordering primitives (GraphContext) to ensure the same layout used during lowering.
   - Serialize using a simple and stable compact format (e.g., newline-delimited node records with escaping or stable JSON with sorted keys and deterministic numeric formatting).

2. Initializer hashing
   - For each initializer: hash the tuple `(name || b"\x00" || dtype || b"\x00" || shape_bytes || b"\x00" || raw_bytes)` to avoid collisions.
   - Sort by `name` before building Merkle leaves.
   - Provide streaming BLAKE3 to avoid large memory usage.

3. Merkle tree
   - Use a binary Merkle tree with left/right concatenation order defined (e.g., left||right).
   - Store only root in `inits_merkle` by default; per-init entries optional.

4. Metadata embedding
   - Set ModelProto.metadata_props.add(key='fuse.seal', value=json.dumps(blob)) before writing file.
   - Keep the metadata size small (omit per-init by default), but make `--seal-inits per-init` available for debugging.

5. Tests
   - Add unit tests for: canonical serializer determinism, initializer hashing, Merkle implementation, CLI `--seal` and `verify` against `examples/golden` models.
   - Add a golden test verifying that running `fuse --seal` on known inputs reproduces the expected metadata.

6. Compatibility & edge cases
   - If a model already has `fuse.seal`, `--seal` should either fail (avoid accidental re-sealing) or replace only if `--force` passed.
   - For models with external tensors, default to excluding external file contents; offer opt-in to include external contents.

## Security and signing follow-ups 🔒
- Sealing provides integrity detection; for non-repudiation add signing: compute seal blob and sign the blob with a private key (e.g., Ed25519), store signature in `fuse.seal.sig` and signer metadata.
- For auditability, provide `fuse sign` / `fuse verify --signature` commands in a follow-up RFC.

## Backwards compatibility & migration ↩️
- Version the `algorithm` field (e.g. `blake3-v1`) so future changes don't break verification.
- If future specs include richer metadata, keep the top-level `fuse.seal` JSON as the single source of truth.

## Proposed milestones (small iterative PRs) 🛠️
1. Add deterministic serializer and unit tests for the graph fingerprint.
2. Add initializer hash and Merkle root computation + tests.
3. CLI `--seal` that embeds `fuse.seal` (defaults: blake3 + merkle) + `fuse verify --seal`.
4. Add per-initializer optional metadata and golden test(s).
5. (Optional) Add signing support and `fuse sign/verify` commands.

## Example metadata (illustrative)

`ModelProto.metadata_props` contains key `fuse.seal` value:

```
{"algorithm": "blake3-v1",
 "graph_hash": "9f12ab...",
 "inits_merkle": "ac4e1e...",
 "init_count": 10
}
```

Per-init optional entry:

```
{"per_init": {"fc1.weight": "fa12...", "fc1.bias": "3c44..."}}
```

## Summary ✅
This design gives us a fast, low-friction way to ensure model artifact integrity and reproducibility. BLAKE3 as the default algorithm keeps sealing cheap and practical for CI; Merkle roots let us scale verification without bloat. The approach is extensible to signing for provenance when needed.