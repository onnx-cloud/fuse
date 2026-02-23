# Fuse Cookbook

 ./scripts/install_local.shIncremental recipes from trivial to real-world.

## 1) Identity model

See [examples/golden/model.fuse](../examples/golden/model.fuse).

Best practice — canonical model identifier:
- We recommend adding a stable `@id` to exported models to aid provenance and inter-repo referencing. Example:

```fuse
@id "urn:my_model.v1"
model my_model(...) { ... }
```

This `@id` is enforced as a lint *recommendation* (warning) and is used by tooling that publishes or indexes models.

## 2) Constants and params

See [examples/golden/params_consts.fuse](../examples/golden/params_consts.fuse).

Key ideas:
- `const` becomes an ONNX initializer.
- `param` declares a typed input (and can be used for weights).

## 3) Elementwise algebra + inline tests

See [examples/golden/algebraic.fuse](../examples/golden/algebraic.fuse).

Run inline tests:
- `python -m src.cli test -f examples/golden/algebraic.fuse`

## 4) Convolution (attributes)

See [examples/golden/conv.fuse](../examples/golden/conv.fuse).

Key idea:
- Attributes are written as `stride@=2` (must be literals).

## 5) Metadata + docs + more ops

See [examples/golden/meta_doc_ops.fuse](../examples/golden/meta_doc_ops.fuse).

Key ideas:
- Module `@meta key=value` -> ONNX `model.metadata_props`
- `@note "..."` inside a function attaches to the graph and to the next emitted node
- `@meta` on args/returns is encoded as JSON in ValueInfo `doc_string`

## 6) Import fusion (real-world ONNX)

See [jupyter/cookbook/finbert.fuse](../jupyter/cookbook/finbert.fuse) and the ONNX artifacts under [onnx/nlp/finbert](../onnx/nlp/finbert).

Key idea:
- Imported ONNX graphs are fused and called like normal functions.

Run & validate:
- `./scripts/run_examples.sh --validate`
