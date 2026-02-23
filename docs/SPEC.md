# **Fuse to ONNX:  Spec**

cli-first, modular tooling for .fuse and .onnx

**Type:** contract | test | data-driven
**Status:** DRAFT

## Stakeholders
- **org_exec**: Executive sponsor
- **org_ops**: Release / packaging owners
- **org_role**: Compiler / product owner
- **org_gov**: Legal / governance
- **org_self**: Fuse maintainers

fuse ast test.fuse

## **1. Module / Metadata Syntax**

```fuse
@fuse 0.7
@opset onnx 13
@domain examples
```

* `@fuse` → version of the Fuse language
* `@opset` → default ONNX opset
* `@domain` → logical module / domain
  

## **2. Imports from remote or local resources**

- import from local Zoo , fail fast if dtype mismatch
```
@import catalog.vision.classifier @1.0 as Eye: f32[B,3,H,W]
```

```fuse
@import nlp.finbert @1.0 as FinBert from "https://huggingface.co/Xenova/finbert/blob/main/onnx/model_q4f16.onnx"
```

* **Single fn per import**: groups all variants
* **`default` keyword**: variant used if no runtime selection
* Variants may include optional metadata:

  * `external_data="..."` → for large tensors
  * `opset=13` → override default opset
  * `alias="finbert_fp32"` → alternative domain

### **Variant Inference**

* **Primary:** `.fuse` `@variant default`
* **Fallback:** inspect ONNX initializers/dtypes:

  * INT8 → `int8`
  * FLOAT16 → `f16` / `q4f16`
  * FLOAT → `fp32`
* Optional: embed `variant` in `metadata_props` inside ONNX for self-describing models
  

## **3. Params and Constants**

```fuse
@param w: f32[64,3,7,7]
@const eps: f32 = 1e-5
```

* `param` → graph input, must match dtype and shape
* `const` → initializer, embedded or external
* All shapes may include **symbolic dimensions** `_` for dynamic axes
  

## **4. Function / Model Declaration**

```fuse
fn norm(x: f32[64]) -> f32[64] {
    x
}

graph classify(input_ids: i32[1,32],
               attention_mask: i32[1,32],
               variant: str="fp32") -> f32[3] {
    y = FinBert(input_ids, attention_mask, variant=variant)
    y
}
```

* `block` → local computation (source-level keyword; `node` is accepted for compatibility) 
* `model` → top-level subgraph for export/fusion
* Optional `variant` argument allows **runtime selection** of imported model variant
  

## **5. Expressions and Control Flow**

```fuse
x = Add(a, b)
y = MatMul(x, w)
if mask {
    y = Mul(y, mask)
} else {
    y = Identity(y)
}
```

* Supports standard operators, function calls, and `if` / `static if`
* Binary ops: `+ - * / @ ⊕`
* Call arguments support named params and literal attributes: `Conv(x, w, stride@=2)`
  

## **6. ONNX Lowering Strategy**

1. **Parse `.fuse` → AST** using `lark` 
2. **Typed AST nodes:**

   * `Param` → graph input (`ValueInfoProto`)
   * `Const` → initializer (`TensorProto`)
   * `nodeDecl` / `ModelDecl` → nodes (`NodeProto`)
   * `ImportDecl` → fused subgraph domain
3. **Variant selection**:

   * Default from `@variant default`
   * Runtime selection via model argument
   * Fallback from ONNX dtypes or metadata
4. **domain management**: prefix imported tensor names to avoid collisions
5. **Multi-fn graph building**: wire outputs → inputs across fused models
6. **Shape inference**: run `onnx.shape_inference.infer_shapes`
7. **Validation**: `onnx.checker.check_model`
  

## **7. Dtype Mapping Table**

| Fuse scalar | ONNX TensorProto |
| ---- | ---- |
| f32         | FLOAT            |
| f64         | DOUBLE           |
| i64         | INT64            |
| i32         | INT32            |
| i16         | INT16            |
| i8          | INT8             |
| u64         | UINT64           |
| u32         | UINT32           |
| u16         | UINT16           |
| u8          | UINT8            |
| bool        | BOOL             |
| f16         | FLOAT16          |
| bf16        | BFLOAT16         |
| complex64   | COMPLEX64        |
| complex128  | COMPLEX128       |
  

## **8. Large Tensors / External Data**

* Constants or pre-trained weights can be stored externally:

```fuse
@variant fp32 file="..." external_data="weights_fp32.bin"
```

* Explicit external constants (MVP syntax):

```fuse
# Load a tensor from an external binary file located relative to the fuse file
const W: f32[2, 2] = @import("weights/f32_2x2.bin")
const W1: f32[2, 2] = @import("weights/f32_2x2.onnx")
```

* `@import(...)` declares that the tensor's data lives in an external file. During lowering the file is validated and the emitted ONNX initializer will be marked with `external_data` and a `location` pointing at the binary file. The CLI supports `--externalize N` to automatically write large embedded tensors into external files and update the `external_data` references. Use the `--bake` flag to force embedding imported tensor bytes directly into the ONNX initializer (`raw_data`) instead of emitting `external_data` references.

* During lowering, external data is referenced instead of embedded to save memory
  

## **9. Fusion Strategy**

* Fused models get **prefixed domains**
* Inputs/outputs wired deterministically
* Multiple variants can coexist without collisions
* Imported constants become initializers in fused graph
  

## **10. Tooling Ergonomics**

* **Explicit metadata** (`@import` + `@variant`) → deterministic, parseable
* **Default variant** → ensures runtime safety
* **AST + manifest registry** → deterministic fusion, multi-model wiring
* **Optional external files** → manage large weights elegantly
* **Runtime variant selection** → enables dynamic precision switching

* **Assertions**: `assert expr` statements are evaluated at lowering; if an assertion can be resolved statically it is enforced (failing lowering on false). Non-evaluable assertions are recorded as textual checks in the model metadata under the key `fuse.asserts` for runtime tooling or debug checks.

## **11. Training**

Fuse supports a lightweight training metadata and annotation system intended to document training-time intent and preserve simple trainability metadata through lowering.

- `@training { ... }` — a module-level metadata block describing training configuration. Supports both flat kv pairs and small nested blocks (e.g., `optimizer: { type = adamw, lr = 0.001 }`). The transformer produces a `meta` fn `{ name: "fuse.training", value: {...} }`.

- `@train <param_decl>` — syntactic sugar that marks a parameter or weight as *trainable* by attaching `trainable = True` to the parsed declaration node.

- `@frozen <param_decl|const_decl>` — marks the declaration as *frozen* with `trainable = False`.

Example:

```fuse
@training { optimizer: { type = adamw, lr = 0.0003 }, schedule: { type = cosine, warmup = 1000 } }
@train weight W: f32[features,64]
@frozen const B: f32 = 1e-3
```

Lowering behavior and metadata:

- The `trainable` flag is preserved in the AST. Lowering does not mandate a particular training workflow (it is orthogonal to ONNX execution), but Fuse records trainability information under model metadata `trainables` when lowering declares trainable/frozen constants or params. This metadata is serialized in `ModelProto.metadata_props` as a stable JSON string mapping qualified names to booleans, e.g. `{ "ns.model.W": true, "ns.model.B": false }`.

- Consumers may use this metadata for tooling (e.g., to extract trainable parameters, freeze weights, or integrate with training pipelines).

Testing & examples:

- Tests covering parsing, lowering, and export are provided under `tests/training/` (`test_parsing_training.py`, `test_lowering_trainables_metadata.py`, `test_export_training_example.py`).
- Suggested example: a notebook that trains toy weights (NumPy), writes them into a Fuse snippet as `@train weight` or `const` declarations, exports to ONNX, and validates using `onnx.ReferenceEvaluator`.

