# Fuse — Graph-First Cognitive Coding

Write semantically typed programs that generate correct ONNX code and tests.

Runs efficiently on `CPU and GPU`, leveraging zero-copy and persistent memory.

Semantic types capture intent, ensuring safety, verifiability, and optimization.

`.fuse` programs can reason about their own structure and data flow.

Bridges human intent and AI pipelines seamlessly, enabling cognitive computation end-to-end.

## Canonical Example

```fuse
fn l1_score(x: <f32>[3]) -> <f32>[1] {
  c: <f32>[3] = [0,0,0]
  total = ReduceSum(Abs(Sub(x,c)), axes: <i64>[1]=[0], keepdims@=0)
  Reshape(Div(1.0, Add(total,1.0)), [1])
}

@proof
fn test_l1_score() -> <f32>[1] {
    x: <f32>[3] = [0.0, 0.0, 0.0]
    y = l1_score(x)
    assert Equal(y, [1.0])
    y
}
```

---

## 1. Structure

* **Top-level:** `@opset`, `@meta`, `param/weight`, `const`, `import`, `fn/block`, `graph/model`, `export`, `@proof`
* **Comments:** `#`, `//`, `/* */`
* **Attributes:** Postfix `@` only, e.g., `keepdims@=0`

---

## 2. Types & Shapes

* Scalars: `f32`, `f64`, `i64`, `i32`, `bool`
* Tensors: `<scalar>[dim1, dim2,...]`
* Functions declare typed args and return types; dims can be symbolic

---

## 3. Core Constructs

| Construct      | Mental model        |
| -------------- | ------------------- |
| `graph`        | Executable graph    |
| `fn`           | Reusable subgraph   |
| `weight`       | Frozen param        |
| `const`        | Literal             |
| `static if`    | Compile-time branch |
| `if`           | Runtime branch      |
| `Loop`         | Iteration           |
| `assert`       | Runtime invariant   |
| `stride@=2`    | Callsite attr       |
| `name@=value`  | Attr sugar          |
| `<f32>(x)`     | Cast                |
| `list[tensor]` | Variadic            |

---

## 4. Directives

| Directive     | Mental model      |
| ------------- | ----------------- |
| `@fuse`       | Lang version      |
| `@opset`      | Target opset      |
| `@namespace`  | Symbol ownership  |
| `@id`         | Stable identity   |
| `@meta`       | Annotations       |
| `@doc`        | Human docs        |
| `@proof`      | Executable spec   |
| `@quantize`   | Quantize          |
| `@dequantize` | Restore precision |

---

## 5. Calling & Sugar Operators

* Binary: `+ - * / @ ⊕` → maps to ONNX (`Add/Sub/Mul/Div/MatMul/Sum`)
* Calls: `Add(x,y)` or infix `x+y`
* Reduce axes: tensors `<i64>[...]`

---

## 6. Patterns & Best Practices

* Pure, small functions
* Typed constants for axes/shapes
* `@proof` blocks for deterministic tests
* Explicit casts (`<f32>(x)`)
* Postfix attributes only

---

## 7. Pitfalls

* Attributes vs tensor inputs (`ReduceSum(x, axes: <i64>[...])`)
* Implicit shape/type conversions not allowed

---

