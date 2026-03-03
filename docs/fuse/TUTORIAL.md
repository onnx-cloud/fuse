# Fuse Language Tutorial

Welcome to the Fuse tutorial! This document walks through the core concepts of the Fuse DSL (domain-specific language for ONNX fusion), including syntax, common operations, control flow, functions, pragmas, and examples.

Fuse is a small cognitive compiler that lowers to ONNX. You write `*.fuse` source files and the compiler generates deterministic ONNX models. The language is designed for clarity and reproducibility.

---
## Table of Contents
1. [Getting Started](#getting-started)
2. [Basic Syntax](#basic-syntax)
3. [Primitive Operations](#primitive-operations)
4. [Variables and Types](#variables-and-types)
5. [Control Flow](#control-flow)
   - [If/Else](#ifelse)
   - [Loops](#loops)
6. [Functions](#functions)
7. [Pragmas and Metadata](#pragmas-and-metadata)
8. [Imports and Fusion](#imports-and-fusion)
9. [Worked Examples](#worked-examples)
10. [Tips & Best Practices](#tips--best-practices)

---
## Getting Started

To run a Fuse file, install the python package and use the CLI:

```sh
python -m fuse --input your_model.fuse --output model.onnx
```

Tests can be run with `make test`.  Separately, `make gold` will compile every file in `examples/golden/` via the CLI (optionally tracing with `make gold-trace`) and verify that an ONNX model is produced; it does **not** execute the full test suite.


## Basic Syntax

Fuse source is whitespace-insensitive and uses a colon (`:`) to separate declarations from type annotations or bodies.

```fuse
@domain("example")
param x: tensor(float, [1, 4]) = [[1, 2, 3, 4]]
```

Comments use `//` or `/* ... */`.

## Primitive Operations

Fuse defines operations that correspond to ONNX operators. Common ops include:

- `add`, `sub`, `mul`, `div`
- `matmul`, `conv`, `relu`, `softmax`, etc.

Example usage:

```fuse
v = add(x, y)
w = relu(v)
```

The names are lower-case, and argument order mirrors ONNX.

## Variables and Types

Variables are immutable SSA-style once bound. Use `param` for external inputs and `const` for compile-time constants.

Type annotations are optional but recommended:

```
const t: tensor(int64, [3]) = [1,2,3]
```

Type inference will propagate shapes and dtypes.

## Control Flow

### If/Else

```fuse
if x > 0 {
  y = relu(x)
} else {
  y = neg(x)
}
```

Conditions must be tensors that broadcast to a boolean value.

### Loops

`while` loops are supported and unrolled or lowered depending on static shape information.

```fuse
i = 0
sum = 0
while i < 10 {
  sum = sum + i
  i = i + 1
}
```

Loops must have a statically bounded iteration count for deterministic lowering.

## Functions

Functions are defined with `def`. Arguments can have default values and types.

```fuse
def square(x: tensor(float, [1])): tensor(float, [1]) {
  return mul(x, x)
}

z = square(a)
```

Functions support overloading and recursion where feasible. Lambdas are normalized automatically.

## Pragmas and Metadata

Pragmas annotate the program with additional information:

- `@domain` sets the ONNX domain prefix for namespacing
- `@import` and `@variant` for remote models
- `// @pragma: foo` for custom build-time hints

Metadata such as `trainable` or `shape` can be attached to parameters and will be serialized in `ModelProto.metadata_props`.

## Imports and Fusion

Fuse can import external ONNX models and fuse them via deterministic prefixing:

```fuse
@import from "resnet.onnx" {
  @variant(main) inputs=["input"] outputs=["output"]
}
```

Imported graphs are lowered with unique name prefixes and connected to the current context.

## Worked Examples

### Example 1: Simple Linear Layer

```fuse
@domain("example")
param x: tensor(float, [1, 128])
param w: tensor(float, [128, 64])
param b: tensor(float, [64])

y = add(matmul(x, w), b)
```

### Example 2: Conditional ReLU

```
@domain("example")
param x: tensor(float, [1, 10])
if max(x) > 0 {
  y = relu(x)
} else {
  y = x
}
```

### Example 3: For-style Loop with Folding

```
@domain("example")
i = 0
res = const(0)
while i < 5 {
  res = add(res, i)
  i = add(i, 1)
}
```

## Tips & Best Practices

- Use the SSA naming convention to avoid confusion: each variable is assigned once.
- Annotate shapes and types to catch errors early.
- Keep imports deterministic by specifying domains explicitly.
- Use `make gold` frequently when changing lowering behavior to detect regressions in example models.  (Run `make test` separately when you need full coverage.)

---

This tutorial is a starting point. Consult `docs/SPEC.md` for full grammar and `examples/` for more complex patterns. Happy fusing! 🎉
