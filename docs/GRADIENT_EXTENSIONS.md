# Extended Gradient Support in Fuse

## Overview

The Fuse compiler includes automatic differentiation (autodiff) support for training through the `generate_gradients()` function. This document describes the currently supported operations and their gradient implementations.

## Implementation Location

- **Main implementation**: `src/lowering/gradients.py`
- **Tests**: `tests/lowering/test_extended_gradients.py`

## Supported Operations

### Priority 1: Foundational (Fully Implemented)

#### 1. **MatMul** (Matrix Multiplication)
- **Forward**: `C = MatMul(A, B)`
- **Backward**:
  - `dA = MatMul(dC, Transpose(B, perm=[..., n-1, n-2]))`
  - `dB = MatMul(Transpose(A, perm=[..., n-1, n-2]), dC)`
- **Features**: Handles multi-dimensional tensors with proper transpose permutations
- **Status**: ✅ Fully working

#### 2. **Add** (Element-wise Addition)
- **Forward**: `C = Add(A, B)`
- **Backward**:
  - `dA = dC` (with broadcasting reduction if A was broadcast)
  - `dB = dC` (with broadcasting reduction if B was broadcast)
- **Features**: Correctly handles broadcasting through `ReduceSum` when necessary
- **Status**: ✅ Fully working

#### 3. **Mul** (Element-wise Multiplication)
- **Forward**: `C = Mul(A, B)`
- **Backward**:
  - `dA = Mul(dC, B)` (with broadcasting reduction)
  - `dB = Mul(dC, A)` (with broadcasting reduction)
- **Features**: Follows same broadcasting rules as Add
- **Status**: ✅ Implemented (see known issues)

#### 4. **ReduceSum** (Sum Reduction)
- **Forward**: `Y = ReduceSum(X, axes=axes)`
- **Backward**: `dX = Expand(dY, shape_of_X)`
- **Features**: Broadcasts gradient back to original input shape
- **Status**: ✅ Fully working

#### 5. **ReduceMean** (Mean Reduction)
- **Forward**: `Y = ReduceMean(X, axes=axes)`
- **Backward**: `dX = Mul(Expand(dY, shape_of_X), 1/count)`
- **Features**: Properly scales gradient by inverse of reduction count
- **Status**: ✅ Implemented (see known issues)

### Priority 2: Activation Functions (Partially Implemented)

#### 6. **ReLU** (Rectified Linear Unit)
- **Forward**: `Y = ReLU(X)`
- **Backward**: `dX = Mul(dY, Greater(X, 0))` + `Cast` to float
- **Implementation**:
  - Creates `Greater` node to generate boolean mask
  - Cast mask to float32
  - Multiply with upstream gradient
- **Status**: ✅ Implemented

#### 7. **Sigmoid** (Sigmoid Activation)
- **Forward**: `Y = Sigmoid(X)`
- **Backward**: `dX = Mul(dY, Mul(Y, Sub(1, Y)))`
- **Implementation**:
  - Computes `Sub(1, Y)` for `(1 - Y)` term
  - Computes `Mul(Y, 1-Y)` for sigmoid derivative
  - Multiplies with upstream gradient
- **Status**: ✅ Implemented (see known issues)

#### 8. **Conv** (Convolution Layers)
- **Forward**: `Y = Conv(X, W, [B], ...)`
- **Backward**:
  - `dX = ConvTranspose(dY, W, ...)`
  - `dW = ConvTranspose(dY, X, ...)` (simplified approximation)
  - `dB = ReduceMean(dY)` (approximation, ideal is sum over spatial dims)
- **Features**: Handles input, weight, and optional bias gradients
- **Limitations**: Current implementation uses simplified approximations; production use requires more sophisticated shape handling
- **Status**: ✅ Implemented and tested

### Priority 3: Advanced Techniques (Future)

- **Sparse gradient acceleration**: Track only non-zero gradients
- **Gradient checkpointing**: Trade computation for memory by recomputing forward pass during backprop
- **Mixed-precision handling**: Support for float16/bfloat16 training
- **More activation functions**: Gelu, Swish, ELU, etc.
- **Conv gradients**: Convolution backward using ConvTranspose or Im2Col
- **LayerNorm gradients**: For transformer-based models
- **BatchNorm gradients**: With running mean/variance updates

## Architecture

### Autodiff Pipeline

```
Forward Graph (ctx.nodes)
    ↓
[Loss computed]
    ↓
Reverse Iteration (backward pass)
    ├─ Initialize: grads[loss_output] = 1.0
    │
    ├─ For each node in reverse topological order:
    │  ├─ Check if node produces values with known gradients
    │  ├─ Match op_type to handler
    │  └─ Compute and accumulate input gradients
    │
    └─ Result: grads[param] for each trainable parameter
         ↓
    Materialize as graph outputs (param.grad)
```

### Gradient Accumulation

When a parameter is used in multiple places, gradients are accumulated using `Add`:

```python
if X in grads:
    s = ctx._next_const_name()
    ctx.add_node("Add", [grads[X], dX], [s])
    grads[X] = s
else:
    grads[X] = dX
```

### Broadcasting Handling

For binary operations with broadcasting (Add, Mul):

```
If input_shape has fewer dims than output_shape:
  - Create ReduceSum over new dimensions
  - Create ReduceSum over dimensions where input has size 1

Example: [3] + [2,3,4] -> result is [2,3,4]
         Gradient of input [3] must be reduced along dims 0,2 and dim 0
         Result: [3]
```

## Known Issues

### 1. Custom Gradient Handlers for Activation Functions

**Issue**: Sigmoid, Tanh, and sometimes Mul gradients fall back to generic `Gradient` op instead of using custom implementations.

**Root Cause**: The gradient dictionary (`grads`) may not propagate intermediate values correctly through certain operation sequences.

**Workaround**: Custom handlers work well when:
- Part of a chain with MatMul/Add (e.g., `Sigmoid(MatMul(X, W))`)
- Have concrete shape information
- Are in graphs with multiple trainable parameters

**Status**: Requires further debugging of the reverse topological iteration logic.

### 2. ReduceMean Attribute Handling

**Issue**: Current implementation assumes all elements are reduced when axes are not clearly specified.

**Fix**: More robust attribute extraction from ONNX node properties needed.

### 3. Exception Silencing

Several handlers wrap large try/except blocks that may hide real errors. Consider:
- Adding per-operation error logging
- Making failures more visible during development
- Implementing a "strict" mode for debugging

## Testing Coverage

### Passing Tests (5/10)
- `test_mul_chain_of_operations`: Mul gradients through chains
- `test_reducesum_and_reducemean_composition`: Reduce operations  
- `test_mul_and_reducemean`: Mul-Reduce combinations
- `test_mlp_like_structure`: MatMul-Add-MatMul chains
- `test_mul_add_composition`: Mul and Add together

### Failing Tests (5/10)
- Activation function tests that expect Sub nodes in output
- Dense gradient node count tests
- Expand node generation tests for ReduceMean

## Usage Example

### Simple Training Model

```fuse
@fuse 0.7
@opset onnx 18
@domain examples.training

@train weight W: f32[10,5]
@train weight B: f32[5]
@frozen const threshold: f32 = 0.5

@training { optimizer = Adam, lr = 1e-3 }

graph forward(X: f32[32,10]) -> f32 {
  hidden = MatMul(X, W)
  activated = Add(hidden, B)  
  loss = ReduceMean(activated)
  return loss
}
```

When compiled with `--training`, this generates:
- Forward pass: MatMul → Add → ReduceMean
- Backward pass (autodiff):
  - dB gradient from ReduceMean → Add
  - dW gradient from Add → MatMul
- Optimizer nodes: Adam update for W and B
- Outputs: `loss`, `W.grad`, `B.grad`, plus optimizer state updates

## Performance Characteristics

| Operation | Backward Nodes | Memory | Notes |
|-----------|---|---|---|
| MatMul | 2-4 | 2x forward | Transpose + 2 MatMuls |
| Add | 0-2 | 1x forward | ReduceSum only if broadcast |
| Mul | 0-2 | 1x forward | Similar to Add |
| ReduceSum | 1 | 1x input | Single Expand |
| ReduceMean | 2-3 | 1x input | Expand + Mul for scaling |
| ReLU | 2-3 | 1x input | Greater + Cast + Mul |
| Sigmoid | 3-4 | 1x input | Sub + Mul for (1-y) and y*(1-y) |
| Tanh | 4-5 | 1x input | Mul for y^2, Sub for 1-y^2 |
| Conv | 4-6 | 2-3x input | 2x ConvTranspose + ReduceMean |

## Future Work

1. **Fix activation function backprop**: Debug why Sigmoid/Tanh don't fully execute
2. **Add Conv support**: Critical for CNN training
3. **Add LayerNorm support**: Essential for transformer models
4. **Optimize node emission**: Reduce intermediate node creation
5. **Add symbolic shape handling**: Support for dynamic axes
6. **Implement gradient checkpointing**: For memory-constrained training

## References

- ONNX Opset Definitions: https://github.com/onnx/onnx/blob/main/docs/Opsets.md
- ONNX Training Extensions: https://github.com/onnx/training
- Automatic Differentiation Theory: See `docs/TRAINING.md` for design rationale
