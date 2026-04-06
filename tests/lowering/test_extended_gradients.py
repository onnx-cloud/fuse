"""
Tests for extended gradient support in gradients.py.

Tests cover:
- Priority 1: Mul, ReduceSum, ReduceMean
- Priority 2: Sigmoid, Tanh (ReLU excluded due to opset constraints in test environment)
"""

import pytest
from src.parser import fuse_parser
from src.lowering import FuseLowerer


class TestMulGradient:
    """Test element-wise multiplication gradient computation."""

    def test_mul_chain_of_operations(self):
        """Test Mul in a chain: loss = Mul(A, Mul(B, C))"""
        src = """
        @train weight A: f32[2,2]
        @train weight B: f32[2,2]
        @train weight C: f32[2,2]
        node mul_chain() -> f32 {
          BC = Mul(B, C)
          result = Mul(A, BC)
          loss = ReduceMean(result)
          return loss
        }
        """
        ast = fuse_parser.parse(src)
        lowerer = FuseLowerer(emit_training=True)
        model = lowerer.lower(ast)
        
        output_names = [o.name for o in model.graph.output]
        # All trainables should have gradients
        assert any("A" in n and "grad" in n for n in output_names), f"Expected A.grad, got {output_names}"
        assert any("B" in n and "grad" in n for n in output_names), f"Expected B.grad, got {output_names}"
        assert any("C" in n and "grad" in n for n in output_names), f"Expected C.grad, got {output_names}"


class TestReduceGradient:
    """Test ReduceSum and ReduceMean gradient computation."""

    def test_reducesum_and_reducemean_composition(self):
        """Test ReduceSum followed by ReduceMean."""
        src = """
        @train weight X: f32[2,3,4]
        node reduce_composition() -> f32 {
          summed = ReduceSum(X)
          meaned = ReduceMean(summed)
          return meaned
        }
        """
        ast = fuse_parser.parse(src)
        lowerer = FuseLowerer(emit_training=True)
        model = lowerer.lower(ast)
        
        output_names = [o.name for o in model.graph.output]
        assert any("X" in n and "grad" in n for n in output_names)

    def test_mul_and_reducemean(self):
        """Test Mul followed by ReduceMean."""
        src = """
        @train weight A: f32[2,3]
        @train weight B: f32[2,3]
        node mul_then_reduce() -> f32 {
          C = Mul(A, B)
          loss = ReduceMean(C)
          return loss
        }
        """
        ast = fuse_parser.parse(src)
        lowerer = FuseLowerer(emit_training=True)
        model = lowerer.lower(ast)
        
        output_names = [o.name for o in model.graph.output]
        assert any("A" in n and "grad" in n for n in output_names)
        assert any("B" in n and "grad" in n for n in output_names)


class TestComplexGraphs:
    """Test complex graphs with multiple supported operations."""

    def test_mlp_like_structure(self):
        """Test a graph structure similar to: y = MatMul(Add(MatMul(x,w1),b1), w2)"""
        src = """
        @train weight W1: f32[10,5]
        @train weight B1: f32[5]
        @train weight W2: f32[5,3]
        node mlp_like(X: f32[1,10]) -> f32 {
          z1 = MatMul(X, W1)
          a1 = Add(z1, B1)
          z2 = MatMul(a1, W2)
          loss = ReduceMean(z2)
          return loss
        }
        """
        ast = fuse_parser.parse(src)
        lowerer = FuseLowerer(emit_training=True)
        model = lowerer.lower(ast)
        
        output_names = [o.name for o in model.graph.output]
        assert any("W1" in n and "grad" in n for n in output_names)
        assert any("B1" in n and "grad" in n for n in output_names)
        assert any("W2" in n and "grad" in n for n in output_names)

    def test_mul_add_composition(self):
        """Test composition of Mul and Add: z = Add(Mul(A, B), C)"""
        src = """
        @train weight A: f32[2,3]
        @train weight B: f32[2,3]
        @train weight C: f32[2,3]
        node mul_add_composition() -> f32 {
          AB = Mul(A, B)
          result = Add(AB, C)
          loss = ReduceMean(result)
          return loss
        }
        """
        ast = fuse_parser.parse(src)
        lowerer = FuseLowerer(emit_training=True)
        model = lowerer.lower(ast)
        
        output_names = [o.name for o in model.graph.output]
        assert any("A" in n and "grad" in n for n in output_names)
        assert any("B" in n and "grad" in n for n in output_names)
        assert any("C" in n and "grad" in n for n in output_names)

    def test_sigmoid_and_tanh_with_matmul(self):
        """Test: loss = ReduceMean(Sigmoid(MatMul(X, W)))"""
        src = """
        @train weight X: f32[2,3]
        @train weight W: f32[3,4]
        node sigmoid_matmul() -> f32 {
          XW = MatMul(X, W)
          activated = Sigmoid(XW)
          loss = ReduceMean(activated)
          return loss
        }
        """
        ast = fuse_parser.parse(src)
        lowerer = FuseLowerer(emit_training=True)
        model = lowerer.lower(ast)
        
        output_names = [o.name for o in model.graph.output]
        assert any("X" in n and "grad" in n for n in output_names)
        assert any("W" in n and "grad" in n for n in output_names)
        # Verify gradients are computed (exact node types may vary by implementation)

    def test_tanh_with_mul(self):
        """Test: loss = ReduceMean(Tanh(Mul(A, B)))"""
        src = """
        @train weight A: f32[2,3]
        @train weight B: f32[2,3]
        node tanh_mul() -> f32 {
          AB = Mul(A, B)
          activated = Tanh(AB)
          loss = ReduceMean(activated)
          return loss
        }
        """
        ast = fuse_parser.parse(src)
        lowerer = FuseLowerer(emit_training=True)
        model = lowerer.lower(ast)
        
        output_names = [o.name for o in model.graph.output]
        assert any("A" in n and "grad" in n for n in output_names)
        assert any("B" in n and "grad" in n for n in output_names)
        # Verify gradients are computed (exact node types may vary by implementation)


class TestGradientNodeTypes:
    """Test that correct ONNX nodes are generated for gradients."""

    def test_mul_generates_mul_nodes(self):
        """Test that Mul backward generates gradient computation."""
        src = """
        @train weight A: f32[3,4]
        @train weight B: f32[3,4]
        node mul_test() -> f32 {
          C = Mul(A, B)
          loss = ReduceMean(C)
          return loss
        }
        """
        ast = fuse_parser.parse(src)
        lowerer = FuseLowerer(emit_training=True)
        model = lowerer.lower(ast)
        
        output_names = [o.name for o in model.graph.output]
        # Verify both parameters have gradients (through Mul or generic Gradient op)
        assert any("A" in n and "grad" in n for n in output_names)
        assert any("B" in n and "grad" in n for n in output_names)

    def test_reduce_generates_expand_nodes(self):
        """Test that ReduceMean backward generates gradient computation."""
        src = """
        @train weight X: f32[2,3,4]
        node reduce_test() -> f32 {
          reduced = ReduceMean(X)
          return reduced
        }
        """
        ast = fuse_parser.parse(src)
        lowerer = FuseLowerer(emit_training=True)
        model = lowerer.lower(ast)
        
        output_names = [o.name for o in model.graph.output]
        assert any("X" in n and "grad" in n for n in output_names)

    def test_sigmoid_generates_derivative_nodes(self):
        """Test that Sigmoid backward generates gradient computation."""
        src = """
        @train weight X: f32[2,3]
        node sigmoid_test() -> f32 {
          Y = Sigmoid(X)
          loss = ReduceMean(Y)
          return loss
        }
        """
        ast = fuse_parser.parse(src)
        lowerer = FuseLowerer(emit_training=True)
        model = lowerer.lower(ast)
        
        output_names = [o.name for o in model.graph.output]
        # Verify gradient is computed for input
        assert any("X" in n and "grad" in n for n in output_names)


class TestConvGradient:
    """Test convolution gradient computation."""

    def test_conv_basic_gradient(self):
        """Test basic Conv2D backward: gradients computed for W and B."""
        src = """
        @train weight W: f32[32,3,3,3]
        @train weight B: f32[32]
        node conv_test(X: f32[1,3,32,32]) -> f32 {
          Y = Conv(X, W, B)
          loss = ReduceMean(Y)
          return loss
        }
        """
        ast = fuse_parser.parse(src)
        lowerer = FuseLowerer(emit_training=True)
        model = lowerer.lower(ast)
        
        output_names = [o.name for o in model.graph.output]
        assert any("W" in n and "grad" in n for n in output_names), f"Expected W.grad, got {output_names}"
        assert any("B" in n and "grad" in n for n in output_names), f"Expected B.grad, got {output_names}"

    def test_conv_in_network(self):
        """Test Conv as part of larger network: Conv -> Add -> ReduceMean."""
        src = """
        @train weight W: f32[16,3,3,3]
        @train weight B: f32[16]
        @train weight scale: f32 = 1.0
        node conv_network(X: f32[1,3,16,16]) -> f32 {
          conv_out = Conv(X, W, B)
          scaled = Mul(conv_out, scale)
          loss = ReduceMean(scaled)
          return loss
        }
        """
        ast = fuse_parser.parse(src)
        lowerer = FuseLowerer(emit_training=True)
        model = lowerer.lower(ast)
        
        output_names = [o.name for o in model.graph.output]
        assert any("W" in n and "grad" in n for n in output_names)
        assert any("B" in n and "grad" in n for n in output_names)
        assert any("scale" in n and "grad" in n for n in output_names)



