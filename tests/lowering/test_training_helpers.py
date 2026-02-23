"""Tests for training helper classes."""

import pytest
from src.lowering.training_helpers import KeyValuePair, TrainingBindingHelper


class TestKeyValuePair:
    """Test KeyValuePair class."""
    
    def test_creation(self):
        """Test creating a key-value pair."""
        kv = KeyValuePair("param1", "grad1")
        assert kv.key == "param1"
        assert kv.value == "grad1"
    
    def test_string_conversion(self):
        """Test that keys and values are converted to strings."""
        kv = KeyValuePair(123, 456)
        assert kv.key == "123"
        assert kv.value == "456"
    
    def test_equality(self):
        """Test equality comparison."""
        kv1 = KeyValuePair("a", "b")
        kv2 = KeyValuePair("a", "b")
        kv3 = KeyValuePair("a", "c")
        
        assert kv1 == kv2
        assert kv1 != kv3
    
    def test_repr(self):
        """Test string representation."""
        kv = KeyValuePair("param", "grad")
        repr_str = repr(kv)
        assert "KeyValuePair" in repr_str
        assert "param" in repr_str
        assert "grad" in repr_str
    
    def test_hashable(self):
        """Test that KeyValuePair is hashable."""
        kv1 = KeyValuePair("a", "b")
        kv2 = KeyValuePair("a", "b")
        kv3 = KeyValuePair("c", "d")
        
        # Should be able to add to set
        s = {kv1, kv2, kv3}
        assert len(s) == 2  # kv1 and kv2 are equal


class TestTrainingBindingHelper:
    """Test TrainingBindingHelper class."""
    
    def test_add_loss_binding(self):
        """Test adding loss binding."""
        # Create a mock training info object
        class MockTrainingInfo:
            def __init__(self):
                self.loss_binding = []
        
        ti = MockTrainingInfo()
        TrainingBindingHelper.add_loss_binding(ti, "loss", "loss_output")
        
        # Check that binding was added
        assert len(ti.loss_binding) == 1
        assert ti.loss_binding[0].key == "loss"
        assert ti.loss_binding[0].value == "loss_output"
    
    def test_add_update_binding_fallback(self):
        """Test adding update binding with fallback."""
        # Create a mock training info without update_binding field
        class MockTrainingInfo:
            pass
        
        ti = MockTrainingInfo()
        TrainingBindingHelper.add_update_binding(ti, "param", "updated_param")
        
        # Check fallback storage
        assert hasattr(ti, "_update_bindings")
        assert len(ti._update_bindings) == 1
        assert ti._update_bindings[0].key == "param"
    
    def test_get_loss_bindings(self):
        """Test getting loss bindings."""
        class MockTrainingInfo:
            def __init__(self):
                self.loss_binding = [
                    KeyValuePair("loss1", "out1"),
                    KeyValuePair("loss2", "out2")
                ]
        
        ti = MockTrainingInfo()
        bindings = TrainingBindingHelper.get_loss_bindings(ti)
        assert len(bindings) == 2
        assert bindings[0].key == "loss1"
    
    def test_get_bindings_fallback(self):
        """Test getting bindings with fallback storage."""
        class MockTrainingInfo:
            pass
        
        ti = MockTrainingInfo()
        ti._loss_bindings = [KeyValuePair("loss", "out")]
        
        bindings = TrainingBindingHelper.get_loss_bindings(ti)
        assert len(bindings) == 1
        assert bindings[0].key == "loss"
