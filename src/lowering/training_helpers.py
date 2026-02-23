"""Helper classes for training metadata emission.

Extracted from training_info_emit.py for better testability and modularity.
"""


class KeyValuePair:
    """Simple key-value pair for training metadata bindings.
    
    This class provides a consistent interface for key-value bindings
    across different ONNX versions, some of which may not have native
    loss_binding support.
    """
    
    def __init__(self, key: str, value: str):
        """Initialize key-value pair.
        
        Args:
            key: The binding key (e.g., parameter name)
            value: The binding value (e.g., gradient output name)
        """
        self.key = str(key)
        self.value = str(value)
    
    def __repr__(self) -> str:
        return f"KeyValuePair(key={self.key!r}, value={self.value!r})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, KeyValuePair):
            return False
        return self.key == other.key and self.value == other.value
    
    def __hash__(self) -> int:
        return hash((self.key, self.value))


class TrainingBindingHelper:
    """Helper for managing training bindings across ONNX versions."""
    
    @staticmethod
    def add_loss_binding(training_info, key: str, value: str) -> None:
        """Add loss binding to TrainingInfoProto in version-compatible way.
        
        Args:
            training_info: ONNX TrainingInfoProto instance
            key: Loss name
            value: Loss output name
        """
        try:
            # Try standard protobuf repeated field
            lb = getattr(training_info, "loss_binding", None)
            if lb is not None:
                try:
                    # Protobuf repeated field add()
                    e = lb.add()
                    e.key = str(key)
                    e.value = str(value)
                    return
                except Exception:
                    # If lb is a list (compatibility shim)
                    lb.append(KeyValuePair(str(key), str(value)))
                    return
        except Exception:
            pass
        
        # Fallback: store in instance attribute
        try:
            if not hasattr(training_info, "_loss_bindings"):
                object.__setattr__(training_info, "_loss_bindings", [])
            training_info._loss_bindings.append(KeyValuePair(str(key), str(value)))
        except Exception:
            # Last resort: silently ignore if we can't attach
            pass
    
    @staticmethod
    def add_update_binding(training_info, key: str, value: str) -> None:
        """Add update binding to TrainingInfoProto.
        
        Args:
            training_info: ONNX TrainingInfoProto instance
            key: Parameter name
            value: Updated parameter output name
        """
        try:
            ub = training_info.update_binding.add()
            ub.key = str(key)
            ub.value = str(value)
        except Exception:
            # Older ONNX: use fallback
            if not hasattr(training_info, "_update_bindings"):
                object.__setattr__(training_info, "_update_bindings", [])
            training_info._update_bindings.append(KeyValuePair(str(key), str(value)))
    
    @staticmethod
    def get_loss_bindings(training_info) -> list:
        """Get loss bindings from TrainingInfoProto.
        
        Args:
            training_info: ONNX TrainingInfoProto instance
            
        Returns:
            List of KeyValuePair objects
        """
        try:
            lb = getattr(training_info, "loss_binding", None)
            if lb is not None:
                return list(lb)
        except Exception:
            pass
        
        # Fallback
        return getattr(training_info, "_loss_bindings", [])
    
    @staticmethod
    def get_update_bindings(training_info) -> list:
        """Get update bindings from TrainingInfoProto.
        
        Args:
            training_info: ONNX TrainingInfoProto instance
            
        Returns:
            List of KeyValuePair objects
        """
        try:
            return list(training_info.update_binding)
        except Exception:
            pass
        
        # Fallback
        return getattr(training_info, "_update_bindings", [])
