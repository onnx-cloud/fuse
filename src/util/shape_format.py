from typing import Iterable, Any

def format_shape(dims: Iterable[Any]) -> str:
    """Format shape for display. Handles int tuples and ONNX dim strings."""
    parts = []
    for d in dims:
        if hasattr(d, "dim_value") and hasattr(d, "dim_param"):
            if d.dim_value:
                parts.append(str(d.dim_value))
            elif d.dim_param:
                parts.append(d.dim_param)
            else:
                parts.append("?")
        else:
            parts.append(str(d))
    return "[" + ", ".join(parts) + "]"
