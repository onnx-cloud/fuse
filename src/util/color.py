def get_visualization_color(index: int, style: str = "default") -> str:
    """Get consistent color for visualization elements."""
    palette = {
        "default": ["#667eea", "#e67e22", "#e74c3c", "#2ecc71", "#9b59b6", "#34495e"],
        "dark": ["#3498db", "#e67e22", "#e74c3c", "#2ecc71", "#9b59b6", "#1abc9c"],
        "word": ["#667eea"],
        "subword": ["#e67e22"],
        "special": ["#e74c3c"],
    }
    colors = palette.get(style, palette["default"])
    return colors[index % len(colors)]

def get_continuous_color(val: float) -> str:
    """Get continuous RGB color string for a value in [0, 1]."""
    val = max(0.0, min(1.0, float(val)))
    r = int(255 * val)
    b = int(255 * (1 - val))
    return f"rgb({r},100,{b})"
