"""Core tensor inspection infrastructure.

Provides:
- TensorView: Base class for all visualization views
- TensorProxy: Wrapper enabling emoji shortcuts (x | 🖼️)
- inspect(): Universal tensor inspector with auto-detection
"""

from __future__ import annotations

import html as _html
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from IPython.display import HTML, display
except ImportError:
    HTML = None
    display = print


# Type alias for tensor-like objects
TensorLike = Union[np.ndarray, "Any", List, Any]


def _to_numpy(tensor: TensorLike) -> np.ndarray:
    """Convert any tensor-like to numpy array."""
    if isinstance(tensor, np.ndarray):
        return tensor
    # PyTorch
    if hasattr(tensor, "detach") and hasattr(tensor, "cpu"):
        return tensor.detach().cpu().numpy()
    # TensorFlow
    if hasattr(tensor, "numpy") and callable(tensor.numpy):
        return tensor.numpy()
    # JAX
    if hasattr(tensor, "__array__"):
        return np.asarray(tensor)
    # List/tuple
    if isinstance(tensor, (list, tuple)):
        return np.asarray(tensor)
    # Fallback
    return np.asarray(tensor)


from src.util.shape_format import format_shape as _format_shape


def _format_dtype(dtype) -> str:
    """Format dtype for display."""
    s = str(dtype)
    # Shorten common numpy dtypes
    replacements = {
        "float32": "f32", "float64": "f64", "float16": "f16",
        "int32": "i32", "int64": "i64", "int16": "i16", "int8": "i8",
        "uint8": "u8", "uint16": "u16", "uint32": "u32", "uint64": "u64",
        "bool": "bool", "complex64": "c64", "complex128": "c128",
    }
    for long, short in replacements.items():
        if long in s:
            return short
    return s


def _format_size(nbytes: int) -> str:
    """Format byte size for display."""
    if nbytes < 1024:
        return f"{nbytes} B"
    elif nbytes < 1024 * 1024:
        return f"{nbytes / 1024:.1f} KB"
    elif nbytes < 1024 * 1024 * 1024:
        return f"{nbytes / (1024 * 1024):.1f} MB"
    else:
        return f"{nbytes / (1024 * 1024 * 1024):.2f} GB"


@dataclass
class TensorStats:
    """Statistics for a tensor."""
    shape: Tuple[int, ...]
    dtype: str
    size_bytes: int
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    zeros_pct: float
    nan_count: int
    inf_count: int
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "TensorStats":
        """Compute stats from numpy array."""
        flat = arr.flatten().astype(np.float64)
        nan_count = int(np.isnan(flat).sum())
        inf_count = int(np.isinf(flat).sum())
        
        # Filter for stats computation
        valid = flat[np.isfinite(flat)]
        if len(valid) == 0:
            return cls(
                shape=arr.shape,
                dtype=_format_dtype(arr.dtype),
                size_bytes=arr.nbytes,
                min_val=float("nan"),
                max_val=float("nan"),
                mean_val=float("nan"),
                std_val=float("nan"),
                zeros_pct=0.0,
                nan_count=nan_count,
                inf_count=inf_count,
            )
        
        zeros_count = int((valid == 0).sum())
        zeros_pct = 100.0 * zeros_count / len(flat) if len(flat) > 0 else 0.0
        
        return cls(
            shape=arr.shape,
            dtype=_format_dtype(arr.dtype),
            size_bytes=arr.nbytes,
            min_val=float(valid.min()),
            max_val=float(valid.max()),
            mean_val=float(valid.mean()),
            std_val=float(valid.std()),
            zeros_pct=zeros_pct,
            nan_count=nan_count,
            inf_count=inf_count,
        )


class TensorView(ABC):
    """Base class for tensor visualizations.
    
    All views must implement _repr_html_ for Jupyter display.
    Optionally implement to_json, to_image, to_widget for different outputs.
    """
    
    def __init__(self, tensor: TensorLike, name: Optional[str] = None):
        self._original = tensor
        self._array = _to_numpy(tensor)
        self.name = name or "tensor"
        self._stats: Optional[TensorStats] = None
    
    @property
    def array(self) -> np.ndarray:
        return self._array
    
    @property
    def stats(self) -> TensorStats:
        if self._stats is None:
            self._stats = TensorStats.from_array(self._array)
        return self._stats
    
    @abstractmethod
    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter."""
        pass
    
    def _repr_mimebundle_(self, **kwargs) -> Dict[str, Any]:
        """Return MIME bundle for rich display."""
        return {
            "text/html": self._repr_html_(),
            "text/plain": repr(self),
        }
    
    def to_json(self) -> Dict[str, Any]:
        """Export view data as JSON-serializable dict."""
        s = self.stats
        return {
            "name": self.name,
            "shape": list(s.shape),
            "dtype": s.dtype,
            "size_bytes": s.size_bytes,
            "stats": {
                "min": s.min_val,
                "max": s.max_val,
                "mean": s.mean_val,
                "std": s.std_val,
                "zeros_pct": s.zeros_pct,
                "nan_count": s.nan_count,
                "inf_count": s.inf_count,
            }
        }
    
    def display(self) -> "TensorView":
        """Display in notebook and return self for chaining."""
        if HTML is not None:
            display(HTML(self._repr_html_()))
        else:
            print(self)
        return self
    
    def __repr__(self) -> str:
        s = self.stats
        return f"<{self.__class__.__name__} '{self.name}' shape={_format_shape(s.shape)} dtype={s.dtype}>"


class TensorInspector(TensorView):
    """Universal tensor inspector with stats, histogram, and actions."""
    
    def __init__(
        self,
        tensor: TensorLike,
        name: Optional[str] = None,
        show_histogram: bool = True,
        show_sample: bool = True,
        max_sample_size: int = 100,
    ):
        super().__init__(tensor, name)
        self.show_histogram = show_histogram
        self.show_sample = show_sample
        self.max_sample_size = max_sample_size
    
    def _make_histogram_ascii(self, bins: int = 20) -> str:
        """Generate ASCII histogram."""
        arr = self._array.flatten()
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            return "No valid data"
        
        hist, edges = np.histogram(valid, bins=bins)
        max_count = hist.max() if hist.max() > 0 else 1
        
        # Use block chars for bars
        blocks = " ▁▂▃▄▅▆▇█"
        chars = []
        for count in hist:
            level = int(8 * count / max_count)
            chars.append(blocks[level])
        return "".join(chars)
    
    def _make_sample_html(self) -> str:
        """Generate HTML for a small sample of the data."""
        arr = self._array
        
        # For small arrays, show all
        if arr.size <= self.max_sample_size:
            sample = arr
        else:
            # Show first elements
            flat = arr.flatten()
            sample = flat[:self.max_sample_size].reshape(-1)
        
        # Format values
        if arr.dtype.kind == "f":
            formatted = [f"{v:.4g}" for v in sample.flatten()[:20]]
        else:
            formatted = [str(v) for v in sample.flatten()[:20]]
        
        if len(sample.flatten()) > 20:
            formatted.append("...")
        
        return f'<code style="font-size: 11px; color: #666;">[{", ".join(formatted)}]</code>'
    
    def _guess_decoder(self) -> Optional[str]:
        """Guess the best decoder based on tensor shape."""
        shape = self._array.shape
        ndim = len(shape)
        
        # Image: 3D with small last dim (CHW) or second-to-last (HWC)
        if ndim == 3:
            if shape[-1] in (1, 3, 4):  # HWC
                return "image"
            if shape[0] in (1, 3, 4):  # CHW
                return "image"
        
        # Batch of images: 4D
        if ndim == 4:
            if shape[-1] in (1, 3, 4) or shape[1] in (1, 3, 4):
                return "image"
        
        # Audio: 1D or 2D with one small dim
        if ndim == 1 and shape[0] > 1000:
            return "audio"
        if ndim == 2 and (shape[0] <= 2 or shape[1] <= 2) and max(shape) > 1000:
            return "audio"
        
        # Attention: 2D or 3D square-ish
        if ndim == 2 and shape[0] == shape[1]:
            return "attention"
        if ndim == 3 and shape[1] == shape[2]:
            return "attention"
        
        # Embeddings: 2D with reasonable dims
        if ndim == 2 and 10 <= shape[0] <= 10000 and 10 <= shape[1] <= 2048:
            return "embeddings"
        
        return None
    
    def _repr_html_(self) -> str:
        s = self.stats
        hist = self._make_histogram_ascii() if self.show_histogram else ""
        sample = self._make_sample_html() if self.show_sample else ""
        guess = self._guess_decoder()
        
        # Decoder suggestion buttons
        decoder_btns = ""
        if guess:
            decoder_btns = f'<span style="margin-left: 8px; color: #667eea; font-size: 11px;">💡 Try: <code>%inspect {self.name} as {guess}</code></span>'
        
        # Build action buttons
        actions = """
        <div style="margin-top: 8px; display: flex; gap: 8px; flex-wrap: wrap;">
            <span style="background: #667eea; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; cursor: pointer;">🖼️ Image</span>
            <span style="background: #667eea; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; cursor: pointer;">🔊 Audio</span>
            <span style="background: #667eea; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; cursor: pointer;">📝 Tokens</span>
            <span style="background: #667eea; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; cursor: pointer;">👁️ Attention</span>
            <span style="background: #667eea; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; cursor: pointer;">🎯 Embed</span>
            <span style="background: #667eea; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px; cursor: pointer;">📊 Histogram</span>
        </div>
        """
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px; margin: 8px 0;
                    background: linear-gradient(135deg, #f8f9fa 0%, #fff 100%);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; font-size: 14px;">📦 {_html.escape(self.name)}</span>
                {decoder_btns}
            </div>
            <div style="display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 8px;">
                <span style="background: #e9ecef; padding: 2px 8px; border-radius: 4px; font-size: 12px;">
                    <strong>Shape:</strong> {_format_shape(s.shape)}
                </span>
                <span style="background: #e9ecef; padding: 2px 8px; border-radius: 4px; font-size: 12px;">
                    <strong>Dtype:</strong> {s.dtype}
                </span>
                <span style="background: #e9ecef; padding: 2px 8px; border-radius: 4px; font-size: 12px;">
                    <strong>Size:</strong> {_format_size(s.size_bytes)}
                </span>
            </div>
            <div style="background: #f8f9fa; padding: 8px; border-radius: 4px; margin-bottom: 8px;">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 8px; font-size: 12px;">
                    <div><strong>min:</strong> {s.min_val:.4g}</div>
                    <div><strong>max:</strong> {s.max_val:.4g}</div>
                    <div><strong>mean:</strong> {s.mean_val:.4g}</div>
                    <div><strong>std:</strong> {s.std_val:.4g}</div>
                    <div><strong>zeros:</strong> {s.zeros_pct:.1f}%</div>
                    <div><strong>NaN:</strong> {s.nan_count} <strong>Inf:</strong> {s.inf_count}</div>
                </div>
            </div>
            <div style="font-family: monospace; font-size: 16px; letter-spacing: 1px; margin-bottom: 8px;">
                {hist}
            </div>
            {sample}
            {actions}
        </div>
        """


def inspect(
    tensor: TensorLike,
    name: Optional[str] = None,
    **kwargs
) -> TensorInspector:
    """Inspect a tensor with universal stats and visualization.
    
    Args:
        tensor: Any tensor-like object (numpy, torch, tf, list)
        name: Optional name for display
        **kwargs: Additional options passed to TensorInspector
    
    Returns:
        TensorInspector view object
    
    Example:
        >>> inspect(my_array)
        >>> inspect(my_array, name="hidden_states")
    """
    return TensorInspector(tensor, name=name, **kwargs)


class TensorProxy:
    """Wrapper that enables emoji shortcuts for tensor inspection.
    
    Usage:
        t = TensorProxy(my_tensor)
        t | 🖼️   # Display as image
        t | 🔊   # Display as audio
        t | 📝   # Display as tokens
    """
    
    EMOJI_MAP = {
        "🖼️": "image",
        "🔊": "audio", 
        "📝": "tokens",
        "👁️": "attention",
        "🎯": "embed",
        "🎬": "video",
        "☁️": "points",
        "📦": "boxes",
        "🔍": "inspect",
        "📊": "histogram",
    }
    
    def __init__(self, tensor: TensorLike, name: Optional[str] = None):
        self._tensor = tensor
        self._array = _to_numpy(tensor)
        self.name = name
    
    @property
    def array(self) -> np.ndarray:
        return self._array
    
    def __or__(self, other: str) -> TensorView:
        """Enable emoji shortcuts: tensor | 🖼️"""
        # Normalize emoji (handle variation selectors)
        other_clean = other.replace("\ufe0f", "").strip()
        
        for emoji, decoder_name in self.EMOJI_MAP.items():
            emoji_clean = emoji.replace("\ufe0f", "")
            if other_clean == emoji_clean or other == decoder_name:
                return self.as_(decoder_name)
        
        # Unknown - return basic inspect
        return inspect(self._tensor, name=self.name)
    
    def as_(self, decoder: str, **kwargs) -> TensorView:
        """Apply a decoder by name."""
        from . import registry
        
        decoder_func = registry.get_decoder(decoder)
        if decoder_func is None:
            # Fallback to basic inspect
            return inspect(self._tensor, name=self.name)
        
        return decoder_func(self._tensor, name=self.name, **kwargs)
    
    def inspect(self, **kwargs) -> TensorInspector:
        """Basic tensor inspection."""
        return inspect(self._tensor, name=self.name, **kwargs)
    
    def __repr__(self) -> str:
        arr = self._array
        return f"<TensorProxy shape={_format_shape(arr.shape)} dtype={_format_dtype(arr.dtype)}>"
    
    def _repr_html_(self) -> str:
        """Default to inspect view."""
        return inspect(self._tensor, name=self.name)._repr_html_()
