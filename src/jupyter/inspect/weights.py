"""Weight and parameter inspection tools.

Provides:
- WeightsView: Overview of all model parameters
- FilterView: Visualization of convolution filters
- weight analysis functions
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .core import TensorView, TensorLike, _to_numpy, _format_shape, _format_dtype, _format_size

try:
    from IPython.display import HTML, display
except ImportError:
    HTML = None
    display = print

try:
    import onnx
    from onnx import ModelProto, TensorProto
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    ModelProto = Any
    TensorProto = Any


def _get_initializers(model: Any) -> Dict[str, np.ndarray]:
    """Extract initializers from model as numpy arrays."""
    if not HAS_ONNX:
        return {}
    
    # Handle various model wrappers
    if hasattr(model, '_model'):
        model = model._model
    if not isinstance(model, onnx.ModelProto):
        return {}
    
    initializers = {}
    for init in model.graph.initializer:
        try:
            arr = onnx.numpy_helper.to_array(init)
            initializers[init.name] = arr
        except Exception:
            pass
    
    return initializers


@dataclass
class ParamInfo:
    """Information about a single parameter."""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    size_bytes: int
    mean: float
    std: float
    min_val: float
    max_val: float
    zeros_pct: float
    is_trainable: bool = True
    
    @classmethod
    def from_array(cls, name: str, arr: np.ndarray, trainable: bool = True) -> "ParamInfo":
        flat = arr.flatten().astype(np.float64)
        zeros = (flat == 0).sum()
        
        return cls(
            name=name,
            shape=arr.shape,
            dtype=_format_dtype(arr.dtype),
            size_bytes=arr.nbytes,
            mean=float(flat.mean()),
            std=float(flat.std()),
            min_val=float(flat.min()),
            max_val=float(flat.max()),
            zeros_pct=100.0 * zeros / len(flat) if len(flat) > 0 else 0,
            is_trainable=trainable,
        )
    
    @property
    def numel(self) -> int:
        result = 1
        for d in self.shape:
            result *= d
        return result


class WeightsView:
    """Overview of all model parameters."""
    
    def __init__(
        self,
        model: Any,
        name: str = "",
        trainables: Optional[Dict[str, bool]] = None,
        sort_by: str = "size",  # size, name, sparsity
    ):
        self._initializers = _get_initializers(model)
        self.name = name or "model"
        self.sort_by = sort_by
        
        # Get trainable info from model metadata
        self._trainables = trainables or {}
        if hasattr(model, '_model'):
            model = model._model
        if HAS_ONNX and isinstance(model, onnx.ModelProto):
            for prop in model.metadata_props:
                if prop.key == "trainables":
                    import json
                    try:
                        self._trainables = json.loads(prop.value)
                    except Exception:
                        pass
        
        # Build param info list
        self._params: List[ParamInfo] = []
        for name, arr in self._initializers.items():
            trainable = self._trainables.get(name, True)
            self._params.append(ParamInfo.from_array(name, arr, trainable))
        
        # Sort
        if sort_by == "size":
            self._params.sort(key=lambda p: -p.size_bytes)
        elif sort_by == "sparsity":
            self._params.sort(key=lambda p: -p.zeros_pct)
        elif sort_by == "name":
            self._params.sort(key=lambda p: p.name)
    
    @property
    def total_params(self) -> int:
        return sum(p.numel for p in self._params)
    
    @property
    def total_size(self) -> int:
        return sum(p.size_bytes for p in self._params)
    
    @property
    def trainable_params(self) -> int:
        return sum(p.numel for p in self._params if p.is_trainable)
    
    def _make_histogram_ascii(self, arr: np.ndarray, bins: int = 20) -> str:
        """Generate ASCII histogram."""
        valid = arr.flatten()
        if len(valid) == 0:
            return ""
        
        hist, _ = np.histogram(valid, bins=bins)
        max_count = hist.max() if hist.max() > 0 else 1
        
        blocks = " ▁▂▃▄▅▆▇█"
        chars = []
        for count in hist:
            level = int(8 * count / max_count)
            chars.append(blocks[level])
        return "".join(chars)
    
    def _repr_html_(self) -> str:
        # Summary stats
        total_params = self.total_params
        total_size = self.total_size
        trainable = self.trainable_params
        frozen = total_params - trainable
        
        # Average sparsity
        avg_sparsity = sum(p.zeros_pct * p.numel for p in self._params) / total_params if total_params > 0 else 0
        
        # Parameter table (top 20)
        rows = []
        for p in self._params[:20]:
            # Mini histogram
            arr = self._initializers.get(p.name)
            hist = self._make_histogram_ascii(arr) if arr is not None else ""
            
            trainable_badge = "🔥" if p.is_trainable else "❄️"
            
            rows.append(f"""
            <tr>
                <td style="font-size: 11px; max-width: 200px; overflow: hidden; text-overflow: ellipsis;" 
                    title="{_html.escape(p.name)}">
                    {trainable_badge} {_html.escape(p.name[:40])}
                </td>
                <td style="font-size: 11px;">{_format_shape(p.shape)}</td>
                <td style="font-size: 11px; text-align: right;">{p.numel:,}</td>
                <td style="font-size: 11px; text-align: right;">{_format_size(p.size_bytes)}</td>
                <td style="font-size: 11px; text-align: right;">{p.zeros_pct:.1f}%</td>
                <td style="font-family: monospace; font-size: 12px; letter-spacing: 0;">{hist}</td>
            </tr>
            """)
        
        if len(self._params) > 20:
            rows.append(f"""
            <tr>
                <td colspan="6" style="text-align: center; font-size: 11px; color: #666;">
                    ... and {len(self._params) - 20} more parameters
                </td>
            </tr>
            """)
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; margin: 8px 0;">
            <div style="background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
                        color: white; padding: 12px; border-radius: 8px 8px 0 0;">
                <span style="font-weight: 600; font-size: 16px;">⚖️ Weights: {_html.escape(self.name)}</span>
            </div>
            
            <div style="padding: 12px;">
                <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; margin-bottom: 12px;">
                    <div style="text-align: center; padding: 8px; background: #f8f9fa; border-radius: 4px;">
                        <div style="font-size: 16px; font-weight: 600;">{len(self._params)}</div>
                        <div style="font-size: 10px; color: #666;">Tensors</div>
                    </div>
                    <div style="text-align: center; padding: 8px; background: #f8f9fa; border-radius: 4px;">
                        <div style="font-size: 16px; font-weight: 600;">{total_params:,}</div>
                        <div style="font-size: 10px; color: #666;">Total Params</div>
                    </div>
                    <div style="text-align: center; padding: 8px; background: #d4edda; border-radius: 4px;">
                        <div style="font-size: 16px; font-weight: 600;">{trainable:,}</div>
                        <div style="font-size: 10px; color: #155724;">🔥 Trainable</div>
                    </div>
                    <div style="text-align: center; padding: 8px; background: #cce5ff; border-radius: 4px;">
                        <div style="font-size: 16px; font-weight: 600;">{frozen:,}</div>
                        <div style="font-size: 10px; color: #004085;">❄️ Frozen</div>
                    </div>
                    <div style="text-align: center; padding: 8px; background: #f8f9fa; border-radius: 4px;">
                        <div style="font-size: 16px; font-weight: 600;">{_format_size(total_size)}</div>
                        <div style="font-size: 10px; color: #666;">Total Size</div>
                    </div>
                </div>
                
                <div style="margin-bottom: 8px; font-size: 11px; color: #666;">
                    Average Sparsity: <strong>{avg_sparsity:.1f}%</strong>
                </div>
                
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="background: #f8f9fa; font-size: 11px; font-weight: 600;">
                        <th style="text-align: left; padding: 4px;">Parameter</th>
                        <th style="text-align: left; padding: 4px;">Shape</th>
                        <th style="text-align: right; padding: 4px;">Elements</th>
                        <th style="text-align: right; padding: 4px;">Size</th>
                        <th style="text-align: right; padding: 4px;">Sparse</th>
                        <th style="text-align: left; padding: 4px;">Distribution</th>
                    </tr>
                    {"".join(rows)}
                </table>
            </div>
        </div>
        """
    
    def get(self, name: str) -> Optional[np.ndarray]:
        """Get a specific parameter by name."""
        return self._initializers.get(name)
    
    def filter(self, pattern: str) -> List[ParamInfo]:
        """Filter parameters by name pattern."""
        import fnmatch
        return [p for p in self._params if fnmatch.fnmatch(p.name, pattern)]


class FilterView(TensorView):
    """Visualization of convolution filters."""
    
    def __init__(
        self,
        tensor: TensorLike,
        name: Optional[str] = None,
        max_filters: int = 64,
        normalize: bool = True,
    ):
        super().__init__(tensor, name or "filters")
        self.max_filters = max_filters
        self.normalize = normalize
    
    def _get_filter_images(self) -> List[np.ndarray]:
        """Extract filter kernels as images."""
        arr = self._array
        
        # Common shapes: (out_ch, in_ch, H, W) or (out_ch, in_ch, D, H, W)
        if arr.ndim == 4:
            # 2D conv: (out, in, H, W)
            n_filters = min(arr.shape[0], self.max_filters)
            filters = []
            for i in range(n_filters):
                # Take first input channel or average
                if arr.shape[1] == 3:
                    # RGB input - show as color
                    f = arr[i].transpose(1, 2, 0)  # HWC
                else:
                    f = arr[i, 0]  # First channel, HW
                filters.append(f)
            return filters
        elif arr.ndim == 2:
            # Dense layer weights - reshape to square-ish
            h = int(np.sqrt(arr.shape[0]))
            w = arr.shape[0] // h
            return [arr[:h*w, :min(arr.shape[1], 64)].reshape(h, w, -1)[:, :, 0]]
        else:
            return [arr.reshape(-1)[:64*64].reshape(64, 64)]
    
    def _normalize_filter(self, f: np.ndarray) -> np.ndarray:
        """Normalize filter to 0-255 range."""
        if not self.normalize:
            return f
        
        f = f.astype(np.float32)
        f_min, f_max = f.min(), f.max()
        if f_max - f_min > 0:
            f = (f - f_min) / (f_max - f_min) * 255
        else:
            f = np.zeros_like(f) + 128
        return f.astype(np.uint8)
    
    def _make_grid_svg(self, cell_size: int = 40) -> str:
        """Generate SVG grid of filters."""
        filters = self._get_filter_images()
        n = len(filters)
        cols = min(8, n)
        rows = (n + cols - 1) // cols
        
        width = cols * (cell_size + 4) + 4
        height = rows * (cell_size + 4) + 4
        
        cells = []
        for i, f in enumerate(filters):
            row, col = i // cols, i % cols
            x = 4 + col * (cell_size + 4)
            y = 4 + row * (cell_size + 4)
            
            # Normalize and create color representation
            f_norm = self._normalize_filter(f)
            
            # Simple color based on mean value
            if f.ndim == 2:
                mean_val = int(f_norm.mean())
                # Grayscale gradient
                color = f"rgb({mean_val},{mean_val},{mean_val})"
            else:
                # RGB
                r = int(f_norm[:, :, 0].mean()) if f_norm.shape[2] > 0 else 128
                g = int(f_norm[:, :, 1].mean()) if f_norm.shape[2] > 1 else 128
                b = int(f_norm[:, :, 2].mean()) if f_norm.shape[2] > 2 else 128
                color = f"rgb({r},{g},{b})"
            
            cells.append(f"""
            <rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" 
                  fill="{color}" stroke="#ccc" rx="2">
                <title>Filter {i}: {f.shape}</title>
            </rect>
            """)
        
        return f"""
        <svg width="{width}" height="{height}" style="background: #f8f9fa; border-radius: 4px;">
            {"".join(cells)}
        </svg>
        """
    
    def _repr_html_(self) -> str:
        arr = self._array
        n_filters = arr.shape[0] if arr.ndim >= 2 else 1
        
        grid = self._make_grid_svg()
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px; margin: 8px 0;
                    background: linear-gradient(135deg, #f8f8fc 0%, #fff 100%);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; font-size: 14px;">🔲 {_html.escape(self.name)}</span>
                <span style="font-size: 11px; color: #666;">
                    {n_filters} filters | Shape: {_format_shape(arr.shape)}
                </span>
            </div>
            <div style="overflow-x: auto;">
                {grid}
            </div>
            <div style="margin-top: 8px; font-size: 11px; color: #999;">
                Range: [{arr.min():.4g}, {arr.max():.4g}] | 
                μ={arr.mean():.4g} σ={arr.std():.4g}
            </div>
        </div>
        """


def weights(model: Any, name: str = "", **kwargs) -> WeightsView:
    """Inspect all weights/parameters in a model.
    
    Args:
        model: ONNX model or Model wrapper
        name: Display name
        **kwargs: Options passed to WeightsView
    """
    return WeightsView(model, name, **kwargs)


def filters(tensor: TensorLike, name: str = "", **kwargs) -> FilterView:
    """Visualize convolution filter weights.
    
    Args:
        tensor: Filter tensor (out_ch, in_ch, H, W)
        name: Display name
        **kwargs: Options passed to FilterView
    """
    return FilterView(tensor, name, **kwargs)


def sparsity_analysis(model: Any) -> Dict[str, float]:
    """Analyze sparsity of all parameters.
    
    Returns dict mapping parameter name to sparsity percentage.
    """
    initializers = _get_initializers(model)
    result = {}
    for name, arr in initializers.items():
        zeros = (arr == 0).sum()
        result[name] = 100.0 * zeros / arr.size if arr.size > 0 else 0
    return dict(sorted(result.items(), key=lambda x: -x[1]))


def magnitude_analysis(model: Any) -> Dict[str, Dict[str, float]]:
    """Analyze weight magnitudes for each parameter.
    
    Returns dict with min, max, mean, std for each parameter.
    """
    initializers = _get_initializers(model)
    result = {}
    for name, arr in initializers.items():
        flat = arr.flatten().astype(np.float64)
        result[name] = {
            "min": float(flat.min()),
            "max": float(flat.max()),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "abs_mean": float(np.abs(flat).mean()),
        }
    return result
