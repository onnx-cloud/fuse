"""Tensor slice exploration tools.

Provides:
- SliceView: Interactive tensor slicing
- slice_tensor(): Create a slice view
"""

from __future__ import annotations

import html as _html
from typing import Any, Optional, Tuple, Union

import numpy as np

from .core import TensorView, TensorLike, _to_numpy, _format_shape, _format_dtype

try:
    from IPython.display import HTML, display
except ImportError:
    HTML = None
    display = print


class SliceView(TensorView):
    """Interactive tensor slice visualization."""
    
    def __init__(
        self,
        tensor: TensorLike,
        name: Optional[str] = None,
        slices: Optional[Tuple[Any, ...]] = None,
        max_rows: int = 20,
        max_cols: int = 20,
        precision: int = 4,
        show_indices: bool = True,
    ):
        super().__init__(tensor, name or "slice")
        self.slices = slices
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.precision = precision
        self.show_indices = show_indices
        
        # Apply slicing if specified
        if slices is not None:
            self._sliced = self._array[slices]
        else:
            self._sliced = self._array
    
    def _format_value(self, val: Any) -> str:
        """Format a single value for display."""
        if isinstance(val, (float, np.floating)):
            if np.isnan(val):
                return '<span style="color: #e74c3c;">NaN</span>'
            if np.isinf(val):
                sign = "+" if val > 0 else "-"
                return f'<span style="color: #e74c3c;">{sign}∞</span>'
            return f"{val:.{self.precision}g}"
        elif isinstance(val, (int, np.integer)):
            return str(val)
        elif isinstance(val, (bool, np.bool_)):
            return "T" if val else "F"
        else:
            return str(val)
    
    def _value_color(self, val: Any) -> str:
        """Get background color based on value."""
        if isinstance(val, (bool, np.bool_)):
            return "#d4edda" if val else "#f8d7da"
        
        try:
            v = float(val)
            if np.isnan(v) or np.isinf(v):
                return "#f8d7da"
            # Normalize to color
            arr = self._sliced.flatten()
            valid = arr[np.isfinite(arr)]
            if len(valid) == 0:
                return "#fff"
            vmin, vmax = valid.min(), valid.max()
            if vmax == vmin:
                return "#fff"
            normalized = (v - vmin) / (vmax - vmin)
            # Blue to red gradient
            r = int(255 * normalized)
            b = int(255 * (1 - normalized))
            return f"rgba({r}, 100, {b}, 0.15)"
        except (ValueError, TypeError):
            return "#fff"
    
    def _make_1d_html(self) -> str:
        """Render 1D array as horizontal row."""
        arr = self._sliced.flatten()
        n = len(arr)
        show_n = min(n, self.max_cols * 2)
        
        cells = []
        for i in range(show_n):
            if i == self.max_cols and n > self.max_cols * 2:
                cells.append('<td style="padding: 2px 4px; color: #999;">...</td>')
                continue
            
            idx = i if i < self.max_cols else n - (show_n - i)
            val = arr[idx]
            color = self._value_color(val)
            cells.append(
                f'<td style="padding: 2px 4px; background: {color}; text-align: right; font-size: 11px;"'
                f' title="[{idx}]">{self._format_value(val)}</td>'
            )
        
        idx_row = ""
        if self.show_indices:
            idx_cells = []
            for i in range(show_n):
                if i == self.max_cols and n > self.max_cols * 2:
                    idx_cells.append('<td style="padding: 1px 4px; font-size: 9px; color: #999;">...</td>')
                    continue
                idx = i if i < self.max_cols else n - (show_n - i)
                idx_cells.append(f'<td style="padding: 1px 4px; font-size: 9px; color: #999;">{idx}</td>')
            idx_row = f'<tr>{"".join(idx_cells)}</tr>'
        
        return f"""
        <table style="border-collapse: collapse; margin: 4px 0;">
            {idx_row}
            <tr>{"".join(cells)}</tr>
        </table>
        """
    
    def _make_2d_html(self) -> str:
        """Render 2D array as table."""
        arr = self._sliced
        rows, cols = arr.shape
        
        show_rows = min(rows, self.max_rows)
        show_cols = min(cols, self.max_cols)
        
        truncate_rows = rows > self.max_rows
        truncate_cols = cols > self.max_cols
        
        html_rows = []
        
        # Header row with column indices
        if self.show_indices:
            header_cells = ['<th style="padding: 2px 4px; font-size: 9px; color: #999;"></th>']
            for j in range(show_cols):
                if j == show_cols // 2 and truncate_cols:
                    header_cells.append('<th style="padding: 2px 4px; font-size: 9px; color: #999;">...</th>')
                col_idx = j if j < show_cols // 2 or not truncate_cols else cols - (show_cols - j)
                header_cells.append(f'<th style="padding: 2px 4px; font-size: 9px; color: #999;">{col_idx}</th>')
            html_rows.append(f'<tr>{"".join(header_cells)}</tr>')
        
        # Data rows
        for i in range(show_rows):
            if i == show_rows // 2 and truncate_rows:
                # Ellipsis row
                cell_count = show_cols + (1 if self.show_indices else 0) + (1 if truncate_cols else 0)
                html_rows.append(f'<tr><td colspan="{cell_count}" style="text-align: center; color: #999;">⋮</td></tr>')
                continue
            
            row_idx = i if i < show_rows // 2 or not truncate_rows else rows - (show_rows - i)
            
            cells = []
            if self.show_indices:
                cells.append(f'<td style="padding: 2px 4px; font-size: 9px; color: #999;">{row_idx}</td>')
            
            for j in range(show_cols):
                if j == show_cols // 2 and truncate_cols:
                    cells.append('<td style="padding: 2px 4px; color: #999;">⋯</td>')
                    continue
                
                col_idx = j if j < show_cols // 2 or not truncate_cols else cols - (show_cols - j)
                val = arr[row_idx, col_idx]
                color = self._value_color(val)
                cells.append(
                    f'<td style="padding: 2px 4px; background: {color}; text-align: right; font-size: 11px;"'
                    f' title="[{row_idx}, {col_idx}]">{self._format_value(val)}</td>'
                )
            
            html_rows.append(f'<tr>{"".join(cells)}</tr>')
        
        return f"""
        <table style="border-collapse: collapse; border: 1px solid #eee; margin: 4px 0;">
            {"".join(html_rows)}
        </table>
        """
    
    def _make_nd_html(self) -> str:
        """Render N-D array by showing first 2D slice."""
        arr = self._sliced
        
        # Show info about dimensions
        info = f"Showing [:, :] slice of shape {_format_shape(arr.shape)}"
        
        # Take first 2D slice
        while arr.ndim > 2:
            arr = arr[0]
        
        # Create temporary 2D view
        view_2d = SliceView(
            arr,
            max_rows=self.max_rows,
            max_cols=self.max_cols,
            precision=self.precision,
            show_indices=self.show_indices,
        )
        
        return f"""
        <div style="font-size: 11px; color: #666; margin-bottom: 4px;">{info}</div>
        {view_2d._make_2d_html()}
        """
    
    def _repr_html_(self) -> str:
        arr = self._sliced
        
        # Choose rendering based on dimensionality
        if arr.ndim == 0:
            # Scalar
            content = f'<div style="font-size: 14px; font-weight: 600;">{self._format_value(arr.item())}</div>'
        elif arr.ndim == 1:
            content = self._make_1d_html()
        elif arr.ndim == 2:
            content = self._make_2d_html()
        else:
            content = self._make_nd_html()
        
        # Slice info
        slice_info = ""
        if self.slices is not None:
            slice_str = ", ".join(str(s) for s in self.slices) if isinstance(self.slices, tuple) else str(self.slices)
            slice_info = f' <code style="font-size: 11px; background: #f0f0f0; padding: 1px 4px; border-radius: 2px;">[{slice_str}]</code>'
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px; margin: 8px 0;
                    background: linear-gradient(135deg, #f8f9fa 0%, #fff 100%);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; font-size: 14px;">
                    ✂️ {_html.escape(self.name)}{slice_info}
                </span>
                <span style="font-size: 11px; color: #666;">
                    Shape: {_format_shape(arr.shape)} | Dtype: {_format_dtype(arr.dtype)}
                </span>
            </div>
            <div style="overflow-x: auto;">
                {content}
            </div>
            <div style="margin-top: 8px; font-size: 11px; color: #999;">
                Range: [{arr.min():.4g}, {arr.max():.4g}] | 
                μ={arr.mean():.4g} σ={arr.std():.4g}
            </div>
        </div>
        """


def slice_tensor(
    tensor: TensorLike,
    slices: Optional[Union[Tuple, slice, int]] = None,
    name: str = "",
    **kwargs
) -> SliceView:
    """Create an interactive slice view of a tensor.
    
    Args:
        tensor: Input tensor
        slices: Slice specification (e.g., (0, slice(10), slice(10)))
        name: Display name
        **kwargs: Options passed to SliceView
        
    Example:
        slice_tensor(x, (0, slice(None, 10), slice(None, 10)))
        slice_tensor(x, name="first_10x10")
    """
    if isinstance(slices, (int, slice)):
        slices = (slices,)
    return SliceView(tensor, name or "tensor", slices=slices, **kwargs)


def head(tensor: TensorLike, n: int = 10, name: str = "") -> SliceView:
    """Show first n elements along first dimension.
    
    Args:
        tensor: Input tensor
        n: Number of elements to show
        name: Display name
    """
    arr = _to_numpy(tensor)
    slices = (slice(None, n),) + (slice(None),) * (arr.ndim - 1)
    return SliceView(tensor, name or f"head({n})", slices=slices)


def tail(tensor: TensorLike, n: int = 10, name: str = "") -> SliceView:
    """Show last n elements along first dimension.
    
    Args:
        tensor: Input tensor
        n: Number of elements to show
        name: Display name
    """
    arr = _to_numpy(tensor)
    slices = (slice(-n, None),) + (slice(None),) * (arr.ndim - 1)
    return SliceView(tensor, name or f"tail({n})", slices=slices)


def sample(tensor: TensorLike, n: int = 100, name: str = "") -> SliceView:
    """Show random sample of n elements.
    
    Args:
        tensor: Input tensor
        n: Number of elements to sample
        name: Display name
    """
    arr = _to_numpy(tensor)
    flat = arr.flatten()
    
    if len(flat) <= n:
        indices = np.arange(len(flat))
    else:
        indices = np.random.choice(len(flat), n, replace=False)
        indices.sort()
    
    sampled = flat[indices]
    return SliceView(sampled, name or f"sample({n})")
