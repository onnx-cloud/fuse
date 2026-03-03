from __future__ import annotations
import html as _html
from typing import List, Optional, Tuple

import numpy as np

from ..core import TensorView, TensorLike
from ..registry import register_decoder

@register_decoder("attention")
def as_attention(
    tensor: TensorLike,
    name: Optional[str] = None,
    tokens: Optional[List[str]] = None,
    head: Optional[int] = None,
    layer: Optional[int] = None,
    aggregate: str = "none",  # none, mean, max
) -> "AttentionView":
    """Display tensor as attention heatmap.
    
    Args:
        tensor: Attention tensor (seq, seq), (heads, seq, seq), or (layers, heads, seq, seq)
        name: Display name
        tokens: Token labels for axes
        head: Specific head to show (None = all or first)
        layer: Specific layer to show (for multi-layer)
        aggregate: How to aggregate heads ('none', 'mean', 'max')
    """
    return AttentionView(
        tensor, name, tokens=tokens, head=head, layer=layer, aggregate=aggregate
    )


class AttentionView(TensorView):
    """Attention matrix visualization as heatmap."""
    
    def __init__(self, tensor: TensorLike, name: Optional[str] = None, **kwargs):
        super().__init__(tensor, name or "attention")
        self.tokens = kwargs.get("tokens")
        self.head = kwargs.get("head")
        self.layer = kwargs.get("layer")
        self.aggregate = kwargs.get("aggregate", "none")
    
    def _get_attention_matrix(self) -> Tuple[np.ndarray, int, int]:
        """Extract attention matrix and return (matrix, n_heads, seq_len)."""
        arr = self._array
        
        # Handle different shapes
        if arr.ndim == 2:
            # (seq, seq)
            return arr, 1, arr.shape[0]
        elif arr.ndim == 3:
            # (heads, seq, seq)
            n_heads = arr.shape[0]
            seq_len = arr.shape[1]
            
            if self.head is not None:
                return arr[self.head], 1, seq_len
            elif self.aggregate == "mean":
                return arr.mean(axis=0), 1, seq_len
            elif self.aggregate == "max":
                return arr.max(axis=0), 1, seq_len
            else:
                return arr, n_heads, seq_len
        elif arr.ndim == 4:
            # (layers, heads, seq, seq) or (batch, heads, seq, seq)
            layer_idx = self.layer if self.layer is not None else 0
            arr = arr[layer_idx]
            return self._get_attention_matrix_from_3d(arr)
        else:
            return arr.reshape(arr.shape[-2:]), 1, arr.shape[-1]
    
    def _get_attention_matrix_from_3d(self, arr: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Helper to process 3D attention tensor."""
        n_heads = arr.shape[0]
        seq_len = arr.shape[1]
        
        if self.head is not None:
            return arr[self.head], 1, seq_len
        elif self.aggregate == "mean":
            return arr.mean(axis=0), 1, seq_len
        elif self.aggregate == "max":
            return arr.max(axis=0), 1, seq_len
        else:
            return arr, n_heads, seq_len
    
    def _make_heatmap_svg(
        self, matrix: np.ndarray, size: int = 200, title: str = ""
    ) -> str:
        """Generate SVG heatmap for a single attention matrix."""
        seq_len = matrix.shape[0]
        cell_size = max(4, min(20, size // seq_len))
        actual_size = cell_size * seq_len
        
        # Normalize matrix
        normalized = (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1e-8)
        
        cells = []
        for i in range(seq_len):
            for j in range(seq_len):
                val = normalized[i, j]
                # Blue color scale
                r = int(255 * (1 - val))
                g = int(255 * (1 - val * 0.5))
                b = 255
                color = f"rgb({r},{g},{b})"
                cells.append(
                    f'<rect x="{j * cell_size}" y="{i * cell_size}" '
                    f'width="{cell_size}" height="{cell_size}" fill="{color}" '
                    f'stroke="#fff" stroke-width="0.5"/>'
                )
        
        title_html = f'<text x="{actual_size/2}" y="-5" text-anchor="middle" font-size="10" fill="#666">{title}</text>' if title else ""
        
        return f"""
        <svg width="{actual_size}" height="{actual_size + (15 if title else 0)}" 
             style="margin: 4px;">
            <g transform="translate(0, {15 if title else 0})">
                {title_html}
                {"".join(cells)}
            </g>
        </svg>
        """
    
    def _repr_html_(self) -> str:
        matrix, n_heads, seq_len = self._get_attention_matrix()
        
        # Generate heatmaps
        if matrix.ndim == 2:
            # Single matrix
            heatmap_html = self._make_heatmap_svg(matrix, size=300)
        else:
            # Multiple heads - show grid
            heatmaps = []
            for h in range(min(n_heads, 12)):  # Max 12 heads
                heatmaps.append(self._make_heatmap_svg(matrix[h], size=150, title=f"Head {h}"))
            heatmap_html = f"""
            <div style="display: flex; flex-wrap: wrap; gap: 8px;">
                {"".join(heatmaps)}
            </div>
            """
        
        shape_info = f"{n_heads} heads × {seq_len} × {seq_len}"
        token_info = f" | {len(self.tokens)} tokens" if self.tokens else ""
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px; margin: 8px 0;
                    background: linear-gradient(135deg, #f3f3fe 0%, #fff 100%);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; font-size: 14px;">👁️ {_html.escape(self.name)}</span>
                <span style="font-size: 11px; color: #666;">{shape_info}{token_info}</span>
            </div>
            <div style="overflow-x: auto;">
                {heatmap_html}
            </div>
            <div style="margin-top: 8px; font-size: 11px; color: #999;">
                Range: [{self._array.min():.3g}, {self._array.max():.3g}]
            </div>
        </div>
        """

