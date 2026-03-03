from __future__ import annotations
import html as _html
from typing import List, Optional

import numpy as np

from ..core import TensorView, TensorLike
from src.util.color import get_continuous_color, get_visualization_color
from ..registry import register_decoder

@register_decoder("embeddings")
def as_embeddings(
    tensor: TensorLike,
    name: Optional[str] = None,
    labels: Optional[List[str]] = None,
    method: str = "pca",  # pca, tsne, umap
    n_components: int = 2,
    perplexity: int = 30,
    color_by: Optional[np.ndarray] = None,
) -> "EmbeddingView":
    """Display embeddings as 2D/3D scatter plot.
    
    Args:
        tensor: Embedding tensor (n_samples, embedding_dim)
        name: Display name
        labels: Labels for each point
        method: Projection method ('pca', 'tsne', 'umap')
        n_components: Number of dimensions (2 or 3)
        perplexity: t-SNE perplexity parameter
        color_by: Array for coloring points
    """
    return EmbeddingView(
        tensor, name, labels=labels, method=method,
        n_components=n_components, perplexity=perplexity, color_by=color_by
    )


class EmbeddingView(TensorView):
    """Embedding visualization with dimensionality reduction."""
    
    def __init__(self, tensor: TensorLike, name: Optional[str] = None, **kwargs):
        super().__init__(tensor, name or "embeddings")
        self.labels = kwargs.get("labels")
        self.method = kwargs.get("method", "pca")
        self.n_components = kwargs.get("n_components", 2)
        self.perplexity = kwargs.get("perplexity", 30)
        self.color_by = kwargs.get("color_by")
        self._projected: Optional[np.ndarray] = None
    
    def _project(self) -> np.ndarray:
        """Project embeddings to lower dimensions."""
        if self._projected is not None:
            return self._projected
        
        arr = self._array
        if arr.ndim != 2:
            arr = arr.reshape(-1, arr.shape[-1])
        
        n_samples = arr.shape[0]
        
        if self.method == "pca":
            from .analysis import pca
            self._projected = pca(arr, n_components=self.n_components)
        elif self.method == "tsne":
            from .analysis import tsne
            perp = min(self.perplexity, n_samples - 1)
            self._projected = tsne(arr, n_components=self.n_components, perplexity=perp)
        elif self.method == "umap":
            from .analysis import umap_project
            self._projected = umap_project(arr, n_components=self.n_components)
        else:
            # Fallback to PCA
            from .analysis import pca
            self._projected = pca(arr, n_components=self.n_components)
        
        return self._projected
    
    def _make_scatter_svg(self, width: int = 400, height: int = 300) -> str:
        """Generate SVG scatter plot."""
        points = self._project()
        
        # Normalize to SVG coordinates
        x = points[:, 0]
        y = points[:, 1]
        
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        x_range = x_max - x_min or 1
        y_range = y_max - y_min or 1
        
        margin = 20
        plot_w = width - 2 * margin
        plot_h = height - 2 * margin
        
        x_norm = margin + (x - x_min) / x_range * plot_w
        y_norm = margin + (1 - (y - y_min) / y_range) * plot_h
        
        # Colors


        if self.color_by is not None:
            colors = self.color_by
            c_min, c_max = colors.min(), colors.max()
            c_range = c_max - c_min or 1
            
            def get_color(i):
                return get_continuous_color((colors[i] - c_min) / c_range)
        else:
            def get_color(i):
                return get_visualization_color(0)
        
        # Generate points
        circles = []
        for i in range(len(x_norm)):
            color = get_color(i)
            label = self.labels[i] if self.labels and i < len(self.labels) else f"Point {i}"
            circles.append(
                f'<circle cx="{x_norm[i]:.1f}" cy="{y_norm[i]:.1f}" r="4" '
                f'fill="{color}" fill-opacity="0.7" stroke="#fff" stroke-width="0.5">'
                f'<title>{_html.escape(str(label))}</title></circle>'
            )
        
        return f"""
        <svg width="{width}" height="{height}" style="background: #fafafa; border-radius: 4px;">
            {"".join(circles)}
        </svg>
        """
    
    def _repr_html_(self) -> str:
        arr = self._array
        if arr.ndim != 2:
            arr = arr.reshape(-1, arr.shape[-1])
        
        n_samples, embed_dim = arr.shape
        scatter_html = self._make_scatter_svg()
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px; margin: 8px 0;
                    background: linear-gradient(135deg, #f3fef3 0%, #fff 100%);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; font-size: 14px;">🎯 {_html.escape(self.name)}</span>
                <span style="font-size: 11px; color: #666;">
                    {n_samples} points × {embed_dim}d → {self.method.upper()} 2D
                </span>
            </div>
            <div style="text-align: center;">
                {scatter_html}
            </div>
        </div>
        """

