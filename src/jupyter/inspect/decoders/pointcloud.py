from __future__ import annotations
import html as _html
from typing import Optional

import numpy as np

from ..core import TensorView, TensorLike
from ..registry import register_decoder

@register_decoder("points")
def as_points(
    tensor: TensorLike,
    name: Optional[str] = None,
    colors: Optional[TensorLike] = None,
    max_points: int = 10000,
) -> "PointCloudView":
    """Display tensor as 3D point cloud.
    
    Args:
        tensor: Point cloud (N, 3) for XYZ coordinates
        name: Display name
        colors: Optional colors (N, 3) RGB
        max_points: Maximum points to display
    """
    return PointCloudView(tensor, name, colors=colors, max_points=max_points)


class PointCloudView(TensorView):
    """Point cloud visualization."""
    
    def __init__(self, tensor: TensorLike, name: Optional[str] = None, **kwargs):
        super().__init__(tensor, name or "pointcloud")
        self.colors = kwargs.get("colors")
        self.max_points = kwargs.get("max_points", 10000)
    
    def _repr_html_(self) -> str:
        arr = self._array
        if arr.ndim != 2 or arr.shape[1] < 3:
            return f"<div>Invalid point cloud shape: {arr.shape}</div>"
        
        n_points = arr.shape[0]
        
        # Create a simple 2D projection (top-down view)
        x = arr[:, 0]
        y = arr[:, 1]
        z = arr[:, 2]
        
        # Sample if too many points
        if n_points > self.max_points:
            idx = np.random.choice(n_points, self.max_points, replace=False)
            x, y, z = x[idx], y[idx], z[idx]
            n_display = self.max_points
        else:
            n_display = n_points
        
        # Normalize to SVG
        width, height = 400, 300
        margin = 20
        
        x_norm = margin + (x - x.min()) / (x.max() - x.min() + 1e-8) * (width - 2*margin)
        y_norm = margin + (y - y.min()) / (y.max() - y.min() + 1e-8) * (height - 2*margin)
        
        # Color by z-depth
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
        
        points = []
        for i in range(len(x_norm)):
            r = int(50 + 150 * z_norm[i])
            g = int(100 + 100 * (1 - z_norm[i]))
            b = int(200 - 100 * z_norm[i])
            points.append(
                f'<circle cx="{x_norm[i]:.1f}" cy="{y_norm[i]:.1f}" r="2" '
                f'fill="rgb({r},{g},{b})" fill-opacity="0.7"/>'
            )
        
        svg = f"""
        <svg width="{width}" height="{height}" style="background: #1a1a2e; border-radius: 4px;">
            {"".join(points)}
        </svg>
        """
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px; margin: 8px 0;
                    background: linear-gradient(135deg, #f3f3f8 0%, #fff 100%);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; font-size: 14px;">☁️ {_html.escape(self.name)}</span>
                <span style="font-size: 11px; color: #666;">
                    {n_points:,} points ({n_display:,} shown) | 3D
                </span>
            </div>
            <div style="text-align: center;">
                {svg}
            </div>
            <div style="margin-top: 8px; font-size: 11px; color: #999;">
                X: [{x.min():.2f}, {x.max():.2f}] | 
                Y: [{y.min():.2f}, {y.max():.2f}] | 
                Z: [{z.min():.2f}, {z.max():.2f}]
            </div>
        </div>
        """

