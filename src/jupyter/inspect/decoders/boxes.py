from __future__ import annotations
import html as _html
from typing import List, Optional, Tuple

import numpy as np
from .image import ImageView

from ..core import TensorView, TensorLike
from ..registry import register_decoder

@register_decoder("boxes")
def as_boxes(
    tensor: TensorLike,
    name: Optional[str] = None,
    format: str = "xyxy",  # xyxy, xywh, cxcywh
    image: Optional[TensorLike] = None,
    labels: Optional[List[str]] = None,
    scores: Optional[TensorLike] = None,
    image_size: Optional[Tuple[int, int]] = None,
) -> "BoxView":
    """Display tensor as bounding boxes.
    
    Args:
        tensor: Box coordinates (N, 4)
        name: Display name
        format: Box format ('xyxy', 'xywh', 'cxcywh')
        image: Optional background image
        labels: Optional labels for each box
        scores: Optional confidence scores
        image_size: Image dimensions if no image provided (W, H)
    """
    return BoxView(
        tensor, name, format=format, image=image,
        labels=labels, scores=scores, image_size=image_size
    )


class BoxView(TensorView):
    """Bounding box visualization."""
    
    COLORS = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#34495e", "#16a085", "#c0392b",
    ]
    
    def __init__(self, tensor: TensorLike, name: Optional[str] = None, **kwargs):
        super().__init__(tensor, name or "boxes")
        self.format = kwargs.get("format", "xyxy")
        self.image = kwargs.get("image")
        self.labels = kwargs.get("labels")
        self.scores = kwargs.get("scores")
        self.image_size = kwargs.get("image_size", (400, 300))
    
    def _to_xyxy(self, box: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert box to xyxy format."""
        if self.format == "xyxy":
            return tuple(box[:4])
        elif self.format == "xywh":
            x, y, w, h = box[:4]
            return (x, y, x + w, y + h)
        elif self.format == "cxcywh":
            cx, cy, w, h = box[:4]
            return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)
        else:
            return tuple(box[:4])
    
    def _repr_html_(self) -> str:
        boxes = self._array
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        
        n_boxes = len(boxes)
        width, height = self.image_size
        
        # Background
        if self.image is not None:
            img_view = ImageView(self.image, name="bg")
            bg_html = f'<image href="{img_view._to_data_uri(img_view._prepare_image(img_view._get_images()[0]))}" width="{width}" height="{height}"/>'
            h, w = img_view._get_images()[0].shape[:2]
            scale_x, scale_y = width / w, height / h
        else:
            bg_html = f'<rect width="{width}" height="{height}" fill="#f8f9fa"/>'
            scale_x, scale_y = 1, 1
        
        # Draw boxes
        box_elements = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = self._to_xyxy(box)
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            
            color = self.COLORS[i % len(self.COLORS)]
            label = self.labels[i] if self.labels and i < len(self.labels) else f"Box {i}"
            score = f" ({self.scores[i]:.2f})" if self.scores is not None else ""
            
            box_elements.append(f"""
            <rect x="{x1:.1f}" y="{y1:.1f}" width="{x2-x1:.1f}" height="{y2-y1:.1f}"
                  fill="none" stroke="{color}" stroke-width="2"/>
            <text x="{x1:.1f}" y="{y1 - 4:.1f}" fill="{color}" font-size="10" font-weight="bold">
                {_html.escape(label)}{score}
            </text>
            """)
        
        svg = f"""
        <svg width="{width}" height="{height}" style="border-radius: 4px;">
            {bg_html}
            {"".join(box_elements)}
        </svg>
        """
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px; margin: 8px 0;
                    background: linear-gradient(135deg, #fef3f5 0%, #fff 100%);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; font-size: 14px;">📦 {_html.escape(self.name)}</span>
                <span style="font-size: 11px; color: #666;">
                    {n_boxes} boxes | Format: {self.format}
                </span>
            </div>
            <div style="text-align: center;">
                {svg}
            </div>
        </div>
        """

