from __future__ import annotations
import html as _html
from typing import List, Optional, Tuple

import numpy as np
from .image import ImageView

from ..core import TensorView, TensorLike
from ..registry import register_decoder

@register_decoder("video")
def as_video(
    tensor: TensorLike,
    name: Optional[str] = None,
    fps: int = 30,
    max_frames: int = 64,
    grid_size: Optional[Tuple[int, int]] = None,
) -> "VideoView":
    """Display tensor as video frames.
    
    Args:
        tensor: Video tensor (T, H, W, C) or (T, C, H, W)
        name: Display name
        fps: Frames per second
        max_frames: Maximum frames to display
        grid_size: Thumbnail grid size (rows, cols)
    """
    return VideoView(
        tensor, name, fps=fps, max_frames=max_frames, grid_size=grid_size
    )


class VideoView(TensorView):
    """Video tensor visualization as frame grid."""
    
    def __init__(self, tensor: TensorLike, name: Optional[str] = None, **kwargs):
        super().__init__(tensor, name or "video")
        self.fps = kwargs.get("fps", 30)
        self.max_frames = kwargs.get("max_frames", 64)
        self.grid_size = kwargs.get("grid_size")
    
    def _get_frames(self) -> List[np.ndarray]:
        """Extract frames as list of HWC arrays."""
        arr = self._array
        
        # Detect format
        if arr.ndim == 4:
            if arr.shape[-1] in (1, 3, 4):  # THWC
                frames = [arr[i] for i in range(min(arr.shape[0], self.max_frames))]
            elif arr.shape[1] in (1, 3, 4):  # TCHW
                frames = [np.transpose(arr[i], (1, 2, 0)) for i in range(min(arr.shape[0], self.max_frames))]
            else:
                frames = [arr[i] for i in range(min(arr.shape[0], self.max_frames))]
        else:
            frames = [self._array]
        
        return frames
    
    def _repr_html_(self) -> str:
        frames = self._get_frames()
        n_frames = len(frames)
        
        # Create thumbnails
        image_view = ImageView(
            np.stack(frames), name=self.name,
            max_images=min(16, n_frames), grid_cols=4
        )
        
        duration = n_frames / self.fps
        h, w = frames[0].shape[:2]
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px; margin: 8px 0;
                    background: linear-gradient(135deg, #fef3fe 0%, #fff 100%);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; font-size: 14px;">🎬 {_html.escape(self.name)}</span>
                <span style="font-size: 11px; color: #666;">
                    {n_frames} frames | {w}×{h} | {duration:.2f}s @ {self.fps}fps
                </span>
            </div>
            {image_view._repr_html_().split('<div style="font-family')[1].rsplit('</div>', 1)[0]}
        </div>
        """

