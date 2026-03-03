from __future__ import annotations
from typing import Any
import base64
import html as _html
import io
from typing import Dict, List, Optional

import numpy as np

from ..core import TensorView, TensorLike, _format_shape
from ..registry import register_decoder

@register_decoder("image")
def as_image(
    tensor: TensorLike,
    name: Optional[str] = None,
    format: Optional[str] = None,  # RGB, BGR, grayscale
    layout: Optional[str] = None,  # CHW, HWC, NCHW, NHWC
    normalize: bool = True,
    denormalize: Optional[Dict[str, List[float]]] = None,  # {"mean": [...], "std": [...]}
    imagenet: bool = False,
    max_images: int = 16,
    grid_cols: int = 4,
) -> "ImageView":
    """Display tensor as image(s).
    
    Args:
        tensor: Image tensor (HWC, CHW, NHWC, NCHW)
        name: Display name
        format: Color format (RGB, BGR, grayscale), auto-detected
        layout: Tensor layout, auto-detected from shape
        normalize: Whether to normalize values to 0-255
        denormalize: Custom mean/std for denormalization
        imagenet: Use ImageNet mean/std for denormalization
        max_images: Max images to show from batch
        grid_cols: Number of columns in image grid
    """
    return ImageView(
        tensor, name, format=format, layout=layout, normalize=normalize,
        denormalize=denormalize, imagenet=imagenet, max_images=max_images,
        grid_cols=grid_cols
    )


class ImageView(TensorView):
    """Image tensor visualization."""
    
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(
        self, tensor: TensorLike, name: Optional[str] = None, **kwargs
    ):
        super().__init__(tensor, name or "image")
        self.format = kwargs.get("format")
        self.layout = kwargs.get("layout")
        self.normalize = kwargs.get("normalize", True)
        self.denormalize = kwargs.get("denormalize")
        self.imagenet = kwargs.get("imagenet", False)
        self.max_images = kwargs.get("max_images", 16)
        self.grid_cols = kwargs.get("grid_cols", 4)
    
    def _detect_layout(self) -> str:
        """Detect tensor layout from shape."""
        shape = self._array.shape
        ndim = len(shape)
        
        if ndim == 2:
            return "HW"  # Grayscale
        elif ndim == 3:
            if shape[-1] in (1, 3, 4):
                return "HWC"
            elif shape[0] in (1, 3, 4):
                return "CHW"
            else:
                return "HWC"  # Default guess
        elif ndim == 4:
            if shape[-1] in (1, 3, 4):
                return "NHWC"
            elif shape[1] in (1, 3, 4):
                return "NCHW"
            else:
                return "NHWC"  # Default guess
        else:
            return "unknown"
    
    def _to_hwc(self, img: np.ndarray) -> np.ndarray:
        """Convert any layout to HWC."""
        if img.ndim == 2:
            return img[:, :, np.newaxis]
        
        layout = self.layout or self._detect_layout()
        
        if layout == "CHW":
            return np.transpose(img, (1, 2, 0))
        elif layout in ("HWC", "HW"):
            return img
        else:
            return img
    
    def _prepare_image(self, img: np.ndarray) -> np.ndarray:
        """Prepare single image for display (denormalize, convert to uint8)."""
        img = img.astype(np.float32)
        
        # Denormalize
        if self.imagenet:
            mean = np.array(self.IMAGENET_MEAN).reshape(1, 1, -1)
            std = np.array(self.IMAGENET_STD).reshape(1, 1, -1)
            img = img * std + mean
        elif self.denormalize:
            mean = np.array(self.denormalize.get("mean", [0])).reshape(1, 1, -1)
            std = np.array(self.denormalize.get("std", [1])).reshape(1, 1, -1)
            img = img * std + mean
        
        # Normalize to 0-255
        if self.normalize:
            if img.max() <= 1.0 and img.min() >= 0.0:
                img = img * 255.0
            elif img.max() > 1.0 or img.min() < 0.0:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
        
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Handle channels
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        elif img.shape[-1] == 4:
            img = img[:, :, :3]  # Drop alpha for now
        
        return img
    
    def _to_data_uri(self, img: np.ndarray) -> str:
        """Convert numpy image to base64 data URI."""
        try:
            from PIL import Image
            pil_img = Image.fromarray(img)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{b64}"
        except ImportError:
            # Fallback: simple PPM format (no PIL needed)
            h, w, c = img.shape
            header = f"P6\n{w} {h}\n255\n".encode()
            data = header + img.tobytes()
            b64 = base64.b64encode(data).decode("utf-8")
            return f"data:image/x-portable-pixmap;base64,{b64}"
    
    def _get_images(self) -> List[np.ndarray]:
        """Extract list of HWC images from tensor."""
        arr = self._array
        layout = self.layout or self._detect_layout()
        
        if layout in ("HW", "HWC", "CHW"):
            # Single image
            return [self._to_hwc(arr)]
        elif layout in ("NHWC", "NCHW"):
            # Batch
            images = []
            n = min(arr.shape[0], self.max_images)
            for i in range(n):
                img = arr[i]
                if layout == "NCHW":
                    img = np.transpose(img, (1, 2, 0))
                images.append(img)
            return images
        else:
            return [self._to_hwc(arr)]
    
    def to_pil(self) -> "Any":
        """Convert to PIL Image (first image if batch)."""
        from PIL import Image
        images = self._get_images()
        img = self._prepare_image(images[0])
        return Image.fromarray(img)
    
    def _repr_html_(self) -> str:
        images = self._get_images()
        layout = self.layout or self._detect_layout()
        
        # Prepare and convert to data URIs
        img_html_parts = []
        for i, img in enumerate(images):
            prepared = self._prepare_image(img)
            uri = self._to_data_uri(prepared)
            h, w = prepared.shape[:2]
            img_html_parts.append(
                f'<img src="{uri}" style="max-width: 200px; max-height: 200px; '
                f'border-radius: 4px; margin: 4px;" title="Image {i}: {w}x{h}">'
            )
        
        # Grid layout
        grid_html = f"""
        <div style="display: flex; flex-wrap: wrap; gap: 8px; justify-content: flex-start;">
            {"".join(img_html_parts)}
        </div>
        """
        
        shape_str = _format_shape(self._array.shape)
        info = f"{len(images)} image{'s' if len(images) > 1 else ''} | Layout: {layout}"
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px; margin: 8px 0;
                    background: linear-gradient(135deg, #fef3f3 0%, #fff 100%);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; font-size: 14px;">🖼️ {_html.escape(self.name)}</span>
                <span style="font-size: 11px; color: #666;">Shape: {shape_str}</span>
            </div>
            <div style="font-size: 11px; color: #666; margin-bottom: 8px;">{info}</div>
            {grid_html}
            <div style="margin-top: 8px; font-size: 11px; color: #999;">
                Range: [{self._array.min():.3g}, {self._array.max():.3g}]
            </div>
        </div>
        """

