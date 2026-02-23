"""Domain-specific decoders for tensor visualization.

Each decoder converts a tensor into a rich visual representation
optimized for a specific data type (images, audio, text, etc.).
"""

from __future__ import annotations

import base64
import html as _html
import io
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .core import TensorView, TensorLike, _to_numpy, _format_shape, _format_dtype, _format_size
from .registry import register_decoder

try:
    from IPython.display import HTML, Audio, display
except ImportError:
    HTML = None
    Audio = None
    display = print


# ============================================================================
# Image Decoder
# ============================================================================

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
    
    def to_pil(self) -> "PIL.Image.Image":
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


# ============================================================================
# Audio Decoder
# ============================================================================

@register_decoder("audio")
def as_audio(
    tensor: TensorLike,
    name: Optional[str] = None,
    sample_rate: int = 16000,
    show_waveform: bool = True,
    show_spectrogram: bool = False,
) -> "AudioView":
    """Display tensor as audio with waveform and optional spectrogram.
    
    Args:
        tensor: Audio tensor (samples,) or (channels, samples)
        name: Display name
        sample_rate: Sample rate in Hz
        show_waveform: Show waveform visualization
        show_spectrogram: Show spectrogram visualization
    """
    return AudioView(
        tensor, name, sample_rate=sample_rate,
        show_waveform=show_waveform, show_spectrogram=show_spectrogram
    )


class AudioView(TensorView):
    """Audio tensor visualization with player and waveform."""
    
    def __init__(self, tensor: TensorLike, name: Optional[str] = None, **kwargs):
        super().__init__(tensor, name or "audio")
        self.sample_rate = kwargs.get("sample_rate", 16000)
        self.show_waveform = kwargs.get("show_waveform", True)
        self.show_spectrogram = kwargs.get("show_spectrogram", False)
    
    def _get_audio_data(self) -> np.ndarray:
        """Get audio as 1D or 2D array (channels, samples)."""
        arr = self._array
        if arr.ndim == 1:
            return arr
        elif arr.ndim == 2:
            # Could be (channels, samples) or (samples, channels)
            if arr.shape[0] <= 2:
                return arr  # (channels, samples)
            elif arr.shape[1] <= 2:
                return arr.T  # Transpose to (channels, samples)
            else:
                return arr[0]  # Take first channel
        else:
            return arr.flatten()
    
    def _make_waveform_svg(self, width: int = 600, height: int = 80) -> str:
        """Generate SVG waveform visualization."""
        audio = self._get_audio_data()
        if audio.ndim > 1:
            audio = audio[0]  # Use first channel
        
        # Downsample for display
        target_points = width
        if len(audio) > target_points:
            step = len(audio) // target_points
            audio = audio[::step]
        
        # Normalize
        max_val = np.abs(audio).max() or 1
        normalized = audio / max_val
        
        # Generate path
        mid_y = height / 2
        points = []
        for i, val in enumerate(normalized):
            x = i * width / len(normalized)
            y = mid_y - val * (height / 2 - 4)
            points.append(f"{x:.1f},{y:.1f}")
        
        path = "M" + " L".join(points)
        
        return f"""
        <svg width="{width}" height="{height}" style="background: #f8f9fa; border-radius: 4px;">
            <line x1="0" y1="{mid_y}" x2="{width}" y2="{mid_y}" stroke="#ccc" stroke-width="1"/>
            <path d="{path}" fill="none" stroke="#667eea" stroke-width="1.5"/>
        </svg>
        """
    
    def _to_audio_element(self) -> str:
        """Generate HTML5 audio element with data URI."""
        audio = self._get_audio_data()
        if audio.ndim > 1:
            audio = audio[0]  # Mono for playback
        
        # Normalize to int16
        max_val = np.abs(audio).max() or 1
        audio_int16 = (audio / max_val * 32767).astype(np.int16)
        
        # Create WAV in memory
        try:
            import wave
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(self.sample_rate)
                wav.writeframes(audio_int16.tobytes())
            buffer.seek(0)
            b64 = base64.b64encode(buffer.read()).decode("utf-8")
            return f'<audio controls src="data:audio/wav;base64,{b64}" style="width: 100%;"></audio>'
        except Exception:
            return '<em style="color: #999;">Audio playback not available</em>'
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        audio = self._get_audio_data()
        samples = audio.shape[-1] if audio.ndim > 1 else len(audio)
        return samples / self.sample_rate
    
    def _repr_html_(self) -> str:
        audio = self._get_audio_data()
        channels = 1 if audio.ndim == 1 else audio.shape[0]
        samples = audio.shape[-1] if audio.ndim > 1 else len(audio)
        duration = self.duration
        
        waveform_html = self._make_waveform_svg() if self.show_waveform else ""
        audio_element = self._to_audio_element()
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px; margin: 8px 0;
                    background: linear-gradient(135deg, #f3f8fe 0%, #fff 100%);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; font-size: 14px;">🔊 {_html.escape(self.name)}</span>
                <span style="font-size: 11px; color: #666;">
                    {duration:.2f}s | {self.sample_rate}Hz | {channels}ch
                </span>
            </div>
            <div style="margin-bottom: 8px;">
                {waveform_html}
            </div>
            <div>
                {audio_element}
            </div>
            <div style="margin-top: 8px; font-size: 11px; color: #999;">
                {samples:,} samples | Range: [{audio.min():.3g}, {audio.max():.3g}]
            </div>
        </div>
        """


# ============================================================================
# Token Decoder
# ============================================================================

@register_decoder("tokens")
def as_tokens(
    tensor: TensorLike,
    name: Optional[str] = None,
    vocab: Optional[Union[str, Dict[int, str], List[str]]] = None,
    tokenizer: Optional[str] = None,
    max_tokens: int = 100,
    show_ids: bool = True,
) -> "TokenView":
    """Display tensor as decoded tokens.
    
    Args:
        tensor: Token ID tensor
        name: Display name
        vocab: Vocabulary (dict mapping id->str, list, or path to JSON)
        tokenizer: Tokenizer name (e.g., 'gpt2', 'bert-base-uncased')
        max_tokens: Maximum tokens to display
        show_ids: Show token IDs below tokens
    """
    return TokenView(
        tensor, name, vocab=vocab, tokenizer=tokenizer,
        max_tokens=max_tokens, show_ids=show_ids
    )


class TokenView(TensorView):
    """Token tensor visualization with decoded text."""
    
    # Simple built-in vocabularies for common special tokens
    COMMON_TOKENS = {
        0: "[PAD]", 1: "[UNK]", 2: "[CLS]", 3: "[SEP]", 4: "[MASK]",
        101: "[CLS]", 102: "[SEP]", 103: "[MASK]", 0: "[PAD]",
    }
    
    def __init__(self, tensor: TensorLike, name: Optional[str] = None, **kwargs):
        super().__init__(tensor, name or "tokens")
        self.vocab = kwargs.get("vocab")
        self.tokenizer_name = kwargs.get("tokenizer")
        self.max_tokens = kwargs.get("max_tokens", 100)
        self.show_ids = kwargs.get("show_ids", True)
        self._tokenizer = None
        self._vocab_dict: Optional[Dict[int, str]] = None
    
    def _load_vocab(self) -> Dict[int, str]:
        """Load vocabulary for decoding."""
        if self._vocab_dict is not None:
            return self._vocab_dict
        
        # Try to load tokenizer
        if self.tokenizer_name:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
                self._vocab_dict = {v: k for k, v in self._tokenizer.get_vocab().items()}
                return self._vocab_dict
            except Exception:
                pass
        
        # Use provided vocab
        if isinstance(self.vocab, dict):
            self._vocab_dict = self.vocab
        elif isinstance(self.vocab, list):
            self._vocab_dict = {i: t for i, t in enumerate(self.vocab)}
        elif isinstance(self.vocab, str):
            try:
                with open(self.vocab) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._vocab_dict = {i: t for i, t in enumerate(data)}
                else:
                    self._vocab_dict = {int(k): v for k, v in data.items()}
            except Exception:
                self._vocab_dict = {}
        else:
            self._vocab_dict = {}
        
        return self._vocab_dict
    
    def _decode_token(self, token_id: int) -> str:
        """Decode a single token ID to string."""
        vocab = self._load_vocab()
        
        if self._tokenizer:
            try:
                return self._tokenizer.decode([token_id])
            except Exception:
                pass
        
        if token_id in vocab:
            return vocab[token_id]
        if token_id in self.COMMON_TOKENS:
            return self.COMMON_TOKENS[token_id]
        
        return f"[{token_id}]"
    
    def _get_token_type(self, token_id: int, text: str) -> str:
        """Classify token type for coloring."""
        if text.startswith("[") and text.endswith("]"):
            return "special"
        if text.startswith("##") or text.startswith("▁"):
            return "subword"
        return "word"
    
    def decode(self) -> str:
        """Decode all tokens to string."""
        arr = self._array.flatten().astype(int)
        tokens = [self._decode_token(int(t)) for t in arr]
        return " ".join(tokens).replace(" ##", "").replace("▁", " ")
    
    def _repr_html_(self) -> str:
        arr = self._array.flatten().astype(int)
        n_tokens = min(len(arr), self.max_tokens)
        
        token_html_parts = []
        type_colors = {
            "word": "#667eea",
            "subword": "#e67e22",
            "special": "#e74c3c",
        }
        
        for i in range(n_tokens):
            token_id = int(arr[i])
            text = self._decode_token(token_id)
            token_type = self._get_token_type(token_id, text)
            color = type_colors.get(token_type, "#667eea")
            
            id_part = f'<div style="font-size: 9px; color: #999;">{token_id}</div>' if self.show_ids else ""
            
            token_html_parts.append(f"""
            <div style="display: inline-flex; flex-direction: column; align-items: center;
                        margin: 2px; padding: 2px 6px; background: {color}22; border-radius: 4px;
                        border: 1px solid {color}44;">
                <div style="font-size: 12px; font-weight: 500; color: {color};">{_html.escape(text)}</div>
                {id_part}
            </div>
            """)
        
        if len(arr) > self.max_tokens:
            token_html_parts.append(f'<div style="padding: 4px; color: #999;">... +{len(arr) - self.max_tokens} more</div>')
        
        vocab_info = self.tokenizer_name or ("custom vocab" if self.vocab else "no vocab")
        
        return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    border: 1px solid #e1e4e8; border-radius: 8px; padding: 12px; margin: 8px 0;
                    background: linear-gradient(135deg, #fef9f3 0%, #fff 100%);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 600; font-size: 14px;">📝 {_html.escape(self.name)}</span>
                <span style="font-size: 11px; color: #666;">
                    {len(arr)} tokens | {vocab_info}
                </span>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 2px; margin-bottom: 8px;">
                {"".join(token_html_parts)}
            </div>
            <div style="font-size: 11px; color: #999;">
                Legend: <span style="color: #667eea;">■ word</span> 
                <span style="color: #e67e22;">■ subword</span>
                <span style="color: #e74c3c;">■ special</span>
            </div>
        </div>
        """


# ============================================================================
# Attention Decoder
# ============================================================================

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


# ============================================================================
# Embeddings Decoder
# ============================================================================

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
                val = (colors[i] - c_min) / c_range
                r = int(255 * val)
                b = int(255 * (1 - val))
                return f"rgb({r},100,{b})"
        else:
            def get_color(i):
                return "#667eea"
        
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


# ============================================================================
# Video Decoder
# ============================================================================

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


# ============================================================================
# Point Cloud Decoder
# ============================================================================

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


# ============================================================================
# Bounding Box Decoder
# ============================================================================

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
