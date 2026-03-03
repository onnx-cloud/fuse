from __future__ import annotations
import base64
import html as _html
import io
from typing import Optional

import numpy as np

from ..core import TensorView, TensorLike
from ..registry import register_decoder

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

