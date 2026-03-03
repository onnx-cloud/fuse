"""Decoder registry for tensor visualizations.

Allows registering custom decoders and retrieving them by name.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import TensorView

# Global registry of decoders
_DECODERS: Dict[str, Callable[..., "TensorView"]] = {}


def register_decoder(name: str):
    """Decorator to register a decoder function.
    
    Example:
        @register_decoder('my_decoder')
        def my_decoder(tensor, name=None, **kwargs):
            return MyDecoderView(tensor, name, **kwargs)
    """
    def decorator(func: Callable[..., "TensorView"]) -> Callable[..., "TensorView"]:
        _DECODERS[name] = func
        # Also register common aliases
        aliases = {
            "image": ["img", "picture", "photo"],
            "audio": ["sound", "wav", "waveform"],
            "tokens": ["text", "token", "words"],
            "attention": ["attn", "att"],
            "embeddings": ["embed", "embedding", "emb"],
            "video": ["vid", "movie", "clip"],
            "points": ["pointcloud", "cloud", "pc"],
            "boxes": ["bbox", "bboxes", "box"],
        }
        if name in aliases:
            for alias in aliases[name]:
                _DECODERS[alias] = func
        return func
    return decorator


def get_decoder(name: str) -> Optional[Callable[..., "TensorView"]]:
    """Get a decoder function by name."""
    return _DECODERS.get(name)


def list_decoders() -> Dict[str, Callable[..., "TensorView"]]:
    """List all registered decoders."""
    return dict(_DECODERS)
