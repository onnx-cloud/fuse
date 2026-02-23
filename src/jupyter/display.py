"""Helpers for rich MIME-aware notebook outputs.

Provides a simple `mime_bundle` helper that returns a mapping suitable for
`IPython.display.display` (a MIME bundle). Also a convenience `display_mime`
function to call display for you.
"""
from __future__ import annotations

from typing import Optional, Dict, Any
# MIMEBundle was removed/changed in some IPython versions; guard import and
# fall back to only importing `display` for compatibility.
try:
    from IPython.display import display, MIMEBundle  # type: ignore
except Exception:
    from IPython.display import display  # type: ignore
    MIMEBundle = None


def mime_bundle(
    text: Optional[str] = None,
    html: Optional[str] = None,
    png_bytes: Optional[bytes] = None,
    json_obj: Optional[dict] = None,
) -> Dict[str, Any]:
    m: Dict[str, Any] = {}
    if text is not None:
        m["text/plain"] = text
    if html is not None:
        m["text/html"] = html
    if png_bytes is not None:
        m["image/png"] = png_bytes
    if json_obj is not None:
        m["application/json"] = json_obj
    return m


def display_mime(bundle: Dict[str, Any]):
    """Display a MIME bundle in the active IPython kernel."""
    # IPython accepts a dict mapping mime -> value
    display(bundle)
