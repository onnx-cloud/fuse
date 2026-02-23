"""Error mapping utilities to convert exceptions into structured error info."""
from __future__ import annotations

import traceback
from typing import Dict, Any


def map_exception(exc: Exception) -> Dict[str, Any]:
    tb = traceback.TracebackException.from_exception(exc)
    # Extract last stack frame info if available
    frames = list(tb.stack)
    last = frames[-1] if frames else None
    line = getattr(last, "lineno", None)
    filename = getattr(last, "filename", None)
    func = getattr(last, "name", None)
    return {
        "message": str(exc),
        "filename": filename,
        "line": line,
        "function": func,
        "stacktrace": "\n".join(tb.format()),
        "friendly": _friendly_message(exc),
    }


def _friendly_message(exc: Exception) -> str:
    # Basic mapping; extend for specific known exceptions
    cls = exc.__class__.__name__
    if cls.endswith("Error"):
        return f"{cls}: {str(exc)}"
    return str(exc)
