"""IPython magics for Fuse — minimal MVP.

Provides `%fuse` (line) and `%%fuse` (cell) magics to parse, lower, and emit ONNX models
into the active IPython session namespace as `_fuse_model`.
"""

from __future__ import annotations







# IPython extension hooks


from .session import SessionManager
from .errors import map_exception
from .introspection import list_symbols, list_ops, op_attributes, op_doc
from IPython.display import JSON, display


def _install_exception_hook(ip):
    """Install exception handler that renders ErrorCard in cell output."""
    from IPython.display import HTML
    import json as _json
    import html as _html

    def _hook(etype, value, tb):
        info = map_exception(value)
        
        # Render as HTML widget for rich display
        error_html = f"""
        <div style="border: 1px solid #e74c3c; border-radius: 4px; padding: 12px; margin: 8px 0; background: #ffe6e6;">
            <div style="font-weight: bold; color: #c0392b; margin-bottom: 8px;">
                ⚠️ {_html.escape(info.get('error', 'Error'))}
            </div>
            <div style="color: #555; margin-bottom: 8px;">
                {_html.escape(info.get('message', str(value)))}
            </div>
            {f'<div style="background: #fff3cd; padding: 8px; border-radius: 4px; color: #856404; margin-top: 8px;">💡 <strong>Suggestion:</strong> {_html.escape(info["suggestion"])}</div>' if info.get('suggestion') else ''}
            {f'<details style="margin-top: 8px;"><summary style="cursor: pointer; color: #3498db;">Show Details</summary><pre style="background: #f8f9fa; padding: 8px; overflow-x: auto; font-size: 11px;">{_html.escape(info.get("detail", ""))}</pre></details>' if info.get('detail') else ''}
        </div>
        """
        display(HTML(error_html))
        
        # Also display structured JSON for programmatic access
        display(JSON(info, root='fuse-error'))
        
        # Print minimal traceback to stderr for debugging
        import traceback
        traceback.print_exception(etype, value, tb, limit=3)

    ip.set_custom_exc((Exception,), lambda etype, value, tb: _hook(etype, value, tb))
    # For testability and older IPython versions, ensure a mapping attribute exists
    # so tests can assert that our hook was installed.
    try:
        cur = getattr(ip, '_custom_exceptions', None)
        if cur is None:
            ip._custom_exceptions = {Exception: _hook}
        else:
            # Ensure Exception is present in mapping/keys
            try:
                if isinstance(cur, dict):
                    cur[Exception] = _hook
                else:
                    # if it's a set or list-like, convert to dict for clarity
                    ip._custom_exceptions = dict((k, None) for k in cur) if hasattr(cur, '__iter__') else {Exception: _hook}
                    ip._custom_exceptions[Exception] = _hook
            except Exception:
                ip._custom_exceptions = {Exception: _hook}
    except Exception:
        pass


def load_ipython_extension(ip):
    # Load the complete Fuse magics module (%%fuse, %fuse_compile, %fuse_run, etc.)
    from src.jupyter import magics as _magics_mod
    _magics_mod.load_ipython_extension(ip)

    # Add IPython-specific infrastructure: session manager, introspection, exception hooks
    # This separation keeps magics modular while ipython.py provides the integration layer.
    session = SessionManager()
    ip.push({"_fuse_session": session})

    # Register simple completion/introspection endpoints in the user namespace
    def _fuse_list_symbols():
        return list_symbols(ip.user_ns)

    def _fuse_list_ops():
        return list_ops()

    def _fuse_op_attributes(name: str):
        return op_attributes(name)

    def _fuse_op_doc(name: str):
        return op_doc(name)

    ip.push({
        "_fuse_list_symbols": _fuse_list_symbols,
        "_fuse_list_ops": _fuse_list_ops,
        "_fuse_op_attributes": _fuse_op_attributes,
        "_fuse_op_doc": _fuse_op_doc,
    })

    _install_exception_hook(ip)

    # Emit brief environment checks for interactive startup so automated
    # notebook smoke tests can assert the kernel is correctly prepared.
    try:
        try:
            import onnx
            print("ONNX library: available", flush=True)
        except Exception:
            print("ONNX library: missing", flush=True)
        try:
            import onnxruntime
            print("ONNX Runtime: available", flush=True)
        except Exception:
            print("ONNX Runtime: missing", flush=True)
        try:
            import numpy as _np
            print("NumPy: available", flush=True)
        except Exception:
            print("NumPy: missing", flush=True)
        print("Fuse IPython magics: registered", flush=True)
    except Exception:
        pass


def unload_ipython_extension(ip):
    # Nothing to cleanup; IPython will tear down magics and user_ns on exit
    pass
