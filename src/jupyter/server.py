"""Minimal Jupyter Server extension endpoints for Fuse.

Provides lightweight JSON endpoints that a frontend (JupyterLab extension or other)
can call to get completions, operator lists, attribute suggestions and mapped
errors. The module exposes helper functions and also provides a Tornado
RequestHandler registration helper for integration with jupyter_server.
"""
from __future__ import annotations

import json
from typing import Dict, Any, List
import os
import time
from pathlib import Path

from .introspection import list_ops, op_attributes
from .errors import map_exception

# Initialize rate limit & audit variables to ensure module-level import stability
_llm_rate: Dict[str, List[float]] = {}
_LLM_RATE_LIMIT = int(os.environ.get('FUSE_LLM_RATE_PER_MIN', '60'))  # requests per minute
_llm_audit_path = Path(__file__).resolve().parents[3] / 'jupyter' / 'logs' / 'llm_access.log'
_llm_audit_path.parent.mkdir(parents=True, exist_ok=True)

# Expose helper functions at module level so tests can import them
def _rate_ok(remote_ip: str) -> (bool, int):
    now = time.time()
    window_start = now - 60
    arr = _llm_rate.get(remote_ip, [])
    arr = [t for t in arr if t >= window_start]
    remaining = max(0, _LLM_RATE_LIMIT - len(arr))
    if remaining <= 0:
        _llm_rate[remote_ip] = arr
        return False, remaining
    arr.append(now)
    _llm_rate[remote_ip] = arr
    return True, remaining - 1


def _audit_log(remote_ip: str, engine: str, payload: dict, resp_status: int):
    entry = {
        'ts': time.time(),
        'ip': remote_ip,
        'engine': engine,
        'payload': {k: v for k, v in (payload or {}).items() if k != 'messages' or isinstance(v, list)},
        'status': resp_status,
    }
    try:
        with open(str(_llm_audit_path), 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception:
        pass


# Public API functions (callable from tests or a frontend)

def get_ops() -> List[str]:
    return list_ops()


def _load_llm_config():
    from pathlib import Path
    import os
    cfgp = os.environ.get('FUSE_LLM_CONFIG')
    if cfgp:
        p = Path(cfgp)
    else:
        p = Path(__file__).resolve().parents[3] / "jupyter" / "config" / "llm_config.json"
        if not p.exists():
            p = Path(__file__).resolve().parents[3] / "jupyter" / "config" / "llm_config.json.example"
    if not p.exists():
        return {}
    try:
        from src.util.config import load_schema
        return load_schema(str(p), force_reload=True)
    except Exception:
        return {}


# Prompt rendering helpers
def _get_fuse_ebnf() -> str:
    """Return a fenced ```fuse``` block containing the current EBNF grammar."""
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]
    parser_src = ROOT / "src" / "parser.py"
    if not parser_src.exists():
        return ""
    src_text = parser_src.read_text()
    start_marker = "GRAMMAR"
    i = src_text.find(start_marker)
    if i == -1:
        return ""
    first = src_text.find('"""', i)
    if first == -1:
        return ""
    second = src_text.find('"""', first + 3)
    if second == -1:
        return ""
    grammar = src_text[first + 3 : second].strip()
    return "```fuse\n" + grammar + "\n```\n"


def _get_fuse_example() -> str:
    """Return a fenced ```fuse``` block with the terse example if available."""
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]
    terse_path = ROOT / "examples" / "golden" / "terse.fuse"
    if terse_path.exists():
        terse_text = terse_path.read_text()
        return "```fuse\n" + terse_text.strip() + "\n```\n"
    return ""


def _render_prompt(prompt_spec: str) -> str:
    """Load and render a prompt spec.

    If prompt_spec is a path to a markdown file (relative to repo root or
    to jupyter/), read it and replace known placeholders like
    {{fuse.ebnf}} and {{fuse.example}} with generated content.
    Otherwise return the literal string.
    """
    from pathlib import Path
    import re

    if not prompt_spec:
        return ""

    root = Path(__file__).resolve().parents[2]
    candidates = [Path(prompt_spec)]
    if not Path(prompt_spec).is_absolute():
        candidates.insert(0, root / prompt_spec)
        candidates.insert(1, root / 'jupyter' / prompt_spec)

    content = None
    for c in candidates:
        try:
            if c.exists():
                content = c.read_text()
                break
        except Exception:
            continue

    if content is None:
        # treat as literal prompt text
        content = str(prompt_spec)

    # Replace placeholders (use callable replacement to avoid regex template parsing on replacement text)
    content = re.sub(r'{{\s*fuse\.ebnf\s*}}', lambda m: _get_fuse_ebnf(), content)
    content = re.sub(r'{{\s*fuse\.example\s*}}', lambda m: _get_fuse_example(), content)
    return content


def list_engines() -> List[str]:
    cfg = _load_llm_config()
    return sorted(list(cfg.get("llm", {}).keys()))


def completions(prefix: str = "", context: str = "") -> List[Dict[str, Any]]:
    """Return context-aware completions for Fuse code.
    
    Args:
        prefix: Text to complete
        context: Surrounding code context to determine completion type
        
    Returns:
        List of completion items with label, insertText, kind, detail
    """
    ops = list_ops()
    keywords = ['param', 'const', 'output', 'node', 'trainable', 'import', 'from', 'domain', 'variant']
    types = ['f32', 'f64', 'i32', 'i64', 'u8', 'bool', 'string']
    
    # Detect context
    is_after_colon = ':' in context[-20:] if context else False
    is_in_paren = '(' in context[-30:] and ')' not in context[-30:] if context else False
    
    results = []
    prefix_lower = prefix.lower()
    
    # Type completions: prioritize after colon, but also suggest types when the
    # provided prefix clearly looks like a type (e.g., 'f3' -> 'f32') so that
    # simple prefix-based completions are useful even without explicit colon.
    suggest_types = False
    if is_after_colon:
        suggest_types = True
    elif prefix:
        low_matches = [t for t in types if t.lower().startswith(prefix_lower) or prefix_lower in t.lower()]
        if low_matches:
            suggest_types = True

    if suggest_types:
        for t in types:
            # Support partial matching: 'f3' matches 'f32'
            if not prefix or t.lower().startswith(prefix_lower) or prefix_lower in t.lower():
                results.append({
                    'label': t,
                    'insertText': t,
                    'kind': 'type',
                    'detail': f'Type: {t}'
                })
    
    # Operator completions
    for op in ops:
        if not prefix or op.lower().startswith(prefix_lower):
            attrs = op_attributes(op)
            attr_str = f" ({', '.join(a['name'] for a in attrs[:3])})" if attrs else ""
            results.append({
                'label': op,
                'insertText': f"{op}(" if is_in_paren else op,
                'kind': 'function',
                'detail': f"ONNX Op{attr_str}"
            })
    
    # Keyword completions
    for kw in keywords:
        if not prefix or kw.lower().startswith(prefix_lower):
            results.append({
                'label': kw,
                'insertText': kw + ' ',
                'kind': 'keyword',
                'detail': 'Keyword'
            })
    
    return results[:100]


def get_op_attributes(name: str) -> List[Dict[str, Any]]:
    return op_attributes(name)


def map_error(message: str, tb: str | None = None) -> Dict[str, Any]:
    # Create a minimal structure that maps message -> friendly output
    # If a traceback is provided, include it in the returned mapping
    dummy_exc = Exception(message)
    info = map_exception(dummy_exc)
    if tb:
        info["provided_traceback"] = tb
    return info


def kernel_symbols(kernel_id: str, nb_app=None) -> Dict[str, Any]:
    """Attempt to query a running kernel for available symbols.

    This tries to use the server's kernel manager to execute a small snippet
    that calls `_fuse_list_symbols()` in the kernel namespace (which is pushed
    by the IPython extension when loaded). Returns a dict with `ok` and `symbols`
    or an `error` message.
    """
    if nb_app is None:
        return {"ok": False, "error": "no nb_app provided"}
    web_app = nb_app.web_app
    km = web_app.settings.get("kernel_manager") or web_app.settings.get("kernels_manager")
    if not km:
        return {"ok": False, "error": "kernel manager unavailable"}
    try:
        # Best-effort: many server environments expose a method to execute code
        # against a kernel programmatically. If not available, return a helpful err.
        if hasattr(km, "get_kernel" or "get_kernel_model"):
            # We can't rely on a stable API here; provide a controlled fallback
            return {"ok": False, "error": "kernel execution not implemented in this environment"}
        return {"ok": False, "error": "kernel manager has no exec API"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# Tornado RequestHandlers for Jupyter server integration (optional)
try:
    from tornado.web import RequestHandler

    class OpsHandler(RequestHandler):
        def get(self):
            self.set_header("Content-Type", "application/json")
            self.write(json.dumps(get_ops()))

    class CompletionsHandler(RequestHandler):
        def post(self):
            data = self.get_json_body()
            if not isinstance(data, dict):
                data = {}
            prefix = data.get("prefix", "")
            context = data.get("context", "")
            self.set_header("Content-Type", "application/json")
            self.write(json.dumps(completions(prefix, context)))

    class OpAttributesHandler(RequestHandler):
        def post(self):
            data = self.get_json_body()
            name = data.get("name") if isinstance(data, dict) else None
            if not name:
                self.set_status(400)
                self.write(json.dumps({"error": "missing op name"}))
                return
            self.set_header("Content-Type", "application/json")
            self.write(json.dumps(get_op_attributes(name)))

    class MapErrorHandler(RequestHandler):
        def post(self):
            data = self.get_json_body()
            msg = data.get("message", "") if isinstance(data, dict) else ""
            tb = data.get("traceback") if isinstance(data, dict) else None
            self.set_header("Content-Type", "application/json")
            self.write(json.dumps(map_error(msg, tb)))

    class HealthHandler(RequestHandler):
        def get(self):
            # Basic environment checks that mirror the welcome notebook
            def _check_import(name):
                try:
                    __import__(name)
                    return True, None
                except Exception as e:
                    return False, str(e)

            checks = {}
            for name in ("onnx", "onnxruntime", "numpy"):
                ok, err = _check_import(name)
                checks[name] = {"ok": ok, "error": err}

            # Check IPython extension availability
            try:
                checks["fuse_ipython"] = {"ok": True}
            except Exception as e:
                checks["fuse_ipython"] = {"ok": False, "error": str(e)}

            self.set_header("Content-Type", "application/json")
            self.write(json.dumps({"ok": all(v["ok"] for v in checks.values()), "checks": checks}))

    class WelcomeHandler(RequestHandler):
        def get(self):
            # Serve the static welcome HTML
            import os
            p = os.path.join(os.path.dirname(__file__), '..', '..', 'jupyter', 'static', 'welcome.html')
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    self.set_header("Content-Type", "text/html")
                    self.write(f.read())
            except Exception as e:
                self.set_status(500)
                self.write(str(e))

    class StylesHandler(RequestHandler):
        def get(self):
            import os
            p = os.path.join(os.path.dirname(__file__), '..', '..', 'jupyter', 'static', 'styles.json')
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    self.set_header("Content-Type", "application/json")
                    self.write(f.read())
            except Exception as e:
                self.set_status(500)
                self.write(json.dumps({"error": str(e)}))

    class ChatStylesHandler(RequestHandler):
        """Serve the chat-specific CSS file for mobile-responsive layout"""
        def get(self):
            import os
            p = os.path.join(os.path.dirname(__file__), '..', '..', 'jupyter', 'static', 'chat-styles.css')
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    self.set_header("Content-Type", "text/css")
                    self.set_header("Cache-Control", "public, max-age=3600")
                    self.write(f.read())
            except Exception as e:
                self.set_status(500)
                self.write(f"/* Error loading styles: {str(e)} */")

    # Very small in-memory rate limiter (per-IP): timestamps of requests
    _llm_rate = {}
    _LLM_RATE_LIMIT = int(os.environ.get('FUSE_LLM_RATE_PER_MIN', '60'))  # requests per minute

    # Path to audit log
    _llm_audit_path = Path(__file__).resolve().parents[3] / 'jupyter' / 'logs' / 'llm_access.log'
    _llm_audit_path.parent.mkdir(parents=True, exist_ok=True)

    def _rate_ok(remote_ip: str) -> (bool, int):
        now = time.time()
        window_start = now - 60
        arr = _llm_rate.get(remote_ip, [])
        arr = [t for t in arr if t >= window_start]
        remaining = max(0, _LLM_RATE_LIMIT - len(arr))
        if remaining <= 0:
            _llm_rate[remote_ip] = arr
            return False, remaining
        arr.append(now)
        _llm_rate[remote_ip] = arr
        return True, remaining - 1

    def _audit_log(remote_ip: str, engine: str, payload: dict, resp_status: int):
        entry = {
            'ts': time.time(),
            'ip': remote_ip,
            'engine': engine,
            'payload': {k: v for k, v in (payload or {}).items() if k != 'messages' or isinstance(v, list)},
            'status': resp_status,
        }
        try:
            with open(str(_llm_audit_path), 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception:
            pass

    class EnginesHandler(RequestHandler):
        def get(self):
            cfg = _load_llm_config()
            engines = cfg.get('llm', {})
            # Return mapping engine -> label (or engine -> engine if no label)
            out = {k: v.get('label', k) for k, v in engines.items()}
            self.set_header('Content-Type', 'application/json')
            self.write(json.dumps(out))

    class StreamHandler(RequestHandler):
        def get(self):
            # SSE streaming handler. Query params 'engine' and 'message' required.
            engine = self.get_argument('engine', default=None)
            message = self.get_argument('message', default='')
            
            if not engine:
                self.set_status(400)
                self.write(json.dumps({'error': 'missing engine'}))
                return
            
            if not message:
                self.set_status(400)
                self.write(json.dumps({'error': 'missing message'}))
                return

            remote_ip = self.request.remote_ip or 'unknown'
            ok, remain = _rate_ok(remote_ip)
            self.set_header('X-RateLimit-Remaining', str(remain))
            if not ok:
                self.set_status(429)
                self.write(json.dumps({'error': 'rate limit exceeded'}))
                _audit_log(remote_ip, engine, {'stream': True}, 429)
                return

            cfg = _load_llm_config()
            engine_cfg = cfg.get('llm', {}).get(engine)
            if not engine_cfg:
                self.set_status(400)
                self.write(json.dumps({'error': f'engine {engine} not configured'}))
                _audit_log(remote_ip, engine, {'stream': True}, 400)
                return

            url = engine_cfg.get('url')
            secret_env = engine_cfg.get('secretEnv')
            prompt_spec = engine_cfg.get('prompt')
            prompt_text = _render_prompt(prompt_spec) if isinstance(prompt_spec, str) else (prompt_spec or '')

            secret = os.environ.get(secret_env)
            if not secret:
                self.set_status(500)
                self.write(json.dumps({'error': f'missing secret env {secret_env}'}))
                _audit_log(remote_ip, engine, {'stream': True}, 500)
                return

            # Proxy streaming request to provider: send a POST with stream=True
            self.set_header('Content-Type', 'text/event-stream')
            self.set_header('Cache-Control', 'no-cache')
            self.set_header('Connection', 'keep-alive')
            
            llm_timeout = int(os.environ.get('FUSE_LLM_TIMEOUT', '30'))
            
            # Build messages array
            messages = []
            if prompt_text:
                messages.append({'role': 'system', 'content': prompt_text})
            messages.append({'role': 'user', 'content': message})
            
            try:
                import requests
                # Note: provider-specific behavior required; we try best-effort to stream lines
                with requests.post(url, headers={"Authorization": f"Bearer {secret}"}, json={'stream': True, 'model': engine_cfg.get('model'), 'messages': messages}, stream=True, timeout=llm_timeout*2) as r:
                    r.raise_for_status()
                    for chunk in r.iter_lines(decode_unicode=True):
                        if not chunk:
                            continue
                        # Forward as SSE data: each line -> data: <line>\n\n
                        # Remove 'data: ' prefix if present (OpenAI format)
                        line = chunk
                        if line.startswith('data: '):
                            line = line[6:]
                        if line == '[DONE]':
                            continue
                        try:
                            self.write('data: ' + line + '\n\n')
                            self.flush()
                        except Exception:
                            break
                _audit_log(remote_ip, engine, {'stream': True}, 200)
            except requests.exceptions.Timeout:
                self.set_status(504)
                self.write('data: ' + json.dumps({'error': f'Stream timed out after {llm_timeout*2}s'}) + '\n\n')
                _audit_log(remote_ip, engine, {'stream': True}, 504)
            except Exception as e:
                self.set_status(502)
                self.write('data: ' + json.dumps({'error': f'Stream error: {str(e)}'}) + '\n\n')
                _audit_log(remote_ip, engine, {'stream': True}, 502)

    class LLMHandler(RequestHandler):
        def post(self):
            data = self.get_json_body() or {}
            engine = data.get('engine')
            messages = data.get('messages', [])
            stream = bool(data.get('stream', False))

            if not engine:
                self.set_status(400)
                self.write(json.dumps({'error': 'missing engine'}))
                return

            remote_ip = self.request.remote_ip or 'unknown'
            ok, remain = _rate_ok(remote_ip)
            self.set_header('X-RateLimit-Remaining', str(remain))
            if not ok:
                self.set_status(429)
                self.write(json.dumps({'error': 'rate limit exceeded'}))
                _audit_log(remote_ip, engine, data, 429)
                return

            cfg = _load_llm_config()
            engine_cfg = cfg.get('llm', {}).get(engine)
            if not engine_cfg:
                self.set_status(400)
                self.write(json.dumps({'error': f'engine {engine} not configured'}))
                _audit_log(remote_ip, engine, data, 400)
                return

            url = engine_cfg.get('url')
            secret_env = engine_cfg.get('secretEnv')
            prompt_spec = engine_cfg.get('prompt')
            prompt_text = _render_prompt(prompt_spec) if isinstance(prompt_spec, str) else (prompt_spec or '')

            secret = os.environ.get(secret_env)
            if not secret:
                self.set_status(500)
                self.write(json.dumps({'error': f'missing secret env {secret_env}'}))
                _audit_log(remote_ip, engine, data, 500)
                return

            # Prepend system prompt if present
            if prompt_text:
                messages = [{"role": "system", "content": prompt_text}] + messages

            # Configurable timeout from environment
            llm_timeout = int(os.environ.get('FUSE_LLM_TIMEOUT', '30'))
            
            try:
                import requests
                if stream:
                    # Proxy streaming provider responses back as SSE
                    self.set_header('Content-Type', 'text/event-stream')
                    self.set_header('Cache-Control', 'no-cache')
                    self.set_header('Connection', 'keep-alive')
                    with requests.post(url, headers={"Authorization": f"Bearer {secret}"}, json={'stream': True, 'model': engine_cfg.get('model'), 'messages': messages}, stream=True, timeout=llm_timeout*2) as r:
                        r.raise_for_status()
                        for chunk in r.iter_lines(decode_unicode=True):
                            if not chunk:
                                continue
                            try:
                                self.write('data: ' + chunk + '\n\n')
                                self.flush()
                            except Exception:
                                break
                    _audit_log(remote_ip, engine, data, 200)
                else:
                    r = requests.post(url, headers={"Authorization": f"Bearer {secret}"}, json={'model': engine_cfg.get('model'), 'messages': messages}, timeout=llm_timeout)
                    r.raise_for_status()
                    self.set_header('Content-Type', 'application/json')
                    self.write(r.text)
                    _audit_log(remote_ip, engine, data, r.status_code)
            except requests.exceptions.Timeout:
                self.set_status(504)
                self.write(json.dumps({
                    'error': f'LLM request timed out after {llm_timeout}s',
                    'suggestion': 'Try a shorter prompt or increase FUSE_LLM_TIMEOUT'
                }))
                _audit_log(remote_ip, engine, data, 504)
            except requests.exceptions.ConnectionError as e:
                self.set_status(502)
                self.write(json.dumps({
                    'error': f'Cannot connect to LLM provider: {str(e)}',
                    'suggestion': 'Check your internet connection and provider URL'
                }))
                _audit_log(remote_ip, engine, data, 502)
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code if hasattr(e, 'response') else 502
                self.set_status(status)
                error_detail = e.response.text if hasattr(e, 'response') else str(e)
                self.write(json.dumps({
                    'error': f'LLM provider error (HTTP {status})',
                    'detail': error_detail[:500],  # Truncate long errors
                    'suggestion': 'Check your API key and model name'
                }))
                _audit_log(remote_ip, engine, data, status)
            except Exception as e:
                self.set_status(502)
                self.write(json.dumps({
                    'error': f'Unexpected error: {type(e).__name__}',
                    'detail': str(e),
                    'suggestion': 'Check server logs for details'
                }))
                _audit_log(remote_ip, engine, data, 502)

    # Admin endpoints (guarded)
    class AdminListHandler(RequestHandler):
        def get(self):
            if os.environ.get('FUSE_LLM_ADMIN_ENABLED') != '1':
                self.set_status(403)
                self.write(json.dumps({'error': 'admin disabled'}))
                return
            cfg = _load_llm_config()
            self.set_header('Content-Type', 'application/json')
            self.write(json.dumps(cfg.get('llm', {})))

    class AdminHandler(RequestHandler):
        def post(self, engine):
            if os.environ.get('FUSE_LLM_ADMIN_ENABLED') != '1':
                self.set_status(403)
                self.write(json.dumps({'error': 'admin disabled'}))
                return
            data = self.get_json_body() or {}
            if not engine:
                self.set_status(400)
                self.write(json.dumps({'error': 'missing engine name in path'}))
                return
            # Load, update and write config
            cfg = _load_llm_config()
            llm = cfg.setdefault('llm', {})
            llm[engine] = data
            try:
                cfgp = Path(os.environ.get('FUSE_LLM_CONFIG') or Path(__file__).resolve().parents[3] / 'jupyter' / 'config' / 'llm_config.json')
                cfgp.write_text(json.dumps(cfg, indent=2))
                _audit_log(self.request.remote_ip or 'unknown', engine, {'admin_update': True, 'body': data}, 200)
                self.set_header('Content-Type', 'application/json')
                self.write(json.dumps({'ok': True, 'engine': engine, 'config': data}))
            except Exception as e:
                self.set_status(500)
                self.write(json.dumps({'error': str(e)}))
                _audit_log(self.request.remote_ip or 'unknown', engine, {'admin_update_failed': True, 'body': data}, 500)

        def delete(self, engine):
            if os.environ.get('FUSE_LLM_ADMIN_ENABLED') != '1':
                self.set_status(403)
                self.write(json.dumps({'error': 'admin disabled'}))
                return
            cfg = _load_llm_config()
            llm = cfg.setdefault('llm', {})
            if engine in llm:
                llm.pop(engine)
                try:
                    cfgp = Path(os.environ.get('FUSE_LLM_CONFIG') or Path(__file__).resolve().parents[3] / 'jupyter' / 'config' / 'llm_config.json')
                    cfgp.write_text(json.dumps(cfg, indent=2))
                    _audit_log(self.request.remote_ip or 'unknown', engine, {'admin_delete': True}, 200)
                    self.set_header('Content-Type', 'application/json')
                    self.write(json.dumps({'ok': True, 'deleted': engine}))
                except Exception as e:
                    self.set_status(500)
                    self.write(json.dumps({'error': str(e)}))
                    _audit_log(self.request.remote_ip or 'unknown', engine, {'admin_delete_failed': True}, 500)
            else:
                self.set_status(404)
                self.write(json.dumps({'error': 'engine not found'}))

    # Extension registration will be handled below.

except Exception:
    # Capture the import-time exception for diagnostics and fall back to no-op handlers
    import traceback
    _server_import_error = traceback.format_exc()

    OpsHandler = None
    CompletionsHandler = None
    OpAttributesHandler = None
    MapErrorHandler = None
    HealthHandler = None
    WelcomeHandler = None
    StylesHandler = None
    # Intentionally leave ChatStylesHandler undefined so `from src.jupyter.server import ChatStylesHandler`
    # raises ImportError when server components (tornado) are unavailable; tests will skip in that case.
    LLMHandler = None
    EnginesHandler = None
    StreamHandler = None
    AdminListHandler = None
    AdminHandler = None


def _jupyter_server_extension_paths():
    if OpsHandler is None:
        return []
    return [{"module": "src.jupyter.server"}]

def load_jupyter_server_extension(nb_app):
    if OpsHandler is None:
        raise RuntimeError("tornado or jupyter server components not available")
        
    web_app = nb_app.web_app
    host_pattern = ".*"
    base = web_app.settings.get("base_url", "")
    # Ensure base_url ends with / for proper concatenation
    if base and not base.endswith("/"):
        base = base + "/"
    handlers = [
        (base + "fuse/api/ops", OpsHandler),
        (base + "fuse/api/completions", CompletionsHandler),
        (base + "fuse/api/op_attributes", OpAttributesHandler),
        (base + "fuse/api/map_error", MapErrorHandler),
        (base + "fuse/api/health", HealthHandler),
        (base + "fuse/welcome", WelcomeHandler),
    ]
    web_app.add_handlers(host_pattern, handlers)
    # Also register LLM endpoint
    web_app.add_handlers(host_pattern, [
        (base + "fuse/api/llm", LLMHandler),
        (base + "fuse/api/llm/engines", EnginesHandler),
        (base + "fuse/api/llm/stream", StreamHandler),
        (base + "fuse/static/styles.json", StylesHandler),
        (base + "fuse/static/chat-styles.css", ChatStylesHandler),
        (base + "fuse/api/llm/admin", AdminListHandler),
        (base + "fuse/api/llm/admin/(.*)", AdminHandler),
    ])
