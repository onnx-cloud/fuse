from typing import Any, Dict, List, Tuple

from .opcodes import default_opcodes
from .onnx_opset import latest_onnx_opset

# Small helpers for config discovery and parsing
from pathlib import Path as _Path
import os as _os


def repo_root() -> _Path:
    return _Path(__file__).resolve().parents[1]


def _simple_toml_parse_bool(text: str, key: str) -> dict:
    import re

    res = {}
    m = re.search(r"enable_training_state_checks\s*=\s*(true|false)", text, re.IGNORECASE)
    if m:
        res[key] = m.group(1).lower() == "true"
    return res


def load_sanitizer_config() -> dict:
    cfg = {"enable_training_state_checks": True}
    env_cfg = _os.getenv("FUSE_SANITIZER_CONFIG")
    if env_cfg:
        p = _Path(env_cfg)
        if p.exists():
            try:
                try:
                    import tomllib as _toml

                    cfg_doc = _toml.loads(p.read_text())
                    t = cfg_doc.get("tool", {}).get("fuse", {}).get("sanitizer", {})
                    if t:
                        cfg.update(t)
                        return cfg
                except Exception:
                    try:
                        import toml as _toml

                        cfg_doc = _toml.loads(p.read_text())
                        t = cfg_doc.get("tool", {}).get("fuse", {}).get("sanitizer", {})
                        if t:
                            cfg.update(t)
                            return cfg
                    except Exception:
                        try:
                            txt = p.read_text()
                            cfg.update(_simple_toml_parse_bool(txt, "enable_training_state_checks"))
                            return cfg
                        except Exception:
                            return cfg
            except Exception:
                return cfg
        return cfg

    prj = repo_root().joinpath("pyproject.toml")
    if prj.exists():
        try:
            try:
                import tomllib as _toml

                cfg_doc = _toml.loads(prj.read_text())
                t = cfg_doc.get("tool", {}).get("fuse", {}).get("sanitizer", {})
                if t:
                    cfg.update(t)
                    return cfg
            except Exception:
                try:
                    import toml as _toml

                    cfg_doc = _toml.loads(prj.read_text())
                    t = cfg_doc.get("tool", {}).get("fuse", {}).get("sanitizer", {})
                    if t:
                        cfg.update(t)
                        return cfg
                except Exception:
                    try:
                        cfg.update(_simple_toml_parse_bool(prj.read_text(), "enable_training_state_checks"))
                        return cfg
                    except Exception:
                        return cfg
        except Exception:
            return cfg
    return cfg


# load once on import
_s_cfg = load_sanitizer_config()


class SanityIssue:
    def __init__(self, kind: str, message: str, node: Any = None, code: str = None, param: str = None, state: str = None, extra: Dict[str, Any] = None):
        self.kind = kind  # 'error' | 'warning'
        self.message = message
        self.node = node
        self.code = code
        self.param = param
        self.state = state
        self.extra = extra or {}

    def as_dict(self):
        out = {"kind": self.kind, "message": self.message}
        if self.node is not None:
            out["node"] = self.node
        if self.code is not None:
            out["code"] = self.code
        if self.param is not None:
            out["param"] = self.param
        if self.state is not None:
            out["state"] = self.state
        out.update(self.extra)
        return out


def _find_meta(ast: List[Any], name: str):
    for node in ast:
        if isinstance(node, dict) and node.get("type") == "meta" and node.get("name") == name:
            return node
    return None


def sanitize_ast(ast: List[Any], opset: int = None, strict: bool = False) -> Dict[str, List[Dict]]:
    """Validate the parsed AST and return structured issues.

    Returns dict: { errors: [...], warnings: [...] }

    Args:
        ast: Parsed AST list
        opset: optional opset version for certain checks
        strict: when True, treat certain metadata issues (e.g., invalid @id/@type)
                as errors instead of warnings.
    """
    warnings: List[SanityIssue] = []
    errors: List[SanityIssue] = []

    ops = default_opcodes()
    opset = opset or latest_onnx_opset()

    # Collect param declarations and train/frozen flags
    params: Dict[str, Dict] = {}
    any_train = False
    for node in ast:
        if not isinstance(node, dict):
            continue
        if node.get("type") in ("param", "const"):
            name = node.get("name")
            if name is None:
                continue
            prev = params.get(name, {})
            # preserve trainable flag
            if "trainable" in node:
                prev_train = prev.get("trainable")
                if prev_train is not None and prev_train != node.get("trainable"):
                    errors.append(SanityIssue("error", f"Conflicting training flags for parameter '{name}'", node))
                prev["trainable"] = node.get("trainable")
            if "requires_grad" in node:
                if node.get("requires_grad"):
                    prev["trainable"] = True
                    any_train = True
                else:
                    prev["trainable"] = False
            # capture declared type info if present for pre-lowering checks
            if "type_decl" in node:
                prev["type_decl"] = node.get("type_decl")
            elif "type" in node:
                # legacy/simple form: store scalar type or leave None
                prev["type_decl"] = node.get("type")
            params[name] = prev
        if node.get("type") == "param" and node.get("requires_grad"):
            any_train = True

    any_train = any((p.get("trainable") is True or p.get("requires_grad") is True) for p in ast if isinstance(p, dict) and p.get("type") in ("param",))

    # Module-level training metadata
    training_meta_node = _find_meta(ast, "fuse.training")
    has_training_meta = training_meta_node is not None
    if any_train and not has_training_meta:
        warnings.append(SanityIssue("warning", "@train used but no module-level @training config present"))
    if has_training_meta and not any_train:
        warnings.append(SanityIssue("warning", "@training metadata present but no @train parameters found"))

    # Analyze optimizer field and emit config-driven advisory warnings
    if has_training_meta:
        try:
            cfg = load_sanitizer_config()
            enable_checks = cfg.get("enable_training_state_checks", True)
        except Exception:
            enable_checks = True

        try:
            val = training_meta_node.get("value") if isinstance(training_meta_node, dict) else None
            optimizer = None
            if isinstance(val, dict):
                optimizer = val.get("optimizer")
            elif isinstance(val, str):
                optimizer = val

            # if optimizer omitted or empty, warn about default
            if not optimizer:
                warnings.append(SanityIssue("warning", "Defaulting to 'adam' optimizer"))
            elif enable_checks:
                opt_name = optimizer if isinstance(optimizer, str) else (optimizer.get("call") if isinstance(optimizer, dict) else None)
                try:
                    reg_path = _Path(__file__).resolve().parents[1].joinpath("schemas/training_optimizers.json")
                    registry = {}
                    if reg_path.exists():
                        import json as _json

                        registry = _json.loads(reg_path.read_text())
                    key = opt_name.lower() if opt_name else ""
                    item = registry.get(key)

                    # if optimizer not recognized by registry, allow referring to a node defined in the AST or imports
                    known_by_node = any(isinstance(n, dict) and n.get("type") == "fn" and n.get("name") == opt_name for n in ast)
                    known_by_import = any(isinstance(n, dict) and n.get("type") == "import" and n.get("name") == opt_name for n in ast)
                    if not item and not known_by_node and not known_by_import:
                        warnings.append(SanityIssue("warning", f"Optimizer '{opt_name}' is not a known builtin nor a defined node", training_meta_node))

                    if item and item.get("require_state") and params:
                        for pname, pinfo in params.items():
                            if not pinfo.get("trainable"):
                                continue
                            for suff in item.get("state_suffixes", []):
                                state_name = f"{pname}.{suff}"
                                warnings.append(SanityIssue("warning", f"Optimizer '{item.get('canon')}' expects state initializer '{state_name}' for parameter '{pname}' (should be present after lowering)", training_meta_node, code="TRAIN.MISSING_STATE", param=pname, state=state_name))
                except Exception:
                    pass
                # Pre-lowering advisory: inspect declared param shapes and emit expectations
                try:
                    rules_path = _Path(__file__).resolve().parents[1].joinpath("schemas/training_param_shape_rules.json")
                    rules = {}
                    if rules_path.exists():
                        import json as _json

                        rules = _json.loads(rules_path.read_text())
                    import fnmatch
                    for pname2, pinfo2 in params.items():
                        if not pinfo2.get("trainable"):
                            continue
                        tdecl = pinfo2.get("type_decl")
                        if not isinstance(tdecl, dict):
                            continue
                        dims = tdecl.get("dims") or []
                        if not dims:
                            continue
                        for r in rules.get("rules", []):
                            pat = r.get("pattern")
                            if pat and fnmatch.fnmatch(pname2, pat):
                                desc = r.get("description") or r.get("note") or r.get("accept")
                                msg = f"Parameter '{pname2}' looks like rule '{r.get('name')}': {desc}. Post-lowering optimizer state tensors should follow this rule (e.g., 'W.m' dims [C] or similar)."
                                warnings.append(SanityIssue("warning", msg, training_meta_node, code="TRAIN.PARAM_STATE_EXPECTATION", param=pname2, extra={"rule": r.get("name")}))
                                break
                except Exception:
                    pass
        except Exception:
            warnings.append(SanityIssue("warning", "Failed to analyze @training optimizer field"))

    # Validate module-level @id and @type metadata early and emit friendly diagnostics
    def _looks_like_absolute_iri_or_curie(v: object) -> bool:
        if not isinstance(v, str):
            return False
        vs = v.strip()
        if vs.startswith("http://") or vs.startswith("https://"):
            return True
        # Accept CURIEs like 'prefix:local' as allowed forms
        import re

        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_\-]*:[^\s]+", vs):
            return True
        return False

    # Check for standalone @id/@type meta declarations
    for name in ("id", "type"):
        m = _find_meta(ast, name)
        if m is not None:
            val = m.get("value")
            if not _looks_like_absolute_iri_or_curie(val):
                issue_kind = "error" if strict else "warning"
                msg = f"@{name} metadata value appears to be non-IRI/non-CURIE: {val!r}. TTL export accepts absolute http(s) IRIs or CURIEs 'prefix:local'."
                if strict:
                    errors.append(SanityIssue(issue_kind, msg, m, code="META.INVALID_IRI"))
                else:
                    warnings.append(SanityIssue(issue_kind, msg, m, code="META.INVALID_IRI"))

    # Also check for @meta { id = ..., type = ... } forms
    for node in ast:
        if isinstance(node, dict) and node.get("type") == "meta" and isinstance(node.get("value"), dict):
            for k, v in node.get("value", {}).items():
                if k in ("id", "@id", "type", "@type"):
                    if not _looks_like_absolute_iri_or_curie(v):
                        issue_kind = "error" if strict else "warning"
                        msg = f"@{k} metadata (via @meta) value appears to be non-IRI/non-CURIE: {v!r}. TTL export accepts absolute http(s) IRIs or CURIEs 'prefix:local'."
                        if strict:
                            errors.append(SanityIssue(issue_kind, msg, node, code="META.INVALID_IRI"))
                        else:
                            warnings.append(SanityIssue(issue_kind, msg, node, code="META.INVALID_IRI"))
    # Other checks (duplicates, imports, unused consts, type aliases, call validation)
    decls: Dict[Tuple[str, str], int] = {}
    for node in ast:
        if isinstance(node, dict) and node.get("type") and node.get("name"):
            key = (node.get("type"), node.get("name"))
            decls[key] = decls.get(key, 0) + 1
            if decls[key] > 1:
                errors.append(SanityIssue("error", f"Duplicate declaration of {node.get('type')} named '{node.get('name')}'", node))

    fn_count = len([n for n in ast if isinstance(n, dict) and n.get("type") == "fn"])
    has_ns = _find_meta(ast, "namespace") or _find_meta(ast, "module") or _find_meta(ast, "domain")
    if fn_count > 1 and not has_ns:
        warnings.append(SanityIssue("warning", "Multiple top-level functions declared with no @domain/@module/@domain; consider adding module metadata to avoid naming collisions"))

    for node in ast:
        if isinstance(node, dict) and node.get("type") in ("fn", "node", "model", "export"):
            params_list = node.get("params") or []
            seen_params = {}
            for p in params_list:
                pname = p.get("name")
                if pname in seen_params:
                    errors.append(SanityIssue("error", f"Function '{node.get('name')}' has duplicate parameter '{pname}'", node))
                seen_params[pname] = True

    for node in ast:
        if isinstance(node, dict) and node.get("type") == "import":
            if not node.get("source") and not node.get("variants"):
                warnings.append(SanityIssue("warning", f"Import '{node.get('name')}' has no 'source' or 'variants' specified", node))

    # Validate fused_tensors declarations reference a file name
    for node in ast:
        if isinstance(node, dict) and node.get("type") == "const":
            val = node.get("value")
            if isinstance(val, dict) and "fused_tensors" in val:
                fname = val.get("fused_tensors", {}).get("file")
                if not fname or not isinstance(fname, str):
                    warnings.append(SanityIssue("warning", f"Fused tensor declaration for '{node.get('name')}' missing a valid file path", node))

    import_names = set()
    const_names = set()
    type_aliases = {}
    for node in ast:
        if isinstance(node, dict):
            if node.get("type") == "import":
                import_names.add(node.get("name"))
                if node.get("alias"):
                    import_names.add(node.get("alias"))
            if node.get("type") == "const":
                if node.get("name"):
                    const_names.add(node.get("name"))
            if node.get("type") == "type_alias":
                type_aliases[node.get("name")] = node.get("type_decl")

    used_calls = set()

    def _check_call(cname: str):
        used_calls.add(cname)
        if cname in import_names or cname in const_names:
            return
        valid, canon, casemap = ops.is_valid(cname, opset)
        if valid:
            if casemap:
                warnings.append(SanityIssue("warning", f"Operator '{cname}' used with non-canonical case; canonical name '{canon}'", cname))
            return
        # fallback to onnx builtin schemas when OpCodes.json is missing or incomplete
        try:
            from onnx.defs import get_all_schemas

            schema_map = {s.name.lower(): s.name for s in get_all_schemas()}
            if cname.lower() in schema_map:
                canon2 = schema_map[cname.lower()]
                if canon2 != cname:
                    warnings.append(SanityIssue("warning", f"Operator '{cname}' used with non-canonical case; canonical name '{canon2}'", cname))
                return
        except Exception:
            pass

        if cname in ops.TRAINING_OPS:
            return
        errors.append(SanityIssue("error", f"Unknown operator '{cname}' for opset {opset}", cname))

    def _walk(x: Any, string_collector: set = None):
        if string_collector is None:
            string_collector = set()
        if isinstance(x, dict):
            if "call" in x and isinstance(x.get("call"), str):
                _check_call(x.get("call"))
            for k, v in x.items():
                if isinstance(v, str):
                    string_collector.add(v)
                _walk(v, string_collector)
        elif isinstance(x, list):
            for i in x:
                if isinstance(i, str):
                    string_collector.add(i)
                _walk(i, string_collector)
        return string_collector

    for node in ast:
        if isinstance(node, dict):
            _walk(node)

    for name in sorted(import_names):
        if name not in used_calls:
            warnings.append(SanityIssue("warning", f"Import '{name}' appears unused", name))

    for cname in sorted(const_names):
        if cname not in used_calls:
            warnings.append(SanityIssue("warning", f"Const '{cname}' appears unused", cname))

    for ta, decl in type_aliases.items():
        if isinstance(decl, dict):
            dims = decl.get("dims")
            if dims:
                for d in dims:
                    if not (isinstance(d, int) or isinstance(d, str)):
                        warnings.append(SanityIssue("warning", f"Type alias '{ta}' has invalid dimension spec '{d}'", decl))
        elif isinstance(decl, str):
            if not decl:
                warnings.append(SanityIssue("warning", f"Type alias '{ta}' has empty scalar type", decl))
        else:
            warnings.append(SanityIssue("warning", f"Type alias '{ta}' has unexpected type form: {type(decl)}", decl))

    # collect user-defined top-level symbols to validate 'return' forms
    user_defined = set()
    for node in ast:
        if isinstance(node, dict) and node.get("type") in ("fn", "node", "model", "export") and node.get("name"):
            user_defined.add(node.get("name"))

    for node in ast:
        if isinstance(node, dict) and node.get("type") in ("fn", "node", "model", "export"):
            params_list = node.get("params") or []
            param_names = {p.get("name") for p in params_list if p.get("name")}
            body = node.get("body") or []
            if not body and node.get("type") in ("node", "model", "export"):
                body = node.get("body") or node.get("block") or node.get("block_body") or []
            used_strings = set()
            for stmt in body:
                _walk(stmt, used_strings)
            unused = param_names - used_strings
            for u in sorted(unused):
                warnings.append(SanityIssue("warning", f"Parameter '{u}' in function '{node.get('name')}' appears unused", node))
            for stmt in body:
                if isinstance(stmt, dict) and stmt.get("return") is not None:
                    ret = stmt.get("return")
                    if isinstance(ret, str):
                        if ret not in param_names and ret not in const_names and ret not in user_defined:
                            warnings.append(SanityIssue("warning", f"Return value '{ret}' in function '{node.get('name')}' is not defined", stmt))

    return {"errors": [e.as_dict() for e in errors], "warnings": [w.as_dict() for w in warnings]}

    for node in ast:
        if isinstance(node, dict) and node.get("type") in ("fn", "node", "model", "export"):
            params_list = node.get("params") or []
            param_names = {p.get("name") for p in params_list if p.get("name")}
            body = node.get("body") or []
            if not body and node.get("type") in ("node", "model", "export"):
                body = node.get("body") or node.get("block") or node.get("block_body") or []
            used_strings = set()
            for stmt in body:
                _walk(stmt, used_strings)
            unused = param_names - used_strings
            for u in sorted(unused):
                warnings.append(SanityIssue("warning", f"Parameter '{u}' in function '{node.get('name')}' appears unused", node))
            for stmt in body:
                if isinstance(stmt, dict) and stmt.get("return") is not None:
                    ret = stmt.get("return")
                    if isinstance(ret, str):
                        if ret not in param_names and ret not in const_names and ret not in user_defined:
                            warnings.append(SanityIssue("warning", f"Return value '{ret}' in function '{node.get('name')}' is not defined", stmt))

    return {"errors": [e.as_dict() for e in errors], "warnings": [w.as_dict() for w in warnings]}

