
from typing import List
from src.parser import ParseError
from src.lowering.utils import LoweringError
from .context import LintMessage, LintResult

def cmd_lint(
    paths: List[str], fail_on_warn: bool = False, check_remote: bool = False, check_training: bool = False, check_meta_strict: bool = False
) -> LintResult:
    """Lint files and return a list of structured messages.

    Each message is a dict: { file: str, kind: 'warning'|'error', message: str, node: optional }

    Args:
        check_meta_strict: when True, treat metadata issues (invalid @id/@type)
                           as errors instead of warnings by forwarding strict
                           to the sanitizer.
    """
    messages: List[LintMessage] = []

    # Ensure `cli_helpers` is available (some import paths can be shadowed by the
    # top-level `src.cli` entrypoint). Import lazily and provide a clear error
    # if the helper cannot be loaded so tests get actionable diagnostics.
    try:
        from src import cli_helpers as cli_helpers
    except (ValueError, TypeError, OSError, RuntimeError, SyntaxError, ImportError, AttributeError, KeyError, ParseError, LoweringError):
        import importlib

        try:
            cli_helpers = importlib.import_module("src.cli.cli_helpers")
        except (ValueError, TypeError, OSError, RuntimeError, SyntaxError, ImportError, AttributeError, KeyError, ParseError, LoweringError) as e:
            for p in paths:
                messages.append({"file": p, "kind": "error", "message": f"cli_helpers import failed: {e}"})
            return messages

    for p in paths:
        # Parse file
        try:
            ast = cli_helpers.parse_fuse_file(p)
        except (ValueError, TypeError, OSError, RuntimeError, SyntaxError, ImportError, AttributeError, KeyError, ParseError, LoweringError) as e:
            # parse_fuse_file raises an Exception with filename/line/column
            # context when parsing fails. Use the provided information so
            # the lint output contains a precise location for the error.
            messages.append({"file": p, "kind": "error", "message": f"parse error: {e}"})
            continue

        # missing @domain when file declares top-level nodes/models
        has_domain = any(
            isinstance(d, dict) and d.get("type") == "meta" and d.get("name") == "domain"
            for d in (ast or [])
        )
        has_decl = any(
            isinstance(d, dict) and d.get("type") in ("node", "model", "export") for d in (ast or [])
        )
        if has_decl and not has_domain:
            messages.append({"file": p, "kind": "warning", "message": "missing @domain declaration (recommended; required when namespacing is enabled)"})

        # validate imports exist when local
        for decl in ast:
            if not isinstance(decl, dict):
                continue
            if decl.get("type") == "import":
                src = decl.get("source")
                if src and not str(src).startswith("http"):
                    from pathlib import Path

                    if not Path(src).exists():
                        messages.append({"file": p, "kind": "error", "message": f"import source not found: {src}"})
            if decl.get("type") in ("node", "model", "export"):
                for param in decl.get("params", []):
                    typ = param.get("type") or param.get("type_decl")
                    if cli_helpers.symbolic_dim_in_type(typ):
                        messages.append({"file": p, "kind": "warning", "message": f"function {decl.get('name')} uses symbolic dims in parameter {param.get('name')} (will be lowered to dynamic dims)"})

        # Run AST sanitizer for more comprehensive checks
        from src.sanitizer import sanitize_ast
        try:
            san = sanitize_ast(ast, strict=check_meta_strict)
            for w in san.get("warnings", []):
                m = {"file": p, "kind": "warning", "message": w.get("message")}
                if w.get("node") is not None:
                    m["node"] = w.get("node")
                if w.get("code") is not None:
                    m["code"] = w.get("code")
                if w.get("param") is not None:
                    m["param"] = w.get("param")
                if w.get("state") is not None:
                    m["state"] = w.get("state")
                messages.append(m)
            for e in san.get("errors", []):
                m = {"file": p, "kind": "error", "message": e.get("message")}
                if e.get("node") is not None:
                    m["node"] = e.get("node")
                if e.get("code") is not None:
                    m["code"] = e.get("code")
                if e.get("param") is not None:
                    m["param"] = e.get("param")
                if e.get("state") is not None:
                    m["state"] = e.get("state")
                messages.append(m)
        except (ValueError, TypeError, OSError, RuntimeError, SyntaxError, ImportError, AttributeError, KeyError, ParseError, LoweringError):
            # Best-effort sanitizer: do not fail lint on sanitizer exceptions
            messages.append({"file": p, "kind": "warning", "message": "sanitizer failed to run"})

        # Optional: perform heavier training checks via lowering when requested
        if check_training:
            try:
                from pathlib import Path
                from src.parser import fuse_parser
                from src.lowering import FuseLowerer
                from src.lowering.training_checks import check_training_model, validate_training_info

                decls = fuse_parser.parse(Path(p).read_text(), filename=p)
                fl = FuseLowerer(emit_training=True)
                try:
                    model = fl.lower(decls)
                    # Run the main training checks
                    tc = check_training_model(model)
                    for w in tc.get("warnings", []):
                        if isinstance(w, dict):
                            messages.append({"file": p, "kind": "warning", "message": w.get("message"), "code": w.get("code"), "param": w.get("param"), "state": w.get("state")})
                        else:
                            messages.append({"file": p, "kind": "warning", "message": str(w)})
                    for e in tc.get("errors", []):
                        if isinstance(e, dict):
                            messages.append({"file": p, "kind": "error", "message": e.get("message"), "code": e.get("code"), "param": e.get("param"), "state": e.get("state"), "expected_output": e.get("expected_output")})
                        else:
                            messages.append({"file": p, "kind": "error", "message": str(e)})

                    # Run TrainingInfo validation (may raise ValueError on fatal issues)
                    try:
                        validate_training_info(model)
                    except ValueError as ve:
                        messages.append({"file": p, "kind": "error", "message": str(ve), "code": "TRAIN.VALIDATION_ERROR"})
                except (ValueError, TypeError, OSError, RuntimeError, SyntaxError, ImportError, AttributeError, KeyError, ParseError, LoweringError) as e:
                    # Do not fail lint if lowering/training checks cannot be run
                    messages.append({"file": p, "kind": "warning", "message": f"training check failed: {e}"})
            except (ValueError, TypeError, OSError, RuntimeError, SyntaxError, ImportError, AttributeError, KeyError, ParseError, LoweringError) as e:
                messages.append({"file": p, "kind": "warning", "message": f"training check setup failed: {e}"})
                messages.append({"file": p, "kind": "warning", "message": f"training check setup failed: {e}"})

    # If the caller requested 'fail_on_warn', upgrade warnings to errors
    if fail_on_warn:
        for m in messages:
            if m.get("kind") == "warning":
                m["kind"] = "error"

    return messages

