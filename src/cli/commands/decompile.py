
from src.parser import ParseError
from src.lowering.utils import LoweringError

def cmd_decompile(
    files,
    out_dir=None,
    fuse=True,
    ast=True,
    proto=False,
    force=False,
    dry_run=False,
):
    """Decompile ONNX models to a best-effort Fuse wrapper and AST.

    Returns list of (src_path, [out_paths...], error_str)
    """
    from pathlib import Path
    from src.decompile import onnx_to_fuse

    results = []
    for f in files:
        try:
            src_path = Path(f)
            target_dir = Path(out_dir) if out_dir else Path(src_path.with_suffix("").name + ".decompile")
            target_dir.mkdir(parents=True, exist_ok=True)
            written = []

            # Decompile to Fuse wrapper
            try:
                fuse_src = onnx_to_fuse(str(src_path))
                if fuse and not dry_run:
                    fuse_file = target_dir / f"{src_path.stem}.fuse"
                    fuse_file.write_text(fuse_src, encoding="utf-8")
                    written.append(str(fuse_file))
            except (ValueError, TypeError, OSError, RuntimeError, SyntaxError, ImportError, AttributeError, KeyError, ParseError, LoweringError):
                # best-effort: continue and still attempt AST/proto
                fuse_src = None

            # AST (if requested): parse decompiled fuse source
            if ast:
                try:
                    if fuse_src is None:
                        # try to decompile first (if not already decompiled)
                        fuse_src = onnx_to_fuse(str(src_path))
                    from src.parser import fuse_parser

                    parsed = fuse_parser.parse(fuse_src)
                    ast_file = target_dir / f"{src_path.stem}.ast.json"
                    if not dry_run:
                        import json

                        ast_file.write_text(json.dumps(parsed, indent=2, sort_keys=False), encoding="utf-8")
                    written.append(str(ast_file))
                except (ValueError, TypeError, OSError, RuntimeError, SyntaxError, ImportError, AttributeError, KeyError, ParseError, LoweringError):
                    written.append(str(target_dir / f"{src_path.stem}.ast.error.txt"))

            # Emit proto text (if requested)
            if proto:
                try:
                    from google.protobuf import text_format
                    import onnx

                    m = onnx.load(str(src_path))
                    if m.graph and m.graph.initializer:
                        del m.graph.initializer[:]
                    pfile = target_dir / f"{src_path.stem}.proto"
                    if not dry_run:
                        pfile.write_text(text_format.MessageToString(m) + "\n", encoding="utf-8")
                    written.append(str(pfile))
                except (ValueError, TypeError, OSError, RuntimeError, SyntaxError, ImportError, AttributeError, KeyError, ParseError, LoweringError):
                    written.append(str(target_dir / f"{src_path.stem}.proto.error.txt"))

            results.append((f, written, None))
        except (ValueError, TypeError, OSError, RuntimeError, SyntaxError, ImportError, AttributeError, KeyError, ParseError, LoweringError) as e:
            results.append((f, None, str(e)))
    return results

