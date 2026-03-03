
from src.parser import ParseError
from src.lowering.utils import LoweringError

def cmd_ttl(
    files,
    out=None,
    ns="",
    ns_uri="",
    no_initializers=False,
    no_metadata=False,
):
    """Convert ONNX model files to RDF/Turtle format.

    Returns list of (src_path, out_path, error_str).
    """
    from pathlib import Path

    import onnx

    from src.export.ttl import save_ttl

    results = []
    for f in files:
        try:
            model = onnx.load(str(f))

            # Determine output path
            if out:
                out_path = Path(out)
                # If out is a directory, create a .ttl file inside it
                if out_path.is_dir() or (len(files) > 1 and not out_path.suffix):
                    out_path.mkdir(parents=True, exist_ok=True)
                    out_path = out_path / (Path(f).stem + ".ttl")
            else:
                # Default: same directory as source, with .ttl extension
                out_path = Path(f).with_suffix(".ttl")

            save_ttl(
                model,
                out_path,
                user_ns=ns,
                user_ns_uri=ns_uri,
                include_initializers=not no_initializers,
                include_metadata=not no_metadata,
            )
            results.append((str(f), str(out_path), None))
        except (ValueError, TypeError, OSError, RuntimeError, SyntaxError, ImportError, AttributeError, KeyError, ParseError, LoweringError) as e:
            results.append((str(f), None, str(e)))
    return results

