
from src.parser import ParseError
from src.lowering.utils import LoweringError

def cmd_metrics(files):
    """Compute metrics for Fuse files and return YAML-like outputs.

    Returns list of (src_path, [yaml_str], error_str)
    """
    from src.metrics import compute_metrics_for_file, format_metrics

    results = []
    for f in files:
        try:
            metrics = compute_metrics_for_file(f)
            out = format_metrics(metrics)
            results.append((f, [out], None))
        except (ValueError, TypeError, OSError, RuntimeError, SyntaxError, ImportError, AttributeError, KeyError, ParseError, LoweringError) as e:
            results.append((f, None, str(e)))
    return results

