
from typing import List
from src.parser import ParseError
from src.lowering.utils import LoweringError
from .context import VerifyResult

def cmd_verify(paths: List[str]) -> List[VerifyResult]:
    """Verify fuse files for compatibility with installed Fuse manifest.

    Returns list of (path, error) where error is None on success else string.
    """
    results: List[VerifyResult] = []
    for p in paths:
        try:
            from src import cli_helpers

            ast = cli_helpers.parse_fuse_file(p)
            res = cli_helpers.check_fuse_compat(ast, source_file=p)
            if res:
                status, req, cur = res
                if status == "fail":
                    msg = (
                        f"Fuse compatibility error: required fuse {req} "
                        f"is incompatible with current fuse {cur}"
                    )
                    results.append((p, msg))
                    continue
                if status == "warn":
                    msg = (
                        f"Warning: file {p} requests fuse {req} "
                        f"which is older than current fuse {cur}"
                    )
                    results.append((p, msg))
                    continue
            results.append((p, None))
        except (ValueError, TypeError, OSError, RuntimeError, SyntaxError, ImportError, AttributeError, KeyError, ParseError, LoweringError) as e:
            results.append((p, str(e)))
    return results

