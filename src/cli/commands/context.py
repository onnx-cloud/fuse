from typing import List, Optional, Tuple, Dict

class CliContext:
    refresh_cache: bool = False
    refresh_import: Optional[List[str]] = None
    folds: int = 8
    externalize: int = 0
    external_dir: Optional[str] = None
    preserve_external: bool = False


VerifyResult = Tuple[str, Optional[str]]  # (path, error_message or None)


LintMessage = Dict[str, object]


LintResult = List[LintMessage]

