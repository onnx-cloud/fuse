import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import onnx
from onnx.defs import get_all_schemas


class OpCodes:
    """Load an op catalogue (OpCodes.json) and provide simple validity checks.

    The format expected is a list of objects {name, since, ...}. We expose
    a case-insensitive lookup and `is_valid` which returns whether an op is
    supported at the requested opset. Training-domain ops are handled by a
    small whitelist within this helper.
    """

    TRAINING_OPS = {"GenerateGradients", "Gradient", "Adam", "AdamW", "Adagrad", "Momentum"}

    def __init__(self, json_path: Optional[str] = None):
        if json_path is None:
            json_path = str(Path(__file__).resolve().parent / "OpCodes.json")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = []
        self._by_name: Dict[str, Dict] = {}
        self._lower_map: Dict[str, str] = {}
        for item in data:
            name = item.get("name")
            if not name:
                continue
            self._by_name[name] = item
            self._lower_map[name.lower()] = name

    def find_canonical(self, name: str) -> Optional[str]:
        if name in self._by_name:
            return name
        return self._lower_map.get(str(name).lower())

    def is_valid(self, name: str, opset: int) -> Tuple[bool, Optional[str], bool]:
        """Return (valid, canonical_name, case_insensitive_match)

        - valid: whether the op is present and in <= opset
        - canonical_name: canonical op name if known
        - case_insensitive_match: True if the input name matched case-insensitively
        """
        # Training ops are accepted by name (domain filtered elsewhere)
        if name in self.TRAINING_OPS:
            return True, name, False

        canon = self.find_canonical(name)
        if canon is None:
            return False, None, False
        item = self._by_name.get(canon)
        since = item.get("since") if isinstance(item, dict) else None
        try:
            ok = since is None or int(since) <= int(opset)
        except Exception:
            ok = True
        return ok, canon, canon != name


# Singleton cache
_default = None


def default_opcodes() -> OpCodes:
    global _default
    if _default is None:
        _default = OpCodes()
    return _default
