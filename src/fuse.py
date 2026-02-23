import json
from pathlib import Path
from typing import Tuple


class FuseError(Exception):
    pass


def _manifest_paths():
    # Default: look for manifest files at repo root (one level up from src)
    root = Path(__file__).resolve().parents[1]
    return {
        "toml": root / "pyproject.toml",
    }


def load_manifest(path: Path | str | None = None):
    """Load manifest info.

    `pyproject.toml` project.version (fallback)

    Returns a dict with at least the key ``fuse_version``.
    """
    p = Path(path) if path else None
    if p:
        if not p.exists():
            raise FuseError(f"fuse manifest not found at {p}")
        # Assume explicit path points to manifest
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    paths = _manifest_paths()
    if paths["toml"].exists():
        try:
            import tomllib

            with open(paths["toml"], "rb") as f:
                data = tomllib.load(f)
            version = data.get("project", {}).get("version")
            if not version:
                raise FuseError("project.version not found in pyproject.toml")
            return {"fuse_version": version}
        except Exception as e:
            raise FuseError(f"failed to load pyproject.toml: {e}") from e

    raise FuseError("no manifest found (pyproject.toml)")


def parse_version(v: str) -> Tuple[int, int, int]:
    parts = v.split(".")
    nums = [int(p) for p in parts]
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])


def compare_required(required: str, current: str) -> str:
    """Compare two versions (required vs current).

    Returns:
      - 'ok' if required major == current major and minor >= current minor or higher
      - 'warn' if required major == current major and minor < current minor
      - 'fail' if required major < current major
      - 'ok' also if required major > current major (future), allow for now
    """
    req = parse_version(required)
    cur = parse_version(current)
    req_major, req_minor, _ = req
    cur_major, cur_minor, _ = cur
    if req_major < cur_major:
        return "fail"
    if req_major == cur_major and req_minor < cur_minor:
        return "warn"
    return "ok"
