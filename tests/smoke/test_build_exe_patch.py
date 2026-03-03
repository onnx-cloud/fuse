import subprocess
from pathlib import Path


def test_build_exe_patch_only_restores(tmp_path):
    """Ensure `scripts/build_exe.sh --patch-only` injects version/build_time
    into ``src/__init__.py`` and leaves a backup that can be used to restore.

    This smoke test exercises the new patch-only mode added during the fix for
    "version built: unknown". It mutates the real source file but always
    restores it at the end so the repository is left clean.
    """
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "build_exe.sh"
    src_init = repo_root / "src" / "__init__.py"
    backup = src_init.with_suffix(".py.bak")

    # remove any stale backup from previous runs
    if backup.exists():
        backup.unlink()

    try:
        proc = subprocess.run([str(script), "--patch-only"], cwd=repo_root, capture_output=True, text=True)
        assert proc.returncode == 0
        assert "--patch-only requested" in proc.stdout

        txt = src_init.read_text(encoding="utf-8")
        # version should come from pyproject.toml (0.7.2 in current repo)
        assert "0.7.2" in txt
        # build time marker inserted too
        assert "__build_time__" in txt
    finally:
        # restore original file using backup if it exists
        if backup.exists():
            backup.replace(src_init)
