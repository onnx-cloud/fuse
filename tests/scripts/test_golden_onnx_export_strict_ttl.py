import subprocess
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def test_golden_export_fails_with_ttl_strict(tmp_path: Path):
    # Locate the project scripts folder by walking parents (robust to test layout)
    cur = Path(__file__).resolve()
    script = None
    for p in cur.parents:
        cand = p / "scripts" / "golden_onnx_export.py"
        if cand.exists():
            script = cand
            break
    assert script is not None, "could not find scripts/golden_onnx_export.py in parent tree"
    p = tmp_path / "badmeta.fuse"
    p.write_text('@meta type = "not-an-iri"\nnode f(a: f32) -> f32 { a }\n')
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Prefer running the script under the project's venv Python when available
    python_bin = Path(sys.executable)
    # Try to locate the repository root (by looking for pyproject.toml or Makefile)
    repo_root = None
    for q in Path(__file__).resolve().parents:
        if (q / "pyproject.toml").exists() or (q / "Makefile").exists():
            repo_root = q
            break
    if repo_root:
        venv_py = repo_root / ".venv" / "bin" / "python"
        if venv_py.exists():
            python_bin = venv_py
    cmd = [str(python_bin), str(script), "--process-file", str(p), "--out-dir", str(out_dir), "--ttl-strict"]
    logger.debug("running %s", cmd)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    logger.debug("returncode %d stderr=%s stdout=%s", proc.returncode, proc.stderr, proc.stdout)

    # Script should exit with non-zero when TTL strict mode rejects metadata
    assert proc.returncode != 0
    stderr = (proc.stderr or "") + (proc.stdout or "")
    assert "invalid @type value for TTL export" in stderr or "unknown prefix" in stderr
