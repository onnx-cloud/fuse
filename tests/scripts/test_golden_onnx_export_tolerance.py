import subprocess
import sys
from pathlib import Path


def test_golden_export_tolerant_to_ttl_invalid_meta(tmp_path: Path):
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
    repo_root = None
    for q in Path(__file__).resolve().parents:
        if (q / "pyproject.toml").exists() or (q / "Makefile").exists():
            repo_root = q
            break
    if repo_root:
        venv_py = repo_root / ".venv" / "bin" / "python"
        if venv_py.exists():
            python_bin = venv_py
    proc = subprocess.run([str(python_bin), str(script), "--process-file", str(p), "--out-dir", str(out_dir)], capture_output=True, text=True)

    # Script should exit successfully (TTL rejection should be non-fatal)
    assert proc.returncode == 0

    # Warning emitted to stderr about TTL export failing
    stderr = (proc.stderr or "") + (proc.stdout or "")
    assert "TTL export failed" in stderr or "non-IRI" in stderr
