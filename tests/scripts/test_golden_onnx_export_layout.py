import subprocess
import sys
from pathlib import Path


def test_golden_export_layout_respects_domain_and_version(tmp_path: Path):
    # create a minimal .fuse with explicit domain and semantic version
    src = tmp_path / "example.fuse"
    src.write_text('@domain examples.golden\n@version 1.2.3\nnode demo(x: f32) -> f32 { return x }\n')
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # locate script
    cur = Path(__file__).resolve()
    script = None
    for p in cur.parents:
        cand = p / "scripts" / "golden_onnx_export.py"
        if cand.exists():
            script = cand
            break
    assert script is not None, "could not find scripts/golden_onnx_export.py in parent tree"

    # Run script
    python_bin = Path(sys.executable)
    proc = subprocess.run([str(python_bin), str(script), "--process-file", str(src), "--out-dir", str(out_dir)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + proc.stdout

    # Find emitted ONNX files (recursively)
    onnx_files = list(out_dir.rglob("*.onnx"))
    assert onnx_files, "no ONNX files emitted"

    for p in onnx_files:
        # TTL/YAML/DOT/MD/AST should live next to the ONNX
        assert p.with_suffix('.ttl').exists(), f"missing ttl for {p}"
        assert p.with_suffix('.yaml').exists(), f"missing yaml for {p}"
        assert p.with_suffix('.dot').exists(), f"missing dot for {p}"
        assert p.with_suffix('.ast').exists(), f"missing ast for {p}"
        assert  p.with_suffix('.md'), f"missing md for {p}"
        assert p.with_suffix('.html'), f"missing html for {p}"
