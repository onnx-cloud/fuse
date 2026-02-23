import subprocess
import sys
import shutil
from pathlib import Path


def test_gold_exports_and_flat_fixture_exists():
    """Ensure each examples/golden/*.fuse can be exported and that a flat
    top-level golden ONNX fixture exists under tmp/onnx/golden.onnx.

    This acts as a guard-rail for `make gold` ensuring CI and local builds
    have a predictable flat artifact for legacy tests that still look for
    `tmp/onnx/golden.onnx`.
    """
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "golden_onnx_export.py"
    assert script.exists(), "could not find scripts/golden_onnx_export.py"

    fuse_dir = root / "examples" / "golden"
    fuse_files = sorted(fuse_dir.glob("*.fuse"))
    assert fuse_files, "no examples/golden/*.fuse files found"

    out_dir = root / "tmp" / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    python_bin = Path(sys.executable)
    # Export each golden file in isolated child-mode to avoid surprising crashes
    # from one example aborting the whole export run.
    failures = []
    for f in fuse_files:
        proc = subprocess.run(
            [str(python_bin), str(script), "--process-file", str(f), "--out-dir", str(out_dir)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            # Record failure and continue to collect failures for a single clear assertion
            failures.append((f, proc.returncode, proc.stdout, proc.stderr))

    # Confirm we emitted at least one ONNX model under tmp/onnx
    onnx_files = list(out_dir.rglob("*.onnx"))
    assert onnx_files, f"no ONNX artifacts found under {out_dir} after export; failures={[(str(f), rc) for f, rc, _, _ in failures]}"

    # STRICT: fail if any per-file export failed
    if failures:
        msgs = []
        for f, rc, out, err in failures:
            msgs.append(f"{f.name}: exit={rc}\nSTDOUT:\n{out.strip()}\nSTDERR:\n{err.strip()}")
        full = "\n\n".join(msgs)
        assert False, f"golden export had failures ({len(failures)}):\n\n{full}"

    # Ensure legacy flat fixture exists for downstream tests: tmp/onnx/golden.onnx
    flat = out_dir / "golden.onnx"
    if not flat.exists():
        # Prefer a named canonical model if present (train_golden.onnx), else pick first emitted ONNX
        candidates = list(out_dir.rglob("**/train_golden.onnx"))
        if not candidates:
            candidates = list(out_dir.rglob("**/*.onnx"))
        assert candidates, "no candidate ONNX files available to create flat golden.onnx"
        shutil.copy2(str(candidates[0]), str(flat))

    assert flat.exists(), "failed to ensure tmp/onnx/golden.onnx exists"
