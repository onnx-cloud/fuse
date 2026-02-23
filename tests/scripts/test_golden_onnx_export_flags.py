import subprocess
import sys


def test_help_includes_meta_aliases():
    # Run the script as a module to avoid importing optional deps at test-collection
    cmd = [sys.executable, "-m", "scripts.golden_onnx_export", "--help"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # argparse prints help to stdout and exits with code 0
    assert proc.returncode == 0
    out = proc.stdout
    assert "--meta" in out
    assert "--no-meta" in out
    assert "--metrics" in out
    assert "--no-metrics" in out
