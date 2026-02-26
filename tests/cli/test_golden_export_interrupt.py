import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def test_golden_export_handles_interrupt():
    """Spawn the child-mode exporter and send SIGINT; it should exit cleanly
    with a short 'user terminated' message and non-zero exit code.
    """
    # use the same Python interpreter that's running the test to ensure
    # required deps (like lark) are present
    cmd = [sys.executable, "-m", "scripts.golden_onnx_export", "--process-file", "examples/golden/strange.fuse", "--out-dir", "tmp/onnx"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        # Give the child a moment to start
        time.sleep(0.2)
        # Send SIGINT
        proc.send_signal(signal.SIGINT)
        # Wait for it to terminate
        stdout, stderr = proc.communicate(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    # Expect the exporter to report a succinct termination message
    assert proc.returncode != 0
    # stderr should contain our concise message
    assert "user terminated" in (stderr or ""), f"expected 'user terminated' in stderr, got: {stderr!r}"
