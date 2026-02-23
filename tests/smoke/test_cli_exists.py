import subprocess
import sys


def test_cli_version_exits_zero():
    # Run the CLI version command as a lightweight smoke check
    proc = subprocess.run([sys.executable, "-m", "src.cli", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.returncode == 0, f"CLI version command failed: stderr={proc.stderr}"
    assert proc.stdout.strip(), "CLI version should print something"
