import subprocess
import sys


def test_entrypoint_version():
    # Call the module entrypoint to ensure it handles 'version' gracefully
    rc = subprocess.call([sys.executable, "-m", "src.cli", "version"])
    assert rc == 0
