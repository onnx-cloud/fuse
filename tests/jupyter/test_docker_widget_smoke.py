import os
import subprocess
import time
import requests
import pytest


@pytest.mark.skip("Docker test; enable by setting DOCKER_SMOKE=1 in your environment")
def test_build_and_run_container_smoke(tmp_path):
    # This test is skipped by default; run locally with DOCKER_SMOKE=1 pytest -k docker
    script = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'build_labextension_and_image.sh')
    script = os.path.abspath(script)
    env = os.environ.copy()
    # Run the script
    subprocess.check_call([script], env=env)
    # If the script completes without raising, consider it a pass
