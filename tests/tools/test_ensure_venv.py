import os
import subprocess
from pathlib import Path


SCRIPT = Path('scripts/ensure-venv.sh').resolve()


def test_ensure_venv_fails_when_not_activated(monkeypatch, tmp_path):
    env = os.environ.copy()
    env.pop('VIRTUAL_ENV', None)

    p = subprocess.run([str(SCRIPT)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert p.returncode != 0
    assert 'Virtualenv not activated' in p.stderr


def test_ensure_venv_passes_when_active(monkeypatch, tmp_path):
    # create a fake project .venv directory and set VIRTUAL_ENV to it
    venv_dir = Path('.venv')
    try:
        venv_dir.mkdir(exist_ok=True)
        env = os.environ.copy()
        env['VIRTUAL_ENV'] = str(venv_dir.resolve())
        p = subprocess.run([str(SCRIPT)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert p.returncode == 0
        assert 'Virtualenv active' in p.stdout
    finally:
        # cleanup placeholder
        try:
            for child in venv_dir.iterdir():
                child.unlink()
        except Exception:
            pass
        try:
            venv_dir.rmdir()
        except Exception:
            pass
