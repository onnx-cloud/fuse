from pathlib import Path
import os
import sys
import pytest


def test_install_startup_writes_file_and_contains_fallback(tmp_path, monkeypatch):
    # Force startup dir to temp path BEFORE importing the installer so
    # module-level defaults pick up the override.
    monkeypatch.setenv("IPYTHON_STARTUP_DIR", str(tmp_path))

    from jupyter.scripts.install_startup import install_startup, STARTUP_DIR, STARTUP_FILE

    # Ensure target dir points to our tmp startup dir
    assert str(STARTUP_DIR).startswith(str(tmp_path))

    # Simulate a minimal importable extension so install_startup's import test passes
    import types
    fake_magics = types.SimpleNamespace(load_ipython_extension=lambda ip: None)
    sys.modules['src'] = types.SimpleNamespace()
    sys.modules['src.jupyter'] = types.SimpleNamespace()
    sys.modules['src.jupyter.magics'] = fake_magics

    install_startup()
    assert STARTUP_FILE.exists()
    text = STARTUP_FILE.read_text()
    # Should include direct import and guidance, and NOT include sys.path fallback
    assert "from src.jupyter.magics import load_ipython_extension" in text
    assert "sys.path.insert(0, root)" not in text
    assert "pip install -e" in text or "install the package" in text

    # cleanup our fake modules
    for k in ('src.jupyter.magics', 'src.jupyter', 'src'):
        sys.modules.pop(k, None)


def test_install_startup_fails_when_src_unimportable(tmp_path, monkeypatch):
    # Ensure startup dir is tmp before importing installer
    monkeypatch.setenv("IPYTHON_STARTUP_DIR", str(tmp_path))
    from importlib import reload
    import sys

    # Temporarily remove 'src' from sys.modules and ensure repo root is not on sys.path
    saved_src = sys.modules.pop('src', None)
    saved_sys_path = list(sys.path)
    # Remove repository root from sys.path if present
    repo_root = str((Path(__file__).resolve().parents[2]))
    sys.path = [p for p in sys.path if p != repo_root]

    from jupyter.scripts.install_startup import install_startup
    try:
        with pytest.raises(SystemExit):
            install_startup()
    finally:
        # restore
        if saved_src is not None:
            sys.modules['src'] = saved_src
        sys.path = saved_sys_path
