from pathlib import Path
import sys
import types
import builtins
import pytest

import importlib


def test_install_kernel_writes_startup_with_fallback(tmp_path, monkeypatch):
    # Mock KernelSpecManager so we don't need ipykernel available
    class FakeKSM:
        def __init__(self):
            pass
        def install_kernel_spec(self, path, kernel_name, user=True):
            # no-op
            return
        def get_kernel_spec(self, name):
            return types.SimpleNamespace(resource_dir=str(path_to_kernelspec))

    # Create a fake kernelspec candidate on disk so install_kernel picks it
    path_to_kernelspec = tmp_path / "jupyter" / "kernelspec"
    path_to_kernelspec.mkdir(parents=True)

    # Monkeypatch KernelSpecManager used in install.py
    monkeypatch.setitem(sys.modules, 'ipykernel.kernelspec', types.SimpleNamespace(KernelSpecManager=FakeKSM))
    # Also ensure jupyter_client.kernelspec is present so startup script section runs
    class FakeKSM2:
        def get_kernel_spec(self, name):
            return types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'jupyter_client.kernelspec', types.SimpleNamespace(KernelSpecManager=FakeKSM2))

    # Ensure Path.home() points to our tmp dir so profile_dir is isolated
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)

    from src.jupyter.install import install_kernel

    # Simulate importable magics to satisfy strict validation
    fake_magics = types.SimpleNamespace(load_ipython_extension=lambda ip: None)
    sys.modules['src'] = types.SimpleNamespace()
    sys.modules['src.jupyter'] = types.SimpleNamespace()
    sys.modules['src.jupyter.magics'] = fake_magics

    # Call install_kernel; it should create a startup script in profile_kernel/startup
    install_kernel(user=True, name='fuse-test', display_name='Fuse Test')

    # cleanup fake modules
    for k in ('src.jupyter.magics', 'src.jupyter', 'src'):
        sys.modules.pop(k, None)

    startup_script = tmp_path / ".ipython" / "profile_kernel" / "startup" / "00_fuse_magics.py"
    assert startup_script.exists()
    content = startup_script.read_text()
    # Should contain direct import and guidance, not repo-root fallback
    assert "from src.jupyter.magics import load_ipython_extension" in content
    assert "sys.path.insert(0, root)" not in content
    assert "pip install -e" in content or "install the package" in content


def test_install_kernel_fails_when_src_unimportable(tmp_path, monkeypatch):
    # Mock KernelSpecManager so we don't need ipykernel available
    class FakeKSM:
        def __init__(self):
            pass
        def install_kernel_spec(self, path, kernel_name, user=True):
            # no-op
            return
        def get_kernel_spec(self, name):
            return types.SimpleNamespace(resource_dir=str(path_to_kernelspec))

    # Setup fake kernelspec path
    path_to_kernelspec = tmp_path / "jupyter" / "kernelspec"
    path_to_kernelspec.mkdir(parents=True)
    monkeypatch.setitem(sys.modules, 'ipykernel.kernelspec', types.SimpleNamespace(KernelSpecManager=FakeKSM))
    monkeypatch.setitem(sys.modules, 'jupyter_client.kernelspec', types.SimpleNamespace(KernelSpecManager=FakeKSM))

    # Ensure Path.home() points to our tmp dir so profile_dir is isolated
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)

    # Temporarily remove 'src' from sys.modules and repo root from sys.path
    saved_src = sys.modules.pop('src', None)
    saved_sys_path = list(sys.path)
    repo_root = str((Path(__file__).resolve().parents[2]))
    sys.path = [p for p in sys.path if p != repo_root]

    from src.jupyter.install import install_kernel
    try:
        with pytest.raises(SystemExit):
            install_kernel(user=True, name='fuse-test2', display_name='Fuse Test')
    finally:
        if saved_src is not None:
            sys.modules['src'] = saved_src
        sys.path = saved_sys_path


def test_install_kernel_make_default_installs_python3(tmp_path, monkeypatch):
    installed = []
    class FakeKSM:
        def __init__(self):
            pass
        def install_kernel_spec(self, path, kernel_name, user=True):
            installed.append(kernel_name)
            return
        def get_kernel_spec(self, name):
            return types.SimpleNamespace(resource_dir=str(path_to_kernelspec))

    # Create a fake kernelspec candidate on disk so install_kernel picks it
    path_to_kernelspec = tmp_path / "jupyter" / "kernelspec"
    path_to_kernelspec.mkdir(parents=True)

    # Monkeypatch KernelSpecManager used in install.py
    monkeypatch.setitem(sys.modules, 'ipykernel.kernelspec', types.SimpleNamespace(KernelSpecManager=FakeKSM))
    # Also ensure jupyter_client.kernelspec is present so startup script section runs
    monkeypatch.setitem(sys.modules, 'jupyter_client.kernelspec', types.SimpleNamespace(KernelSpecManager=FakeKSM))

    # Ensure Path.home() points to our tmp dir so profile_dir is isolated
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)

    # Simulate importable magics to satisfy strict validation
    fake_magics = types.SimpleNamespace(load_ipython_extension=lambda ip: None)
    sys.modules['src'] = types.SimpleNamespace()
    sys.modules['src.jupyter'] = types.SimpleNamespace()
    sys.modules['src.jupyter.magics'] = fake_magics

    from src.jupyter.install import install_kernel

    # Call install_kernel with make_default; it should request both names
    install_kernel(user=True, name='fuse-test', display_name='Fuse Test', make_default=True)

    # cleanup fake modules
    for k in ('src.jupyter.magics', 'src.jupyter', 'src'):
        sys.modules.pop(k, None)

    assert 'fuse-test' in installed
    assert 'python3' in installed
