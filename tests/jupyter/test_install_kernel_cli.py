import runpy
import sys
import types
from pathlib import Path


def test_cli_make_default_installs_python3(monkeypatch, tmp_path):
    installed = []

    class FakeKSM:
        def __init__(self):
            pass
        def install_kernel_spec(self, path, kernel_name, user=True):
            installed.append(kernel_name)
            return
        def get_kernel_spec(self, name):
            return types.SimpleNamespace(resource_dir=str(tmp_path / 'jupyter' / 'kernelspec'))

    # Ensure our fake modules are used when the CLI runs
    monkeypatch.setitem(sys.modules, 'ipykernel.kernelspec', types.SimpleNamespace(KernelSpecManager=FakeKSM))
    monkeypatch.setitem(sys.modules, 'jupyter_client.kernelspec', types.SimpleNamespace(KernelSpecManager=FakeKSM))

    saved_argv = sys.argv[:]
    # Ensure Path.home is isolated so install creates startup scripts under tmp
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    # Provide a fake magics module so strict import validation passes
    fake_magics = types.SimpleNamespace(load_ipython_extension=lambda ip: None)
    sys.modules['src.jupyter.magics'] = fake_magics
    try:
        # When run as a script, argv[0] is the program name; provide the module name as argv[0]
        sys.argv[:] = ['src.jupyter.install', '--make-default']
        # run the module as a script entrypoint
        runpy.run_module('src.jupyter.install', run_name='__main__')
    finally:
        sys.argv[:] = saved_argv
        # cleanup
        for k in ('src.jupyter.magics',):
            sys.modules.pop(k, None)

    assert 'fuse' in installed or 'fuse-test' in installed or len(installed) > 0
    assert 'python3' in installed
