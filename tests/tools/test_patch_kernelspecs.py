import shutil
import json
from pathlib import Path
import subprocess


SCRIPT = Path('scripts/patch_kernelspecs.py').resolve()


def test_patch_notebook_in_temp_dir(tmp_path):
    src = Path('jupyter/notebooks/jupyer_converter.ipynb')
    tdir = tmp_path / 'nbdir'
    tdir.mkdir()
    dst = tdir / 'jupyer_converter.ipynb'
    shutil.copy(src, dst)

    # Run the patch script in-place on the temp dir
    proc = subprocess.run(['python', str(SCRIPT), '--in-place', '--dir', str(tdir)], capture_output=True, text=True)
    assert proc.returncode == 0

    # Verify metadata was updated
    with dst.open('r', encoding='utf-8') as fh:
        nb = json.load(fh)
    ks = nb.get('metadata', {}).get('kernelspec')
    assert ks is not None
    assert ks.get('name') == 'fuse'
    assert 'Fuse' in ks.get('display_name', '')
