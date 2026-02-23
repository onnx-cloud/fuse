# Simple helper to ensure notebooks use the Fuse kernelspec
"""
Usage:
  python scripts/patch_kernelspecs.py [--in-place] [--dir path]

By default this prints the notebooks that would be changed. Pass --in-place to overwrite files.
"""
from pathlib import Path
import argparse
import json

FUSE_KERNEL = {'name': 'fuse', 'display_name': 'Fuse (ONNX)', 'language': 'python'}


def patch_notebook(path: Path, in_place: bool = False) -> bool:
    # Use plain JSON to avoid requiring nbformat during basic patching
    with path.open('r', encoding='utf-8') as fh:
        nb = json.load(fh)
    meta = nb.setdefault('metadata', {})
    old = meta.get('kernelspec')
    if old == FUSE_KERNEL:
        return False
    meta['kernelspec'] = FUSE_KERNEL
    if in_place:
        with path.open('w', encoding='utf-8') as fh:
            json.dump(nb, fh, indent=1)
    return True


def main():
    ap = argparse.ArgumentParser(description='Patch jupyter notebooks to use the Fuse kernelspec')
    ap.add_argument('--in-place', action='store_true', help='Overwrite notebooks in-place')
    ap.add_argument('--dir', default='jupyter/notebooks', help='Directory containing notebooks')
    args = ap.parse_args()

    d = Path(args.dir)
    changed = []
    for p in sorted(d.glob('*.ipynb')):
        try:
            if patch_notebook(p, in_place=args.in_place):
                changed.append(str(p))
        except Exception as e:
            print('Skipping', p, 'due to error:', e)
    if changed:
        print('Patched the following notebooks:')
        for c in changed:
            print('  ', c)
    else:
        print('No notebooks needed patching')


if __name__ == '__main__':
    main()
