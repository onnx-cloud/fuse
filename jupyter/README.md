Fuse Jupyter integration

To enable the `%%fuse` cell magic persistently in kernels launched from this environment run:

  python jupyter/scripts/install_startup.py

This script creates `00_fuse_magics.py` in the IPython startup directory (defaults to `~/.ipython/profile_default/startup`) which attempts to import and register the Fuse magics on kernel start. The loader is robust in development checkouts and will attempt to add the repository root to `sys.path` before importing the extension.

If the magic still fails at kernel startup:

- Ensure the `src` package is importable in the kernel environment (e.g., `pip install -e .`).
- You can also load the extension in a running kernel manually:

  %load_ext src.jupyter.magics

Or, in Python:

  from IPython import get_ipython
  from src.jupyter.magics import load_ipython_extension
  load_ipython_extension(get_ipython())

For advanced users, the Jupyter kernel spec installer `src.jupyter.install.install_kernel` will also create a kernel-specific startup loader; run `python -m src.jupyter.install` to install the kernel and startup hooks. To make Fuse the default Python kernel (install a copy under the name `python3`), add the `--make-default` flag when running the installer. Additionally, you can patch notebooks in-place to embed the Fuse kernelspec using the helper script:

```
python scripts/patch_kernelspecs.py --in-place
```

Note: Some notebooks in this repo may be owned by root in certain build images and may require appropriate permissions to overwrite.

## Visualizations & Magics 🎨

Fuse provides a set of lightweight Jupyter magics and a small visual toolkit for inspecting tensors, embeddings, attention maps, audio, and models inside notebooks.

Quick starter:

- Load the magics once per kernel:

  ```python
  %load_ext src.jupyter.magics
  %load_ext src.jupyter.inspect.magics
  ```

- Common magics:
  - `%image <expr>` — Render image-like tensors (HWC/CHW/NHWC/NCHW).
  - `%pca <expr>` / `%tsne <expr>` — Project and display embeddings.
  - `%attention <expr>` — Heatmap for attention tensors (heads x seq x seq).
  - `%inspect <expr>` — Auto-detect best visualization for a tensor (or use `as <decoder>` to force one).
  - `%graph <model>` — Display a compiled model graph (also available via `%fuse.show <name>`).

- Cookbook recipes with examples:
  - `jupyter/cookbook/visuals_image.ipynb`
  - `jupyter/cookbook/visuals_attention.ipynb`

If you want to add a new decoder or visualization, see `jupyter/todo/INSPECT.md` for design notes and the decoder registry API.

