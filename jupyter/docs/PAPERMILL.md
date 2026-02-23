# Using Papermill with Fuse Jupyter 📦✨

## Quick install

install papermill in the Docker environment:

```bash
uv pip install papermill
# or in our dev image: add to requirements-dev.txt or run inside Docker
```

---

## Notebook conventions

- Mark a cell that sets parameters with the `parameters` tag (Papermill convention). Example cell contents:

```python
# Parameters
input_value = 2
model_name = 'demo'
```

Then tag the cell with the `parameters` tag in the notebook UI or using nbformat.

- Keep parameter names simple and serializable (strings, numbers, dicts).
- Use relative paths for datasets (e.g., `jupyter/data/iris.csv`) and set working dir via `--cwd` when invoking papermill.

---

## Basic CLI usage

Run a notebook and write the executed copy to an `artifacts/` folder:

```bash
papermill jupyter/notebooks/quick_fuse.ipynb artifacts/quick_fuse.run.ipynb -p input_value 4 --kernel python3
```

Options of interest:
- `-p <name> <value>`: pass a parameter (repeatable)
- `--parameters-file <file.yml>`: pass a YAML/JSON file with many parameters
- `--cwd <path>`: run the notebook with a specific working directory
- `--kernel <name>`: force the kernel (e.g., `python3`)

Example using YAML:

```bash
papermill jupyter/notebooks/quick_fuse.ipynb artifacts/quick_fuse.run.ipynb --parameters-file run_params.yml
```

---

## Python API

```python
import papermill as pm

pm.execute_notebook(
    'jupyter/notebooks/quick_fuse.ipynb',
    'artifacts/quick_fuse.param_run.ipynb',
    parameters={'input_value': 4, 'model_name': 'ci-run'},
    kernel_name='python3',
    cwd='.'
)
```

Papermill returns the path to the executed notebook (and writes it), which you can open with `nbformat` and check outputs.

---

## Testing notebooks in CI (recommended pattern)

1. Use Papermill to execute the notebook into `artifacts/` in CI.
2. Use `nbformat` or `nbclient` to inspect outputs and assert expected results (e.g., a numeric result or presence of a success message).

Example pytest snippet to run & assert:

```python
import papermill as pm
import nbformat


def test_quick_fuse_runs(tmp_path):
    out = tmp_path / 'quick_fuse.out.ipynb'
    pm.execute_notebook('jupyter/notebooks/quick_fuse.ipynb', str(out), parameters={'input_value': 2})

    nb = nbformat.read(str(out), as_version=4)
    # collect stdout from executed cells
    outputs = []
    for cell in nb.cells:
        if cell.get('cell_type') != 'code':
            continue
        for out_item in cell.get('outputs', []):
            if out_item.get('output_type') == 'stream' and out_item.get('name') == 'stdout':
                outputs.append(out_item.get('text', ''))
    stdout_text = ''.join(outputs)
    assert '4.0' in stdout_text or 'All checks passed' in stdout_text
```

Notes:
- Mark tests as skipped when heavy runtime dependencies like `onnxruntime` aren't present (see existing tests pattern with `pytest.importorskip("onnxruntime")`).
- Use small deterministic parameters to keep CI fast.

---

## Using Papermill in Docker / CI (Fuse dev image)

We provide a dev image that includes a Jupyter server and our labextension. To execute notebooks inside the image:

- Build and run the image (either let Docker build the labextension via the multi-stage `docker/jupyter/Dockerfile` or run `scripts/build_labextension_and_image.sh` locally to produce `jupyter/labextensions/fuse/lib` and then run `make jupyter-image`). The final image copies built assets into `/fused/labextensions/fuse/lib` and does not contain Node/npm.
- Exec into the container and run papermill:

```bash
# inside container
python -m uv pip install papermill
papermill jupyter/notebooks/welcome.ipynb /tmp/welcome.executed.ipynb
```

In CI (GitHub Actions), add a job step:

```yaml
- name: Install papermill & execute notebooks
  run: |
    python -m uv pip install papermill
    papermill jupyter/notebooks/quick_fuse.ipynb artifacts/quick_fuse.ci.ipynb -p input_value 2
    papermill jupyter/notebooks/welcome.ipynb artifacts/welcome.ci.ipynb
```

Collect `artifacts/*.ipynb` as action artifacts for debugging.

---

## Advanced patterns

- Parameter sweep: implement a small loop or use a matrix job in CI to run multiple parameter sets and collect outputs.
- Hyperparameter jobs: combine Papermill with `dask` or job schedulers to orchestrate many notebook runs.
- Clean artifact naming: include timestamp and parameter hashes to avoid collisions: `artifacts/quick_fuse.input=2.20250202T1234.ipynb`.

---

## Security & best practices ⚠️
- Executing arbitrary notebooks runs code: only execute trusted notebooks or run them inside isolated containers.
- Use `--cwd` to restrict working directories and avoid executing in sensitive filesystem locations.
- Prefer CI runners or containerized environments with ephemeral storage when running notebook executions automatically.

---

## Small examples we can add to the repo (suggestions)
- `scripts/run_examples_papermill.sh` that executes a curated set of notebooks and stores outputs in `artifacts/`.
- A param file `jupyter/params/quick_fuse.yml` demonstrating common runs.
- A pytest that executes the welcome notebook via papermill and checks the health JSON via the welcome page.

---

If you'd like, I can add one or more of these examples (script, param file, and a pytest) to the repository and wire a small GitHub Actions step to save executed notebooks as artifacts. Which example should I add next? 👇
- Add `scripts/run_examples_papermill.sh` (recommended first)
- Add a sample `jupyter/params/quick_fuse.yml`
- Add an example pytest that uses papermill and is skipped in lightweight CI

