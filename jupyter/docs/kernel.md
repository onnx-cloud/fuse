# Jupyter Kernel Design — In-Memory Compilation + Python Expressions

Requirements
------------
Functional
- Cells can contain Fuse source and be compiled in-memory into an ONNX ModelProto.
- Cells can contain normal Python and execute arbitrarily (with access to kernel-provided helper objects).
- Bindings: Python -> Fuse inputs (e.g., numpy arrays) and Fuse params may be set from Python variables.
- Running: ability to run the compiled model and return numpy arrays, summaries, and visualization-ready objects.
- Magics: `%fuse.compile`, `%%fuse` (cell magic), `%fuse.run`, `%fuse.show`, `%fuse.clear`.
- Deterministic naming and stable metadata for reproducibility and tests.
- Clear error reporting that maps compile/runtime errors back to Fuse source lines.
- Fast incremental compilation for short edit-compile-run cycles.

Architecture overview
---------------------
High-level components
- Kernel process (a standard Jupyter kernel using ipykernel): handles execution requests.
- Fuse in-memory compiler: wraps existing `FuseLowerer` and `GraphContext` to produce an `onnx.ModelProto` in memory.
- Execution backend: default to ONNX Runtime (ORT) but pluggable (e.g., ORT, onnxruntime-training, backend stubs for testing).
- Session state manager: holds named compiled models, parameter values, and caches to support incremental recompilation.
- Dockerfile that launches Jupter + Fuse fully workingjobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build image
        run: docker build -t fuse-jupyter:ci .
      - name: Run smoke notebook
        run: docker run --rm -v $GITHUB_WORKSPACE:/workspace fuse-jupyter:ci bash -lc "python -m nbclient jupyter/notebooks/quick_fuse.ipynb"

Execution model
---------------
1. Cell magic `%%fuse` or `%fuse.compile my_model` parses cell content as Fuse source. The kernel:
   - Parses and normalizes using existing parser utilities.
   - Lowers to GraphContext + ONNX ModelProto in memory (no files by default).
   - Stores the compiled model under a session name and exposes a `Model` wrapper object in Python.
2. Python cells can reference compiled `Model` objects and call `model.run(inputs=..., params=..., backend='ort')`.
3. Data flow between Python and Fuse types:
   - Provide converters: numpy <-> ONNX tensor; shape/dtype checking; helpful error messages on mismatches.
4. Incremental edit/compile: model registry tracks source/AST hash; `compile` only rebuilds when sources change.
5. add a heuristics mode to auto-detect @ffuse DSL  when cell lacks valid Python

Kernel API (Python)
--------------------
Expose a small library object `fuse_kernel` available automatically inside the kernel namespace.
Example usage:

Magics
------
- `%%fuse [name]` — compile cell content into model named `name` (or default name). Returns compile summary or error trace.
- `%fuse.compile [name_or_path]` — compile a file or named model programmatically and return the `Model` wrapper.
- `%fuse.run [name_or_path] [--input <json-or-py-expr>] [--provider <runtime>]` — run a named/compiled model (or source file) and return outputs; supports inline JSON or Python expressions for inputs.
- `%fuse.show [name_or_path]` — quick visualization (Graphviz/HTML) or text summary of a compiled model or source file.
- `%fuse.clear [name_or_path]` — remove a compiled model from session cache.

CLI command magics (mapping) 🔁
- `%fuse.lint [paths] [--fail-on-warn]` — run `fuse lint` on given paths and return structured lint messages.
- `%fuse.verify [paths]` — run `fuse verify` to check Fuse file compatibility and report errors/warnings.
- `%fuse.onnx [paths] [--out-dir <dir>] [--training]` — export models to ONNX (mirrors `fuse onnx` options like `--bake`, `--externalize`, `--seal`).
- `%fuse.run_cli [paths] [--input path.npz] ` — run the `fuse run` command to execute a source file end-to-end and return outputs.
- `%fuse.dot [paths] [--dot-dir <dir>] [--render]` — produce DOT and optional SVG/PNG artifacts using `fuse dot`.
- `%fuse.docs [paths] [--out-dir <dir>] [--md]` — generate documentation artifacts for sources or ONNX models.
- `%fuse.metrics [paths]` — compute and return model metrics (YAML-like strings or structured dicts).
- `%fuse.models [paths] [--publish]` — process model files and optionally publish manifests (maps to `fuse models`).
- `%fuse.zoo [op ...]` — wrapper around `fuse zoo` operations (`list`, `publish`, `show`).

Notes:
- Magics accept keyword-style arguments and forward them to the corresponding `src.cli.commands` functions; results are returned as Python objects (lists/dicts) and printed for convenience.
- For commands that produce files (docs, graphviz, inspect, onnx), magics return a list of generated file paths and can optionally provide rich HTML/graphical outputs using `IPython.display`.
- Errors should capture and display friendly diagnostics (parse/lowering traces) and include clickable links to generated artifacts when available.

Design details & trade-offs
---------------------------
1. Cell detection strategy
   - Use explicit magics for clarity (`%%fuse` and plain cells remain Python). This keeps semantics unambiguous and minimizes surprises.
2. In-memory compilation and caching
   - Key on a deterministic digest of the canonicalized source (parser -> AST -> normalized IR); reuse compiled ModelProto when digest matches.
   - Maintain a small LRU cache and keep the last N models for quick re-run .

3. Python expression support
   - Rely on ipykernel to evaluate Python cells normally.
   - Provide the `Model` wrapper with intuitive Python API (run, run_async, to_onnx, show).
   - Ensure that users can set parameters from Python (e.g., model.params['w'] = numpy.array(...)). Changes to params should not require full recompilation unless param shapes/types differ.

4. Error mapping
   - Capture exceptions from parsing/lowering and return traceback-like messages that include source line:col ranges.
   - For runtime errors from ONNX Runtime, translate into user-friendly messages including fn names and input shapes.

5. Security
   - Kernel runs arbitrary Python code (standard Jupyter model); recommend users run untrusted notebooks in isolated environments.

6. Extensibility
   - Backends: implement a backend interface (run_model) to allow different runtimes and hardware accelerators.
   - Visualizations: `model.show()` can provide Graphviz or an HTML widget using `IPython.display`.

Implementation plan & milestones
-------------------------------
- Implement simple `ipykernel` subclass that recognizes `%%fuse` and compiles cell to ONNX in memory using existing `FuseLowerer`.
- Expose `Model` object with `.run(inputs)` that uses onnxruntime.InferenceSession.
- Add a minimal example notebook in `examples/` and a smoke test.
- Add caching, incremental compilation, improved error messages.
- Implement magics `%fuse.run`, `%fuse.show`, `%fuse.clear`.
- Add unit tests for kernel compile/run semantics and integration tests using pytest + a headless jupyter client.
- Add `setup.py`/`pyproject` extras: `fuse[jupyter]` that installs `ipykernel`, `jupyter-client`, and `onnxruntime`.
- Write docs in `docs/` and add notebook examples in `examples/` (cookbook style).

Testing strategy
----------------
- Unit tests for the compile API: feed Fuse sources and assert ModelProto contents and deterministic metadata.
- Integration tests that start the kernel (e.g., using `jupyter_client` or `nbclient`), send a `%%fuse` cell, then run Python cells that call `model.run()` and assert outputs.
- Add golden notebooks under `examples/golden/` and `make gold` checks to ensure model outputs remain stable.

Dependencies
------------
- dev: `ipykernel`, `jupyter_client`, `nbclient` (for testing), `notebook` (optional for manual testing)
- runtime: `onnx`, `onnxruntime` (or pluggable backends)
- existing repo: use `src/fuse` modules (`FuseLowerer`, `GraphContext`, parsers)

Decisions
--------------
- we use ipykernel — much lower work.
- For parameter persistence, we support a `@persistence` fuse DSL pragma

Security/Operational Notes
--------------------------
- Kernel runs untrusted code; recommend documentation advising use of isolated environments or ephemeral containers when running third-party notebooks.
- Keep default behavior non-persistent (no writing external data) unless user opts in.

Appendix: Minimal interactive example
------------------------------------
In a notebook:

```text
# Cell 1 (Fuse)
@fuse 
@domain my_mlp
# simple model source ...
graph demo(x: f32[1,3]) {}

# Cell 2 (Python)
from fused import get_model
m = get_model('my_mlp.demo')
import numpy as np
out = m.run({'x': np.ones((1,3), dtype=np.float32)})
print(out)
```
