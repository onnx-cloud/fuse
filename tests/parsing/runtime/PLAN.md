Runtime tests — PLAN

Goals

- Validate runtime artifacts (ORT Web bundles or ORT runner integration) for correctness and reproducible packaging.

Scope

- Unit tests for packaging helpers: manifest generation, `model.json`, and runtime asset lookups.
- Integration smoke tests that validate exported `model.onnx` using `onnx.checker` and verify `model.json` points to correct WASM assets.

Approach

- Mock ORT Web asset availability with an `ORTAssetProvider` test double.
- For heavier tests that run inference, mark them slow and run optionally (CI can opt-in).

Fixtures & DI

- Inject `ORTAssetProvider` and `BundleWriter` so tests use in-memory writer or temp dirs.

Files to consult

- `src/ort_web_install.py`, `scripts/install_ort_web.sh`