# Extending Fuse Jupyter: Autocomplete & UI Integration ✅

This document describes the pieces added to support:
 - Structured JSON endpoints for ops/completions/attributes/error mapping
 - A small scaffold for a JupyterLab frontend extension (see `jupyter/labextensions/`)
 - A simple project scaffolder script to create notebook templates under `projects/`

Files of interest:
- `src/jupyter/server.py` — Provides `get_ops()`, `completions(prefix)`, `get_op_attributes(name)`, and `map_error()` plus optional Tornado request handlers registered under `/fuse/api/...`.
- `src/jupyter/introspection.py` — Uses ONNX schemas when available for richer attribute/type information.
- `scripts/scaffold_project.py` — CLI to create a small project skeleton from the template notebook.

Frontend scaffold:
- `jupyter/labextensions/fuse` — contains a minimal package.json and a placeholder JS extension that can be developed to call the server endpoints and provide completions or an error card UI.

Build & image notes:
- The Jupyter image build uses a **multi-stage Docker build**. A `node-builder` stage runs `npm ci` + `npm run build` and the resulting `lib/` bundle is copied into the final Python image under `/fused/labextensions/fuse/lib`.
- The final image intentionally does **not** include fn or npm; the built assets are copied in. If you prefer to build assets locally and have Docker pick them up, run `scripts/build_labextension_and_image.sh` first (this creates `jupyter/labextensions/fuse/lib`) and then run `make jupyter-image` — the Dockerfile will copy local artifacts into the image if present.
- CI includes a check that the built labextension artifacts exist in the final image (see `.github/workflows/jupyter-image-smoke.yml`).

How to enable server endpoints:
1. Start Jupyter Server/Lab normally. If Tornado and jupyter server components are available, the extension will register endpoints under `/fuse/api/*`.
2. Verify: `curl <server>/fuse/api/ops` should return a JSON array (or run the tests below).

Next steps for a full UX:
- Implement a full completer provider using the JupyterLab Completion API and call `/fuse/api/completions` for suggestions (returns JSON array).
- Implement an ErrorCard React/Phosphor widget that POSTs exception details to `/fuse/api/map_error` and renders a formatted card.
- Add a small CI job that builds and (optionally) tests the labextension bundle (we added a presence check for the built artifacts).
