Fuse JupyterLab extension scaffold

This folder contains a minimal scaffold for a JupyterLab extension that can
call the server endpoints exposed at `/fuse/api/*`.

To develop:
 - Implement the frontend (preferably TypeScript + React) as a standard
   JupyterLab extension.
 - Use the server endpoints: `/fuse/api/ops`, `/fuse/api/completions`, `/fuse/api/op_attributes`, `/fuse/api/map_error`.
 - Local build: run `scripts/build_labextension_and_image.sh` to run `npm ci` and `npm run build` locally — this produces `jupyter/labextensions/fuse/lib`.
 - Image build: `make jupyter-image` runs a Docker multi-stage build that will either build the extension inside a `node-builder` stage or copy your local `lib/` artifacts into the final image if present.
 - CI: our smoke workflow now verifies the built labextension artifacts exist in the final image; add further packaging tests as needed.
