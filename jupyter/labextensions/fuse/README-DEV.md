Development notes for Fuse JupyterLab extension

Ideas to complete the extension:
- Build steps: use TypeScript and Rollup or Webpack to produce an ES module bundle.
- Register the extension in `package.json` with `jupyterlab` entry points and publish to npm or include via federated extension.
- Implement a CompletionProvider that adapts the server response into the format expected by JupyterLab's completion API.
- Implement the ErrorCard widget to POST to `/fuse/api/map_error` and render the returned structured error data.
