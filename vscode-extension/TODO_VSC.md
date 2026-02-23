# TODO — VS Code extension (vscode-extension) 🧩

Short, prioritized list of suggested improvements for the Fuse VS Code extension. Each item includes a brief rationale and a suggested implementation starting point.

## Summary

The extension currently provides syntax highlighting, snippets and a Python-based LSP with **diagnostics** and **hover** (see `src/lsp_server.py`). The README advertises more LSP features than are implemented. The items below are prioritized to get the most value with the least friction first.

## Quick wins ✅ (low effort)

1. **Fix package name typo** (critical)
   - Problem: `package.json` uses `"name": "onxx-fuse-dsl"` ( `onxx-fuse`).
   - Why: avoids confusion in marketplace and packaging.
   - Files: `vscode-extension/package.json`

2. **Align README with current implementation** (low)
   - Update `vscode-extension/README.md` so features list matches what's implemented (or mark planned features "(planned)").
   - Files: `vscode-extension/README.md`

3. **Add npm scripts** for development & CI (low)
   - Examples: `build`, `lint`, `test`, `vscode:prepublish`.
   - Benefit: standardizes contribution workflow and CI automation.

4. **Add a minimal test that asserts LSP fallback** (low)
   - Ensure extension picks bundled binary or Python fallback (`resolveServerCommand`) in CI.

## High-impact LSP features (medium effort)

1. **Document Symbols / Workspace Symbols** (medium)
   - Implement `textDocument/documentSymbol` and `workspace/symbol` using the parser/AST symbol table.
   - Benefit: enables outline, Go to Symbol, workspace search.

2. **Go-to Definition / References / Rename / Find References** (medium)
   - Implement location providers using resolved AST and name allocation logic (`name_allocator.py`).
   - Add tests validating symbol resolution across files.

3. **Completion & Signature Help** (medium)
   - Provide completions for known constructs (op names, model names, params) and signature help for function-like constructs.

4. **Code Actions & Quick Fixes** (medium)
   - Use analysis to offer fixes (e.g., remove unused attribute, convert const to param).

5. **Formatting & Formatting on Save** (low–medium)
   - Add `textDocument/formatting` using a stable formatting function (consider a `fuse.format` module or small formatter).

6. **Semantic tokens (semantic highlighting)** (medium)
   - Provide richer highlighting beyond the TextMate grammar.

7. **Folding ranges** (low)
   - Implement `textDocument/foldingRange` from AST blocks.

## Tests & CI (high ROI)

1. **Integration tests using the VS Code Test harness** (medium)
   - Create e2e tests that launch the extension (using `@vscode/test-electron`), open a `.fuse` document, and assert diagnostics, hover, go-to-def, and completions.
   - Add a CI job to run these tests on PRs.

2. **Unit tests for the LSP server** (low)
   - Add tests that start the `FuseLanguageServer` in-process (or simulated client) and assert diagnostics and hover outputs.

3. **Snapshot tests for grammar/semantic tokens** (low)
   - When grammar changes, ensure snapshots exist and are checked in (see `syntaxes/fuse.tmLanguage.json`).

## Packaging & Releases (medium)

1. **CI automation to build cross-platform `fuse-lsp` binary** and include it in the VSIX.
   - Integrate with existing `tools/package_vscode_extension.sh` and CI workflow (`.github/workflows/ci.yml`).

2. **Publish workflow for VSIX** (low)
   - Automate creation of `.vsix` and attach as release artifact; consider `vsce`/`@vscode/cli` for publish.

3. **Ensure consistent engine target** (low)
   - Update `engines.vscode` (currently `^1.60.0`) to a more modern baseline and confirm compatibility.

## UX, settings & commands (medium)

1. **Add actionable commands** (e.g., `Fuse: Compile to ONNX`, `Fuse: Run Example`, `Fuse: Show Graph`) and expose them in the command palette and explorer context menus.

2. **Improve activation events**
   - Add `onCommand` activations for heavy features; keep `onLanguage:fuse` for lightweight features.

3. **Add settings** beyond `languageServerCommand`:
   - `fuse.enableFormatting`, `fuse.hoverDetailLevel`, `fuse.telemetryEnabled` (opt-in only), `fuse.logLevel`.

4. **Logging & debug helpers**
   - Add a dev command to dump the server's symbol table or open a debug output channel.

## Docs & contributor experience (low–medium)

1. **Contributing guide for extension** (low)
   - How to run, test, bundle `fuse-lsp`, and debug the extension.

2. **Examples workspace**
   - Add a small `examples/` set demonstrating features, and include these in the extension tests.

3. **Changelog & release notes**
   - Keep extension-specific changelog on releases.

## Performance, reliability & security (higher effort)

1. **Incremental parsing & caching** (higher)
   - If parsing large files becomes slow, adopt incremental parsing or cache symbol tables.

2. **Timeouts & robust startup**
   - Add startup timeouts and friendly diagnostics if the server fails to start.

3. **Privacy & telemetry**
   - If collecting telemetry, make it explicit and opt-in. Prefer logging to console for developers.

## Nice-to-haves / Integrations (optional)

- Jupyter Notebook / VS Code Notebook integration (see `TODO_JUPYTER_KERNEL.md`) — LSP-based completion/hover inside notebooks.
- Web (vscode.dev) compatibility — ensure server fallback works with web (maybe via language client in web extension host).
- Model visualization integration (show ONNX graph) as a panel or webview.

## First recommended tasks (small, actionable order)

1. Fix the package name typo in `package.json`. (5–10m)
2. Update README to reflect implemented features. (10–20m)
3. Add `scripts` + a minimal unit test for server fallback. (30–60m)
4. Add one integrated e2e test for diagnostics + hover with `@vscode/test-electron`. (1–2d)

If you'd like, I can open a PR with the `TODO_VSC.md` file added to `vscode-extension/` and implement the typo fix + README alignment as a first small PR. ✅
