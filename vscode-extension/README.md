# Fuse VS Code Extension

Provides syntax highlighting, snippets, and LSP integration for the Fuse ONNX cognitive compiler.

## Development

1. Prepare: `cd vscode-extension && npm install`
2. Debug: Press **F5** from `vscode-extension/` folder to launch Extension Development Host
3. Package: `../tools/package_vscode_extension.sh` (builds bundled LSP binary and packages)

## LSP Features

- **Diagnostics** — parsing and linker errors
- **Hover** — function/model/parameter signatures
- **Go to Definition** — jump to symbols
- **Document Symbols** — workspace symbol search
- **References & Rename** — AST-powered precise refactoring
- **Code Actions** — quick-fix attribute removal

## Runtime

The extension uses a bundled `fuse-lsp` binary (built from `src/lsp_server.py` via PyInstaller). During development, it falls back to `python -m src.lsp_server`.

## Maintenance

When grammar changes: update `syntaxes/fuse.tmLanguage.json` and add corresponding test examples.
