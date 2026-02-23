# CONFIG.md

The CLI accepts a `--config` argument. When supplied, Fuse will attempt to read the JSON file, validate it against `schemas/fuse.config.schema.json` (if `jsonschema` is installed), and merge values into the parsed CLI arguments. CLI flags always take precedence over values in the config file.

## Structure

Top-level keys are grouped by command (or global), for example:

- `global`: Global CLI-level settings
- `onnx`, `graphviz`, `inspect`, `run`, `lint`, `verify`, `golden`, `models`, `version`, `completion`: Per-command default options
- `vscode`: VS Code extension settings

All fields are optional; missing fields fall back to the CLI defaults.

## Global options

- `verbose`: integer (0 = normal, 1 = -v, 2+ = -vv)
- `quiet`: boolean
- `version`: boolean

## Command options

### onnx
- `files`: array of strings
- `out_dir`: string | null
- `refresh_cache`: boolean
- `refresh_import`: string | null
- `folds`: integer (default 8)
- `externalize`: integer (default 0)
- `external_dir`: string | null
- `preserve_external`: boolean
- `wasm`: boolean

### dot/graphviz
- `files`: array
- `dot`: string | null
- `out_dir`: string | null
- `name_pattern`: string | null
- `filter`: string | null (regex)
- `rankdir`: string (default `LR`)
- `force`: boolean
- `dry_run`: boolean

### inspect
- `files`: array
- `out_dir`: string | null
- `dot`: boolean
- `svg`: boolean
- `png`: boolean
- `interactive`: boolean
- `plots`: boolean
- `filter`: string | null
- `force`: boolean
- `dry_run`: boolean

### run
- `files`: array
- `input_path`: string | null
- `output`: string | null
- `entry`: string | null
- `provider`: string | null

### lint
- `files`: array
- `fail_on_warn`: boolean
- `check_remote`: boolean

### verify
- `files`: array

### golden
- `files`: array
- `quiet`: boolean
- `fail_fast`: boolean

### models
- `path`: string | null
- `root`: string | null
- `refresh_cache`: boolean
- `refresh_import`: string | null
- `externalize`: integer
- `manifest_only`: boolean
- `manifest_dir`: string | null
- `overwrite`: boolean
- `variant`: string | null
- `metadata`: string | null

### version
- `short`: boolean
- `json`: boolean

### completion
- `shell`: string (one of `bash|zsh|fish`)

## VS Code extension settings

- `fuse.languageServerCommand`: string — override for the Fuse language server command. Empty string uses the bundled `fuse-lsp` when available. See `vscode-extension/package.json`.

## Example usage

- `fuse --config schemas/fuse.config.example.json onnx`
- `fuse --config schemas/fuse.config.example.json run` (with defaults taken from file)

## Notes & TODO

- The project now automatically reads and applies `--config` (merge precedence: CLI flags > config). Validation is attempted if `jsonschema` is available and `schemas/fuse.config.schema.json` exists. Add `jsonschema` to your dev environment to enable strict validation (`uv pip install -r requirements-dev.txt`).

Thank you! ✅
