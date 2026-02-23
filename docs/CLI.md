# Fuse CLI Reference

```
fuse [-h] [-v] [-q] [--config CONFIG] [--version] [--strict]
     {verify,lint,compile,ttl,run,graphviz,inspect,docs,ebnf,golden,completion,models,version} ...
```

| Option | Description |
|--------|-------------|
| `-h, --help` | Show help message and exit |
| `-v, --verbose` | Increase verbosity (use `-vv` for more) |
| `-q, --quiet` | Suppress non-error output |
| `--config CONFIG` | Path to config file (JSON) |
| `--version` | Print package version and exit |
| `--strict` | Enable strict validation (fail on invalid metadata such as non-semantic `@version`) |

## Commands

### `verify`

Verify Fuse files for compatibility with the installed Fuse manifest.

```bash
fuse verify [-h] [-f [FILES ...]]
```

| Option | Description |
|--------|-------------|
| `-f, --files` | Fuse source files or glob patterns |

**Example:**
```bash
fuse verify -f examples/**/*.fuse
```

### `lint`

Lint Fuse source files for errors and warnings.

```bash
fuse lint [-h] [-f [FILES ...]] [--fail-on-warn] [--check-remote] [--check-training] [--json]
```

| Option | Description |
|--------|-------------|
| `-f, --files` | Fuse source files or glob patterns |
| `--fail-on-warn` | Treat warnings as errors |
| `--check-remote` | Validate remote import URLs |
| `--check-training` | Run additional training metadata checks |
| `--json` | Output machine-readable JSON (validates against `lint_schema.json`) |

**Example:**
```bash
fuse lint -f src/*.fuse --fail-on-warn
fuse lint -f examples/golden/*.fuse --json --check-training
```

### `compile`

Compile Fuse source files to ONNX format.

```bash
fuse compile [-h] [-f [FILES ...]] [-o OUT] [--refresh-cache] [--import IMPORT]
             [--folds N] [--externalize SIZE] [--external-dir DIR] [--preserve-external]
             [--bake] [--wasm] [--compact] [--training]
             [--tf] [--tfl] [--pt]
             [-S] [--seal-algo {blake3,sha256,sha1}] [--seal-inits {none,merkle,per-init,full}]
             [--seal-include-external] [--seal-force]
             [--ttl [PATH]] [--ttl-ns PREFIX] [--ttl-ns-uri URI]
```

| Option | Description |
|--------|-------------|
| `-f, --files` | Fuse source files or glob patterns |
| `-o, --out` | Output directory |
| `--refresh-cache` | Force refresh of cached imports |
| `--import` | Refresh specific import |
| `--folds` | Constant folding passes (default: 8) |
| `--externalize` | Externalize initializers larger than N bytes |
| `--external-dir` | Directory for external tensor data |
| `--preserve-external` | Keep existing external data references |
| `--bake` | Embed external/imported tensor data into `.onnx` initializers |
| `--wasm` | Optimize output for WASM runtimes |
| `--compact` | Emit compact model (suppress initial identity node) |
| `--training` | Emit training metadata when present |
| `--proto` | Emit text-format protobufs for generated models (excludes initializers) |

**Export Targets:**

| Option | Description |
|--------|-------------|
| `--tf` | Export TensorFlow SavedModel alongside ONNX
| `--tfl` | Export TensorFlow Lite (`.tflite`) alongside ONNX|
| `--pt` | Export PyTorch `.pt` file alongside ONNX |

**Sealing Options (deterministic hashing):**

| Option | Description |
|--------|-------------|
| `-S, --seal` | Embed deterministic seal into ModelProto metadata |
| `--seal-algo` | Hash algorithm: `blake3` (default), `sha256`, or `sha1` |
| `--seal-inits` | Initializer inclusion mode: `none`, `merkle` (default), `per-init`, `full` |
| `--seal-include-external` | Include external initializer contents when computing seal |
| `--seal-force` | Replace existing seal if present |

**TTL/RDF Export:**

| Option | Description |
|--------|-------------|
| `--ttl [PATH]` | Export RDF/Turtle alongside ONNX (optionally specify output path) |
| `--ttl-ns` | User namespace prefix for TTL export (e.g., `my:`) |
| `--ttl-ns-uri` | User namespace URI for TTL export (e.g., `https://example.org/#`) |

**Example:**
```bash
fuse compile -f model.fuse -o ./out
fuse compile -f model.fuse --training --seal -S --seal-algo blake3
fuse compile -f model.fuse --externalize 1024 --external-dir ./weights
fuse compile -f model.fuse --ttl --ttl-ns "ex:" --ttl-ns-uri "https://example.org/#"
# Emit documentation artifacts (Markdown/TTL/DOT/AST) for compiled models
fuse compile -f model.fuse --docs --proto -o ./out  # include .proto outputs (text-format)
```

### `ttl`

Convert ONNX model(s) to RDF/Turtle format (standalone command).

```bash
fuse ttl [-h] [-f [FILES ...]] [-o OUT] [--ns PREFIX] [--ns-uri URI]
         [--no-initializers] [--no-metadata]
```

| Option | Description |
|--------|-------------|
| `-f, --files` | ONNX files to convert |
| `-o, --out` | Output file or directory |
| `--ns` | User namespace prefix (e.g., `my:`) |
| `--ns-uri` | User namespace URI (e.g., `https://example.org/#`) |
| `--no-initializers` | Exclude initializer details from output |
| `--no-metadata` | Exclude model metadata from output |

**Example:**
```bash
fuse ttl -f model.onnx -o model.ttl --ns "ex:" --ns-uri "https://example.org/#"
```

### `run`

Run inference on a compiled ONNX model.

```bash
fuse run [-h] [-f [FILES ...]] [--input PATH] [--output PATH] [--entry NAME] [--provider PROV]
```

| Option | Description |
|--------|-------------|
| `-f, --files` | Fuse source files (compiled to ONNX on-the-fly) |
| `--input` | Path to input data (JSON) |
| `--output` | Path to write output (JSON) |
| `--entry` | Entry function name |
| `--provider` | ONNX Runtime execution provider |

**Example:**
```bash
fuse run -f model.fuse --input data.json --output result.json
```

### `graphviz`

Generate Graphviz visualizations from Fuse source files.

```bash
fuse dot [-h] [-f [FILES ...]] [--dot DIR] [--out-dir DIR]
              [--name-pattern PATTERN] [--filter REGEX]
              [--rankdir {LR,TB,RL,BT}] [--force] [--dry-run]
```

| Option | Description |
|--------|-------------|
| `-f, --files` | Fuse source files or glob patterns |
| `--dot` | Output directory for DOT files |
| `--render` | Attempt to render DOT to SVG/PNG (safe; failures produce `.error.txt`) |
| `--out-dir` | Common output directory |
| `--name-pattern` | Filename pattern for outputs |
| `--filter` | Regex filter for graph nodes |
| `--rankdir` | Graph direction: `LR` (default), `TB`, `RL`, `BT` |
| `--force` | Overwrite existing files |
| `--dry-run` | Show what would be generated without writing |

**Example:**
```bash
fuse dot -f model.fuse --dot ./graphs --rankdir TB
```

### `inspect`


Inspect ONNX models and emit artifacts (AST, `.fuse`, DOT, TTL, metadata).

### `docs`

Generate documentation artifacts (Markdown, TTL, DOT, AST) from Fuse source or ONNX models.

```bash
fuse docs -f model.fuse --md --dot --ast --ttl --proto -o ./docs  # emit .proto (text-format) for models as well (excludes initializers)
```

| Option | Description |
|--------|-------------|
| `-f, --files` | Fuse source or ONNX files to document |
| `-o, --out` | Output directory (per-file subdirs created) |
| `--md` | Generate Markdown using `src/template/fuse.md` |
| `--md-template` | Path to a custom MD template |
| `--ttl` | Generate TTL (RDF/Turtle) from compiled ONNX |
| `--dot` | Generate Graphviz DOT |
| `--ast` | Emit AST JSON and compact AST |
| `--proto` | Emit protobuf (text-format) for models, excluding initializers |
| `--render` | Attempt to render DOT to SVG/PNG (best-effort) |
| `--force` | Overwrite existing output directories |
| `--dry-run` | Show what would be generated without writing |


### `decompile` / `audit`

Decompile an ONNX model to a best-effort Fuse wrapper and write canonical artifacts (Fuse source, AST, optional protobuf without initializers).

```bash
fuse decompile -f model.onnx --out ./out --fuse --ast --proto
# alias: `fuse audit` is equivalent to `fuse decompile`
```

### `ebnf`

Emit the Fuse runtime EBNF grammar as Markdown. The output mirrors the
content emitted by `scripts/generate_gold.py`: a header, a fenced
```fuse``` block with the grammar body, and an appended terse example
from `examples/golden/terse.fuse` when available.

```bash
fuse ebnf [-h] [--out PATH] [--asts PATH]
```

| Option | Description |
|--------|-------------|
| `--out` | Write EBNF Markdown to `PATH` instead of printing to stdout |
| `--asts` | Write canonical AST schema JSON to `PATH` (copies `schemas/fuse.ast.schema.json`) |

**Examples:**
```bash
# Print grammar to stdout
fuse ebnf

# Write EBNF Markdown to a file
fuse ebnf --out docs/ebnf.md

# Write canonical AST schema to a file
fuse ebnf --asts schemas/fuse.ast.schema.json

# Do both at once
fuse ebnf --out docs/ebnf.md --asts schemas/fuse.ast.schema.json
```

```bash
fuse inspect [-h] [-f [FILES ...]] [-o OUT] [--dot] [--ttl]
             [--interactive] [--plots] [--filter REGEX] [--force] [--dry-run]
```

| Option | Description |
|--------|-------------|
| `-f, --files` | ONNX files to inspect |
| `-o, --out` | Output directory |
| `--dot` | Generate DOT graph output |
| `--render` | Attempt to render DOT to SVG/PNG (safe; failures produce `.error.txt`) |
| `--ttl` | Generate TTL (RDF/Turtle) output |
| `--interactive` | Launch interactive inspection mode |
| `--plots` | Generate weight distribution plots |
| `--filter` | Regex filter for nodes |
| `--force` | Overwrite existing files |
| `--dry-run` | Show what would be generated without writing |

**Example:**
```bash
fuse inspect -f model.onnx -o ./inspect_out --dot --plots
```

### `golden`

Run golden tests against Fuse source files with `@golden` blocks.

```bash
fuse golden [-h] [-f [FILES ...]] [--quiet] [--fail-fast]
```

| Option | Description |
|--------|-------------|
| `-f, --files` | Files or glob patterns to include |
| `--quiet` | Suppress per-file output |
| `--fail-fast` | Stop on first failure |

**Example:**
```bash
fuse golden -f examples/golden/*.fuse
fuse golden -f examples/golden/*.fuse --fail-fast
```

### `models`

Manage and publish Fuse models to a local zoo.

```bash
fuse models [-h] [--path PATH] [--root ROOT] [--refresh_cache]
            [--refresh_import IMPORT] [--externalize SIZE]
            [--manifest-only] [--manifest-dir DIR] [--overwrite]
            [--variant NAME] [--metadata JSON]
```

| Option | Description |
|--------|-------------|
| `--path` | Path to Fuse file or directory |
| `--root` | Root directory for the local zoo |
| `--refresh_cache` | Force refresh of cached imports |
| `--refresh_import` | Refresh specific import |
| `--externalize` | Externalize initializers larger than N bytes |
| `--manifest-only` | Generate manifest JSON only (don't publish) |
| `--manifest-dir` | Directory for manifest files |
| `--overwrite` | Overwrite existing published models |
| `--variant` | Variant name |
| `--metadata` | Additional metadata (JSON string) |

**Example:**
```bash
fuse models --path ./models --root ./zoo --manifest-only --manifest-dir ./manifests
```

### `version`

Print package version information.

```bash
fuse version [-h] [--short] [--json]
```

| Option | Description |
|--------|-------------|
| `--short` | Print short version string only |
| `--json` | Output JSON: `{"version": ..., "build_time": ...}` |

**Example:**
```bash
fuse version
fuse version --json
```

### `completion`

Print shell completion helper for bash, zsh, or fish.

```bash
fuse completion [-h] [{bash,zsh,fish}]
```

| Argument | Description |
|----------|-------------|
| `shell` | Shell type: `bash` (default), `zsh`, or `fish` |

**Example:**
```bash
# Add to your shell profile:
eval "$(fuse completion bash)"
```

## Configuration File

The CLI supports a JSON configuration file via `--config`. The schema is defined in `fuse.config.schema.json`.

**Example (`fuse.config.json`):**
```json
{
  "global": {
    "verbose": 1
  },
  "compile": {
    "out_dir": "./build",
    "folds": 8,
    "externalize": 1024
  },
  "lint": {
    "fail_on_warn": true
  }
}
```

Usage:
```bash
fuse --config fuse.config.json onnx -f model.fuse
```

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Command-specific failure (e.g., lint errors, compilation errors) |
| `2` | Invalid usage or unknown command |

## See Also

- [SPEC.md](../SPEC.md) — Language specification
- [CONFIG.md](../CONFIG.md) — Configuration reference
- [TRAINING.md](../TRAINING.md) — Training metadata guide
- [SEALED_MODELS.md](../SEALED_MODELS.md) — Model sealing and verification
- [EXTERNAL_DATA.md](../EXTERNAL_DATA.md) — External data handling
- [RDFS.md](../RDFS.md) — RDF/Turtle export format
