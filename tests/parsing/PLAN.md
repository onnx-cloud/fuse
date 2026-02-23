Parsing tests — PLAN

Goals

- Validate full grammar coverage and corner cases from `SPEC.md`.
- Ensure parser emits stable AST nodes for identical input (idempotence).
- Catch regressions in tokenization and EBNF changes.

Scope

- Unit tests for lexer/tokenizer: whitespace, comments, identifiers, literals.
- Parser tests: each top-level production and many combined constructs.
- Error tests: assert clear error messages and location spans for invalid inputs.

Approach

- Table-driven tests mapping source snippets → expected parse-tree skeletons (not full serialization).
- Use small, focused snippets rather than whole files for grammar coverage.
- Add fuzz-style property tests that confirm parse(tree) → pretty-print → parse(tree) idempotence for the grammar subset.

Fixtures & DI

- Provide a `parse()` test helper that accepts an injected `ParserConfig` (e.g., `allow_experimental`, `domain`), not global flags.
- Inject an alternate `ErrorReporter` to capture and assert errors without raising.
- Keep token stream and AST transformer separate; test them independently.

Test artifacts

- Golden token lists and minimal parse-tree JSON for representative inputs stored under `./`.

Files to consult

- `SPEC.md` (grammar)
- `src/parser.py` (lexer + parser + AST transformer)