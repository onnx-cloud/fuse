from pathlib import Path
from src.cli.commands import cmd_lint


def test_cmd_lint_includes_sanitizer_warnings(tmp_path: Path):
    p = tmp_path / "fuse.fuse"
    p.write_text('''node f(a: f32) -> f32 { 1.0 }
''')
    messages = cmd_lint([str(p)])
    warnings = [m for m in messages if m.get("kind") == "warning"]
    assert any("appears unused" in w.get("message").lower() for w in warnings)


def test_cmd_lint_reports_sanitizer_errors(tmp_path: Path):
    p = tmp_path / "fuse2.fuse"
    p.write_text('''node f(a: f32, a: f32) -> f32 { 1.0 }
''')
    messages = cmd_lint([str(p)])
    errors = [m for m in messages if m.get("kind") == "error"]
    assert any("duplicate parameter" in e.get("message").lower() or "duplicate declaration" in e.get("message").lower() for e in errors)


def test_cmd_lint_reports_invalid_meta_iri(tmp_path: Path):
    p = tmp_path / "badmeta.fuse"
    p.write_text('''@meta type = "not-an-iri"
node m() -> f32 { 0.0 }
''')
    messages = cmd_lint([str(p)])
    warnings = [m for m in messages if m.get("kind") == "warning"]
    assert any("@type" in w.get("message") and ("non-iri" in w.get("message").lower() or "non-iri/non-curie" in w.get("message").lower()) for w in warnings)


def test_cmd_lint_accepts_meta_curie(tmp_path: Path):
    p = tmp_path / "curie.fuse"
    p.write_text('''@meta type = "my:Thing"
node m() -> f32 { 0.0 }
''')
    messages = cmd_lint([str(p)])
    # Should not warn about invalid IRI/CURIE
    assert not any(m.get("kind") == "warning" and "@type" in m.get("message") for m in messages)


def test_cmd_lint_meta_strict_errors(tmp_path: Path):
    p = tmp_path / "badmeta.fuse"
    p.write_text('''@meta type = "not-an-iri"
node m() -> f32 { 0.0 }
''')
    messages = cmd_lint([str(p)], check_meta_strict=True)
    errors = [m for m in messages if m.get("kind") == "error"]
    assert any("@type" in e.get("message") and ("non-iri" in e.get("message").lower() or "non-iri/non-curie" in e.get("message").lower()) for e in errors)
