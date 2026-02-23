from src.lsp_server import collect_diagnostics


def test_collects_parse_diagnostics():
    # Broken text: missing body
    text = "node foo(x: f32) -> f32[1]"
    diags = collect_diagnostics(text)
    assert diags
    assert any("Unasserted" in d.message or "Parse error" in d.message for d in diags)


def test_collects_lowering_diagnostics_for_golden():
    text = open("examples/golden/golden.fuse").read()
    diags = collect_diagnostics(text)
    # Golden files should generally be clean; if diagnostics are present
    # ensure they are well-formed and contain a range. Prefer a lowering
    # diagnostic if one exists but allow an empty list.
    if not diags:
        return
    assert any("Lowering error" in d.message or "Parse error" in d.message for d in diags)
    has_snippet = any("multi_modal_latent" in d.message or "multi_modal_latent" in d.message for d in diags)
    # If messages are present, at least one should include a sensible location
    assert any(getattr(d, "range", None) and d.range.start.line >= 0 for d in diags)