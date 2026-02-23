from pathlib import Path
from src.metrics import compute_metrics_for_file


def test_weight_dims_expr_evaluated(tmp_path: Path):
    p = tmp_path / "weights.fuse"
    p.write_text('''param W_patch: f32[768, 3*32*32] = 0
model m() { return 0 }
''')
    m = compute_metrics_for_file(str(p))
    w = [w for w in m.get("weights", []) if w.get("name") == "W_patch"]
    assert len(w) == 1
    w0 = w[0]
    assert w0.get("elements") == 768 * (3 * 32 * 32)
    assert w0.get("bytes") == w0.get("elements") * 4
