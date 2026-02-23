from src.metrics import compute_metrics_for_file, format_metrics
from pathlib import Path


def test_format_includes_totals(tmp_path: Path):
    p = tmp_path / "weights.fuse"
    # simple model with one param of 100 elements
    p.write_text('''param W: f32[100] = 0.0\nnode f(x: f32[1]) -> f32[1] { x }\n''')
    m = compute_metrics_for_file(str(p))
    s = format_metrics(m)

    assert "metrics:" in s
    assert "total_nodes:" in s
    assert "total_parameters:" in s
    assert "total_bytes_moved:" in s
    assert "total_bytes:" in s
    assert "total_flops:" in s
