from src.metrics import compute_metrics_for_file, format_metrics
from pathlib import Path


def test_format_does_not_include_source(tmp_path: Path):
    p = tmp_path / "simple.fuse"
    p.write_text('''node f(x: f32[1]) -> f32[1] { x }\n''')
    m = compute_metrics_for_file(str(p))
    s = format_metrics(m)
    assert "source:" not in s
