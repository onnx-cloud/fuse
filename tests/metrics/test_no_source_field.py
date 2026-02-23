from pathlib import Path
from src.metrics import compute_metrics_for_file


def test_compute_metrics_does_not_include_source(tmp_path: Path):
    p = tmp_path / "f.fuse"
    p.write_text('''node f(x: f32[1]) -> f32[1] { x }\n''')
    m = compute_metrics_for_file(str(p))
    assert "source" not in m
