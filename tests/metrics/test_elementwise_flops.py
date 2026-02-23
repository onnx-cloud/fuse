from pathlib import Path
from src.metrics import compute_metrics_for_file


def test_mul_elementwise_flops(tmp_path: Path):
    p = tmp_path / "mul.fuse"
    p.write_text('''fn f(a: f32[2,3], b: f32[2,3]) -> f32[2,3] {
  return Mul(a, b)
}
''')
    m = compute_metrics_for_file(str(p))
    # find the Mul node
    muls = [n for n in m.get("per_node", []) if n.get("op") == "Mul"]
    assert len(muls) == 1
    assert muls[0].get("flops") == 2 * 3  or muls[0].get("flops") == 6


def test_mul_broadcast_flops(tmp_path: Path):
    p = tmp_path / "mulb.fuse"
    p.write_text('''fn f(a: f32[2,3], b: f32[3]) -> f32[2,3] {
  return Mul(a, b)
}
''')
    m = compute_metrics_for_file(str(p))
    muls = [n for n in m.get("per_node", []) if n.get("op") == "Mul"]
    assert len(muls) == 1
    assert muls[0].get("flops") == 6
