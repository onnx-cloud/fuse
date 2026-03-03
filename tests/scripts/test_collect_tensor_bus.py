import pytest

pytest.importorskip("lark")

from scripts.collect_tensor_bus import collect


def test_collect_transitive_callee_exposed(tmp_path):
    src = tmp_path / "mod.fuse"
    src.write_text('''@domain mymod

model encode_text(P_txt: f32[77,512]) {
  return P_txt
}

model clip_demo() {
  x = encode_text()
  return x
}
''')
    bus = collect(src)
    # exact keys for callee and replicated under caller
    assert "mymod.encode_text.P_txt" in bus
    assert "mymod.clip_demo.P_txt" in bus
    assert bus["mymod.clip_demo.P_txt"]["name"] == "P_txt"
    assert bus["mymod.clip_demo.P_txt"]["tensor"] == bus["mymod.encode_text.P_txt"]["tensor"]
