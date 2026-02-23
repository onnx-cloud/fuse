import json
from pathlib import Path

import onnx
from onnx import helper
from src.cli.helpers import (
    find_fuse_files,
    parse_fuse_file,
    save_json,
    save_onnx,
)


def test_find_fuse_files_dir(tmp_path: Path):
    a = tmp_path / "a.fuse"
    b = tmp_path / "b.fuse"
    other = tmp_path / "readme.txt"
    a.write_text("node a() {}")
    b.write_text("node b() {}")
    other.write_text("ignore")

    got = find_fuse_files(str(tmp_path))
    assert isinstance(got, list)
    assert str(a) in got and str(b) in got
    # ensure ordering is deterministic (sorted)
    assert got == sorted(got)


def test_parse_fuse_file(tmp_path: Path):
    f = tmp_path / "one.fuse"
    f.write_text("node one() -> f32 { return 1.0 }")
    ast = parse_fuse_file(str(f))
    assert isinstance(ast, list)
    assert any(
        (
            isinstance(d, dict)
            and d.get("type") == "node"
            and d.get("name") == "one"
        )
        for d in ast
    )


def test_save_json(tmp_path: Path):
    dest = tmp_path / "out" / "file.json"
    save_json({"a": 1}, str(dest))
    assert dest.exists()
    data = json.loads(dest.read_text())
    assert data == {"a": 1}


def make_model_with_external(src_path: str, dest_name: str):
    # make a minimal ONNX model and attach external_files metadata pointing
    # to src_path with dest filename `dest_name`.
    graph = helper.make_graph([], "g", [], [])
    model = helper.make_model(graph)
    files = [{"src": str(src_path), "dest": dest_name, "init_name": "x"}]
    proto = onnx.onnx_pb.StringStringEntryProto(
        key="external_files", value=json.dumps(files)
    )
    model.metadata_props.append(proto)
    return model


def test_save_onnx_copies_external(tmp_path: Path):
    src = tmp_path / "data.bin"
    src.write_bytes(b"\x00\x01\x02")
    model = make_model_with_external(str(src), "data.bin")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_path = out_dir / "m.onnx"

    save_onnx(model, str(out_path))

    copied = out_dir / "data.bin"
    assert copied.exists()
    assert copied.read_bytes() == src.read_bytes()
