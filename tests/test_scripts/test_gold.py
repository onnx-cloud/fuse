import subprocess
import sys
from pathlib import Path

from scripts import gold


def make_fake_file(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_text("dummy")
    return p


def test_successful_compilation(monkeypatch, tmp_path, capsys):
    # two fake fuse files
    files = [make_fake_file(tmp_path, "a.fuse"), make_fake_file(tmp_path, "b.fuse")]

    # capture any runs and create corresponding .onnx
    def fake_run(cmd, capture_output=False, text=False):
        # invoked with '-f' <fusefile>
        idx = cmd.index("-f")
        fuse_path = Path(cmd[idx + 1])
        out_dir = Path(cmd[cmd.index("-o") + 1])
        dest = out_dir / (fuse_path.stem + ".onnx")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"ok")
        class Dummy:
            returncode = 0
            stdout = ""
            stderr = ""
        return Dummy()

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = gold.main([
        "--out-dir",
        str(tmp_path / "onnx"),
        "--files",
        str(files[0]),
        "--files",
        str(files[1]),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "compiled successfully" in out


def test_trace_flag(monkeypatch, tmp_path, capsys):
    fuse = make_fake_file(tmp_path, "t.fuse")

    # trace mode uses subprocess.call
    def fake_call(cmd):
        print("running compile")
        # create the expected output file so compile_file succeeds
        od = Path(cmd[cmd.index("-o") + 1])
        fuse_path = Path(cmd[cmd.index("-f") + 1])
        dest = od / (fuse_path.stem + ".onnx")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"ok")
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)

    rc = gold.main([
        "--trace",
        "--out-dir",
        str(tmp_path / "onnx"),
        "--files",
        str(fuse),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "--- Running: compile t.fuse" in out
    assert "running compile" in out


def test_compile_error(monkeypatch, tmp_path, capsys):
    bad = make_fake_file(tmp_path, "bad.fuse")
    good = make_fake_file(tmp_path, "good.fuse")

    def fake_run(cmd, capture_output=False, text=False):
        fuse = cmd[cmd.index("-f") + 1]
        class Dummy:
            pass
        if "bad.fuse" in fuse:
            Dummy.returncode = 5
            Dummy.stdout = ""
            Dummy.stderr = "oops"
        else:
            out_dir = Path(cmd[cmd.index("-o") + 1])
            Path(out_dir / (Path(fuse).stem + ".onnx")).write_bytes(b"ok")
            Dummy.returncode = 0
            Dummy.stdout = ""
            Dummy.stderr = ""
        return Dummy()

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = gold.main([
        "--out-dir",
        str(tmp_path / "onnx"),
        "--files",
        str(bad),
        "--files",
        str(good),
    ])
    assert rc != 0
    out = capsys.readouterr().out
    assert "ERROR: step 'compile bad.fuse' failed" in out


def test_missing_output(monkeypatch, tmp_path, capsys):
    fuse = make_fake_file(tmp_path, "x.fuse")
    od = tmp_path / "onnx"
    od.mkdir()

    def fake_run(cmd, capture_output=False, text=False):
        # pretend success; ensure no onnx file remains
        fuse_path = Path(cmd[cmd.index("-f") + 1])
        dest = od / (fuse_path.stem + ".onnx")
        if dest.exists():
            dest.unlink()
        class Dummy:
            returncode = 0
            stdout = ""
            stderr = ""
        return Dummy()

    monkeypatch.setattr(subprocess, "run", fake_run)

    rc = gold.main([
        "--out-dir",
        str(od),
        "--files",
        str(fuse),
    ])
    assert rc != 0
    out = capsys.readouterr().out
    assert "missing" in out.lower()
