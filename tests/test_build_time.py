import importlib
from pathlib import Path


def test_build_time_prefers_build_file(tmp_path: Path, monkeypatch):
    # Create a temporary src/_build_time.txt and ensure src.__build_time__ reads it
    build_file = Path("src") / "_build_time.txt"

    # write a deterministic timestamp
    ts = "2000-01-02T03:04:05Z"
    build_file.write_text(ts)

    # reload src to pick up the build time file
    import src
    importlib.reload(src)

    assert getattr(src, "__build_time__", None) == ts

    # cleanup
    build_file.unlink()
    importlib.reload(src)  # cleanup import state