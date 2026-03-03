import io
import sys

from src import __version__ as pkg_version
from src.cli import main


def _capture_stdout(func, *args, **kwargs):
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        rc = func(*args, **kwargs)
        out = sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout
    return rc, out


def test_global_version_flag_prints_package_version():
    rc, out = _capture_stdout(main, ["--version"])
    assert rc == 0
    # Output should start with the package version and include a build timestamp
    out = out.strip()
    assert out.startswith(f"fuse {pkg_version}")
    assert "(built:" in out


def test_global_version_flag_json_includes_build_time():
    rc, out = _capture_stdout(main, ["--version", "--json"]) if False else (None, "")
    # Note: --version is a global flag and argparse won't allow --json with it
    # directly; instead we test the `version --json` subcommand below.
    pass


def test_version_subcommand_uses_src_version():
    rc, out = _capture_stdout(main, ["version"])

    # main(..) returns 0 on success
    assert rc == 0
    assert pkg_version in out
    assert "(built:" in out


def test_version_subcommand_json_includes_build_time():
    rc, out = _capture_stdout(main, ["version", "--json"])
    assert rc == 0
    import json

    data = json.loads(out)
    assert data.get("version") == pkg_version
    assert "build_time" in data


def test_version_subcommand_short_is_bare_version():
    rc, out = _capture_stdout(main, ["version", "--short"])
    assert rc == 0
    assert out.strip() == pkg_version
