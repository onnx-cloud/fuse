import json
from pathlib import Path

import pytest


def test_config_merge_applies(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"global": {"verbose": 1}, "compile": {"folds": 4}}))

    def fake_dispatch(args):
        # merged values should be applied
        assert getattr(args, "folds") == 4
        assert getattr(args, "verbose") == 1
        return 0

    monkeypatch.setattr("src.cli.cli_dispatch.dispatch", fake_dispatch)
    from src.cli import main

    rc = main(["--config", str(cfg), "compile"])
    assert rc == 0


def test_cli_overrides_config(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"compile": {"folds": 4}}))

    def fake_dispatch(args):
        # CLI flag should override config
        assert getattr(args, "folds") == 7
        return 0

    monkeypatch.setattr("src.cli.cli_dispatch.dispatch", fake_dispatch)
    from src.cli import main

    rc = main(["--config", str(cfg), "compile", "--folds", "7"])
    assert rc == 0
