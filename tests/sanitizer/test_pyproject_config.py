import json
from pathlib import Path
import os

from src.sanitizer import sanitize_ast


def test_pyproject_disable_training_checks(tmp_path, monkeypatch):
    # Create a fake project root with pyproject.toml containing disable flag
    pr = tmp_path
    py = pr / "pyproject.toml"
    py.write_text('[tool.fuse.sanitizer]\nenable_training_state_checks = false\n')

    # Monkeypatch the repo root locator used by sanitizer
    import src.sanitizer as sanitizer_mod

    # monkeypatch repo_root helper
    monkeypatch.setattr(sanitizer_mod, "repo_root", lambda: pr)

    # reload config (module cached) by forcing load via function
    monkeypatch.setattr(sanitizer_mod, "load_sanitizer_config", lambda: {"enable_training_state_checks": False})

    ast = [
        {"type": "meta", "name": "fuse.training", "value": {"optimizer": "Adam"}},
        {"type": "param", "name": "W", "trainable": True},
    ]

    res = sanitize_ast(ast)
    warnings = res.get("warnings", [])
    # With checks disabled, there should be no TRAIN.MISSING_STATE
    assert not any(w.get("code") == "TRAIN.MISSING_STATE" for w in warnings)


def test_env_override_config(tmp_path, monkeypatch):
    cfg = tmp_path / "san.toml"
    cfg.write_text('[tool.fuse.sanitizer]\nenable_training_state_checks = false\n')
    os.environ["FUSE_SANITIZER_CONFIG"] = str(cfg)

    ast = [
        {"type": "meta", "name": "fuse.training", "value": {"optimizer": "Adam"}},
        {"type": "param", "name": "W", "trainable": True},
    ]

    res = sanitize_ast(ast)
    warnings = res.get("warnings", [])
    assert not any(w.get("code") == "TRAIN.MISSING_STATE" for w in warnings)

    # cleanup
    del os.environ["FUSE_SANITIZER_CONFIG"]
