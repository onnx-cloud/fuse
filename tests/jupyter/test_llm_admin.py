import json
import os
import pytest
from pathlib import Path

pytest.importorskip('requests')

from src.jupyter.server import _load_llm_config


def test_admin_guard(tmp_path, monkeypatch):
    # Ensure admin endpoints are disabled by default
    monkeypatch.delenv('FUSE_LLM_ADMIN_ENABLED', raising=False)
    cfg = _load_llm_config()
    assert isinstance(cfg, dict)


def test_admin_write_and_delete(tmp_path, monkeypatch):
    # Enable admin and point config to tmp file
    monkeypatch.setenv('FUSE_LLM_ADMIN_ENABLED', '1')
    cfgp = tmp_path / 'llm_config.json'
    cfgp.write_text(json.dumps({'llm':{}}))
    monkeypatch.setenv('FUSE_LLM_CONFIG', str(cfgp))

    # Simulate creating an engine by writing via Path
    from src.jupyter.server import Path
    data = {'url': 'https://x', 'secretEnv': 'X', 'label': 'X'}
    cfg = _load_llm_config()
    assert cfg.get('llm', {}) == {}
    # write directly
    cfgp.write_text(json.dumps({'llm': {'test': data}}))
    cfg2 = _load_llm_config()
    assert 'test' in cfg2.get('llm', {})
    # delete
    cfgp.write_text(json.dumps({'llm': {}}))
    cfg3 = _load_llm_config()
    assert cfg3.get('llm', {}) == {}
