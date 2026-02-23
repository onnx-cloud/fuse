import os
import json
import time
from src.jupyter.server import list_engines, _llm_rate, _LLM_RATE_LIMIT, _llm_audit_path


def test_list_engines_reads_example(tmp_path, monkeypatch):
    # Ensure list_engines returns keys from example config
    p = tmp_path / 'llm_config.json'
    cfg = {"llm": {"think":{"url":"https://x","secretEnv":"X"}, "deep": {"url":"https://y","secretEnv":"Y"}}}
    p.write_text(json.dumps(cfg))
    monkeypatch.setenv('FUSE_LLM_CONFIG', str(p))
    engines = list_engines()
    assert 'think' in engines and 'deep' in engines


def test_rate_limiting_and_audit(tmp_path, monkeypatch):
    # Clear counters
    _llm_rate.clear()
    ip = '1.2.3.4'
    # simulate hitting the limit
    for _ in range(_LLM_RATE_LIMIT):
        ok, rem = (lambda ip: (True, _LLM_RATE_LIMIT - len(_llm_rate.get(ip, []))))(ip)
        _llm_rate.setdefault(ip, []).append(time.time())
    # next call should be over limit
    # call rate function
    from src.jupyter.server import _rate_ok, _audit_log
    ok, rem = _rate_ok(ip)
    assert ok is False
    # write an audit log entry
    monkeypatch.setenv('FUSE_LLM_CONFIG', str(tmp_path / 'llm_config.json'))
    _audit_log(ip, 'think', {'messages': [{'role':'user','content':'hi'}]}, 200)
    assert _llm_audit_path.exists()
    lines = _llm_audit_path.read_text().strip().splitlines()
    assert len(lines) >= 1
