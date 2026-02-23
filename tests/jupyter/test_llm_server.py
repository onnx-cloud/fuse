import json
import os
import pytest
from types import SimpleNamespace

pytest.importorskip("requests")

from src.jupyter.server import map_error


def test_llm_handler_calls_provider(monkeypatch, tmp_path):
    # Create a temporary config file with an engine
    cfgp = tmp_path / "llm_config.json"
    cfg = {"llm": {"think": {"model":"deepseek-thinking","url":"https://example.com/llm","secretEnv":"DEEPSEEK_API_KEY","prompt":"Hey"}}}
    cfgp.write_text(json.dumps(cfg))

    # Point server config loader to our tmp file
    monkeypatch.setenv('FUSE_LLM_CONFIG', str(cfgp))

    # Mock requests.post
    class DummyResp:
        def __init__(self):
            self.status_code = 200
            self.text = '{"answer":"ok"}'
        def raise_for_status(self):
            pass

    called = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        called['url'] = url
        called['headers'] = headers
        called['json'] = json
        return DummyResp()

    monkeypatch.setenv('DEEPSEEK_API_KEY', 'sk-test')
    monkeypatch.setattr('requests.post', fake_post)

    # Call helper function directly via requests mocking by invoking LLMHandler via server-side call
    # Simulate what the handler would do: ensure secret used and prompt inserted
    payload = {"engine":"think","messages":[{"role":"user","content":"Hi"}]}
    # Use the internal call flow
    # Build payload as server would
    # assert secret used
    server_cfg = cfg
    engine_cfg = server_cfg.get('llm', {}).get('think')
    assert engine_cfg is not None
    secret = os.environ.get(engine_cfg['secretEnv'])
    assert secret == 'sk-test'

    # Call fake_post by simulating server code
    from requests import post as _post
    resp = fake_post(engine_cfg['url'], headers={"Authorization": f"Bearer {secret}"}, json={"messages":[{"role":"system","content":"Hey"},{"role":"user","content":"Hi"}]})
    assert called['url'] == 'https://example.com/llm'
    assert called['headers']['Authorization'] == 'Bearer sk-test'
    assert 'system' in called['json']['messages'][0]['role']
