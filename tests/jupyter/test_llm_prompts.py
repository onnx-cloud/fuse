import os
from src.jupyter.server import _load_llm_config, _render_prompt


def test_render_prompt_file_replaces_placeholders(monkeypatch):
    # Use the shipped jupyter config which references a prompt path with placeholders
    monkeypatch.setenv('FUSE_LLM_CONFIG', os.path.join(os.path.dirname(__file__), '..', '..', 'jupyter', 'config', 'llm_config.json'))
    cfg = _load_llm_config()
    engine_cfg = cfg.get('llm', {}).get('think')
    assert engine_cfg is not None
    prompt_spec = engine_cfg.get('prompt')
    rendered = _render_prompt(prompt_spec)
    # Should include fenced EBNF and the terse example (MatMul from terse.fuse)
    assert '```fuse' in rendered
    assert 'MatMul' in rendered or 'node dense' in rendered


def test_render_prompt_literal_returns_literal():
    txt = 'You are a helpful assistant.'
    assert _render_prompt(txt) == txt
