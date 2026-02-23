import json
from pathlib import Path


def test_llm_config_has_expected_structure():
    p = Path(__file__).resolve().parents[2] / "jupyter" / "config" / "llm_config.json"
    assert p.exists(), "llm_config.json should exist"
    cfg = json.loads(p.read_text())
    assert "llm" in cfg and isinstance(cfg["llm"], dict)
    # Ensure each entry has required keys and label startswith @fuse
    for name, entry in cfg["llm"].items():
        assert isinstance(entry, dict), f"Entry {name} must be an object"
        for k in ("model", "url", "secretEnv", "prompt", "label"):
            assert k in entry, f"{name} missing key: {k}"
        assert entry["label"].startswith("@fuse"), f"LLM label should start with '@fuse' (entry {name})"
