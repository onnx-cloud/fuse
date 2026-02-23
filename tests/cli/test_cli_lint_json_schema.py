import json
import pytest
from pathlib import Path

from src.cli import cli_dispatch

try:
    import jsonschema
except Exception:  # pragma: no cover - dev env may not have jsonschema
    jsonschema = None


@pytest.mark.skipif(
    not Path(__file__).resolve().parents[2].joinpath("src/cli/lint_schema.json").exists(),
    reason="lint schema not present",
)
def test_cli_lint_json_conforms_to_schema(tmp_path, capsys):
    # write a small fuse file that will trigger a sanitizer warning (missing module)
    p = tmp_path / "bad.fuse"
    p.write_text('''
    node foo(a: f32) -> f32 {
      return a
    }
    ''')

    args = type("A", (), {"command": "lint", "f": [str(p)], "json": True, "fail_on_warn": False, "check_training": False})
    rc = cli_dispatch.dispatch(args)
    assert rc == 0

    out = capsys.readouterr().out
    data = json.loads(out)

    # Basic schema sanity: top-level messages array and at least one message
    assert "messages" in data
    assert isinstance(data["messages"], list)
    assert len(data["messages"]) >= 1

    m = data["messages"][0]
    assert "kind" in m and m["kind"] in ("warning", "error")
    assert "message" in m and isinstance(m["message"], str)

    # If jsonschema is available, validate against the canonical schema
    schema_path = Path(__file__).resolve().parents[2].joinpath("src/cli/lint_schema.json")
    if jsonschema and schema_path.exists():
        schema = json.loads(schema_path.read_text())
        # This will raise a ValidationError if invalid
        jsonschema.validate(instance=data, schema=schema)
