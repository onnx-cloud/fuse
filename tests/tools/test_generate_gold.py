import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "generate_gold.py"
EBNF = ROOT / "docs" / "ebnf.md"
SCHEMA_GOLD = ROOT / "tests" / "parsing" / "golden" / "fuse.ast.schema.json"


def test_generate_gold_executes(tmp_path):
    # Run script
    res = subprocess.run(
        [str(SCRIPT)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert res.returncode == 0, f"Script failed: {res.stderr}"
    assert EBNF.exists(), "EBNF output not created"
    assert SCHEMA_GOLD.exists(), "Schema gold not created"
    # basic checks
    ebnf_txt = EBNF.read_text()
    assert "GRAMMAR" not in ebnf_txt, "EBNF should contain only grammar body"
    # Should include an appended example snippet header and the terse example node
    assert (
        "## Example: examples/golden/terse.fuse" in ebnf_txt
    ), "EBNF should include terse.fuse example"
    assert "node dense(" in ebnf_txt, "terse example should include 'node dense'"
    data = SCHEMA_GOLD.read_text()
    assert data.strip().startswith("{") and data.strip().endswith(
        "}"
    ), "Schema file not valid JSON-ish"
