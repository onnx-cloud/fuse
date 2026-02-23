import hashlib
from pathlib import Path


def test_opcodes_json_checksum_is_unchanged():
    """Prevent accidental edits to the golden `src/OpCodes.json` by asserting its checksum.

    If you intentionally update `src/OpCodes.json` (e.g., to sync with upstream ONNX
    changes), update the expected checksum below.
    """
    p = Path(__file__).resolve().parents[1] / "src" / "OpCodes.json"
    data = p.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    expected = "db603fe50c928838788a19259b743c198cae85a5f6c548decc6cff8bdbd3a11e"
    assert (
        digest == expected
    ), (
        "src/OpCodes.json has changed. If this is intentional, update the test's"
        " expected checksum. Otherwise revert the change."
    )
