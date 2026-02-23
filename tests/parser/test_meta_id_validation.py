import pytest
from src.parser import fuse_parser


def test_meta_id_rejects_invalid_values():
    bad = '@id "not-an-iri"\nnode n() { return 0 }\n'
    with pytest.raises(Exception):
        fuse_parser.parse(bad)

    # valid absolute IRI accepted
    ok = '@id "https://example.org/myid"\nnode n() { return 0 }\n'
    ast = fuse_parser.parse(ok)
    assert any(d.get("type") == "node" for d in ast)

    # valid CURIE accepted
    ok2 = '@id "ex:myid"\nnode n() { return 0 }\n'
    ast2 = fuse_parser.parse(ok2)
    assert any(d.get("type") == "node" for d in ast2)
