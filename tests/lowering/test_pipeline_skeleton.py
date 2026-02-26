import logging
from src.lowering.passes import lower_ast


def test_pipeline_noop():
    # a trivial AST (empty module) should pass through the pipeline without error
    ast = {}
    class DummyBuilder:
        def __init__(self):
            self.events = []
    builder = DummyBuilder()
    lower_ast(ast, builder)
    # builder remains unchanged since passes are no-ops
    assert hasattr(builder, "events")


def test_normalization_pass():
    from src.lowering.passes import NormalizationPass

    ast = [{"type": "meta", "name": "module", "value": "foo"}]
    normed = NormalizationPass().run(ast)
    assert normed[0]["name"] == "domain"


def test_typeshape_pass():
    from src.lowering.passes import TypeShapePass

    ast = [{"type": "node", "name": "x"}]
    typed = TypeShapePass().run(ast)
    assert typed[0].get("__typed__") is True


def test_lower_emits_module_deprecation_warning(caplog):
    # build an AST directly so the parser doesn't normalize the module meta
    from src.lowering import FuseLowerer

    from src.util.project_version import get_project_version
    version = get_project_version()
    ast = [
        {"type": "meta", "name": "fuse", "value": version},
        {"type": "meta", "name": "module", "value": "olddom"},
        {
            "type": "node",
            "name": "n",
            "params": [],
            "ret_type": "f32",
            "body": [{"return": 0.0}],
            "@id": None,
        },
    ]
    fl = FuseLowerer()
    import pytest
    with pytest.warns(UserWarning) as rec:
        model = fl.lower(ast)
    assert any("deprecated" in str(w.message) for w in rec)


def test_typeshape_literal_inference():
    from src.lowering.passes import TypeShapePass
    ast = {"lit": 42}
    numbered = TypeShapePass().run(ast)
    assert numbered.get("type", {}).get("scalar") == "i64"

    ast2 = {"lit": 3.14}
    dec = TypeShapePass().run(ast2)
    assert dec.get("type", {}).get("scalar") == "f32"

    ast3 = {"lit": True}
    boolv = TypeShapePass().run(ast3)
    assert boolv.get("type", {}).get("scalar") == "bool"
