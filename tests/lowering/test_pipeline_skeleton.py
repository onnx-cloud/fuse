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




def test_typeshape_pass():
    from src.lowering.passes import TypeShapePass

    ast = [{"type": "node", "name": "x"}]
    typed = TypeShapePass().run(ast)
    assert typed[0].get("__typed__") is True




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
