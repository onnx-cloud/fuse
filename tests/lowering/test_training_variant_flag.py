from src.parser import fuse_parser
from src.lowering import FuseLowerer


def _meta_dict(model):
    return {kv.key: kv.value for kv in model.metadata_props}


def test_emit_training_sets_metadata():
    src = """
    node m(x: f32) -> f32 { return x }
    """
    ast = fuse_parser.parse(src)
    fl = FuseLowerer(emit_training=True)
    model = fl.lower(ast)
    md = _meta_dict(model)
    assert "training" in md
    # Booleans are stringified in metadata props; accept typical truthy forms
    assert md["training"] in ("True", "true", "1")


def test_emit_training_default_false():
    ast = fuse_parser.parse("node m(x: f32) -> f32 { return x }")
    fl = FuseLowerer()
    model = fl.lower(ast)
    md = _meta_dict(model)
    assert "training" not in md
