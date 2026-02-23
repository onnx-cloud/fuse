import onnx
from src.lowering import FuseLowerer
from src.parser import fuse_parser


def test_version_meta_is_preserved():
    src = """
    @version 0.7.0
    node id(x: f32) -> f32 { return x }
    """
    ast = fuse_parser.parse(src)
    lowerer = FuseLowerer()
    model = lowerer.lower(ast)

    # Ensure metadata prop 'version' is present and equals the declared value
    metas = {p.key: p.value for p in model.metadata_props}
    assert metas.get("version") == "0.7.0"