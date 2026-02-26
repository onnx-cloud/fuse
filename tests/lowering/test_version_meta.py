import onnx
from src.lowering import FuseLowerer
from src.parser import fuse_parser


def test_version_meta_is_preserved():
    # use the current project version so patch bumps don't break the test
    from src.util.project_version import get_project_version
    version = get_project_version()
    src = f"""
    @version {version}
    node id(x: f32) -> f32 {{ return x }}
    """
    ast = fuse_parser.parse(src)
    lowerer = FuseLowerer()
    model = lowerer.lower(ast)

    # Ensure metadata prop 'version' is present and equals the declared value
    metas = {p.key: p.value for p in model.metadata_props}
    assert metas.get("version") == version

