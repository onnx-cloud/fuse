from src.parser import fuse_parser
from src.lowering import FuseLowerer


def _first_node_type_and_name(model):
    nodes = model.graph.node
    if not nodes:
        return None, None
    return nodes[0].op_type, nodes[0].name


def test_entry_identity_emitted_by_default():
    from tests.test_utils import project_fuse_version
    FUSE_DECL = f"@fuse {project_fuse_version()}\n"
    src = FUSE_DECL + """
    @domain mymod
    graph g(x: f32[2]) -> f32[2] {
      y = MatMul(x, x)
      Add(y, x)
    }
    """
    ast = fuse_parser.parse(src)
    model = FuseLowerer().lower(ast)
    op, name = _first_node_type_and_name(model)
    assert op == "Identity"
    assert name == "mymod.g"


def test_compact_suppresses_entry_identity():
    src = """
    @domain mymod
    graph g(x: f32[2]) -> f32[2] {
      y = MatMul(x, x)
      Add(y, x)
    }
    """
    ast = fuse_parser.parse(src)
    fl = FuseLowerer()
    model = fl.lower(ast, compact=True)
    op, name = _first_node_type_and_name(model)
    # In compact mode we should not see the module-qualified Identity as the first node
    assert not (op == "Identity" and name == "mymod.g")