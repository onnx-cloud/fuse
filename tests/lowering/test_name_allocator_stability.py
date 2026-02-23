from src.lowering import FuseLowerer
from src.name_allocator import StableNameAllocator
from src.parser import fuse_parser


def test_lowering_is_deterministic_with_allocator():
    src = """
    node mm(a: f32[2,3], b: f32[3,4]) -> f32[2,4] {
      return a @ b
    }
    """
    ast = fuse_parser.parse(src)
    na1 = StableNameAllocator(scope_prefix="ns", scope_display="ns.m")
    model1 = FuseLowerer().lower(ast, name_allocator=na1)

    na2 = StableNameAllocator(scope_prefix="ns", scope_display="ns.m")
    model2 = FuseLowerer().lower(ast, name_allocator=na2)

    # Models should be structurally similar in their node naming
    names1 = [n.name for n in model1.graph.node]
    names2 = [n.name for n in model2.graph.node]
    assert names1 == names2
