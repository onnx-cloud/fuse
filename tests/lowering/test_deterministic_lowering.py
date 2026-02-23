from src.lowering import FuseLowerer
from src.name_allocator import StableNameAllocator
from src.parser import fuse_parser


def test_lowering_node_and_initializer_names_are_deterministic():
    src = """
    const C: f32 = 1.0
    node m(x: f32[2]) -> f32[2] {
        y = Add(x, C)
        return y
    }
    """
    ast = fuse_parser.parse(src)
    fl1 = FuseLowerer()
    m1 = fl1.lower(ast, name_allocator=StableNameAllocator())
    names1 = [n.name for n in m1.graph.node]
    inits1 = [i.name for i in m1.graph.initializer]

    fl2 = FuseLowerer()
    m2 = fl2.lower(ast, name_allocator=StableNameAllocator())
    names2 = [n.name for n in m2.graph.node]
    inits2 = [i.name for i in m2.graph.initializer]

    assert names1 == names2
    assert inits1 == inits2
