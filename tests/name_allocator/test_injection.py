from src.graph_context import GraphContext
from src.name_allocator import StableNameAllocator


def test_stable_allocator_names():
    na = StableNameAllocator(scope_prefix="ns", scope_display="ns.m")
    ctx = GraphContext(name="m", name_allocator=na)
    # first node uses scope_display
    n1 = ctx._next_node_name("Add")
    assert n1 == "ns.m"
    n2 = ctx._next_node_name("Mul")
    assert n2 == "ns__Mul_1" or n2 == "ns__Mul_1"  # consistent format
    # constants
    c1 = ctx._next_const_name()
    assert c1 == "ns__const_0"
    c2 = ctx._next_const_name()
    assert c2 == "ns__const_1"
