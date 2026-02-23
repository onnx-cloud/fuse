import onnx
from src.parser import fuse_parser
from src.lowering import FuseLowerer
from src.name_allocator import StableNameAllocator


def test_name_allocator_snapshot_and_invariants():
    src = """
    node mm(a: f32[2,3], b: f32[3,4]) -> f32[2,4] {
      t1 = a @ b
      t2 = t1 + a
      t3 = t2 @ b
      return t3
    }
    """
    ast = fuse_parser.parse(src)

    na1 = StableNameAllocator(scope_prefix="ns", scope_display="ns.m")
    m1 = FuseLowerer().lower(ast, name_allocator=na1)

    names = [n.name for n in m1.graph.node]

    # Invariants: unique names (no collisions)
    assert len(names) == len(set(names))

    # Snapshot: lock the current deterministic naming so regressions are explicit
    expected = ["ns__MatMul_1", "ns__Add_3", "ns__MatMul_5"]
    assert names == expected, f"Node name snapshot mismatch: {names}"

    # Stable ordering across independent allocators
    na2 = StableNameAllocator(scope_prefix="ns", scope_display="ns.m")
    m2 = FuseLowerer().lower(ast, name_allocator=na2)
    assert [n.name for n in m2.graph.node] == names

    # Serialized ONNX artifacts preserve these node names
    b = m1.SerializeToString()
    parsed = onnx.ModelProto()
    parsed.ParseFromString(b)
    assert [n.name for n in parsed.graph.node] == names
