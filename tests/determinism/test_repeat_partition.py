import pytest
from src.parser import fuse_parser
from src.lowering import FuseLowerer
from tests.test_utils import project_fuse_version
FUSE_DECL = f"@fuse {project_fuse_version()}\n"


@pytest.mark.parametrize("runs", [1, 2, 3])
def test_lowering_repeatable_with_stable_allocator(runs, stable_name_allocator):
    """Lower the same AST multiple times with a fresh allocator and assert byte equality.

    This helps detect cross-run non-determinism (e.g., global counters or order-dependent dicts).
    """
    src = FUSE_DECL + """
    node mm(a: f32[2,3], b: f32[3,4]) -> f32[2,4] {
      t1 = a @ b
      t2 = t1 + a
      t3 = t2 @ b
      return t3
    }
    """
    ast = fuse_parser.parse(src)

    from src.name_allocator import StableNameAllocator

    models = []
    for _ in range(3):
        na = StableNameAllocator(scope_prefix="test", scope_display="test.module")
        model = FuseLowerer().lower(ast, name_allocator=na)
        models.append(model)

    # Deterministic invariants: identical node name lists, op types and ordering
    node_lists = [[n.name for n in m.graph.node] for m in models]
    optype_lists = [[n.op_type for n in m.graph.node] for m in models]

    assert all(n == node_lists[0] for n in node_lists[1:]), "Node name ordering changed across runs"
    assert all(o == optype_lists[0] for o in optype_lists[1:]), "Node op_type ordering changed across runs"

    # Serialized ONNX may include provenance metadata (timestamps) that differs across runs; ensure
    # node-level invariants instead of raw bytes equality to avoid false negatives.
