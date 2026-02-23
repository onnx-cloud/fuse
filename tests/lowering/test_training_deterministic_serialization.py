from src.parser import fuse_parser
from src.lowering import FuseLowerer
from src.name_allocator import StableNameAllocator


def test_training_serialization_is_deterministic():
    src = open("examples/golden/training.fuse").read()
    ast = fuse_parser.parse(src)

    fl1 = FuseLowerer(emit_training=True)
    m1 = fl1.lower(ast, name_allocator=StableNameAllocator())

    fl2 = FuseLowerer(emit_training=True)
    m2 = fl2.lower(ast, name_allocator=StableNameAllocator())

    b1 = m1.SerializeToString(deterministic=True)
    b2 = m2.SerializeToString(deterministic=True)

    assert b1 == b2
    # Ensure training_info present
    assert len(m1.training_info) >= 1
    assert len(m2.training_info) >= 1
