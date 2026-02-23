from pathlib import Path

from src.parser import fuse_parser
from src.lowering import FuseLowerer


def test_tuple_return_lowering():
    p = Path("examples/golden/arithmetic.fuse")
    ast = fuse_parser.parse(p.read_text())
    fl = FuseLowerer()
    model = fl.lower(ast, source_file=str(p))
    assert model is not None
    # arithmetic.fuse returns two outputs (r1, r4) from the top-level graph
    assert len(model.graph.output) == 2
