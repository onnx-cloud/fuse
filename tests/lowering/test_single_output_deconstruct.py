import pytest
from src.parser import fuse_parser
from src.lowering.main import FuseLowerer
from src.graph_context import GraphContext
from src.lowering.utils import LoweringError

def test_single_output_deconstruct():
    source = """
    @domain test
    
    fn single_return(a: f32[1]) -> f32[1] {
        return a
    }

    @proof graph my_proof() {
        a1: f32[1] = [1]
        // Error case: unpacking single output into multiple targets
        r1, r2 = single_return(a1)
        return r1
    }
    """
    
    ast = fuse_parser.parse(source)
    ctx = GraphContext()
    lowerer = FuseLowerer()
    with pytest.raises(LoweringError, match="attempt to select non-zero index from single-output call"):
        lowerer.lower(ast, ctx)

