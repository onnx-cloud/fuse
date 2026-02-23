from src.lowering import FuseLowerer
from src.parser import fuse_parser


def test_train_weight_as_initializer_and_values():
    # Simulate a small model with a trainable weight declared with a default
    src = """
    @train weight W: f32[2,2] = [[1.0, 0.0], [0.0, 1.0]]
    node m(x: f32[2]) -> f32[2] { return MatMul(W, x) }
    """
    ast = fuse_parser.parse(src)
    fl = FuseLowerer()
    model = fl.lower(ast)
    inits = {i.name: i for i in model.graph.initializer}
    # Find initializer ending with 'W'
    found = [name for name in inits if name.endswith("W")]
    assert found, f"Expected initializer for W, got: {list(inits.keys())}"
    w_proto = inits[found[0]]
    # Value check asserted below via float_data/raw_data presence
    # We expect a 2x2 identity-like slice to be present as float values (at least one 1.0)
    has_float = any(v == 1.0 for v in (w_proto.float_data or []))
    has_raw = w_proto.raw_data and len(w_proto.raw_data) > 0
    assert has_float or has_raw
