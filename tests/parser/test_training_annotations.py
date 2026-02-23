import pytest
pytest.importorskip("lark")
from src.parser import fuse_parser


def test_parser_recognizes_loss_and_algorithm_decorators():
    src = """
    @loss
    node loss_fn(x: f32[2]) -> f32[2] { return x }

    @algorithm
    node my_alg(W: f32[2], g: f32[2]) -> f32[2] { return W }
    """
    ast = fuse_parser.parse(src)
    # Find declarations by name
    names = {d.get('name'): d for d in ast if isinstance(d, dict) and d.get('name')}
    assert 'loss_fn' in names and names['loss_fn'].get('loss')
    assert 'my_alg' in names and names['my_alg'].get('algorithm')
