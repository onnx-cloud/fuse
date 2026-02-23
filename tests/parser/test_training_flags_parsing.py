import pytest
pytest.importorskip("lark")
from src.parser import fuse_parser


def test_training_flags_parsed():
    src = """
    @training { optimizer = Adam, lr = 1e-3, lr_input = true, step_input = true }
    node m(x: f32[2]) -> f32[2] { y = MatMul(x, x); y }
    """
    ast = fuse_parser.parse(src)
    metas = [d for d in ast if isinstance(d, dict) and d.get('type') == 'meta' and d.get('name') == 'fuse.training']
    assert metas
    cfg = metas[0]['value']
    assert cfg.get('lr_input') is True
    assert cfg.get('step_input') is True
