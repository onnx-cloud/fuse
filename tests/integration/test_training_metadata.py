import json

from src.lowering import FuseLowerer
from src.parser import fuse_parser


def _find_meta_val(model, key):
    for e in model.metadata_props:
        if e.key == key:
            try:
                return json.loads(e.value)
            except Exception:
                return e.value
    return None


def test_train_and_frozen_propagate_to_model_metadata():
    src = """
    @train weight W: f32[2,2]
    @frozen const B: f32 = 1.0
    node m(x: f32[2]) -> f32[2] { return MatMul(W, x) }
    """
    ast = fuse_parser.parse(src)
    fl = FuseLowerer()
    model = fl.lower(ast)
    assert model is not None
    val = _find_meta_val(model, "trainables")
    assert isinstance(val, dict), "trainables metadata missing or not a dict"
    # keys may be qualified; check endswith
    assert any(k.endswith("W") and v is True for k, v in val.items())
    assert any(k.endswith("B") and v is False for k, v in val.items())
