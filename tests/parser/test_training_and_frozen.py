from src.parser import fuse_parser


def _find_meta(ast, name):
    for node in ast:
        if (
            isinstance(node, dict)
            and node.get("type") == "meta"
            and node.get("name") == name
        ):
            return node
    return None


def _find_decl(ast, name):
    for node in ast:
        if isinstance(node, dict) and node.get("name") == name:
            return node
    return None


def test_meta_training_parsed():
    src = """
    @training { optimizer = adamw, lr = 0.001 }
    """
    ast = fuse_parser.parse(src)
    meta = _find_meta(ast, "fuse.training")
    assert meta is not None
    assert "optimizer" in str(meta.get("value"))


def test_meta_training_nested_parsed():
    src = """
    @training { optimizer: { type = adamw, lr = 0.001 }, schedule: { type = cosine } }
    """
    ast = fuse_parser.parse(src)
    meta = _find_meta(ast, "fuse.training")
    assert meta is not None
    v = meta.get("value")
    assert isinstance(v, dict)
    assert "optimizer" in v and isinstance(v["optimizer"], dict)
    assert v["optimizer"].get("type") == "adamw"
    assert v.get("schedule", {}).get("type") == "cosine"


def test_train_and_frozen_params_set_trainable():
    src = """
    @train weight W: f32[2,2]
    @frozen param p: f32
    """
    ast = fuse_parser.parse(src)
    w = _find_decl(ast, "W")
    p = _find_decl(ast, "p")
    assert w is not None and w.get("trainable") is True and w.get("trainable") is True
    assert p is not None and p.get("trainable") is False and p.get("trainable") is False
