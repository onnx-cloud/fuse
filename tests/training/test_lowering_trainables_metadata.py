from src.parser import fuse_parser


def test_train_sets_trainable_true():
    src = "@train weight W: f32[2,2]"
    ast = fuse_parser.parse(src)
    decls = [n for n in ast if isinstance(n, dict) and n.get("name") == "W"]
    assert decls
    assert decls[0].get("trainable") is True


def test_frozen_sets_trainable_false():
    src = "@frozen const B: f32 = 1.0"
    ast = fuse_parser.parse(src)
    decls = [n for n in ast if isinstance(n, dict) and n.get("name") == "B"]
    assert decls
    assert decls[0].get("trainable") is False
