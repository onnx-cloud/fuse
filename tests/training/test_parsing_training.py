from src.parser import fuse_parser


def test_training_flat_parses():
    src = "@training { optimizer = adamw, lr = 0.001 }"
    ast = fuse_parser.parse(src)
    metas = [
        n
        for n in ast
        if isinstance(n, dict)
        and n.get("type") == "meta"
        and n.get("name") == "fuse.training"
    ]
    assert metas
    assert metas[0]["value"].get("optimizer") == "adamw"


def test_training_nested_parses():
    src = "@training { optimizer: { type = adamw, lr = 0.001 }, schedule: { type = cosine } }"
    ast = fuse_parser.parse(src)
    metas = [
        n
        for n in ast
        if isinstance(n, dict)
        and n.get("type") == "meta"
        and n.get("name") == "fuse.training"
    ]
    assert metas
    v = metas[0]["value"]
    assert isinstance(v, dict)
    assert "optimizer" in v and isinstance(v["optimizer"], dict)
    assert v["optimizer"]["type"] == "adamw"
    assert v["schedule"]["type"] == "cosine"
