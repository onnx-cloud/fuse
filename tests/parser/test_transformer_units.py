from src.parser import fuse_parser


def test_cast_generic_shorthand_parses():
    src = "fn c(x: i64) -> f32 { return Cast<f32>(x) }"
    ast = fuse_parser.parse(src)
    node = next(
        (d for d in ast if isinstance(d, dict) and d.get("type") == "node"), None
    )
    assert node is not None
    body = node.get("body")
    # Return normalization should fold to a single call node
    ret = body[0]
    assert isinstance(ret, dict)
    # either {'return': {'call': 'Cast', 'generics': 'f32', 'args': ['x']}} or similar
    r = ret.get("return")
    assert isinstance(r, dict) and r.get("call") == "Cast"
    gens = r.get("generics")
    assert gens == "f32" or (
        isinstance(gens, dict) and gens.get("to") == "f32"
    )


def test_call_generics_with_kwargs_and_scalars():
    src = "fn c() -> f32 { return Zeros<like=x>(1) }"
    ast = fuse_parser.parse(src)
    node = next((d for d in ast if d.get("type") == "node"), None)
    ret = node["body"][0]["return"]
    if isinstance(ret, dict) and ret.get("call") == "Zeros":
        gens = ret.get("generics")
        # generics for <like=x> should be a dict
        assert isinstance(gens, dict) and gens.get("like") == "x"
