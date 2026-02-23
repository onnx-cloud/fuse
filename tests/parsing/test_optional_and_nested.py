from src.parser import fuse_parser


def test_param_defaults_and_optional_args():
    src = """
    node f(x: f32 = 1.0, y: i32 = 2) -> f32 {
        return x
    }
    """
    ast = fuse_parser.parse(src)
    # find first node in AST (auto-injected @fuse may appear at index 0)
    node = next((n for n in ast if isinstance(n, dict) and n.get("type") == "node"), ast[0])
    params = node["params"]
    assert params[0]["name"] == "x"
    assert "value" in params[0]
    # Parser may return numeric tokens or strings; assert textual startswith
    assert str(params[0]["value"]).startswith("1")
    assert "value" in params[1]
    assert str(params[1]["value"]).startswith("2")


def test_static_if_with_nested_block_parses():
    src = """
    node f(cond: bool) -> i32 {
        static if cond {
            return 1
        } else {
            return 2
        }
    }
    """
    ast = fuse_parser.parse(src)
    node = next((n for n in ast if isinstance(n, dict) and n.get("type") == "node"), ast[0])
    body = node["body"]
    # There should be an `if`-shaped node inside the function body
    assert any(isinstance(stmt, dict) and "if" in stmt for stmt in body)
