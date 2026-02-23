from lark import Lark

from src.parser import GRAMMAR, fuse_parser


def test_lexer_tokens_simple():
    # Tokenize a small snippet and assert we see the identifier token 'foo'
    lexer = Lark(GRAMMAR, parser="earley")
    tokens = list(lexer.lex("node foo() { }"))
    # Find a token whose text is 'foo' and of IDENT-like type
    assert any(
        getattr(t, "value", None) == "foo" and t.type == "IDENT"
        for t in tokens
    )


def test_parser_productions_smoke():
    src = """
    node add(x: f32, y: f32) -> f32 {
      return x
    }
    """
    ast = fuse_parser.parse(src)
    assert isinstance(ast, list)
    assert len(ast) >= 1
    # find a node entry among parsed AST nodes (skip top-level @fuse meta if present)
    node = next((n for n in ast if isinstance(n, dict) and n.get("type") == "node"), None)
    assert node is not None and node.get("type") == "node"
    assert node.get("name") == "add"
    params = node.get("params")
    assert isinstance(params, list)
    assert [p["name"] for p in params] == ["x", "y"]
    assert node.get("ret_type") is not None
    assert isinstance(node.get("body"), list)

    # Idempotence: colon-style indented blocks should parse to same AST
    src_colon = """
    model m(x: f32):
      return x
    """
    src_brace = """
    model m(x: f32) { return x }
    """
    a = fuse_parser.parse(src_colon)
    b = fuse_parser.parse(src_brace)
    assert a == b
