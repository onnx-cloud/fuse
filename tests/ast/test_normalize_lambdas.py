from src.ast.normalize_lambdas import normalize_lambdas


def test_normalize_single_lambda():
    src_decl = {
        "type": "model",
        "name": "m",
        "params": [],
        "body": [
            {
                "let": "x",
                "expr": {
                    "call": "Loop",
                    "generics": {
                        "body": {
                            "lambda": {
                                "args": ["i", "s"],
                                "body": [
                                    True,
                                    {
                                        "call": "Add",
                                        "args": [
                                            "s",
                                            {
                                                "call": "Cast",
                                                "args": ["i"],
                                                "generics": {"to": "f32"},
                                            },
                                        ],
                                    },
                                ],
                            }
                        }
                    },
                    "args": ["n", True],
                },
            }
        ],
    }
    normalized = normalize_lambdas([src_decl])
    # Expect 1 generated node followed by the original declaration
    assert normalized[0]["type"] == "node"
    assert normalized[0]["name"].startswith("__lambda_node_")
    assert normalized[1]["type"] == "model"
    # The model should reference the generated node by name in its generics
    gen_name = normalized[0]["name"]
    # Find the Loop call in model body
    body = normalized[1]["body"]
    found = False
    for stmt in body:
        if isinstance(stmt, dict) and stmt.get("let") == "x":
            call = stmt.get("expr")
            gen = call.get("generics")
            assert gen.get("body") == gen_name
            found = True
    assert found
