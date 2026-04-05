import pytest

from src.lowering.passes import TypeShapePass, NormalizationPass


def test_const_folding_basic():
    """TypeShapePass should propagate literal types through a simple body."""
    ast = [
        {
            "type": "node",
            "name": "add_consts",
            "params": [
                {"name": "x", "type_decl": {"scalar": "f32", "dims": [2]}},
            ],
            "body": [
                {"let": "a", "value": {"lit": 1.0}},
                {"return": {"call": "Add", "args": [
                    {"ref": "x"},
                    {"ref": "a"},
                ]}},
            ],
        }
    ]
    result = TypeShapePass().run(ast)
    # The literal 1.0 should be typed as f32
    body = result[0]["body"]
    let_stmt = body[0]
    assert let_stmt["value"]["type"]["scalar"] == "f32"
    # The declaration should be marked as typed
    assert result[0].get("__typed__") is True


def test_normalization_pass_resolves_type_aliases():
    """NormalizationPass should inline type aliases into type_decl fields."""
    ast = [
        {"type": "type_alias", "name": "Vec3", "type_decl": {"scalar": "f32", "dims": [3]}},
        {
            "type": "node",
            "name": "use_alias",
            "params": [{"name": "v", "type_decl": "Vec3"}],
            "body": [{"return": {"ref": "v"}}],
        },
    ]
    result = NormalizationPass().run(ast)
    # The param type_decl should be resolved from alias
    param_td = result[1]["params"][0]["type_decl"]
    assert isinstance(param_td, dict), f"expected resolved type dict, got {param_td!r}"
    assert param_td.get("scalar") == "f32"
    assert param_td.get("dims") == [3]


def test_name_allocation_is_deterministic(stable_namer):
    n1 = stable_namer.next("x")
    n2 = stable_namer.next("x")
    assert n1 != n2
