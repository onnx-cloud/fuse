import pytest
import onnx
from src.parser import fuse_parser
from src.lowering import FuseLowerer


def _lower(src, inline: bool = False, opset: int | None = None):
    # accept either raw AST or string
    if isinstance(src, (list, tuple)):
        ast = src
    else:
        ast = fuse_parser.parse(src)
    fl = FuseLowerer(inline_functions=inline)
    if opset is not None:
        from src.graph_context import GraphContext
        ctx = GraphContext(opset=opset)
        # lower into provided context then build the model ourselves
        fl.lower(ast, ctx=ctx)
        return ctx.build_model()
    return fl.lower(ast)


def test_functionproto_emitted_by_default():
    from src.util.project_version import get_project_version
    version = get_project_version()
    src = f"""\
@fuse {version}
@opset onnx 18
@domain example
fn foo(x: f32) -> f32 {{ Add(x, x) }}
model m(x: f32) -> f32 {{ foo(x) }}
"""
    m = _lower(src, inline=False)
    # model should contain a FunctionProto named "foo"
    assert any(f.name == "foo" for f in m.functions)
    # graph nodes should call op_type "foo" rather than Add
    ops = [n.op_type for n in m.graph.node]
    assert "foo" in ops and "Add" not in ops


def test_inline_flag_inlines_function():
    from src.util.project_version import get_project_version
    version = get_project_version()
    src = f"""\
@fuse {version}
@opset onnx 18
@domain example
fn bar(x: f32) -> f32 {{ Mul(x, x) }}
model m(x: f32) -> f32 {{ bar(x) }}
"""
    m = _lower(src, inline=True)
    # inline mode should produce no FunctionProto
    assert len(m.functions) == 0
    # graph should contain the body node
    assert any(n.op_type == "Mul" for n in m.graph.node)


def test_import_preserves_functions(tmp_path):
    # create a simple onnx model with a FunctionProto and import it from fuse
    import os
    from onnx import helper, FunctionProto as FP, ModelProto

    # build a small function graph
    f = FP()
    f.name = "f2"
    f.input.extend(["a"])
    f.output.extend(["b"])
    node = helper.make_node("Neg", ["a"], ["b"], name="n")
    f.node.extend([node])

    # create model containing function and a simple graph that uses it
    g = helper.make_graph([helper.make_node("f2", ["x"], ["y"])], "", [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])], [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])])
    model = helper.make_model(g, functions=[f], opset_imports=[helper.make_opsetid("", 18)])
    path = tmp_path / "base.onnx"
    onnx.save(model, str(path))

    # fuse import it in a simple fuse file
    from src.util.project_version import get_project_version
    version = get_project_version()
    # build AST manually to avoid parser issues with @import source
    ast = [
        {"type": "meta", "name": "fuse", "value": version},
        {"type": "meta", "name": "module", "value": "d"},
        {
            "type": "import",
            "name": "lib",
            "alias": "lib",
            "source": str(path),
            "variants": [],
        },
        {
            "type": "model",
            "name": "m",
            "params": [{"name": "x", "type": "f32", "value": None}],
            "ret_type": "f32",
            "body": [{"call": "lib", "args": ["x"]}],
        },
    ]
    m2 = _lower(ast, opset=18)
    # imported function should be present (name will be prefixed by module and/or alias)
    assert m2.functions and any("lib" in fn.name for fn in m2.functions)


def test_functionproto_domain_and_call_domain():
    # make sure emitted FunctionProto has non-empty domain derived from @domain
    from src.util.project_version import get_project_version
    version = get_project_version()
    src = f"""\
@fuse {version}
@opset onnx 18
@domain mydom
fn foo(x: f32) -> f32 {{ Add(x, x) }}
model m(x: f32) -> f32 {{ foo(x) }}
"""
    m = _lower(src, inline=False)
    # single function exists
    assert len(m.functions) == 1
    fn = m.functions[0]
    assert fn.name == "foo"
    assert fn.domain == "mydom"
    # call node domain should match
    assert all(n.domain == "mydom" for n in m.graph.node if n.op_type == "foo")


def test_function_name_conflict_with_builtin():
    # define a user function named 'Add' which is also a builtin
    from src.util.project_version import get_project_version
    version = get_project_version()
    src = f"""\
@fuse {version}
@opset onnx 18
@domain xd
fn Add(x: f32) -> f32 {{ Mul(x, x) }}
model m(x: f32) -> f32 {{ Add(x) }}
"""
    m = _lower(src, inline=False)
    # there should still be one FunctionProto for user Add in custom domain
    assert any(f.name == "Add" and f.domain == "xd" for f in m.functions)
    # graph should call Add (domain=xd) not the builtin Multiply
    calls = [(n.op_type, n.domain) for n in m.graph.node]
    assert ("Add", "xd") in calls
    assert all(not (n.op_type == "Mul" and n.domain == "") for n in m.graph.node)


def test_functionproto_default_domain_fallback():
    # even if no model-level domain is declared, emitting via _emit_function_proto
    # should assign a stable "fuse.local" domain.
    from src.lowering.main import FuseLowerer
    from src.graph_context import GraphContext

    fl = FuseLowerer(inline_functions=False)
    ctx = GraphContext(opset=18)
    simple = {
        "type": "node",
        "name": "foo",
        "params": [{"name": "x", "type": "f32"}],
        "ret_type": "f32",
        "body": [{"return": [{"call": "Add", "args": ["x"]}]}],
    }
    fl._emit_function_proto(simple, ctx)
    assert len(ctx.functions) == 1
    assert ctx.functions[0].domain == "fuse.local"
