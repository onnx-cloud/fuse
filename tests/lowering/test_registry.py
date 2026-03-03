import pytest

from src.lowering.ops_pkg import registry


def test_builtin_ops_pre_registered():
    # elementwise and convert submodules should register common ops on import
    assert registry.get_lowerer("Add") is not None
    assert registry.get_lowerer("Cast") is not None


def test_register_and_lookup():
    # ensure registry is clean for test
    registry._LOWERERS.clear()

    @registry.onnx_op("Foo", domain="bar")
    def foo_lower(self, call, ctx, env, types, type_hint=None, out_name=None):
        # mimic OpsLowerer method signature; return a dummy name
        return "foo", None

    assert registry.get_lowerer("Foo", domain="bar") is foo_lower
    # opset filtering
    assert registry.get_lowerer("Foo", domain="bar", opset=21) is foo_lower
    assert registry.get_lowerer("Foo", domain="bar", opset=20) is None


def test_fallback_to_builtin_domain():
    registry._LOWERERS.clear()

    @registry.onnx_op("Add")
    def add_lower(self, call, ctx, env, types, type_hint=None, out_name=None):
        return "add", None

    # when domain-specific not registered, builtin entry should be returned
    assert registry.get_lowerer("Add", domain="custom") is add_lower


@pytest.mark.parametrize("op_type,domain", [("Baz", ""), ("Baz", "qux")])
def test_get_unregistered_returns_none(op_type, domain):
    registry._LOWERERS.clear()
    assert registry.get_lowerer(op_type, domain) is None


def test_lower_call_uses_registry():
    # clear and register
    registry._LOWERERS.clear()

    @registry.onnx_op("Foo")
    def foo_lower(self, call, ctx, env, types, type_hint=None, out_name=None):
        return "bar", {"scalar": "f32"}

    # minimal dummy objects
    class DummyLowerer:
        def __init__(self):
            self._maybe_fold_elementwise = lambda op, inputs, lits: None
            self.ELEMENTWISE_OPS = set()
            self._ensure_same_scalar = lambda op, types: None
            self._user_decls = {}
            self.inline_functions = False
            self.import_manager = type("IM", (), {"fused_signatures": set()})()
            self._lower_expr = lambda a, ctx, env, types, type_hint=None: (None, None)

    class DummyCtx:
        opset = 21


    # Instead, use real OpsLowerer with dummy lowerer
    from src.lowering.ops import OpsLowerer
    dummy = DummyLowerer()
    ol = OpsLowerer(dummy)
    call = {"call": "Foo", "args": []}

    # call lower_onnx_call directly; should hit registry before doing anything
    name, typ = ol._lower_onnx_call(call, DummyCtx(), {}, {}, None, None)
    assert name == "bar"
    assert typ == {"scalar": "f32"}
