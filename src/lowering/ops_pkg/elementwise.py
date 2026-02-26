# Elementwise operator lowerings using the new registry

from src.lowering.ops import OpsLowerer
from .registry import onnx_op


@onnx_op("Add")
def lower_add(ops_lowerer: OpsLowerer, call, ctx, env, types, type_hint=None, out_name=None):
    """Lower Add by delegating to the generic path.

    This implementation exists primarily as a demonstration of how
    builtin operators can be migrated into the registry.  A more
    optimized version could reimplement constant folding or type
    promotion directly here.  To avoid re-entering the registry we set
    a temporary flag on the call.
    """
    call["_registry_skipped"] = True
    try:
        return ops_lowerer._lower_onnx_call(call, ctx, env, types, type_hint, out_name)
    finally:
        call.pop("_registry_skipped", None)


@onnx_op("Sub")
def lower_sub(ops_lowerer: OpsLowerer, call, ctx, env, types, type_hint=None, out_name=None):
    call["_registry_skipped"] = True
    try:
        return ops_lowerer._lower_onnx_call(call, ctx, env, types, type_hint, out_name)
    finally:
        call.pop("_registry_skipped", None)


@onnx_op("Mul")
def lower_mul(ops_lowerer: OpsLowerer, call, ctx, env, types, type_hint=None, out_name=None):
    call["_registry_skipped"] = True
    try:
        return ops_lowerer._lower_onnx_call(call, ctx, env, types, type_hint, out_name)
    finally:
        call.pop("_registry_skipped", None)


@onnx_op("Div")
def lower_div(ops_lowerer: OpsLowerer, call, ctx, env, types, type_hint=None, out_name=None):
    call["_registry_skipped"] = True
    try:
        return ops_lowerer._lower_onnx_call(call, ctx, env, types, type_hint, out_name)
    finally:
        call.pop("_registry_skipped", None)
