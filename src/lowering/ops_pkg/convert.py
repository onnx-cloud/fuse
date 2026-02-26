# Conversion-related operator lowerings

from src.lowering.ops import OpsLowerer
from .registry import onnx_op


@onnx_op("Cast")
def lower_cast(ops_lowerer: OpsLowerer, call, ctx, env, types, type_hint=None, out_name=None):
    """Lower Cast; delegate to generic lowering while avoiding recursion."""
    call["_registry_skipped"] = True
    try:
        return ops_lowerer._lower_onnx_call(call, ctx, env, types, type_hint, out_name)
    finally:
        call.pop("_registry_skipped", None)
