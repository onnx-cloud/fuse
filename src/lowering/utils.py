class LoweringError(Exception):
    """Raised when lowering a declaration fails; wraps the original
    exception and includes the source file, function name and optional
    line/column location when available.
    """

    def __init__(self, message, source=None, function=None, line=None, column=None):
        super().__init__(message)
        self.source = source
        self.function = function
        self.line = line
        self.column = column


def _onnx_to_fuse_scalar(elem_type: int) -> str:
    """Map ONNX TensorProto elem_type to a fuse scalar name.

    Returns a Fuse-style scalar name such as ``'f32'`` or ``'i64'``. This is a
    conservative mapping used when materializing output types from ONNX
    metadata.
    """
    try:
        from onnx import TensorProto

        return {
            TensorProto.FLOAT: "f32",
            TensorProto.DOUBLE: "f64",
            TensorProto.INT64: "i64",
            TensorProto.INT32: "i32",
            TensorProto.INT16: "i16",
            TensorProto.INT8: "i8",
            TensorProto.UINT8: "u8",
            TensorProto.UINT16: "u16",
            TensorProto.UINT32: "u32",
            TensorProto.UINT64: "u64",
            TensorProto.BOOL: "bool",
            TensorProto.FLOAT16: "f16",
        }.get(int(elem_type), "f32")
    except Exception:
        return "f32"
