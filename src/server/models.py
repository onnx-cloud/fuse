from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class LintOptions(BaseModel):
    check_types: Optional[bool] = True
    check_shapes: Optional[bool] = True
    strict: Optional[bool] = False


class LintRequest(BaseModel):
    source: str
    options: Optional[LintOptions] = None


class LintError(BaseModel):
    severity: str
    message: str
    location: Optional[Dict[str, Any]] = None
    code: Optional[str] = None


class LintResponse(BaseModel):
    valid: bool
    warnings: List[str] = []
    errors: List[LintError] = []
    diagnostics: Optional[Dict[str, Any]] = None


class CompileOptions(BaseModel):
    opset: Optional[int] = None
    optimize: Optional[bool] = True
    fold_constants: Optional[bool] = True
    inline_imports: Optional[bool] = False
    format: Optional[str] = "binary"  # binary | text


class CompileRequest(BaseModel):
    source: str
    options: Optional[CompileOptions] = None
    imports: Optional[Dict[str, Any]] = None


class CompileResponse(BaseModel):
    success: bool
    onnx: Optional[str] = None
    onnx_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    diagnostics: Optional[Dict[str, Any]] = None
    warnings: List[str] = []
    errors: Optional[List[Dict[str, Any]]] = None


class DecompileOptions(BaseModel):
    preserve_names: Optional[bool] = True
    infer_types: Optional[bool] = True
    pretty: Optional[bool] = True
    comments: Optional[bool] = True


class DecompileRequest(BaseModel):
    onnx: Optional[str] = None
    options: Optional[DecompileOptions] = None


class DecompileResponse(BaseModel):
    success: bool
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    warnings: List[str] = []
    errors: Optional[List[Dict[str, Any]]] = None
