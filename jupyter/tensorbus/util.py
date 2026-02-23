from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import onnx
from onnx import TensorProto
from typing import Any

# Some onnx installs expose mapping via onnx.mapping; guard for typing
_onnx_mapping: Any = getattr(__import__("onnx"), "mapping", None)
import onnxruntime as ort


def blake3_hash(s: str) -> int:
    # Use blake2b as a portable stand-in for blake3 (takes first 8 bytes)
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def detect_providers(test_gpu: bool = True) -> List[str]:
    available = ort.get_available_providers()
    if not test_gpu:
        return ["CPUExecutionProvider"]

    gpu_order = [
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "DmlExecutionProvider",
        "CoreMLExecutionProvider",
    ]
    providers = [p for p in gpu_order if p in available]
    providers.append("CPUExecutionProvider")
    return providers


def primary_device(providers: List[str]) -> str:
    p = providers[0]
    if p in ("CUDAExecutionProvider", "ROCMExecutionProvider"):
        return "cuda"
    if p == "DmlExecutionProvider":
        return "dml"
    return "cpu"


def tensorproto_to_numpy(t: TensorProto) -> np.ndarray:
    return np.frombuffer(
        t.raw_data, dtype=_onnx_mapping.TENSOR_TYPE_TO_NP_TYPE[t.data_type]
    ).reshape(t.dims)


def numpy_to_tensorproto(array: np.ndarray, name: str) -> TensorProto:
    t = TensorProto()
    t.name = name
    t.dims.extend(array.shape)
    t.data_type = _onnx_mapping.NP_TYPE_TO_TENSOR_TYPE[array.dtype.type]
    t.raw_data = array.tobytes()
    return t

