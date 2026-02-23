from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from .util import detect_providers


class ModelZoo(ABC):
    root: str

    @abstractmethod
    def load_model(self, model_name: str) -> onnx.ModelProto: ...


class Transport(ABC):
    @abstractmethod
    def publish(self, snapshot: "TensorSnapshot") -> None: ...

    @abstractmethod
    def subscribe(self, callback: Callable[["TensorSnapshot"], None]) -> None: ...


@dataclass(slots=True)
class HLC:
    counter: int
    node: str

    def tick(self) -> "HLC":
        return HLC(self.counter + 1, self.node)

    def __gt__(self, other: "HLC") -> bool:
        return (self.counter, self.node) > (other.counter, other.node)

@dataclass(slots=True)
class TensorSnapshot:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    clock: Optional[HLC]
    hash: bytes
    payload: Optional[bytes]


@dataclass(slots=True)
class TensorBinding:
    id: int
    arg: Any
    clock: HLC
    hash: bytes
    values: Dict[str, ort.OrtValue]  # "cpu", "cuda", etc.


@dataclass(slots=True)
class ModelBinding:
    id: int
    name: str
    session: ort.InferenceSession
    epoch: int = 0


class TensorBusInterface(ABC):
    @abstractmethod
    def load_model(self, model_name: str) -> ModelBinding: ...
    @abstractmethod
    def set_tensor(self, name: str, arr: np.ndarray) -> None: ...
    @abstractmethod
    def get_tensor(self, name: str) -> TensorBinding: ...
    @abstractmethod
    def run(self, model_name: str) -> None: ...
    @abstractmethod
    def add_transport(self, transport: Transport) -> None: ...


class SessionResolver(ABC):
    @abstractmethod
    def session(self, model_name: str) -> ort.InferenceSession: ...


class LocalSessionResolver(SessionResolver):
    def __init__(self, test_gpu: bool = True):
        self.providers = detect_providers(test_gpu)

    def session(self, model_name: str) -> ort.InferenceSession:
        return ort.InferenceSession(model_name, providers=self.providers)


__all__ = [
    "TensorBusInterface",
    "Transport",
    "TensorSnapshot",
    "TensorBinding",
    "ModelBinding",
    "HLC",
    "SessionResolver",
    "LocalSessionResolver",
    "blake3_hash",
    "detect_providers",
    "primary_device",
]
