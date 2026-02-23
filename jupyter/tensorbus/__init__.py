"""Fusion TensorBus package
"""

__version__ = "0.2.0"

from .bus import TensorBus
from .core_types import (
    HLC,
    LocalSessionResolver,
    ModelBinding,
    SessionResolver,
    TensorBinding,
    TensorSnapshot,
    Transport,
)

# Fabric (multi-model orchestration)
from .fabric import TensorFabric, ComputeNode, NodeState

# Transport implementations
from .transports.memory import MemTransport
from .transports.posix import PosixTransport
from .transports.posix_onnx import PosixOnnxTransport

# Transport wrappers (v2)
from .transports.filtered import FilteredTransport
from .transports.compressed import CompressedTransport
from .transports.validated import ValidatedTransport

__all__ = [
    "TensorBus",
    # Core types
    "HLC",
    "LocalSessionResolver",
    "ModelBinding",
    "SessionResolver",
    "TensorBinding",
    "TensorSnapshot",
    "Transport",
    # Fabric
    "TensorFabric",
    "ComputeNode",
    "NodeState",
    # Transports
    "MemTransport",
    "PosixTransport",
    "PosixOnnxTransport",
    # Transport wrappers
    "FilteredTransport",
    "CompressedTransport",
    "ValidatedTransport",
]

# Backwards compatibility
from . import core_types as types
