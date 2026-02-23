"""TensorBus transport implementations."""

# Core transports
from .memory import MemTransport
from .posix import PosixTransport
from .posix_onnx import PosixOnnxTransport

# v2 Transport wrappers
from .filtered import FilteredTransport
from .compressed import CompressedTransport
from .validated import ValidatedTransport

# Phase 2: Observability
from .metrics import MetricsTransport
from .traced import TracedTransport

# Phase 3: Advanced features
from .shm import ShmTransport
from .lazy import LazyTransport

# Phase 3.5: CUDA IPC
try:
    from .cuda_ipc import CudaIpcTransport
    HAS_CUDA_IPC = True
except ImportError:
    HAS_CUDA_IPC = False

# Phase 4: Scale & security
from .gossip import GossipTransport
from .encrypted import EncryptedTransport
from .rbac import RBACTransport

# Phase 5: Ecosystem transports
from .kafka import KafkaTransport
from .redis_stream import RedisTransport
from .grpc_transport import GrpcTransport

__all__ = [
    # Core
    "MemTransport",
    "PosixTransport",
    "PosixOnnxTransport",
    # v2
    "FilteredTransport",
    "CompressedTransport",
    "ValidatedTransport",
    # Phase 2
    "MetricsTransport",
    "TracedTransport",
    # Phase 3
    "ShmTransport",
    "LazyTransport",
    # Phase 4
    "GossipTransport",
    "EncryptedTransport",
    "RBACTransport",
    # Phase 5
    "KafkaTransport",
    "RedisTransport",
    "GrpcTransport",
]

# Conditionally add CUDA IPC if available
if HAS_CUDA_IPC:
    __all__.append("CudaIpcTransport")
