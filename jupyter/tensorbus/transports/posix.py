import logging
import os
import pickle
from typing import Optional

from ..core_types import TensorSnapshot
from prometheus_client import CollectorRegistry
from .posix_base import PosixTransportBase

logger = logging.getLogger(__name__)


class PosixTransport(PosixTransportBase):
    def __init__(
        self,
        root_dir: str,
        worker_threads: int = 1,
        queue_size: int = 1024,
        metrics_registry: Optional[CollectorRegistry] = None,
        batch_window_ms: int = 0,
    ) -> None:
        super().__init__(root_dir, worker_threads=worker_threads, queue_size=queue_size, metrics_registry=metrics_registry, batch_window_ms=batch_window_ms)

    def _do_write(self, snap: TensorSnapshot) -> None:
        import time

        path = os.path.join(self.root, f"{snap.name}.onnx")

        t0 = time.perf_counter()
        try:
            data = pickle.dumps(snap)
        except Exception as e:
            logger.exception("Failed to pickle snapshot %s: %s", snap.name, e)
            return
        t1 = time.perf_counter()

        # Use atomic write helper from base
        with self._lock:
            ok = self._atomic_write_bytes(data, path)
            if not ok:
                return
        t2 = time.perf_counter()

        # Record metrics (ms)
        self._metrics["serialize_ms"].append((t1 - t0) * 1000.0)
        self._metrics["write_ms"].append((t2 - t1) * 1000.0)
        self._metrics["total_ms"].append((t2 - t0) * 1000.0)

        # Prometheus
        self._written_counter.inc()
        try:
            self._write_hist.observe(t2 - t0)
        except Exception as e:
            logger.debug("Failed to observe histogram: %s", e)

        # Notify subscribers (log exceptions)
        for cb in list(self._subscribers):
            try:
                cb(snap)
            except Exception as e:
                logger.exception("Subscriber callback raised for %s: %s", snap.name, e)

    def load_snapshot(self, name: str) -> Optional[TensorSnapshot]:
        try:
            self._validate_name(name)
        except ValueError:
            logger.error("load_snapshot rejected: invalid snapshot name: %s", name)
            return None

        path = os.path.join(self.root, f"{name}.onnx")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.exception("Failed to load pickle snapshot %s: %s", path, e)
            return None
