from __future__ import annotations

import os
import threading
import queue
import time
from typing import Optional, List

from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge

import logging
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from ..core_types import Transport
from collections.abc import Callable

from ..core_types import TensorSnapshot

logger = logging.getLogger(__name__)


from .posix_base import PosixTransportBase


class PosixOnnxTransport(PosixTransportBase):
    def __init__(
        self,
        root_dir: str,
        worker_threads: int = 1,
        queue_size: int = 1024,
        metrics_registry: Optional[CollectorRegistry] = None,
        batch_window_ms: int = 0,
    ) -> None:
        super().__init__(root_dir, worker_threads=worker_threads, queue_size=queue_size, metrics_registry=metrics_registry, batch_window_ms=batch_window_ms)
        # transport-specific extra metrics (optional, namespaced)
        self._registry = metrics_registry or self._registry
        self._enqueue_counter_onnx = Counter("tensorbus_posix_onnx_enqueue_total", "Enqueued ONNX snapshots", registry=self._registry)
        self._written_counter_onnx = Counter("tensorbus_posix_onnx_written_total", "Completed ONNX writes", registry=self._registry)
        self._write_hist_onnx = Histogram("tensorbus_posix_onnx_write_seconds", "ONNX write duration seconds", registry=self._registry)
        self._queue_gauge_onnx = Gauge("tensorbus_posix_onnx_queue_size", "Current ONNX queue size", registry=self._registry)



    def _validate_name(self, name: str) -> None:
        if not name or os.path.basename(name) != name or \
           "/" in name or "\\" in name or name.startswith('.'):
            raise ValueError(f"Invalid snapshot name: {name}")

    def _do_write(self, snap: TensorSnapshot):
        import time

        # Validate name to avoid path traversal
        try:
            self._validate_name(snap.name)
        except ValueError as e:
            logger.error("Rejecting snapshot with invalid name: %s", snap.name)
            return

        # Build tensor from raw payload using numpy helper (correct dtype mapping)
        if snap.payload is None:
            logger.error("Snapshot payload missing for %s", snap.name)
            return
        try:
            arr = np.frombuffer(snap.payload, dtype=snap.dtype).reshape(snap.shape)
        except Exception as e:
            logger.exception("Failed to interpret payload for %s: %s", snap.name, e)
            return

        tensor_proto = numpy_helper.from_array(arr, name=snap.name)

        graph = helper.make_graph(
            nodes=[],
            name=f"{snap.name}",
            inputs=[
                helper.make_tensor_value_info(snap.name, tensor_proto.data_type, snap.shape)
            ],
            outputs=[
                helper.make_tensor_value_info(snap.name, tensor_proto.data_type, snap.shape)
            ],
            initializer=[tensor_proto],
        )

        model = helper.make_model(graph, producer_name="tensorbus")
        if snap.clock is not None:
            kv = model.metadata_props.add()
            kv.key = "tensorbus.clock_counter"
            kv.value = str(snap.clock.counter)
            kv = model.metadata_props.add()
            kv.key = "tensorbus.clock_node"
            kv.value = snap.clock.node
        if snap.hash:
            kv = model.metadata_props.add()
            kv.key = "tensorbus.hash_hex"
            kv.value = snap.hash.hex()

        path = os.path.join(self.root, f"{snap.name}.onnx")

        t0 = time.perf_counter()
        serialized = model.SerializeToString()
        t1 = time.perf_counter()

        with self._lock:
            ok = self._atomic_write_bytes(serialized, path)
            if not ok:
                return
        t2 = time.perf_counter()

        # record metrics in ms
        self._metrics["serialize_ms"].append((t1 - t0) * 1000.0)
        self._metrics["write_ms"].append((t2 - t1) * 1000.0)
        self._metrics["total_ms"].append((t2 - t0) * 1000.0)

        self._written_counter.inc()
        try:
            self._write_hist.observe(t2 - t0)
        except Exception as e:
            logger.debug("Failed to observe histogram: %s", e)

        # Notify subscribers, log failures
        for cb in list(self._subscribers):
            try:
                cb(snap)
            except Exception as e:
                logger.exception("Subscriber callback raised for %s: %s", snap.name, e)

    def publish(self, snap: TensorSnapshot):
        # Validate name early
        try:
            self._validate_name(snap.name)
        except ValueError:
            logger.error("publish rejected: invalid snapshot name: %s", snap.name)
            return

        # reuse base publish logic (validation, enqueue, short wait)
        super().publish(snap)
        # increment optional onnx-specific counters
        try:
            self._enqueue_counter_onnx.inc()
            try:
                self._queue_gauge_onnx.set(self._queue.qsize())
            except Exception:
                pass
        except Exception:
            # metrics are best-effort
            logger.debug("Failed to record ONNX-specific enqueue metrics")

    def flush(self, timeout: Optional[float] = None):
        start = time.perf_counter()
        try:
            self._queue.join()
        except Exception:
            return
        if timeout is not None:
            elapsed = time.perf_counter() - start
            if elapsed > timeout:
                raise TimeoutError("flush timed out")

    def close(self):
        self.flush()
        self._stop.set()
        for _ in self._workers:
            try:
                self._queue.put_nowait(None)
            except Exception:
                pass
        for w in self._workers:
            w.join()


    def subscribe(self, cb: Callable[[TensorSnapshot], None]):
        with self._lock:
            self._subscribers.append(cb)

    def load_snapshot(self, name: str) -> Optional[TensorSnapshot]:
        # Validate name to avoid path traversal
        try:
            self._validate_name(name)
        except ValueError:
            logger.error("load_snapshot rejected: invalid snapshot name: %s", name)
            return None

        path = os.path.join(self.root, f"{name}.onnx")
        if not os.path.exists(path):
            return None

        # Load the ONNX model and extract first initializer
        try:
            model = onnx.load(path)
        except Exception as e:
            logger.exception("Failed to load ONNX model %s: %s", path, e)
            return None

        if len(model.graph.initializer) == 0:
            return None
        tensor_proto = model.graph.initializer[0]
        try:
            arr = numpy_helper.to_array(tensor_proto)
        except Exception as e:
            logger.exception("Failed to convert initializer to numpy array for %s: %s", path, e)
            return None

        # Try to recover metadata if present
        counter = None
        node = None
        hash_hex = None
        for kv in model.metadata_props:
            if kv.key == "tensorbus.clock_counter":
                try:
                    counter = int(kv.value)
                except Exception:
                    counter = None
            elif kv.key == "tensorbus.clock_node":
                node = kv.value
            elif kv.key == "tensorbus.hash_hex":
                hash_hex = kv.value

        clock = None
        if counter is not None and node is not None:
            from ..core_types import HLC

            clock = HLC(counter, node)

        h = b""
        if hash_hex:
            try:
                h = bytes.fromhex(hash_hex)
            except Exception:
                h = b""

        return TensorSnapshot(
            name=tensor_proto.name,
            dtype=str(arr.dtype),
            shape=arr.shape,
            clock=clock,
            hash=h,
            payload=arr.tobytes(),
        )
