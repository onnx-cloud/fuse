from __future__ import annotations

import logging
import os
import threading
import queue
import time
from typing import Optional, List, Callable

from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge

from ..core_types import Transport, TensorSnapshot

logger = logging.getLogger(__name__)


class PosixTransportBase(Transport):
    """Base class for POSIX file-backed transports.

    Subclasses must implement:
      - _serialize_snap(snap) -> bytes
      - load_snapshot(name) -> TensorSnapshot | None
    """

    def __init__(
        self,
        root_dir: str,
        worker_threads: int = 1,
        queue_size: int = 1024,
        metrics_registry: Optional[CollectorRegistry] = None,
        batch_window_ms: int = 0,
    ) -> None:
        self._batch_window_ms = int(batch_window_ms)
        self.root = root_dir
        os.makedirs(self.root, exist_ok=True)
        self._subscribers: List[Callable[[TensorSnapshot], None]] = []
        self._lock = threading.Lock()

        # Background worker queue (worker_threads==0 => synchronous mode)
        self._queue: "queue.Queue[Optional[TensorSnapshot]]" = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._workers: List[threading.Thread] = []
        self._sync_mode = worker_threads == 0
        if not self._sync_mode:
            for i in range(max(1, worker_threads)):
                t = threading.Thread(target=self._worker_loop, name=f"posix-base-worker-{i}", daemon=True)
                t.start()
                self._workers.append(t)

        # simple in-memory metrics for profiling (milliseconds)
        from typing import Dict, List

        self._metrics: Dict[str, List[float]] = {"serialize_ms": [], "write_ms": [], "total_ms": []}

        # Prometheus metrics
        self._registry = metrics_registry or CollectorRegistry()
        self._enqueue_counter = Counter("tensorbus_posix_enqueue_total", "Enqueued snapshots", registry=self._registry)
        self._written_counter = Counter("tensorbus_posix_written_total", "Completed writes", registry=self._registry)
        self._write_hist = Histogram("tensorbus_posix_write_seconds", "Write duration seconds", registry=self._registry)
        self._queue_gauge = Gauge("tensorbus_posix_queue_size", "Current queue size", registry=self._registry)

    def snapshot_metrics(self, clear: bool = True):
        """Return metrics summary and optionally clear the buffers."""
        import statistics

        def summarize(lst):
            if not lst:
                return {"count": 0, "mean": 0.0, "stdev": 0.0}
            return {"count": len(lst), "mean": statistics.mean(lst), "stdev": statistics.pstdev(lst)}

        s = {k: summarize(v) for k, v in self._metrics.items()}
        if clear:
            for k in list(self._metrics.keys()):
                self._metrics[k].clear()
        return s

    def _validate_name(self, name: str) -> None:
        if not name or os.path.basename(name) != name or \
           "/" in name or "\\" in name or name.startswith('.'):
            raise ValueError(f"Invalid snapshot name: {name}")

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                self._queue.task_done()
                break

            # Batching/dedup: collect items for a short window and keep only latest per name
            if self._batch_window_ms > 0:
                end_time = time.time() + (self._batch_window_ms / 1000.0)
                latest = {item.name: item}
                while time.time() < end_time:
                    try:
                        more = self._queue.get_nowait()
                    except queue.Empty:
                        time.sleep(0.001)
                        continue
                    if more is None:
                        self._queue.task_done()
                        break
                    latest[more.name] = more
                    self._queue.task_done()
                for snap in latest.values():
                    self._do_write(snap)
            else:
                self._do_write(item)

            self._queue.task_done()
            try:
                self._queue_gauge.set(self._queue.qsize())
            except Exception as e:
                logger.warning("Failed to set queue gauge: %s", e)

    def _atomic_write_bytes(self, data: bytes, path: str) -> bool:
        tmp_path = path + f".tmp.{os.getpid()}"
        try:
            with open(tmp_path, "wb") as f:
                f.write(data)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError as e:
                    logger.warning("fsync failed for %s: %s", tmp_path, e)
            last_exc = None
            for attempt in range(3):
                try:
                    os.replace(tmp_path, path)
                    last_exc = None
                    break
                except OSError as e:
                    last_exc = e
                    logger.warning("os.replace failed (attempt %d) for %s: %s", attempt + 1, tmp_path, e)
                    time.sleep(0.01)
            if last_exc:
                logger.error("Failed to atomically replace %s -> %s: %s", tmp_path, path, last_exc)
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                return False
            return True
        except Exception as e:
            logger.exception("Atomic write failed for %s: %s", path, e)
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            return False

    def _do_write(self, snap: TensorSnapshot) -> None:
        """Serialize and write snapshot to disk. Subclasses may override but
        should reuse _atomic_write_bytes to perform the final write."""
        raise NotImplementedError

    def publish(self, snap: TensorSnapshot) -> None:
        """Enqueue a snapshot for background write. In sync mode writes inline."""
        # Validate name early
        try:
            self._validate_name(snap.name)
        except ValueError:
            logger.error("publish rejected: invalid snapshot name: %s", snap.name)
            return

        if self._sync_mode:
            self._do_write(snap)
            return

        t0 = time.time()
        try:
            self._queue.put_nowait(snap)
        except queue.Full:
            self._queue.put(snap)
        self._enqueue_counter.inc()
        try:
            self._queue_gauge.set(self._queue.qsize())
        except Exception as e:
            logger.warning("Failed to set queue gauge: %s", e)

        # Wait briefly for background worker to materialize file & invoke subscribers
        path = os.path.join(self.root, f"{snap.name}.onnx")
        max_wait = 0.5  # seconds
        waited = 0.0
        interval = 0.005
        while waited < max_wait:
            if os.path.exists(path):
                try:
                    mtime = os.path.getmtime(path)
                    if mtime >= t0:
                        break
                except Exception:
                    break
            time.sleep(interval)
            waited += interval

    def flush(self, timeout: Optional[float] = None) -> None:
        start = time.perf_counter()
        try:
            self._queue.join()
        except Exception:
            return
        if timeout is not None:
            elapsed = time.perf_counter() - start
            if elapsed > timeout:
                raise TimeoutError("flush timed out")

    def close(self) -> None:
        self.flush()
        self._stop.set()
        for _ in self._workers:
            try:
                self._queue.put_nowait(None)
            except Exception:
                pass
        for w in self._workers:
            w.join()

    def subscribe(self, cb: Callable[[TensorSnapshot], None]) -> None:
        with self._lock:
            self._subscribers.append(cb)

    # load_snapshot must be implemented by subclass
    def load_snapshot(self, name: str) -> Optional[TensorSnapshot]:
        raise NotImplementedError
