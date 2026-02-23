import mmap
import pickle
import tempfile
import threading
from typing import Optional
from ..core_types import Transport
from collections.abc import Callable

from ..core_types import TensorSnapshot


class MemTransport(Transport):
    """In-memory pub/sub transport.

    Two modes:
    - default (in-memory bytes): fast and simple, stores the full serialized
      snapshot as bytes in ``self._buf``.
    - mmap-backed: when ``use_mmap=True`` an anonymous file-backed mmap is
      used and will be grown (re-mapped) if a serialized snapshot doesn't fit.

    The mmap implementation avoids unbounded allocations for very large
    snapshots and can be more efficient for repeated large writes.
    """

    def __init__(self, size: int = 1024 * 1024, use_mmap: bool = False):
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[TensorSnapshot], None]] = []
        self._use_mmap = use_mmap

        if self._use_mmap:
            # Create a temporary file and an mmap mapping of the requested size.
            self._tmpfile = tempfile.TemporaryFile()
            self._tmpfile.truncate(size)
            self._mmap = mmap.mmap(self._tmpfile.fileno(), size, access=mmap.ACCESS_WRITE)
            self._mmap_size = size
            self._buf_len = 0
        else:
            # Store the last published bytes directly; this avoids unnecessary
            # preallocation and copying for small/typical workloads.
            self._buf: bytes = b""

    def _ensure_mmap_size(self, n: int):
        """Ensure mmap has capacity for n bytes; grow (and remap) if needed."""
        if n <= self._mmap_size:
            return
        # Grow by at least doubling to amortize remaps.
        new_size = max(n, self._mmap_size * 2)
        self._tmpfile.truncate(new_size)
        # Close old mapping and create a new one with the larger size.
        try:
            self._mmap.close()
        except Exception:
            pass
        self._mmap = mmap.mmap(self._tmpfile.fileno(), new_size, access=mmap.ACCESS_WRITE)
        self._mmap_size = new_size

    def publish(self, snap: TensorSnapshot):
        data = pickle.dumps(snap, protocol=pickle.HIGHEST_PROTOCOL)
        with self._lock:
            if self._use_mmap:
                self._ensure_mmap_size(len(data))
                self._mmap.seek(0)
                self._mmap.write(data)
                self._buf_len = len(data)
            else:
                # Replace the last buffer atomically while holding the lock.
                self._buf = data

        # Notify subscribers without holding the lock to avoid deadlocks.
        for cb in list(self._subscribers):
            cb(snap)

    def subscribe(self, cb: Callable[[TensorSnapshot], None]):
        self._subscribers.append(cb)

    def last_bytes(self) -> bytes:
        """Return the last serialized snapshot as bytes (thread-safe)."""
        with self._lock:
            if self._use_mmap:
                self._mmap.seek(0)
                return self._mmap.read(self._buf_len)
            return self._buf

    def close(self):
        """Release resources used by the mmap-backed transport."""
        if not self._use_mmap:
            return
        try:
            self._mmap.close()
        except Exception:
            pass
        try:
            self._tmpfile.close()
        except Exception:
            pass

    def __del__(self):
        # Best-effort cleanup of temporary resources.
        try:
            self.close()
        except Exception:
            pass
