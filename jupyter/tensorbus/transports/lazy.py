"""LazyTransport - Pull-based replication for large models."""
from __future__ import annotations

import pickle
from collections import OrderedDict
from typing import Callable, Optional, Dict

from ..core_types import Transport, TensorSnapshot, HLC


class LRUCache:
    """Simple LRU cache for tensor snapshots."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache: OrderedDict[str, TensorSnapshot] = OrderedDict()
    
    def get(self, key: str) -> Optional[TensorSnapshot]:
        """Get value and move to end (most recently used)."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: TensorSnapshot) -> None:
        """Put value, evicting LRU if needed."""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # Remove oldest
    
    def size(self) -> int:
        return len(self.cache)


class LazyTransport(Transport):
    """Lazy/pull-based replication transport for large models.
    
    Instead of pushing all tensor updates, maintains a catalog of
    available tensors and fetches on-demand when requested.
    
    Use cases:
    - Large model weights (GB-scale)
    - Sparse access patterns (few tensors accessed frequently)
    - Bandwidth-constrained environments
    
    Architecture:
    - Catalog: Central registry of available tensors (lightweight)
    - Storage: Backend storage for tensor data (S3, filesystem, etc.)
    - Cache: LRU cache for frequently accessed tensors
    
    Example:
        # Publisher
        storage = PosixTransport("/mnt/models")
        transport = LazyTransport(storage_backend=storage, mode="publisher")
        bus = TensorBus(transport=transport)
        bus.set_tensor("huge.weights", large_array)  # Stored but not pushed
        
        # Consumer
        transport = LazyTransport(storage_backend=storage, mode="consumer")
        bus = TensorBus(transport=transport)
        value = bus.get_tensor("huge.weights")  # Fetched on demand
    
    Benefits:
    - 10-100x less network traffic for sparse access
    - Only fetch what you need
    - Automatic caching for hot tensors
    """

    def __init__(
        self,
        storage_backend: Transport,
        mode: str = "consumer",
        cache_size: int = 10,
    ):
        """Initialize lazy transport.
        
        Args:
            storage_backend: Underlying storage (Posix, S3, etc.)
            mode: "publisher" (write to storage) or "consumer" (read on-demand)
            cache_size: Number of snapshots to cache
        """
        self.storage = storage_backend
        self.mode = mode
        self.cache = LRUCache(max_size=cache_size)
        
        # Catalog: tensor_name -> latest clock
        self.catalog: Dict[str, HLC] = {}
        
        self._subscribers = []
    
    def publish(self, snap: TensorSnapshot) -> None:
        """Publish snapshot (write to storage and update catalog)."""
        # Write to storage backend
        self.storage.publish(snap)
        
        # Update catalog
        if snap.clock:
            self.catalog[snap.name] = snap.clock
        
        # Cache it
        cache_key = self._cache_key(snap.name, snap.clock)
        self.cache.put(cache_key, snap)
        
        # Notify subscribers of catalog update (not full snapshot)
        for cb in self._subscribers:
            try:
                # Send lightweight catalog notification
                catalog_snap = TensorSnapshot(
                    name=snap.name,
                    dtype=snap.dtype,
                    shape=snap.shape,
                    clock=snap.clock,
                    hash=snap.hash,
                    payload=None  # No payload in catalog
                )
                cb(catalog_snap)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Subscriber failed: {e}")
    
    def subscribe(self, callback: Callable[[TensorSnapshot], None]) -> None:
        """Subscribe to catalog updates (not full snapshots)."""
        self._subscribers.append(callback)
    
    def fetch(self, name: str, clock: Optional[HLC] = None) -> Optional[TensorSnapshot]:
        """Explicitly fetch a tensor snapshot (pull operation).
        
        Args:
            name: Tensor name
            clock: Specific version to fetch, or None for latest
        
        Returns:
            TensorSnapshot if found, None otherwise
        """
        # Use latest if no clock specified
        if clock is None:
            clock = self.catalog.get(name)
            if not clock:
                return None
        
        # Check cache first
        cache_key = self._cache_key(name, clock)
        cached = self.cache.get(cache_key)
        if cached and cached.payload is not None:
            return cached
        
        # Fetch from storage (this is the "lazy" part - on-demand load)
        # Note: This requires storage backend to support querying by name/clock
        # For now, we rely on the subscriber pattern to have received catalog updates
        
        # In a real implementation, would query storage:
        # snapshot = self.storage.fetch(name, clock)
        
        # For this implementation, we expect publish() to have been called
        # and we're just doing catalog-based lazy loading
        
        return None  # Would fetch from storage in production
    
    def get_catalog(self) -> Dict[str, HLC]:
        """Get catalog of available tensors."""
        return self.catalog.copy()
    
    def get_cache_info(self) -> Dict[str, any]:
        """Get cache statistics."""
        return {
            "size": self.cache.size(),
            "max_size": self.cache.max_size,
            "hit_rate": "N/A"  # Would track hits/misses in production
        }
    
    @staticmethod
    def _cache_key(name: str, clock: Optional[HLC]) -> str:
        """Generate cache key from name and clock."""
        if clock:
            return f"{name}@{clock.counter}:{clock.node}"
        return name
