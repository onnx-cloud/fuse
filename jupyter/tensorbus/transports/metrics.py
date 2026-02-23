"""MetricsTransport - Comprehensive Prometheus metrics wrapper."""
from __future__ import annotations

import time
from typing import Callable, Optional

from ..core_types import Transport, TensorSnapshot

# Optional prometheus dependency
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


class MetricsTransport(Transport):
    """Transport wrapper with comprehensive Prometheus metrics.
    
    Exposes metrics for:
    - Tensor size distribution
    - Publish/subscribe rates
    - Replication lag
    - Error rates
    
    Example:
        transport = MetricsTransport(
            MemTransport(),
            registry=REGISTRY,
            namespace="tensorbus"
        )
        
        # Metrics available at /metrics endpoint:
        # tensorbus_publishes_total
        # tensorbus_publish_bytes
        # tensorbus_publish_duration_seconds
        # tensorbus_subscribers_active
    
    Requires: prometheus_client (already in requirements.txt)
    """

    def __init__(
        self,
        wrapped: Transport,
        registry: Optional[CollectorRegistry] = None,
        namespace: str = "tensorbus",
    ):
        if not HAS_PROMETHEUS:
            raise ImportError(
                "MetricsTransport requires prometheus_client (should be in requirements)"
            )

        self.transport = wrapped
        # If caller doesn't provide a registry, create a local CollectorRegistry to
        # avoid duplicate metric registration when multiple wrappers are constructed
        # within the same process (e.g., in benchmark runs).
        self.registry = registry if registry is not None else CollectorRegistry()
        self.namespace = namespace

        # Counters
        self.publishes_total = Counter(
            f"{namespace}_publishes_total",
            "Total number of tensor snapshots published",
            ["tensor_name", "status"],
            registry=self.registry
        )

        self.subscribes_total = Counter(
            f"{namespace}_subscribes_total",
            "Total number of tensor snapshots received",
            ["tensor_name", "status"],
            registry=self.registry
        )

        # Histograms
        self.publish_bytes = Histogram(
            f"{namespace}_publish_bytes",
            "Size of published tensor snapshots in bytes",
            ["tensor_name"],
            buckets=[100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000],
            registry=self.registry
        )

        self.publish_duration = Histogram(
            f"{namespace}_publish_duration_seconds",
            "Duration of publish operations",
            ["tensor_name"],
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry
        )

        # Gauges
        self.active_subscribers = Gauge(
            f"{namespace}_subscribers_active",
            "Number of active subscribers",
            registry=self.registry
        )

        self.last_publish_timestamp = Gauge(
            f"{namespace}_last_publish_timestamp",
            "Unix timestamp of last publish",
            ["tensor_name"],
            registry=self.registry
        )

        self._subscribers = []

    def publish(self, snap: TensorSnapshot) -> None:
        start = time.time()
        
        try:
            # Record size
            payload_size = len(snap.payload or b"")
            self.publish_bytes.labels(tensor_name=snap.name).observe(payload_size)
            
            # Publish
            self.transport.publish(snap)
            
            # Record success
            self.publishes_total.labels(tensor_name=snap.name, status="success").inc()
            self.last_publish_timestamp.labels(tensor_name=snap.name).set(time.time())
            
        except Exception as e:
            self.publishes_total.labels(tensor_name=snap.name, status="error").inc()
            raise
        finally:
            # Record duration
            duration = time.time() - start
            self.publish_duration.labels(tensor_name=snap.name).observe(duration)

    def subscribe(self, callback: Callable[[TensorSnapshot], None]) -> None:
        # Wrap callback to record metrics
        def wrapped_callback(snap: TensorSnapshot) -> None:
            try:
                callback(snap)
                self.subscribes_total.labels(
                    tensor_name=snap.name,
                    status="success"
                ).inc()
            except Exception as e:
                self.subscribes_total.labels(
                    tensor_name=snap.name,
                    status="error"
                ).inc()
                raise

        self._subscribers.append(wrapped_callback)
        self.active_subscribers.set(len(self._subscribers))
        self.transport.subscribe(wrapped_callback)
