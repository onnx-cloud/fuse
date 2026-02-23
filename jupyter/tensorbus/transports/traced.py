"""TracedTransport - OpenTelemetry distributed tracing wrapper."""
from __future__ import annotations

import time
from typing import Callable, Optional

from ..core_types import Transport, TensorSnapshot

# Optional opentelemetry dependency
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False


class TracedTransport(Transport):
    """Transport wrapper with OpenTelemetry distributed tracing.
    
    Adds tracing spans for publish/subscribe operations with:
    - Tensor metadata (name, size, dtype)
    - Clock information (HLC counter)
    - Operation timing
    - Error tracking
    
    Example:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
        
        # Setup tracer
        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
        
        # Use traced transport
        transport = TracedTransport(
            MemTransport(),
            tracer_name="tensorbus"
        )
    
    Requires: pip install opentelemetry-api opentelemetry-sdk
    """

    def __init__(
        self,
        wrapped: Transport,
        tracer_name: str = "tensorbus",
        tracer_provider: Optional[trace.TracerProvider] = None,
    ):
        if not HAS_OTEL:
            raise ImportError(
                "TracedTransport requires: pip install opentelemetry-api opentelemetry-sdk"
            )

        self.transport = wrapped
        # Store tracer name and provider and resolve a tracer at call time to
        # ensure we pick up the test fixture's tracer provider (which may be
        # installed after module import or if an environment pre-set provider
        # prevented overriding during test setup).
        self._tracer_name = tracer_name
        self._tracer_provider = tracer_provider
        # Keep a convenience tracer if available, but prefer resolving at call
        # time to avoid using a stale tracer bound to a different provider.
        try:
            self.tracer = trace.get_tracer(tracer_name, tracer_provider=tracer_provider)
        except Exception:
            self.tracer = None

    def publish(self, snap: TensorSnapshot) -> None:
        # Resolve tracer at call-time to pick up any test-installed provider
        tracer = None
        try:
            tracer = trace.get_tracer(self._tracer_name, tracer_provider=self._tracer_provider)
        except Exception:
            tracer = self.tracer

        if tracer is None:
            # Fallback: no tracing available, just forward
            self.transport.publish(snap)
            return

        with tracer.start_as_current_span("transport.publish") as span:
            # Add tensor metadata
            span.set_attribute("tensor.name", snap.name)
            span.set_attribute("tensor.dtype", snap.dtype)
            span.set_attribute("tensor.shape", str(snap.shape))
            span.set_attribute("tensor.size_bytes", len(snap.payload or b""))
            
            # Add clock info
            if snap.clock:
                span.set_attribute("tensor.clock.counter", snap.clock.counter)
                span.set_attribute("tensor.clock.node", snap.clock.node)
            
            # Add hash
            span.set_attribute("tensor.hash", snap.hash.hex())
            
            try:
                start = time.time()
                self.transport.publish(snap)
                duration_ms = (time.time() - start) * 1000
                
                span.set_attribute("operation.duration_ms", duration_ms)
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    def subscribe(self, callback: Callable[[TensorSnapshot], None]) -> None:
        # Wrap callback to add tracing
        def wrapped_callback(snap: TensorSnapshot) -> None:
            tracer = None
            try:
                tracer = trace.get_tracer(self._tracer_name, tracer_provider=self._tracer_provider)
            except Exception:
                tracer = self.tracer

            if tracer is None:
                callback(snap)
                return

            with tracer.start_as_current_span("transport.subscribe") as span:
                span.set_attribute("tensor.name", snap.name)
                span.set_attribute("tensor.size_bytes", len(snap.payload or b""))
                
                if snap.clock:
                    span.set_attribute("tensor.clock.counter", snap.clock.counter)
                
                try:
                    start = time.time()
                    callback(snap)
                    duration_ms = (time.time() - start) * 1000
                    
                    span.set_attribute("callback.duration_ms", duration_ms)
                    span.set_status(Status(StatusCode.OK))
                    
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        self.transport.subscribe(wrapped_callback)
