"""Validated transport wrapper for data quality checks."""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

from ..core_types import Transport, TensorSnapshot

logger = logging.getLogger(__name__)


class ValidatedTransport(Transport):
    """Wraps a transport with automatic validation.
    
    Validates tensor snapshots before publishing/after subscribing.
    Catches common issues: NaN, Inf, empty arrays, shape mismatches.
    
    Example:
        transport = ValidatedTransport(
            MemTransport(),
            check_nan=True,
            check_inf=True,
            check_range=(-100.0, 100.0)
        )
    """

    def __init__(
        self,
        wrapped: Transport,
        check_nan: bool = True,
        check_inf: bool = True,
        check_range: Optional[tuple[float, float]] = None,
        check_empty: bool = True,
    ):
        self.transport = wrapped
        self.check_nan = check_nan
        self.check_inf = check_inf
        self.check_range = check_range
        self.check_empty = check_empty
        self._subscribers = []

    def _validate(self, snap: TensorSnapshot) -> None:
        """Validate snapshot, raise ValueError on issues."""
        if not snap.payload:
            if self.check_empty:
                raise ValueError(f"Empty payload in snapshot: {snap.name}")
            return

        # Reconstruct array for validation
        try:
            arr = np.frombuffer(snap.payload, dtype=snap.dtype).reshape(snap.shape)
        except Exception as e:
            raise ValueError(f"Cannot reconstruct tensor {snap.name}: {e}")

        if self.check_nan and np.isnan(arr).any():
            raise ValueError(f"NaN detected in tensor: {snap.name}")

        if self.check_inf and np.isinf(arr).any():
            raise ValueError(f"Inf detected in tensor: {snap.name}")

        if self.check_range:
            min_val, max_val = self.check_range
            if arr.min() < min_val or arr.max() > max_val:
                raise ValueError(
                    f"Value out of range [{min_val}, {max_val}] in tensor {snap.name}: "
                    f"got [{arr.min()}, {arr.max()}]"
                )

    def publish(self, snap: TensorSnapshot) -> None:
        try:
            self._validate(snap)
            self.transport.publish(snap)
        except ValueError as e:
            logger.error(f"Validation failed on publish: {e}")
            raise

    def subscribe(self, callback: Callable[[TensorSnapshot], None]) -> None:
        def validating_callback(snap: TensorSnapshot) -> None:
            try:
                self._validate(snap)
                callback(snap)
            except ValueError as e:
                logger.error(f"Validation failed on subscribe: {e}")
                # Don't propagate invalid data

        self._subscribers.append(callback)
        self.transport.subscribe(validating_callback)
