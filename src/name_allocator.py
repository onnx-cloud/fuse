from typing import Optional


class NameAllocator:
    """Abstract name allocator interface."""

    def next_node_name(self, op_type: str) -> str:
        raise NotImplementedError

    def next_const_name(self) -> str:
        raise NotImplementedError


class StableNameAllocator(NameAllocator):
    """Deterministic allocator that uses simple counters and optional scope info.

    Behavior mirrors existing GraphContext naming when used as a replacement.
    """

    def __init__(
        self,
        scope_prefix: Optional[str] = None,
        scope_display: Optional[str] = None,
    ):
        self._node_id = 0
        self._const_id = 0
        self.scope_prefix = scope_prefix
        self.scope_display = scope_display

    def next_node_name(self, op_type: str) -> str:
        # Special-case the very first emitted node in a scoped graph: use a
        # human-readable "module.node" name when available.
        if self._node_id == 0 and self.scope_display:
            name = self.scope_display
            self._node_id += 1
            return name
        prefix = f"{self.scope_prefix}__" if self.scope_prefix else ""
        name = f"{prefix}{op_type}_{self._node_id}"
        self._node_id += 1
        return name

    def next_const_name(self) -> str:
        prefix = f"{self.scope_prefix}__" if self.scope_prefix else ""
        name = f"{prefix}const_{self._const_id}"
        self._const_id += 1
        return name
