"""Lightweight session manager for Jupyter kernels.

Maintains a small state object to live in the IPython user namespace to provide
stateful sessions that match notebook expectations: graph_context, variables,
imported modules, and training state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Set, Optional


@dataclass
class SessionManager:
    """Simple container to be instantiated per kernel session.

    Typical usage in notebooks:
      %load_ext src.jupyter.ipython
      from fuse_jupyter import session
      session = _fuse_session  # created by extension
    """

    graph_context: Optional[Any] = None
    variables: Dict[str, Any] = field(default_factory=dict)
    modules: Set[str] = field(default_factory=set)
    training_state: Dict[str, Any] = field(default_factory=dict)

    def set_var(self, name: str, value: Any) -> None:
        self.variables[name] = value

    def get_var(self, name: str, default: Any = None) -> Any:
        return self.variables.get(name, default)

    def record_module(self, module_name: str) -> None:
        self.modules.add(module_name)

    def set_training_state(self, key: str, value: Any) -> None:
        self.training_state[key] = value

    def clear(self) -> None:
        self.graph_context = None
        self.variables.clear()
        self.modules.clear()
        self.training_state.clear()
