"""Simple stack of contexts to manage nested scopes during lowering."""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ContextStack:
    """Maintain a stack of mapping-like contexts for name resolution.

    Each context is a dictionary mapping identifiers to graph names or
    other metadata.  Lookup searches the stack from top (most recent)
    to bottom, enabling nested scopes with fall-through.
    """

    def __init__(self):
        self._stack: List[Dict[str, Any]] = []

    def push(self, ctx: Optional[Dict[str, Any]] = None) -> None:
        # Avoid the common Python idiom ``ctx or {}`` which treats any falsy
        # container (including empty dicts or EnvDicts) as equivalent to
        # ``None``.  An empty context is still a legitimate scope and should
        # be preserved rather than replaced with a fresh dict.
        self._stack.append(ctx if ctx is not None else {})

    def pop(self) -> Dict[str, Any]:
        return self._stack.pop()

    def current(self) -> Dict[str, Any]:
        return self._stack[-1] if self._stack else {}

    def lookup(self, name: str) -> Any:
        for ctx in reversed(self._stack):
            if name in ctx:
                return ctx[name]
        raise KeyError(f"name '{name}' not found in any context")

    def set(self, name: str, value: Any) -> None:
        if not self._stack:
            self.push()
        self._stack[-1][name] = value

    def __len__(self) -> int:
        return len(self._stack)

    def __repr__(self) -> str:
        return f"ContextStack(stack={self._stack})"


class EnvDict(dict):
    """Dictionary-like view backed by a ContextStack.

    Lookups traverse the context stack from top to bottom, while assignments
    always record into the current (top) frame.  This allows nested scopes to
    shadow names without losing parent values.

    This class *also* attempts to mimic enough of the built-in dict API to be
    safely used in contexts where callers treat the object as a plain dict
    (e.g. ``dict(env)`` or ``env.get(key)``).  The original implementation
    stored all data in an internal ``ContextStack`` and left the underlying
    ``dict`` itself empty; as a result, ``dict(env)`` produced an empty dict and
    ``env.get(...)`` always returned the default.  These bugs manifested in
    intermittent lowering failures (missing environment entries during inlining)
    and incorrect handling of ``@output`` annotations.
    """

    def __init__(self, initial: Optional[Dict[str, Any]] = None):
        super().__init__()
        self._stack = ContextStack()
        # ``initial`` may legitimately be an empty mapping (including another
        # EnvDict).  The previous idiom ``initial or {}`` treated any empty
        # container as missing, causing environments to be dropped when
        # wrapping an existing EnvDict.  Use an explicit ``is not None`` check
        # instead so that callers can inherit an empty context.
        self._stack.push(initial if initial is not None else {})

    def __getitem__(self, key: str) -> Any:
        return self._stack.lookup(key)

    def __setitem__(self, key: str, value: Any) -> None:
        # Always record into the current (top) frame
        self._stack.set(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style ``get`` that respects the stacked contexts.

        The built-in ``dict.get`` is a descriptor implemented in C and bypasses
        ``__getitem__``; as a result the original implementation always returned
        ``default``.  This override ensures we look through the stack exactly the
        same way ``__getitem__`` does.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def pop(self, key: Optional[str] = None, default: Any = None) -> Any:
        """Pop behaviour with dual semantics:

        - When called *without* arguments we treat it as a stack pop and return
          the top frame (old behaviour).
        - When ``key`` is provided we remove that binding from the current frame
          and return its value, or ``default`` if the name is absent (dict-like
          behaviour).  This mirrors how the built-in ``dict`` behaves and is
          used in a few lowering codepaths (e.g. clearing ``__last_multi_return__``).
        """
        if key is None:
            # stack-pop semantics
            return self._stack.pop()
        # dict-pop semantics: remove from current frame only
        try:
            val = self[key]
        except KeyError:
            return default
        try:
            # attempt to remove from the current frame dict directly
            self._stack.current().pop(key, None)
        except (AttributeError, RuntimeError) as e:
            # Stack may be empty or unavailable; non-critical
            logger.debug(f"Could not pop from context stack: {e}")
        return val

    def update(self, other: Dict[str, Any]) -> None:
        for k, v in other.items():
            self[k] = v

    def __iter__(self):
        # iterate all keys present in any frame, starting from the most recent
        seen = set()
        # traverse from top to bottom so more-recent bindings mask older ones
        for ctx in reversed(self._stack._stack):
            for k in ctx:
                if k not in seen:
                    seen.add(k)
                    yield k

    def keys(self):
        return list(self.__iter__())

    def items(self):
        return [(k, self[k]) for k in self.__iter__()]

    def push(self, ctx: Optional[Dict[str, Any]] = None) -> None:
        self._stack.push(ctx)

    def __contains__(self, key: object) -> bool:
        try:
            self._stack.lookup(key)  # type: ignore
            return True
        except KeyError:
            return False
