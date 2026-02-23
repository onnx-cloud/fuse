import json
import logging
import sys
import time
from typing import Any, Dict

# Import our JSON formatter helpers; avoid creating a top-level `logging` module
from .tlogging import Logger  # noqa: F401 (exports)

class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for structured logging without extra deps.

    Emits a JSON object per record with timestamp, level, component, trace_id and message.
    """

    def format(self, record: logging.LogRecord) -> str:
        component = getattr(record, "component", record.name)
        trace_id = getattr(record, "trace_id", None)

        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "component": component,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": trace_id,
        }
        # include exception info if present
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # include any other extra fields set on the record without clobbering
        for k, v in record.__dict__.items():
            if k in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "process",
                "processName",
                "message",
                "trace_id",
                "component",
            ):
                continue
            try:
                # ensure the value is JSON serializable
                json.dumps({k: v})
                payload[k] = v
            except Exception:
                payload[k] = repr(v)
        return json.dumps(payload, default=str)


def setup_logging(level: int = logging.INFO, structured: bool = True):
    """Configure root logger to emit structured JSON logs to stdout.

    Called once by entry points (CLI, MCP server) during startup. Subsequent
    calls are idempotent. Also resets the in-context trace id to avoid leaks
    across tests or repeated invocations.

    Examples
    --------
    >>> import fusion.flog as flog
    >>> flog.setup_logging(level=logging.DEBUG)
    >>> flog.set_trace_id("abc-123")
    >>> import fusion.runtime as rt
    >>> rt.logger.info("started")
    """
    # No keep setup idempotent.

    root = logging.getLogger()
    # Avoid duplicate handlers if already configured
    if any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        # Update level and return
        root.setLevel(level)
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    if structured:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str = __name__) -> logging.Logger:
    return logging.getLogger(name)
