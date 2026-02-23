# generic structured logging wrapper

import logging
from typing import Any


class Logger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self.logger.debug(msg, extra={"props": kwargs})

    def info(self, msg: str, **kwargs: Any) -> None:
        self.logger.info(msg, extra={"props": kwargs})

    def warning(self, msg: str, **kwargs: Any) -> None:
        self.logger.warning(msg, extra={"props": kwargs})

    def error(self, msg: str, **kwargs: Any) -> None:
        self.logger.error(msg, extra={"props": kwargs})

    def critical(self, msg: str, **kwargs: Any) -> None:
        self.logger.critical(msg, extra={"props": kwargs})
