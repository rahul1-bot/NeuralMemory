from __future__ import annotations

import logging
from typing import Any


class MemoryStorage:
    def __init__(
        self,
        collection: Any,
        logger: logging.Logger
    ) -> None:
        self._collection: Any = collection
        self._logger: logging.Logger = logger
