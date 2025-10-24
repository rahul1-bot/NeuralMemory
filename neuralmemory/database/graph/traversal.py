from __future__ import annotations

import logging
from datetime import timedelta
from typing import Callable

from neuralmemory.core.models import MemoryResult


class GraphTraversal:
    def __init__(
        self,
        read_memory_callback: Callable[[str], MemoryResult | None],
        logger: logging.Logger
    ) -> None:
        self._read_memory: Callable[[str], MemoryResult | None] = read_memory_callback
        self._logger: logging.Logger = logger

    def satisfies_temporal_constraint(
        self,
        memory_id: str,
        anchor_memory_id: str,
        constraint: str
    ) -> bool:
        try:
            mem1: MemoryResult | None = self._read_memory(memory_id)
            mem2: MemoryResult | None = self._read_memory(anchor_memory_id)

            if not mem1 or not mem2:
                return False

            if constraint == "before":
                return mem1.timestamp < mem2.timestamp
            elif constraint == "after":
                return mem1.timestamp > mem2.timestamp
            elif constraint == "during":
                delta: timedelta = abs(mem1.timestamp - mem2.timestamp)
                return delta.total_seconds() < 86400

            return True

        except Exception as e:
            self._logger.error(f"Temporal constraint check failed: {e}")
            return False
