from __future__ import annotations

import logging
from typing import Callable

from neuralmemory.core.models import MemoryResult


class CacheEvictionPolicy:
    def __init__(
        self,
        working_memory: dict[str, MemoryResult],
        max_working_memory_size: int,
        read_memory_callback: Callable[[str], MemoryResult | None],
        logger: logging.Logger
    ) -> None:
        self._working_memory: dict[str, MemoryResult] = working_memory
        self._max_size: int = max_working_memory_size
        self._read_memory: Callable[[str], MemoryResult | None] = read_memory_callback
        self._logger: logging.Logger = logger

    def promote(self, memory_id: str) -> None:
        if memory_id in self._working_memory:
            return

        memory: MemoryResult | None = self._read_memory(memory_id)
        if not memory:
            return

        if len(self._working_memory) >= self._max_size:
            oldest_id: str = next(iter(self._working_memory))
            del self._working_memory[oldest_id]
            self._logger.debug(f"Evicted {oldest_id[:8]}... from working memory (LRU)")

        self._working_memory[memory_id] = memory
        self._logger.debug(f"Promoted {memory_id[:8]}... to working memory")
