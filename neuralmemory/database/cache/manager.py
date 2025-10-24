from __future__ import annotations

import logging

from neuralmemory.core.models import MemoryResult


class CacheManager:
    def __init__(
        self,
        working_memory: dict[str, MemoryResult],
        logger: logging.Logger
    ) -> None:
        self._working_memory: dict[str, MemoryResult] = working_memory
        self._logger: logging.Logger = logger

    def get_all(self) -> list[MemoryResult]:
        return list(self._working_memory.values())

    def clear(self) -> None:
        count: int = len(self._working_memory)
        self._working_memory.clear()
        self._logger.info(f"Cleared {count} memories from working memory")

    def contains(self, memory_id: str) -> bool:
        return memory_id in self._working_memory
