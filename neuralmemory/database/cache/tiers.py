from __future__ import annotations

import logging
from typing import Callable

from neuralmemory.core.models import MemoryResult, SearchResult


class TierAwareRetrieval:
    def __init__(
        self,
        working_memory: dict[str, MemoryResult],
        hybrid_search_callback: Callable[[str, int], list[SearchResult]],
        promote_callback: Callable[[str], None],
        logger: logging.Logger
    ) -> None:
        self._working_memory: dict[str, MemoryResult] = working_memory
        self._hybrid_search: Callable[[str, int], list[SearchResult]] = hybrid_search_callback
        self._promote: Callable[[str], None] = promote_callback
        self._logger: logging.Logger = logger

    def retrieve(self, query: str, n_results: int = 5) -> list[SearchResult]:
        working_mem_results: list[SearchResult] = []
        for mem_id, mem_result in self._working_memory.items():
            if any(keyword.lower() in mem_result.content.lower() for keyword in query.lower().split()):
                working_mem_results.append(
                    SearchResult(
                        rank=len(working_mem_results) + 1,
                        content=mem_result.content,
                        rerank_score=0.95,
                        cosine_distance=0.0,
                        metadata=mem_result.metadata,
                        memory_id=mem_result.memory_id,
                        short_id=mem_result.short_id,
                        enhanced_metadata=mem_result.enhanced_metadata
                    )
                )

        if working_mem_results:
            self._logger.info(f"Found {len(working_mem_results)} results in working memory (0s latency)")
            return working_mem_results[:n_results]

        results: list[SearchResult] = self._hybrid_search(query, n_results)

        for result in results:
            if result.enhanced_metadata and result.enhanced_metadata.access_frequency >= 3:
                if result.memory_id:
                    self._promote(result.memory_id)

        return results
