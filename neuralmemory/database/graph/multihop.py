from __future__ import annotations

import logging
from typing import Callable

from neuralmemory.core.models import MultiHopQuery, MemoryResult, SearchResult


class MultiHopSearchEngine:
    def __init__(
        self,
        smart_search_callback: Callable[[str, int], list[SearchResult]],
        read_memory_callback: Callable[[str], MemoryResult | None],
        get_related_callback: Callable[[str, int], list[MemoryResult]],
        satisfies_temporal_callback: Callable[[str, str, str], bool],
        logger: logging.Logger
    ) -> None:
        self._smart_search: Callable[[str, int], list[SearchResult]] = smart_search_callback
        self._read_memory: Callable[[str], MemoryResult | None] = read_memory_callback
        self._get_related: Callable[[str, int], list[MemoryResult]] = get_related_callback
        self._satisfies_temporal: Callable[[str, str, str], bool] = satisfies_temporal_callback
        self._logger: logging.Logger = logger

    def search(self, query: MultiHopQuery) -> list[MemoryResult]:
        try:
            initial_results: list[SearchResult] = self._smart_search(
                query.starting_query,
                n_results=5
            )

            if not initial_results:
                return []

            visited: set[str] = set()
            final_results: list[MemoryResult] = []

            for result in initial_results:
                if result.memory_id and result.memory_id not in visited:
                    if query.temporal_constraint and query.temporal_anchor_memory_id:
                        if not self._satisfies_temporal(
                            result.memory_id,
                            query.temporal_anchor_memory_id,
                            query.temporal_constraint
                        ):
                            continue

                    memory: MemoryResult | None = self._read_memory(result.memory_id)
                    if memory:
                        final_results.append(memory)
                        visited.add(result.memory_id)

                    if query.max_hops > 1:
                        related: list[MemoryResult] = self._get_related(
                            result.memory_id,
                            max_depth=query.max_hops - 1
                        )

                        for rel_mem in related:
                            if rel_mem.memory_id not in visited:
                                final_results.append(rel_mem)
                                visited.add(rel_mem.memory_id)

            self._logger.info(
                f"Multi-hop search: '{query.starting_query}' -> {len(final_results)} results "
                f"(max_hops={query.max_hops})"
            )

            return final_results[:10]

        except Exception as e:
            self._logger.error(f"Multi-hop search failed: {e}")
            return []
