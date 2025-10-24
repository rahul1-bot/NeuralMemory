from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable

from neuralmemory.core.models import SearchResult, EnhancedMemoryMetadata


class FilteringStrategy:
    def __init__(
        self,
        smart_search_callback: Callable[[str, int], list[SearchResult]],
        logger: logging.Logger
    ) -> None:
        self._smart_search: Callable[[str, int], list[SearchResult]] = smart_search_callback
        self._logger: logging.Logger = logger

    def filtered_search(
        self,
        query: str,
        n_results: int = 3,
        memory_type: str | None = None,
        importance_min: float | None = None,
        importance_max: float | None = None,
        project: str | None = None,
        session_id: str | None = None,
        entities: list[str] | None = None,
        topics: list[str] | None = None,
        has_action_items: bool | None = None,
        outcome: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None
    ) -> list[SearchResult]:
        where_filters: dict[str, Any] = {}

        if memory_type:
            where_filters["memory_type"] = memory_type
        if importance_min is not None:
            where_filters["importance"] = {"$gte": importance_min}
        if importance_max is not None:
            if "importance" in where_filters:
                where_filters["importance"]["$lte"] = importance_max
            else:
                where_filters["importance"] = {"$lte": importance_max}
        if project:
            where_filters["project"] = project
        if session_id:
            where_filters["session_id"] = session_id
        if outcome:
            where_filters["outcome"] = outcome

        results: list[SearchResult] = self._smart_search(query, n_results=n_results * 2)

        filtered_results: list[SearchResult] = []
        for result in results:
            if not result.enhanced_metadata:
                continue

            meta: EnhancedMemoryMetadata = result.enhanced_metadata

            if entities and not any(e in meta.entities for e in entities):
                continue

            if topics and not any(t in meta.topics for t in topics):
                continue

            if has_action_items is not None:
                has_items: bool = len(meta.action_items) > 0
                if has_items != has_action_items:
                    continue

            if start_date and meta.timestamp < start_date:
                continue
            if end_date and meta.timestamp > end_date:
                continue

            filtered_results.append(result)

        self._logger.info(f"Filtered search returned {len(filtered_results)} results from {len(results)} candidates")
        return filtered_results[:n_results]
