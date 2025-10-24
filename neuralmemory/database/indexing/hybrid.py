from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable

from neuralmemory.core.models import SearchResult


class HybridSearch:
    def __init__(
        self,
        enable_hybrid_retrieval: bool,
        bm25_search_callback: Callable[[str, int], list[tuple[str, float]]],
        semantic_search_callback: Callable[[str, int], list[SearchResult]],
        logger: logging.Logger
    ) -> None:
        self._enabled: bool = enable_hybrid_retrieval
        self._bm25_search: Callable[[str, int], list[tuple[str, float]]] = bm25_search_callback
        self._semantic_search: Callable[[str, int], list[SearchResult]] = semantic_search_callback
        self._logger: logging.Logger = logger

    def search(
        self,
        query: str,
        n_results: int = 5,
        importance_weight: float = 0.2,
        recency_weight: float = 0.1
    ) -> list[SearchResult]:
        if not self._enabled:
            return self._semantic_search(query, n_results)

        self._logger.info(f"Hybrid search: '{query[:50]}...'")

        bm25_results: list[tuple[str, float]] = self._bm25_search(query, top_k=n_results * 3)
        bm25_scores: dict[str, float] = {mem_id: score for mem_id, score in bm25_results}

        semantic_results: list[SearchResult] = self._semantic_search(query, n_results * 2)

        combined_scores: list[tuple[SearchResult, float]] = []

        for result in semantic_results:
            memory_id: str = result.memory_id or ""

            score: float = result.rerank_score

            if memory_id in bm25_scores:
                bm25_normalized: float = min(1.0, bm25_scores[memory_id] / 10.0)
                score += 0.3 * bm25_normalized

            if result.enhanced_metadata:
                score += importance_weight * result.enhanced_metadata.importance

                days_old: int = (datetime.now() - result.enhanced_metadata.timestamp).days
                if days_old < 7:
                    recency_boost: float = (7 - days_old) / 7.0
                    score += recency_weight * recency_boost

            combined_scores.append((result, score))

        combined_scores.sort(key=lambda x: x[1], reverse=True)

        final_results: list[SearchResult] = []
        for rank, (result, combined_score) in enumerate(combined_scores[:n_results], start=1):
            final_results.append(
                SearchResult(
                    rank=rank,
                    content=result.content,
                    rerank_score=combined_score,
                    cosine_distance=result.cosine_distance,
                    metadata=result.metadata,
                    memory_id=result.memory_id,
                    short_id=result.short_id,
                    enhanced_metadata=result.enhanced_metadata
                )
            )

        self._logger.info(f"Hybrid search returned {len(final_results)} results with combined scoring")
        return final_results
