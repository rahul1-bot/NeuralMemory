from __future__ import annotations

import logging


class BM25Index:
    def __init__(
        self,
        bm25_corpus: list[str],
        bm25_ids: list[str],
        logger: logging.Logger
    ) -> None:
        self._corpus: list[str] = bm25_corpus
        self._ids: list[str] = bm25_ids
        self._logger: logging.Logger = logger

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self._corpus:
            return []

        try:
            from rank_bm25 import BM25Okapi

            tokenized_corpus: list[list[str]] = [doc.lower().split() for doc in self._corpus]
            tokenized_query: list[str] = query.lower().split()

            bm25 = BM25Okapi(tokenized_corpus)
            scores: list[float] = bm25.get_scores(tokenized_query)

            scored_results: list[tuple[int, float]] = [
                (idx, score) for idx, score in enumerate(scores) if score > 0
            ]
            scored_results.sort(key=lambda x: x[1], reverse=True)

            return [(self._ids[idx], score) for idx, score in scored_results[:top_k]]

        except ImportError:
            self._logger.warning("rank_bm25 not installed, BM25 search unavailable")
            return []
        except Exception as e:
            self._logger.error(f"BM25 search failed: {e}")
            return []
