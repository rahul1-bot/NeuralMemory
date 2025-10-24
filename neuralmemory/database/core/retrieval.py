from __future__ import annotations

import logging
import time
from typing import Any

from neuralmemory.core.models import SearchResult, EnhancedMemoryMetadata, MemoryResult
from neuralmemory.core.exceptions import MemoryValidationError, VectorDatabaseError
from neuralmemory.engines.embedding import Qwen3EmbeddingEngine
from neuralmemory.engines.reranker import Qwen2Reranker


class MemoryRetrieval:
    def __init__(
        self,
        collection: Any,
        embedding_engine: Qwen3EmbeddingEngine,
        reranker_engine: Qwen2Reranker,
        preprocess_query_callback,
        logger: logging.Logger
    ) -> None:
        self._collection: Any = collection
        self._embedding_engine: Qwen3EmbeddingEngine = embedding_engine
        self._reranker: Qwen2Reranker = reranker_engine
        self._preprocess_query = preprocess_query_callback
        self._logger: logging.Logger = logger

    def retrieve(self, query: str, n_results: int = 3) -> list[SearchResult]:
        if not query.strip():
            raise MemoryValidationError("Query cannot be empty")
        if n_results < 1:
            raise ValueError("n_results must be positive")

        self._logger.info(f"Retrieving memories for query: '{query[:50]}...' with n_results={n_results}")

        if self._embedding_engine is None or self._reranker is None or self._collection is None:
            raise VectorDatabaseError("Components not initialized")

        start_time: float = time.time()
        n_candidates: int = min(n_results * 3, 15)

        preprocessed_query: str = self._preprocess_query(query)

        try:
            self._logger.debug(f"Generating query embedding and fetching {n_candidates} candidates")
            query_embedding = self._embedding_engine.encode(preprocessed_query, is_query=True)

            results = self._collection.query(
                query_embeddings=query_embedding.cpu().numpy().tolist(),
                n_results=n_candidates,
                include=["documents", "metadatas", "distances"]
            )

            documents: list[str] = results["documents"][0]
            distances: list[float] = results["distances"][0]
            metadatas: list[dict[str, Any]] = results["metadatas"][0] or [{} for _ in documents]
            ids: list[str] = results["ids"][0]

            self._logger.debug(f"Retrieved {len(documents)} candidates, reranking...")
            reranked_indices: list[tuple[int, float]] = self._reranker.rerank(
                preprocessed_query, documents, top_k=n_results
            )

            search_results: list[SearchResult] = []
            for rank, (idx, score) in enumerate(reranked_indices, 1):
                if idx < len(documents):
                    enhanced_meta: EnhancedMemoryMetadata | None = None
                    if idx < len(metadatas) and metadatas[idx]:
                        try:
                            enhanced_meta = EnhancedMemoryMetadata.from_chromadb_dict(metadatas[idx])
                        except Exception as e:
                            self._logger.warning(f"Failed to parse enhanced metadata: {e}")

                    result: SearchResult = SearchResult(
                        rank=rank,
                        content=documents[idx],
                        rerank_score=score,
                        cosine_distance=distances[idx],
                        metadata=metadatas[idx],
                        memory_id=ids[idx],
                        short_id=metadatas[idx].get("short_id") if idx < len(metadatas) else None,
                        enhanced_metadata=enhanced_meta
                    )
                    search_results.append(result)

            total_time: float = time.time() - start_time
            self._logger.info(f"Retrieved {len(search_results)} results in {total_time:.3f} seconds")

            return search_results

        except Exception as e:
            self._logger.error(f"Memory retrieval failed: {e}")
            raise VectorDatabaseError(f"Memory retrieval failed: {e}")
