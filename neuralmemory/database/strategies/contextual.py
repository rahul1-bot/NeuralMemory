from __future__ import annotations

import logging
from typing import Any

from neuralmemory.core.models import ConflictDetectionResult
from neuralmemory.engines.embedding import Qwen3EmbeddingEngine


class ContextualEmbeddingStrategy:
    def __init__(
        self,
        collection: Any,
        embedding_engine: Qwen3EmbeddingEngine,
        enable_contextual_embeddings: bool,
        conflict_similarity_threshold: float,
        logger: logging.Logger
    ) -> None:
        self._collection: Any = collection
        self._embedding_engine: Qwen3EmbeddingEngine = embedding_engine
        self._enabled: bool = enable_contextual_embeddings
        self._threshold: float = conflict_similarity_threshold
        self._logger: logging.Logger = logger

    def encode_with_context(self, content: str, context_memories: list[str]) -> list[float]:
        if not self._enabled or not context_memories:
            return self._embedding_engine.encode(content).tolist()

        context_text: str = "\n".join(context_memories[:3])
        contextualized_content: str = f"Previous context:\n{context_text}\n\nCurrent memory:\n{content}"

        self._logger.debug(f"Encoding with context: {len(context_memories)} memories")
        return self._embedding_engine.encode(contextualized_content).tolist()

    def detect_conflicts(
        self,
        memory_id: str,
        content: str,
        embedding: list[float] | None = None
    ) -> list[ConflictDetectionResult]:
        if not self._enabled:
            return []

        try:
            if embedding is None:
                embedding_tensor = self._embedding_engine.encode(content)
                embedding = embedding_tensor.tolist()

            results: dict[str, Any] = self._collection.query(
                query_embeddings=[embedding],
                n_results=5
            )

            conflicts: list[ConflictDetectionResult] = []

            if results and results['ids'] and len(results['ids']) > 0:
                for idx, conflicting_id in enumerate(results['ids'][0]):
                    if conflicting_id == memory_id:
                        continue

                    distance: float = results['distances'][0][idx]
                    similarity: float = 1.0 - distance

                    if similarity >= self._threshold:
                        conflict_type: str = "update"
                        if similarity >= 0.98:
                            conflict_type = "duplicate"
                        elif similarity < 0.95:
                            conflict_type = "contradiction"

                        conflicts.append(ConflictDetectionResult(
                            memory_id=memory_id,
                            conflicting_memory_id=conflicting_id,
                            similarity_score=similarity,
                            conflict_type=conflict_type
                        ))

                        self._logger.info(
                            f"Conflict detected: {memory_id[:8]}... vs {conflicting_id[:8]}... "
                            f"(similarity={similarity:.3f}, type={conflict_type})"
                        )

            return conflicts

        except Exception as e:
            self._logger.error(f"Conflict detection failed: {e}")
            return []
