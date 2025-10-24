from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from neuralmemory.core.models import ConsolidationResult, StorageResult


class ConsolidationStrategy:
    def __init__(
        self,
        collection: Any,
        store_memory_callback: Callable[..., StorageResult],
        logger: logging.Logger
    ) -> None:
        self._collection: Any = collection
        self._store_memory: Callable[..., StorageResult] = store_memory_callback
        self._logger: logging.Logger = logger

    def consolidate_advanced(
        self,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 3,
        max_clusters: int = 10
    ) -> list[ConsolidationResult]:
        try:
            all_results: dict[str, Any] = self._collection.get(
                include=["documents", "metadatas", "embeddings"]
            )

            if not all_results or not all_results['ids'] or len(all_results['ids']) < min_cluster_size:
                return []

            clusters: list[list[int]] = self._find_clusters(
                all_results['embeddings'],
                similarity_threshold,
                min_cluster_size
            )

            consolidation_results: list[ConsolidationResult] = []

            for cluster in clusters[:max_clusters]:
                cluster_ids: list[str] = [all_results['ids'][idx] for idx in cluster]
                cluster_contents: list[str] = [all_results['documents'][idx] for idx in cluster]

                summary: str = self._create_summary(cluster_contents)

                summary_result: StorageResult = self._store_memory(
                    content=summary,
                    tags=["consolidated", "summary"],
                    memory_type="semantic",
                    importance=0.85
                )

                for cid in cluster_ids:
                    metadata_result: dict[str, Any] = self._collection.get(ids=[cid], include=["metadatas"])
                    if metadata_result and metadata_result['metadatas']:
                        meta: dict[str, Any] = metadata_result['metadatas'][0]
                        meta['outcome'] = "archived"
                        meta['consolidated_into'] = summary_result.memory_id
                        self._collection.update(ids=[cid], metadatas=[meta])

                consolidation_results.append(ConsolidationResult(
                    consolidated_count=len(cluster_ids),
                    summary_memory_id=summary_result.memory_id,
                    archived_memory_ids=cluster_ids,
                    consolidation_type="similarity"
                ))

                self._logger.info(
                    f"Consolidated {len(cluster_ids)} memories into {summary_result.memory_id[:8]}..."
                )

            return consolidation_results

        except Exception as e:
            self._logger.error(f"Consolidation failed: {e}")
            return []

    def _find_clusters(
        self,
        embeddings: list[list[float]],
        similarity_threshold: float,
        min_size: int
    ) -> list[list[int]]:
        if not embeddings or len(embeddings) < min_size:
            return []

        embeddings_array = np.array(embeddings)
        n: int = len(embeddings)
        visited: set[int] = set()
        clusters: list[list[int]] = []

        for i in range(n):
            if i in visited:
                continue

            cluster: list[int] = [i]
            visited.add(i)

            for j in range(i + 1, n):
                if j in visited:
                    continue

                similarity: float = np.dot(embeddings_array[i], embeddings_array[j]) / (
                    np.linalg.norm(embeddings_array[i]) * np.linalg.norm(embeddings_array[j])
                )

                if similarity >= similarity_threshold:
                    cluster.append(j)
                    visited.add(j)

            if len(cluster) >= min_size:
                clusters.append(cluster)

        return clusters

    def _create_summary(self, contents: list[str]) -> str:
        key_points: set[str] = set()

        for content in contents[:5]:
            first_sentence: str = content.split('.')[0].strip()
            if len(first_sentence) > 20:
                key_points.add(first_sentence)

        summary: str = f"Consolidated summary of {len(contents)} related memories:\n\n"
        summary += "\n".join(f"- {point}" for point in list(key_points)[:5])

        return summary
