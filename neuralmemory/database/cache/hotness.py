from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable

from neuralmemory.core.models import EnhancedMemoryMetadata, MemoryTier


class HotnessCalculator:
    def __init__(
        self,
        collection: Any,
        short_term_days: int,
        logger: logging.Logger
    ) -> None:
        self._collection: Any = collection
        self._short_term_days: int = short_term_days
        self._logger: logging.Logger = logger

    def calculate(self, metadata: EnhancedMemoryMetadata) -> float:
        frequency_score: float = min(1.0, metadata.access_frequency / 10.0)

        days_old: int = (datetime.now() - metadata.timestamp).days
        if days_old < 7:
            recency_score: float = (7 - days_old) / 7.0
        else:
            recency_score = 0.1

        hotness: float = (0.7 * frequency_score) + (0.3 * recency_score)

        return hotness

    def tier_by_age(
        self,
        calculate_hotness_callback: Callable[[EnhancedMemoryMetadata], float]
    ) -> dict[str, int]:
        results = self._collection.get(include=["metadatas"])

        if not results or not results.get("ids"):
            return {"total": 0, "archived": 0, "short_term": 0, "working": 0}

        stats: dict[str, int] = {"total": 0, "archived": 0, "short_term": 0, "working": 0}
        updated_ids: list[str] = []
        updated_metadatas: list[dict[str, Any]] = []

        for memory_id, metadata_dict in zip(results["ids"], results["metadatas"]):
            stats["total"] += 1
            metadata: EnhancedMemoryMetadata = EnhancedMemoryMetadata.from_chromadb_dict(metadata_dict)

            days_old: int = (datetime.now() - metadata.timestamp).days

            new_tier: MemoryTier = metadata.tier

            if days_old > self._short_term_days and metadata.importance < 0.9:
                hotness: float = calculate_hotness_callback(metadata)
                if hotness < 0.3:
                    new_tier = MemoryTier.ARCHIVE
                    stats["archived"] += 1

            elif days_old <= self._short_term_days or metadata.importance >= 0.9:
                new_tier = MemoryTier.SHORT_TERM
                stats["short_term"] += 1

            if new_tier != metadata.tier:
                updated_metadata: EnhancedMemoryMetadata = EnhancedMemoryMetadata(
                    memory_type=metadata.memory_type,
                    importance=metadata.importance,
                    session_id=metadata.session_id,
                    project=metadata.project,
                    entities=metadata.entities,
                    topics=metadata.topics,
                    action_items=metadata.action_items,
                    outcome=metadata.outcome,
                    access_frequency=metadata.access_frequency,
                    last_accessed=metadata.last_accessed,
                    parent_memory_id=metadata.parent_memory_id,
                    related_memory_ids=metadata.related_memory_ids,
                    sequence_num=metadata.sequence_num,
                    timestamp=metadata.timestamp,
                    tier=new_tier,
                    code_references=metadata.code_references,
                    stale=metadata.stale,
                    stale_reason=metadata.stale_reason,
                    decay_counter=metadata.decay_counter,
                    memory_strength=metadata.memory_strength
                )
                updated_ids.append(memory_id)
                updated_metadatas.append(updated_metadata.to_chromadb_dict())

        if updated_ids:
            self._collection.update(ids=updated_ids, metadatas=updated_metadatas)
            self._logger.info(f"Updated tiers for {len(updated_ids)} memories")

        return stats
