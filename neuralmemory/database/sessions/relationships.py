from __future__ import annotations

import logging
from typing import Any, Callable

from neuralmemory.core.models import MemoryResult


class RelationshipManager:
    def __init__(
        self,
        collection: Any,
        read_memory_callback: Callable[[str], MemoryResult | None],
        logger: logging.Logger
    ) -> None:
        self._collection: Any = collection
        self._read_memory: Callable[[str], MemoryResult | None] = read_memory_callback
        self._logger: logging.Logger = logger

    def add_relationship(
        self,
        memory_id: str,
        related_memory_id: str,
        bidirectional: bool = True
    ) -> bool:
        try:
            memory1: MemoryResult | None = self._read_memory(memory_id)
            if not memory1 or not memory1.enhanced_metadata:
                return False

            related_ids: list[str] = list(memory1.enhanced_metadata.related_memory_ids)
            if related_memory_id not in related_ids:
                related_ids.append(related_memory_id)

            if self._collection:
                current_meta: dict[str, Any] = dict(memory1.metadata)
                current_meta["related_memory_ids"] = ",".join(related_ids)
                self._collection.update(
                    ids=[memory_id],
                    metadatas=[current_meta]
                )

            if bidirectional:
                memory2: MemoryResult | None = self._read_memory(related_memory_id)
                if memory2 and memory2.enhanced_metadata:
                    related_ids_2: list[str] = list(memory2.enhanced_metadata.related_memory_ids)
                    if memory_id not in related_ids_2:
                        related_ids_2.append(memory_id)
                        current_meta_2: dict[str, Any] = dict(memory2.metadata)
                        current_meta_2["related_memory_ids"] = ",".join(related_ids_2)
                        self._collection.update(
                            ids=[related_memory_id],
                            metadatas=[current_meta_2]
                        )

            self._logger.info(f"Added relationship: {memory_id} <-> {related_memory_id}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to add related memory: {e}")
            return False

    def get_related(
        self,
        memory_id: str,
        max_depth: int = 2
    ) -> list[MemoryResult]:
        visited: set[str] = set()
        related: list[MemoryResult] = []

        def traverse(current_id: str, depth: int) -> None:
            if depth > max_depth or current_id in visited:
                return

            visited.add(current_id)
            memory: MemoryResult | None = self._read_memory(current_id)

            if memory and memory.enhanced_metadata:
                related.append(memory)
                for related_id in memory.enhanced_metadata.related_memory_ids:
                    if related_id and related_id not in visited:
                        traverse(related_id, depth + 1)

        traverse(memory_id, 0)
        return related[1:]
