from __future__ import annotations

import logging
from typing import Any, Callable

from neuralmemory.core.models import MemoryProvenance, StorageResult


class ProvenanceTracker:
    def __init__(
        self,
        collection: Any,
        store_memory_callback: Callable[..., StorageResult],
        logger: logging.Logger
    ) -> None:
        self._collection: Any = collection
        self._store_memory: Callable[..., StorageResult] = store_memory_callback
        self._logger: logging.Logger = logger

    def store_with_provenance(
        self,
        content: str,
        provenance: MemoryProvenance,
        tags: list[str] | None = None,
        **kwargs
    ) -> StorageResult:
        kwargs['project'] = kwargs.get('project', provenance.created_by)

        result: StorageResult = self._store_memory(content=content, tags=tags, **kwargs)

        try:
            metadata_result: dict[str, Any] = self._collection.get(ids=[result.memory_id], include=["metadatas"])
            if metadata_result and metadata_result['metadatas']:
                meta: dict[str, Any] = metadata_result['metadatas'][0]
                meta['provenance_source'] = provenance.source
                meta['provenance_confidence'] = provenance.confidence
                if provenance.citation:
                    meta['provenance_citation'] = provenance.citation

                self._collection.update(ids=[result.memory_id], metadatas=[meta])

                self._logger.info(
                    f"Stored memory with provenance: {result.memory_id[:8]}... "
                    f"(source={provenance.source}, confidence={provenance.confidence:.2f})"
                )
        except Exception as e:
            self._logger.warning(f"Failed to update provenance: {e}")

        return result
