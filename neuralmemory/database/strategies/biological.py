from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from neuralmemory.core.models import EnhancedMemoryMetadata


class BiologicalDecayStrategy:
    def __init__(
        self,
        collection: Any,
        enable_biological_decay: bool,
        logger: logging.Logger,
        deletion_threshold: float = 0.1
    ) -> None:
        self._collection: Any = collection
        self._enabled: bool = enable_biological_decay
        self._logger: logging.Logger = logger
        self._deletion_threshold: float = deletion_threshold

    def apply_decay(
        self,
        memory_id: str,
        metadata: EnhancedMemoryMetadata
    ) -> EnhancedMemoryMetadata:
        if not self._enabled:
            return metadata

        days_since_created: int = (datetime.now() - metadata.timestamp).days

        decayed_strength: float = metadata.memory_strength * (0.5 ** days_since_created)

        updated_metadata: EnhancedMemoryMetadata = EnhancedMemoryMetadata(
            **{
                **metadata.model_dump(),
                "memory_strength": max(0.0, min(1.0, decayed_strength))
            }
        )

        return updated_metadata

    def apply_decay_to_all(self) -> int:
        if not self._enabled:
            return 0

        try:
            all_results: dict[str, Any] = self._collection.get(include=["metadatas"])

            if not all_results or not all_results['ids']:
                return 0

            deleted_count: int = 0

            for idx, memory_id in enumerate(all_results['ids']):
                metadata_dict: dict[str, Any] = all_results['metadatas'][idx] if all_results['metadatas'] else {}

                timestamp_str: str | None = metadata_dict.get('timestamp')
                if not timestamp_str:
                    continue

                try:
                    timestamp: datetime = datetime.fromisoformat(timestamp_str)
                except (ValueError, TypeError):
                    continue

                days_since_created: int = (datetime.now() - timestamp).days

                memory_strength: float = float(metadata_dict.get('memory_strength', 1.0))

                decayed_strength: float = memory_strength * (0.5 ** days_since_created)

                if decayed_strength < self._deletion_threshold:
                    self._collection.delete(ids=[memory_id])
                    deleted_count += 1
                    self._logger.info(
                        f"Deleted decayed memory: {memory_id[:8]}... "
                        f"(strength: {decayed_strength:.4f} < threshold: {self._deletion_threshold})"
                    )
                else:
                    metadata_dict['memory_strength'] = max(0.0, min(1.0, decayed_strength))
                    self._collection.update(
                        ids=[memory_id],
                        metadatas=[metadata_dict]
                    )

            self._logger.info(f"Applied Ebbinghaus decay: updated all memories, deleted {deleted_count} weak memories")
            return deleted_count

        except Exception as e:
            self._logger.error(f"Failed to apply decay: {e}")
            return 0

    def reinforce(self, memory_id: str) -> bool:
        if not self._enabled:
            return True

        try:
            result: dict[str, Any] = self._collection.get(ids=[memory_id], include=["metadatas"])

            if not result or not result['ids']:
                return False

            metadata_dict: dict[str, Any] = result['metadatas'][0]

            metadata_dict['memory_strength'] = 1.0
            metadata_dict['last_accessed'] = datetime.now().isoformat()
            metadata_dict['access_count'] = metadata_dict.get('access_count', 0) + 1

            self._collection.update(
                ids=[memory_id],
                metadatas=[metadata_dict]
            )

            self._logger.debug(f"Reinforced memory: {memory_id[:8]}... (reset strength to 1.0)")

            return True

        except Exception as e:
            self._logger.error(f"Failed to reinforce memory: {e}")
            return False
