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
        logger: logging.Logger
    ) -> None:
        self._collection: Any = collection
        self._enabled: bool = enable_biological_decay
        self._logger: logging.Logger = logger

    def apply_decay(
        self,
        memory_id: str,
        metadata: EnhancedMemoryMetadata
    ) -> EnhancedMemoryMetadata:
        if not self._enabled or metadata.decay_counter is None:
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
            all_results: dict[str, Any] = self._collection.get()

            if not all_results or not all_results['ids']:
                return 0

            deleted_count: int = 0

            for idx, memory_id in enumerate(all_results['ids']):
                metadata_dict: dict[str, Any] = all_results['metadatas'][idx] if all_results['metadatas'] else {}

                if 'decay_counter' in metadata_dict and metadata_dict['decay_counter'] is not None:
                    decay_counter: int = int(metadata_dict['decay_counter'])

                    new_counter: int = decay_counter - 1

                    if new_counter <= 0:
                        self._collection.delete(ids=[memory_id])
                        deleted_count += 1
                        self._logger.info(f"Deleted expired memory: {memory_id[:8]}... (decay counter reached 0)")
                    else:
                        metadata_dict['decay_counter'] = new_counter
                        self._collection.update(
                            ids=[memory_id],
                            metadatas=[metadata_dict]
                        )

            self._logger.info(f"Applied decay: deleted {deleted_count} memories")
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

            if 'decay_counter' in metadata_dict and metadata_dict['decay_counter'] is not None:
                metadata_dict['decay_counter'] = 5
                metadata_dict['last_accessed'] = datetime.now().isoformat()
                metadata_dict['access_count'] = metadata_dict.get('access_count', 0) + 1

                self._collection.update(
                    ids=[memory_id],
                    metadatas=[metadata_dict]
                )

                self._logger.debug(f"Reinforced memory: {memory_id[:8]}... (reset decay counter)")

            return True

        except Exception as e:
            self._logger.error(f"Failed to reinforce memory: {e}")
            return False
