from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable

from neuralmemory.core.models import MemoryResult
from neuralmemory.core.exceptions import MemoryValidationError, VectorDatabaseError


class MemoryDeletion:
    def __init__(
        self,
        collection: Any,
        read_memory_callback: Callable[[str], MemoryResult | None],
        logger: logging.Logger
    ) -> None:
        self._collection: Any = collection
        self._read_memory: Callable[[str], MemoryResult | None] = read_memory_callback
        self._logger: logging.Logger = logger

    def delete(self, identifier: str, soft_delete: bool = False) -> bool:
        if not identifier:
            raise MemoryValidationError("Identifier cannot be empty")

        if self._collection is None:
            raise VectorDatabaseError("Database not initialized")

        self._logger.info(f"{'Soft' if soft_delete else 'Hard'} deleting memory: {identifier}")

        existing_memory: MemoryResult | None = self._read_memory(identifier)
        if not existing_memory:
            raise MemoryValidationError(f"Memory not found: {identifier}")

        try:
            if soft_delete:
                new_metadata: dict[str, Any] = dict(existing_memory.metadata) if existing_memory.metadata else {}
                new_metadata["deleted"] = True
                new_metadata["deleted_at"] = datetime.now().isoformat()

                self._collection.update(
                    ids=[existing_memory.memory_id],
                    metadatas=[new_metadata]
                )
                self._logger.info(f"Soft deleted memory: {existing_memory.memory_id}")
            else:
                self._collection.delete(ids=[existing_memory.memory_id])
                self._logger.info(f"Hard deleted memory: {existing_memory.memory_id}")

            return True

        except Exception as e:
            self._logger.error(f"Failed to delete memory: {e}")
            raise VectorDatabaseError(f"Failed to delete memory: {e}")

    def batch_delete(self, identifiers: list[str], soft_delete: bool = False) -> dict[str, bool]:
        if not identifiers:
            raise MemoryValidationError("No identifiers provided")

        self._logger.info(f"Batch {'soft' if soft_delete else 'hard'} deleting {len(identifiers)} memories")

        delete_results: dict[str, bool] = {}
        for identifier in identifiers:
            try:
                success: bool = self.delete(identifier, soft_delete)
                delete_results[identifier] = success
            except (MemoryValidationError, VectorDatabaseError) as e:
                self._logger.warning(f"Failed to delete {identifier}: {e}")
                delete_results[identifier] = False

        successful_deletes: int = sum(1 for success in delete_results.values() if success)
        self._logger.info(f"Batch delete completed: {successful_deletes}/{len(identifiers)} successful")
        return delete_results
