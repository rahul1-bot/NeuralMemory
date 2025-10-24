from __future__ import annotations

import logging


class EntityIndex:
    def __init__(
        self,
        entity_index: dict[str, list[str]],
        logger: logging.Logger
    ) -> None:
        self._index: dict[str, list[str]] = entity_index
        self._logger: logging.Logger = logger

    def search(self, entities: list[str]) -> list[str]:
        memory_ids: set[str] = set()

        for entity in entities:
            entity_lower: str = entity.lower()
            if entity_lower in self._index:
                memory_ids.update(self._index[entity_lower])

        return list(memory_ids)
