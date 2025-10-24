from __future__ import annotations
import json
import logging
from datetime import datetime
from typing import Any

from neuralmemory.core.models import MemoryExport
from neuralmemory.core.exceptions import VectorDatabaseError


class MemoryExporter:

    def __init__(
        self,
        collection: Any,
        sessions: dict[str, Any],
        logger: logging.Logger
    ) -> None:
        self._collection = collection
        self._sessions = sessions
        self._logger = logger

    def export_memories(
        self,
        file_path: str,
        session_id: str | None = None,
        project: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None
    ) -> MemoryExport:
        try:
            all_results: dict[str, Any] = self._collection.get(
                include=["documents", "metadatas", "embeddings"]
            )

            if not all_results or not all_results['ids']:
                return MemoryExport(total_memories=0, memories=[])

            memories: list[dict[str, Any]] = []

            for idx, memory_id in enumerate(all_results['ids']):
                metadata: dict[str, Any] = all_results['metadatas'][idx] if all_results['metadatas'] else {}

                if session_id and metadata.get('session_id') != session_id:
                    continue
                if project and metadata.get('project') != project:
                    continue
                if start_date or end_date:
                    timestamp_str: str | None = metadata.get('timestamp')
                    if timestamp_str:
                        ts: datetime = datetime.fromisoformat(timestamp_str)
                        if start_date and ts < start_date:
                            continue
                        if end_date and ts > end_date:
                            continue

                memories.append({
                    "id": memory_id,
                    "content": all_results['documents'][idx],
                    "metadata": metadata,
                    "embedding": all_results['embeddings'][idx] if all_results['embeddings'] else None
                })

            export: MemoryExport = MemoryExport(
                total_memories=len(memories),
                memories=memories,
                sessions=[s.to_dict() for s in self._sessions.values()],
                metadata={"filters": {"session_id": session_id, "project": project}}
            )

            with open(file_path, 'w') as f:
                json.dump(export.to_dict(), f, indent=2)

            self._logger.info(f"Exported {len(memories)} memories to {file_path}")

            return export

        except Exception as e:
            self._logger.error(f"Export failed: {e}")
            raise VectorDatabaseError(f"Export failed: {e}")
