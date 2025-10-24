from __future__ import annotations
import json
import logging
from typing import Any, Callable

from neuralmemory.core.models import MemoryExport, SessionMetadata, ConflictDetectionResult
from neuralmemory.core.exceptions import VectorDatabaseError


class MemoryImporter:

    def __init__(
        self,
        collection: Any,
        sessions: dict[str, Any],
        session_name_to_id: dict[str, str],
        save_sessions_callback: Callable[[], None],
        conflict_detector: Callable[[str, str, list[float]], list[ConflictDetectionResult]] | None,
        logger: logging.Logger
    ) -> None:
        self._collection = collection
        self._sessions = sessions
        self._session_name_to_id = session_name_to_id
        self._save_sessions = save_sessions_callback
        self._detect_conflicts = conflict_detector
        self._logger = logger

    def import_memories(self, file_path: str, merge_duplicates: bool = True) -> int:
        try:
            with open(file_path, 'r') as f:
                data: dict[str, Any] = json.load(f)

            export: MemoryExport = MemoryExport.from_dict(data)

            imported_count: int = 0

            for memory in export.memories:
                if merge_duplicates and memory.get('embedding') and self._detect_conflicts:
                    conflicts: list[ConflictDetectionResult] = self._detect_conflicts(
                        memory['id'],
                        memory['content'],
                        memory['embedding']
                    )

                    if conflicts:
                        self._logger.info(f"Skipping duplicate: {memory['id'][:8]}...")
                        continue

                try:
                    self._collection.add(
                        ids=[memory['id']],
                        documents=[memory['content']],
                        metadatas=[memory['metadata']],
                        embeddings=[memory['embedding']] if memory.get('embedding') else None
                    )
                    imported_count += 1
                except Exception as e:
                    self._logger.warning(f"Failed to import {memory['id'][:8]}...: {e}")

            for session_data in export.sessions:
                try:
                    session: SessionMetadata = SessionMetadata.from_dict(session_data)
                    self._sessions[session.session_id] = session
                    if session.name:
                        self._session_name_to_id[session.name] = session.session_id
                except Exception as e:
                    self._logger.warning(f"Failed to import session: {e}")

            self._save_sessions()

            self._logger.info(f"Imported {imported_count}/{export.total_memories} memories from {file_path}")

            return imported_count

        except Exception as e:
            self._logger.error(f"Import failed: {e}")
            raise VectorDatabaseError(f"Import failed: {e}")
