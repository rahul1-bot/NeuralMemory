from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from typing import Callable

from neuralmemory.core.models import SessionMetadata
from neuralmemory.core.exceptions import MemoryValidationError


class SessionManager:
    def __init__(
        self,
        sessions: dict[str, SessionMetadata],
        session_name_to_id: dict[str, str],
        save_sessions_callback: Callable[[], None],
        logger: logging.Logger
    ) -> None:
        self._sessions: dict[str, SessionMetadata] = sessions
        self._session_name_to_id: dict[str, str] = session_name_to_id
        self._save_sessions: Callable[[], None] = save_sessions_callback
        self._logger: logging.Logger = logger

    def start_new(
        self,
        name: str | None = None,
        project: str | None = None,
        topic: str | None = None,
        participants: list[str] | None = None
    ) -> str:
        if name:
            if not re.match(r'^[a-zA-Z0-9_-]+$', name):
                raise MemoryValidationError(
                    f"Invalid session name '{name}': only alphanumeric, dash, and underscore allowed"
                )
            if name in self._session_name_to_id:
                raise MemoryValidationError(f"Session name '{name}' already exists")

        session_id: str = str(uuid.uuid4())

        session_meta: SessionMetadata = SessionMetadata(
            session_id=session_id,
            name=name,
            project=project,
            topic=topic,
            participants=participants if participants else ["Claude"],
            created_at=datetime.now(),
            last_activity=datetime.now(),
            status="active",
            total_memories=0,
            avg_importance=0.0
        )

        self._sessions[session_id] = session_meta
        if name:
            self._session_name_to_id[name] = session_id

        self._save_sessions()
        self._logger.info(f"Started new session: {name or session_id[:8]}")

        return session_id

    def list_all(self) -> dict[str, SessionMetadata]:
        return dict(self._sessions)

    def get_by_name(self, name: str) -> SessionMetadata | None:
        session_id: str | None = self._session_name_to_id.get(name)
        if session_id:
            return self._sessions.get(session_id)
        return None
