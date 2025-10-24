from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from neuralmemory.core.models import SessionMetadata


class SessionMetadataStore:
    def __init__(
        self,
        sessions_file: Path,
        sessions: dict[str, SessionMetadata],
        session_name_to_id: dict[str, str],
        logger: logging.Logger
    ) -> None:
        self._sessions_file: Path = sessions_file
        self._sessions: dict[str, SessionMetadata] = sessions
        self._session_name_to_id: dict[str, str] = session_name_to_id
        self._logger: logging.Logger = logger

    def load(self) -> None:
        if self._sessions_file.exists():
            try:
                with open(self._sessions_file, 'r') as f:
                    data: dict[str, Any] = json.load(f)
                    for session_id, session_data in data.items():
                        session_meta: SessionMetadata = SessionMetadata.from_dict(session_data)
                        self._sessions[session_id] = session_meta
                        if session_meta.name:
                            self._session_name_to_id[session_meta.name] = session_id
                self._logger.info(f"Loaded {len(self._sessions)} sessions from {self._sessions_file}")
            except Exception as e:
                self._logger.warning(f"Failed to load sessions: {e}")

    def save(self) -> None:
        try:
            data: dict[str, Any] = {
                session_id: session.to_dict()
                for session_id, session in self._sessions.items()
            }
            self._sessions_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._sessions_file, 'w') as f:
                json.dump(data, f, indent=2)
            self._logger.debug(f"Saved {len(self._sessions)} sessions to {self._sessions_file}")
        except Exception as e:
            self._logger.error(f"Failed to save sessions: {e}")
