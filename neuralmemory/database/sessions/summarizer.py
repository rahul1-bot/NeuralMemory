from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable

from neuralmemory.core.models import SessionMetadata, StorageResult


class SessionSummarizer:
    def __init__(
        self,
        collection: Any,
        sessions: dict[str, SessionMetadata],
        save_sessions_callback: Callable[[], None],
        store_memory_callback: Callable[..., StorageResult],
        logger: logging.Logger
    ) -> None:
        self._collection: Any = collection
        self._sessions: dict[str, SessionMetadata] = sessions
        self._save_sessions: Callable[[], None] = save_sessions_callback
        self._store_memory: Callable[..., StorageResult] = store_memory_callback
        self._logger: logging.Logger = logger

    def end_session(
        self,
        session_id: str | None,
        current_session_id: str | None,
        summarize: bool = True
    ) -> tuple[str | None, str | None]:
        if not current_session_id:
            self._logger.warning("No active session to end")
            return None, None

        target_session_id: str = current_session_id

        if target_session_id in self._sessions:
            session: SessionMetadata = self._sessions[target_session_id]
            completed_session: SessionMetadata = SessionMetadata(
                session_id=session.session_id,
                name=session.name,
                project=session.project,
                topic=session.topic,
                participants=session.participants,
                created_at=session.created_at,
                last_activity=datetime.now(),
                status="completed",
                total_memories=session.total_memories,
                avg_importance=session.avg_importance
            )
            self._sessions[target_session_id] = completed_session
            self._save_sessions()

        summary_text: str | None = None
        if summarize:
            summary_text = self._generate_summary(target_session_id)
            if summary_text:
                self._store_memory(
                    content=summary_text,
                    tags=["summary", "session"],
                    memory_type="semantic",
                    importance=0.9,
                    session_id=target_session_id
                )

        self._logger.info(f"Ended session: {target_session_id}")

        return summary_text, target_session_id

    def _generate_summary(self, session_id: str) -> str:
        if self._collection is None:
            return ""

        try:
            results = self._collection.get(
                where={"session_id": session_id},
                include=["documents", "metadatas"]
            )

            if not results or not results.get("documents"):
                return ""

            decisions: list[str] = []
            all_action_items: list[str] = []
            outcomes: list[str] = []

            for idx, doc in enumerate(results["documents"]):
                content_lower: str = doc.lower()
                metadata: dict[str, Any] = results["metadatas"][idx] if results.get("metadatas") else {}

                if any(keyword in content_lower for keyword in ["decided", "chose", "will implement"]):
                    decisions.append(doc[:200])

                action_items_str: str = metadata.get("action_items", "")
                if action_items_str:
                    items: list[str] = action_items_str.split(",")
                    all_action_items.extend(items)

                if metadata.get("outcome") == "completed":
                    outcomes.append(doc[:100])

            summary_parts: list[str] = [f"Session Summary ({session_id[:8]})"]

            if decisions:
                summary_parts.append(f"\nKey Decisions ({len(decisions)}):")
                summary_parts.extend([f"- {d[:150]}" for d in decisions[:3]])

            if all_action_items:
                summary_parts.append(f"\nAction Items ({len(all_action_items)}):")
                summary_parts.extend([f"- {item.strip()}" for item in all_action_items[:5]])

            if outcomes:
                summary_parts.append(f"\nCompleted Outcomes ({len(outcomes)}):")
                summary_parts.extend([f"- {o[:100]}" for o in outcomes[:3]])

            return "\n".join(summary_parts)

        except Exception as e:
            self._logger.error(f"Failed to generate session summary: {e}")
            return ""
