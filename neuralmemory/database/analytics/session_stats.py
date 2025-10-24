from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from neuralmemory.core.models import SessionMetadata


class SessionStatisticsCalculator:
    def __init__(
        self,
        collection: Any,
        sessions: dict[str, SessionMetadata],
        logger: logging.Logger
    ) -> None:
        self._collection: Any = collection
        self._sessions: dict[str, SessionMetadata] = sessions
        self._logger: logging.Logger = logger

    def calculate(
        self,
        session_id: str | None,
        current_session_id: str | None
    ) -> dict[str, Any]:
        target_session_id: str | None = session_id or current_session_id
        if not target_session_id:
            return {}

        if self._collection is None:
            return {}

        try:
            session_meta: SessionMetadata | None = self._sessions.get(target_session_id)

            results = self._collection.get(
                where={"session_id": target_session_id},
                include=["documents", "metadatas"]
            )

            if not results or not results.get("documents"):
                return {
                    "session_id": target_session_id,
                    "total_memories": 0
                }

            total_memories: int = len(results["documents"])
            importances: list[float] = []
            topics_count: dict[str, int] = {}
            entities_count: dict[str, int] = {}
            memory_types: dict[str, int] = {}
            action_items_total: int = 0
            action_items_completed: int = 0

            timestamps: list[datetime] = []

            for idx in range(len(results["documents"])):
                metadata: dict[str, Any] = results["metadatas"][idx] if results.get("metadatas") else {}

                importance: float = float(metadata.get("importance", 0.5))
                importances.append(importance)

                topics_str: str = metadata.get("topics", "")
                if topics_str:
                    for topic in topics_str.split(","):
                        topic = topic.strip()
                        topics_count[topic] = topics_count.get(topic, 0) + 1

                entities_str: str = metadata.get("entities", "")
                if entities_str:
                    for entity in entities_str.split(","):
                        entity = entity.strip()
                        entities_count[entity] = entities_count.get(entity, 0) + 1

                memory_type: str = metadata.get("memory_type", "episodic")
                memory_types[memory_type] = memory_types.get(memory_type, 0) + 1

                action_items_str: str = metadata.get("action_items", "")
                if action_items_str:
                    items: list[str] = [i.strip() for i in action_items_str.split(",") if i.strip()]
                    action_items_total += len(items)

                if metadata.get("outcome") == "completed":
                    action_items_completed += 1

                timestamp_str: str = metadata.get("timestamp")
                if timestamp_str:
                    timestamps.append(datetime.fromisoformat(timestamp_str))

            duration_str: str = "N/A"
            if len(timestamps) >= 2:
                timestamps.sort()
                duration: timedelta = timestamps[-1] - timestamps[0]
                hours: int = int(duration.total_seconds() // 3600)
                minutes: int = int((duration.total_seconds() % 3600) // 60)
                duration_str = f"{hours}h {minutes}m"

            stats: dict[str, Any] = {
                "session_id": target_session_id,
                "session_name": session_meta.name if session_meta else None,
                "total_memories": total_memories,
                "avg_importance": sum(importances) / len(importances) if importances else 0.0,
                "duration": duration_str,
                "topic_distribution": dict(sorted(topics_count.items(), key=lambda x: x[1], reverse=True)[:10]),
                "entity_participation": dict(sorted(entities_count.items(), key=lambda x: x[1], reverse=True)),
                "memory_type_distribution": memory_types,
                "action_items_total": action_items_total,
                "action_items_completed": action_items_completed,
                "completion_ratio": action_items_completed / action_items_total if action_items_total > 0 else 0.0
            }

            self._logger.debug(f"Calculated session stats for {target_session_id}: {stats}")
            return stats

        except Exception as e:
            self._logger.error(f"Error calculating session stats: {e}")
            return {
                "session_id": target_session_id,
                "error": str(e)
            }
