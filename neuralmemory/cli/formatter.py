from __future__ import annotations
from datetime import datetime
from typing import Any

from neuralmemory.core.models import SearchResult, StorageResult, MemoryResult, SessionMetadata


class MemoryFormatter:
    def __init__(self) -> None:
        pass

    def format_header(self, query: str, total_results: int, db_path: str) -> str:
        return (
            f"NEURAL MEMORY SEARCH\n"
            f"Query: '{query}'\n"
            f"Results: {total_results}\n"
            f"Database: {db_path}\n"
            f"{'-' * 60}"
        )

    def format_result(self, result: SearchResult) -> str:
        timestamp_str: str | None = result.metadata.get("timestamp")
        tags_str: str = result.metadata.get("tags", "")

        identifier: str = ""
        if result.short_id:
            identifier = f"ID: {result.short_id}"
        elif result.memory_id:
            identifier = f"ID: {result.memory_id}"

        formatted_date_time: str = "N/A"
        if timestamp_str and isinstance(timestamp_str, str):
            try:
                timestamp: datetime = datetime.fromisoformat(timestamp_str)
                formatted_date_time = timestamp.strftime("%d/%m/%Y | %I:%M %p")
            except (ValueError, TypeError) as e:
                formatted_date_time = "Invalid date"

        if '\n' in result.content:
            content_lines: list[str] = result.content.split('\n')
            formatted_content: str = "Content:\n"
            for line in content_lines:
                formatted_content += f"  {line}\n"
            formatted_content = formatted_content.rstrip()
        elif '\\n' in result.content:
            processed_content: str = result.content.replace('\\n', '\n')
            content_lines: list[str] = processed_content.split('\n')
            if len(content_lines) > 1:
                formatted_content: str = "Content:\n"
                for line in content_lines:
                    formatted_content += f"  {line}\n"
                formatted_content = formatted_content.rstrip()
            else:
                formatted_content: str = f"Content: {processed_content}"
        else:
            formatted_content: str = f"Content: {result.content}"

        return (
            f"\nResult {result.rank} (Score: {result.rerank_score:.3f})\n"
            f"{identifier}\n"
            f"Distance: {result.cosine_distance:.3f}\n"
            f"Date/Time: {formatted_date_time}\n"
            f"Tags: {tags_str if tags_str else 'None'}\n"
            f"{formatted_content}\n"
            f"{'-' * 50}"
        )

    def format_footer(self, execution_time: float) -> str:
        return f"\nExecution time: {execution_time:.3f} seconds"

    def format_store_header(self) -> str:
        return (
            f"NEURAL MEMORY STORE\n"
            f"{'-' * 60}"
        )

    def format_store_result(self, result: StorageResult) -> str:
        return (
            f"[SUCCESS] {result}\n"
            f"Message: {result.message}\n"
            f"Memory ID: {result.memory_id}"
        )

    def format_batch_store_results(self, results: list[StorageResult]) -> str:
        lines: list[str] = [f"[SUCCESS] Stored {len(results)} memories:"]
        for idx, result in enumerate(results):
            lines.append(f"  Memory {idx + 1}: ID={result.memory_id[:8]}... - {result.message}")
        return "\n".join(lines)

    def format_store_footer(self, execution_time: float) -> str:
        return f"\nExecution time: {execution_time:.3f} seconds"

    def format_error(self, error: Exception) -> str:
        return f"[ERROR] {error}"

    def format_read_header(self) -> str:
        return (
            f"NEURAL MEMORY READ\n"
            f"{'-' * 60}"
        )

    def format_memory_result(self, result: MemoryResult) -> str:
        identifier: str = result.short_id if result.short_id else result.memory_id[:8]
        timestamp_str: str = result.timestamp.strftime("%d/%m/%Y | %I:%M %p")
        tags_str: str = ", ".join(result.tags) if result.tags else "None"

        if '\n' in result.content:
            content_lines: list[str] = result.content.split('\n')
            formatted_content: str = "Content:\n"
            for line in content_lines:
                formatted_content += f"  {line}\n"
            formatted_content = formatted_content.rstrip()
        else:
            formatted_content: str = f"Content: {result.content}"

        return (
            f"\nMemory ID: {result.memory_id}\n"
            f"Short ID: {result.short_id if result.short_id else 'None'}\n"
            f"Type: {result.memory_type if result.memory_type else 'None'}\n"
            f"Date/Time: {timestamp_str}\n"
            f"Tags: {tags_str}\n"
            f"{formatted_content}\n"
            f"{'-' * 50}"
        )

    def format_batch_read_results(self, results: list[MemoryResult]) -> str:
        lines: list[str] = [f"[SUCCESS] Read {len(results)} memories:"]
        for idx, result in enumerate(results):
            identifier: str = result.short_id if result.short_id else result.memory_id[:8]
            lines.append(f"  {idx + 1}. [{identifier}] - {len(result.content)} chars")
        return "\n".join(lines)

    # Session-related formatting methods
    def format_session_header(self) -> str:
        return (
            f"NEURAL MEMORY SESSION\n"
            f"{'-' * 60}"
        )

    def format_session_created(self, session_id: str, name: str | None) -> str:
        identifier: str = name if name else session_id[:8]
        return (
            f"{self.format_session_header()}\n"
            f"[SUCCESS] Started new session: {identifier}\n"
            f"Session ID: {session_id}"
        )

    def format_session_list(self, sessions: dict[str, SessionMetadata]) -> str:
        if not sessions:
            return f"{self.format_session_header()}\n[INFO] No sessions found"

        lines: list[str] = [
            self.format_session_header(),
            f"Total sessions: {len(sessions)}\n"
        ]

        for session_id, session in sessions.items():
            identifier: str = session.name if session.name else session_id[:8]
            created: str = session.created_at.strftime("%d/%m/%Y %I:%M %p")
            last_activity: str = session.last_activity.strftime("%d/%m/%Y %I:%M %p")
            participants_str: str = ", ".join(session.participants) if session.participants else "None"

            lines.append(
                f"Session: {identifier}\n"
                f"  Status: {session.status}\n"
                f"  Memories: {session.total_memories}\n"
                f"  Avg Importance: {session.avg_importance:.2f}\n"
                f"  Created: {created}\n"
                f"  Last Activity: {last_activity}\n"
                f"  Project: {session.project if session.project else 'None'}\n"
                f"  Topic: {session.topic if session.topic else 'None'}\n"
                f"  Participants: {participants_str}\n"
                f"  {'-' * 50}"
            )

        return "\n".join(lines)

    def format_session_details(self, session: SessionMetadata) -> str:
        identifier: str = session.name if session.name else session.session_id[:8]
        created: str = session.created_at.strftime("%d/%m/%Y %I:%M %p")
        last_activity: str = session.last_activity.strftime("%d/%m/%Y %I:%M %p")
        participants_str: str = ", ".join(session.participants) if session.participants else "None"

        return (
            f"{self.format_session_header()}\n"
            f"Session: {identifier}\n"
            f"Session ID: {session.session_id}\n"
            f"Name: {session.name if session.name else 'None'}\n"
            f"Status: {session.status}\n"
            f"Project: {session.project if session.project else 'None'}\n"
            f"Topic: {session.topic if session.topic else 'None'}\n"
            f"Participants: {participants_str}\n"
            f"Created: {created}\n"
            f"Last Activity: {last_activity}\n"
            f"Total Memories: {session.total_memories}\n"
            f"Average Importance: {session.avg_importance:.2f}"
        )

    def format_session_stats(self, stats: dict[str, Any]) -> str:
        if not stats:
            return f"{self.format_session_header()}\n[INFO] No statistics available"

        lines: list[str] = [
            self.format_session_header(),
            f"Session Statistics\n"
        ]

        if "session_name" in stats:
            identifier: str = stats["session_name"] if stats["session_name"] else stats.get("session_id", "")[:8]
            lines.append(f"Session: {identifier}")

        lines.append(f"Total Memories: {stats.get('total_memories', 0)}")
        lines.append(f"Average Importance: {stats.get('avg_importance', 0.0):.2f}")
        lines.append(f"Duration: {stats.get('duration', 'N/A')}")

        # Topic distribution
        if "topic_distribution" in stats and stats["topic_distribution"]:
            lines.append(f"\nTop Topics:")
            for topic, count in list(stats["topic_distribution"].items())[:5]:
                lines.append(f"  {topic}: {count}")

        # Entity participation
        if "entity_participation" in stats and stats["entity_participation"]:
            lines.append(f"\nEntity Participation:")
            for entity, count in stats["entity_participation"].items():
                lines.append(f"  {entity}: {count}")

        # Memory type distribution
        if "memory_type_distribution" in stats and stats["memory_type_distribution"]:
            lines.append(f"\nMemory Types:")
            for mem_type, count in stats["memory_type_distribution"].items():
                lines.append(f"  {mem_type}: {count}")

        # Action items
        if "action_items_total" in stats:
            lines.append(f"\nAction Items:")
            lines.append(f"  Total: {stats['action_items_total']}")
            lines.append(f"  Completed: {stats.get('action_items_completed', 0)}")
            lines.append(f"  Completion Ratio: {stats.get('completion_ratio', 0.0):.1%}")

        return "\n".join(lines)

    def format_conversation_thread(self, thread: list[MemoryResult]) -> str:
        if not thread:
            return f"{self.format_session_header()}\n[INFO] No conversation thread found"

        lines: list[str] = [
            self.format_session_header(),
            f"Conversation Thread ({len(thread)} memories)\n"
        ]

        for idx, memory in enumerate(thread, 1):
            identifier: str = memory.short_id if memory.short_id else memory.memory_id[:8]
            timestamp: str = memory.timestamp.strftime("%d/%m/%Y %I:%M %p")
            content_preview: str = memory.content[:100] + "..." if len(memory.content) > 100 else memory.content

            lines.append(
                f"{idx}. [{identifier}] - {timestamp}\n"
                f"   {content_preview}\n"
            )

        return "\n".join(lines)

    def format_context_window(self, context: dict[str, list[MemoryResult]]) -> str:
        lines: list[str] = [self.format_session_header()]

        # Before memories
        before: list[MemoryResult] = context.get("before", [])
        if before:
            lines.append(f"\nContext Before ({len(before)} memories):")
            for memory in before:
                identifier: str = memory.short_id if memory.short_id else memory.memory_id[:8]
                timestamp: str = memory.timestamp.strftime("%d/%m/%Y %I:%M %p")
                content_preview: str = memory.content[:80] + "..." if len(memory.content) > 80 else memory.content
                lines.append(f"  [{identifier}] {timestamp}: {content_preview}")

        # Target memory
        target: list[MemoryResult] = context.get("target", [])
        if target:
            lines.append(f"\n{'=' * 60}")
            lines.append("TARGET MEMORY:")
            lines.append(f"{'=' * 60}")
            for memory in target:
                lines.append(self.format_memory_result(memory))

        # After memories
        after: list[MemoryResult] = context.get("after", [])
        if after:
            lines.append(f"\nContext After ({len(after)} memories):")
            for memory in after:
                identifier: str = memory.short_id if memory.short_id else memory.memory_id[:8]
                timestamp: str = memory.timestamp.strftime("%d/%m/%Y %I:%M %p")
                content_preview: str = memory.content[:80] + "..." if len(memory.content) > 80 else memory.content
                lines.append(f"  [{identifier}] {timestamp}: {content_preview}")

        return "\n".join(lines)

    def format_session_ended(self, summary: str | None) -> str:
        if summary:
            return (
                f"{self.format_session_header()}\n"
                f"[SUCCESS] Session ended with summary\n\n"
                f"Summary:\n{summary}"
            )
        else:
            return (
                f"{self.format_session_header()}\n"
                f"[SUCCESS] Session ended (no summary generated)"
            )
