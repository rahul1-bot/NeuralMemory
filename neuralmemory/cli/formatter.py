from __future__ import annotations
from datetime import datetime

from neuralmemory.core.models import SearchResult, StorageResult, MemoryResult


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
