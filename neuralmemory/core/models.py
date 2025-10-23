from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class SearchResult:
    rank: int
    content: str
    rerank_score: float
    cosine_distance: float
    metadata: dict[str, Any]
    memory_id: str | None = None
    short_id: str | None = None

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("Rank must be positive")
        if not 0.0 <= self.rerank_score <= 1.0:
            raise ValueError("Rerank score must be between 0 and 1")
        if self.cosine_distance < 0.0:
            raise ValueError("Cosine distance cannot be negative")

    def __str__(self) -> str:
        return f"Result(rank={self.rank}, score={self.rerank_score:.3f})"

    def __repr__(self) -> str:
        return f"SearchResult({self.rank}, '{self.content[:50]}...', {self.rerank_score:.3f}, {self.cosine_distance:.3f})"

    def __lt__(self, other: SearchResult) -> bool:
        return self.rerank_score > other.rerank_score

    @property
    def is_high_confidence(self) -> bool:
        return self.rerank_score > 0.8

    @property
    def content_preview(self) -> str:
        return self.content[:100] + "..." if len(self.content) > 100 else self.content


@dataclass(frozen=True, slots=True)
class MemoryContent:
    content: str
    tags: list[str]
    timestamp: datetime
    memory_type: str | None = None
    short_id: str | None = None

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("Content cannot be empty")
        if not isinstance(self.tags, list):
            raise ValueError("Tags must be a list")

    @property
    def metadata(self) -> dict[str, Any]:
        metadata_dict: dict[str, Any] = {
            "tags": ",".join(self.tags),
            "timestamp": self.timestamp.isoformat(),
            "created_at": datetime.now().isoformat()
        }
        if self.memory_type:
            metadata_dict["memory_type"] = self.memory_type
        if self.short_id:
            metadata_dict["short_id"] = self.short_id
        return metadata_dict


@dataclass(frozen=True, slots=True)
class StorageResult:
    memory_id: str
    short_id: str
    success: bool
    message: str

    def __str__(self) -> str:
        identifier: str = self.short_id if self.short_id else self.memory_id[:8]
        return f"Memory {identifier}... {'stored' if self.success else 'failed'}"


@dataclass(frozen=True, slots=True)
class MemoryResult:
    memory_id: str
    short_id: str | None
    content: str
    tags: list[str]
    memory_type: str | None
    timestamp: datetime
    metadata: dict[str, Any]
    success: bool = True

    def __str__(self) -> str:
        identifier: str = self.short_id if self.short_id else self.memory_id[:8]
        return f"Memory [{identifier}] - {len(self.content)} chars"

    @property
    def content_preview(self) -> str:
        return self.content[:200] + "..." if len(self.content) > 200 else self.content
