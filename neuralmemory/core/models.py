from __future__ import annotations
from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, field_validator


class SearchResult(BaseModel):
    rank: int
    content: str
    rerank_score: float
    cosine_distance: float
    metadata: dict[str, Any]
    memory_id: str | None = None
    short_id: str | None = None
    model_config = ConfigDict(frozen=True)

    @field_validator('rank')
    @classmethod
    def validate_rank(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                f"Invalid rank: expected positive integer (>= 1), got {v}. "
                f"Check search result construction and ensure rank starts at 1."
            )
        return v

    @field_validator('rerank_score')
    @classmethod
    def validate_rerank_score(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"Invalid rerank_score: expected value in range [0.0, 1.0], got {v}. "
                f"Check reranker output normalization."
            )
        return v

    @field_validator('cosine_distance')
    @classmethod
    def validate_cosine_distance(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(
                f"Invalid cosine_distance: expected non-negative value, got {v}. "
                f"Check embedding similarity computation."
            )
        return v

    def __str__(self) -> str:
        return f"SearchResult(rank={self.rank}, score={self.rerank_score:.3f})"

    def __repr__(self) -> str:
        content_preview: str = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return (
            f"SearchResult(rank={self.rank}, content='{content_preview}', "
            f"rerank_score={self.rerank_score:.3f}, cosine_distance={self.cosine_distance:.3f})"
        )

    def __lt__(self, other: SearchResult) -> bool:
        return self.rerank_score > other.rerank_score

    @property
    def is_high_confidence(self) -> bool:
        return self.rerank_score > 0.8

    @property
    def content_preview(self) -> str:
        return self.content[:100] + "..." if len(self.content) > 100 else self.content


class MemoryContent(BaseModel):
    content: str
    tags: list[str]
    timestamp: datetime
    memory_type: str | None = None
    short_id: str | None = None
    model_config = ConfigDict(frozen=True)

    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                f"Invalid content: expected non-empty string, got empty or whitespace-only string. "
                f"Provide meaningful memory content."
            )
        return v

    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        if not isinstance(v, list):
            raise ValueError(
                f"Invalid tags: expected list of strings, got {type(v).__name__}. "
                f"Ensure tags are provided as a list."
            )
        return v

    def __str__(self) -> str:
        tag_preview: str = ", ".join(self.tags[:3])
        if len(self.tags) > 3:
            tag_preview += f" (+{len(self.tags) - 3} more)"
        return f"MemoryContent({len(self.content)} chars, tags=[{tag_preview}])"

    def __repr__(self) -> str:
        content_preview: str = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return (
            f"MemoryContent(content='{content_preview}', tags={self.tags}, "
            f"timestamp={self.timestamp.isoformat()})"
        )

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


class StorageResult(BaseModel):
    memory_id: str
    short_id: str
    success: bool
    message: str
    model_config = ConfigDict(frozen=True)

    @field_validator('memory_id')
    @classmethod
    def validate_memory_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                f"Invalid memory_id: expected non-empty string, got empty string. "
                f"Check memory storage operation and UUID generation."
            )
        return v

    def __str__(self) -> str:
        identifier: str = self.short_id if self.short_id else self.memory_id[:8]
        status: str = "stored successfully" if self.success else "failed"
        return f"StorageResult({identifier}... - {status})"

    def __repr__(self) -> str:
        return (
            f"StorageResult(memory_id='{self.memory_id[:8]}...', short_id='{self.short_id}', "
            f"success={self.success}, message='{self.message}')"
        )


class MemoryResult(BaseModel):
    memory_id: str
    short_id: str | None
    content: str
    tags: list[str]
    memory_type: str | None
    timestamp: datetime
    metadata: dict[str, Any]
    success: bool = True
    model_config = ConfigDict(frozen=True)

    @field_validator('memory_id')
    @classmethod
    def validate_memory_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                f"Invalid memory_id: expected non-empty string, got empty string. "
                f"Check memory retrieval operation."
            )
        return v

    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                f"Invalid content: expected non-empty string, got empty or whitespace-only string. "
                f"Check stored memory integrity."
            )
        return v

    def __str__(self) -> str:
        identifier: str = self.short_id if self.short_id else self.memory_id[:8]
        return f"MemoryResult([{identifier}] - {len(self.content)} chars, {len(self.tags)} tags)"

    def __repr__(self) -> str:
        content_preview: str = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return (
            f"MemoryResult(memory_id='{self.memory_id[:8]}...', short_id='{self.short_id}', "
            f"content='{content_preview}', tags={self.tags}, timestamp={self.timestamp.isoformat()})"
        )

    @property
    def content_preview(self) -> str:
        return self.content[:200] + "..." if len(self.content) > 200 else self.content
