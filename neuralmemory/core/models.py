from __future__ import annotations
from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, field_validator


class EnhancedMemoryMetadata(BaseModel):
    """Rich metadata schema for intelligent memory management with classification, relationships, and usage tracking."""

    # Classification
    memory_type: Literal["episodic", "semantic", "procedural", "working"] = "episodic"
    importance: float = 0.5

    # Context
    session_id: str | None = None
    project: str | None = None
    entities: list[str] = []
    topics: list[str] = []

    # Tracking
    action_items: list[str] = []
    outcome: Literal["completed", "pending", "failed", "cancelled"] | None = None
    access_count: int = 0
    last_accessed: datetime | None = None

    # Relationships
    parent_memory_id: str | None = None
    related_memory_ids: list[str] = []
    sequence_num: int = 0

    # Standard fields
    tags: list[str] = []
    timestamp: datetime = datetime.now()
    short_id: str | None = None

    model_config = ConfigDict(frozen=True)

    @field_validator('importance')
    @classmethod
    def validate_importance(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"Invalid importance: expected value in range [0.0, 1.0], got {v}. "
                f"Use 0.0 for low priority, 0.5 for medium, 1.0 for critical memories."
            )
        return v

    @field_validator('access_count')
    @classmethod
    def validate_access_count(cls, v: int) -> int:
        if v < 0:
            raise ValueError(
                f"Invalid access_count: expected non-negative integer, got {v}. "
                f"Access count tracks memory retrieval frequency."
            )
        return v

    @field_validator('entities')
    @classmethod
    def validate_entities(cls, v: list[str]) -> list[str]:
        if not isinstance(v, list):
            raise ValueError(
                f"Invalid entities: expected list of strings, got {type(v).__name__}. "
                f"Provide entity names like ['Rahul', 'Claude', 'NeuralMemory']."
            )
        return v

    @field_validator('topics')
    @classmethod
    def validate_topics(cls, v: list[str]) -> list[str]:
        if not isinstance(v, list):
            raise ValueError(
                f"Invalid topics: expected list of strings, got {type(v).__name__}. "
                f"Provide topic keywords like ['refactoring', 'pydantic', 'architecture']."
            )
        return v

    @field_validator('action_items')
    @classmethod
    def validate_action_items(cls, v: list[str]) -> list[str]:
        if not isinstance(v, list):
            raise ValueError(
                f"Invalid action_items: expected list of strings, got {type(v).__name__}. "
                f"Provide actionable tasks from memory."
            )
        return v

    def __str__(self) -> str:
        entity_preview: str = ", ".join(self.entities[:3])
        if len(self.entities) > 3:
            entity_preview += f" (+{len(self.entities) - 3} more)"
        return f"EnhancedMetadata(type={self.memory_type}, importance={self.importance:.2f}, entities=[{entity_preview}])"

    def __repr__(self) -> str:
        return (
            f"EnhancedMemoryMetadata(memory_type='{self.memory_type}', importance={self.importance:.2f}, "
            f"session_id='{self.session_id}', project='{self.project}', "
            f"entities={self.entities}, topics={self.topics}, access_count={self.access_count})"
        )

    def to_chromadb_dict(self) -> dict[str, Any]:
        """Convert to ChromaDB-compatible metadata dictionary."""
        metadata_dict: dict[str, Any] = {
            "memory_type": self.memory_type,
            "importance": self.importance,
            "tags": ",".join(self.tags),
            "timestamp": self.timestamp.isoformat(),
            "access_count": self.access_count,
            "sequence_num": self.sequence_num,
        }

        if self.session_id:
            metadata_dict["session_id"] = self.session_id
        if self.project:
            metadata_dict["project"] = self.project
        if self.entities:
            metadata_dict["entities"] = ",".join(self.entities)
        if self.topics:
            metadata_dict["topics"] = ",".join(self.topics)
        if self.action_items:
            metadata_dict["action_items"] = ",".join(self.action_items)
        if self.outcome:
            metadata_dict["outcome"] = self.outcome
        if self.last_accessed:
            metadata_dict["last_accessed"] = self.last_accessed.isoformat()
        if self.parent_memory_id:
            metadata_dict["parent_memory_id"] = self.parent_memory_id
        if self.related_memory_ids:
            metadata_dict["related_memory_ids"] = ",".join(self.related_memory_ids)
        if self.short_id:
            metadata_dict["short_id"] = self.short_id

        return metadata_dict

    @classmethod
    def from_chromadb_dict(cls, metadata: dict[str, Any]) -> EnhancedMemoryMetadata:
        """Create from ChromaDB metadata dictionary."""
        return cls(
            memory_type=metadata.get("memory_type", "episodic"),
            importance=float(metadata.get("importance", 0.5)),
            session_id=metadata.get("session_id"),
            project=metadata.get("project"),
            entities=metadata.get("entities", "").split(",") if metadata.get("entities") else [],
            topics=metadata.get("topics", "").split(",") if metadata.get("topics") else [],
            action_items=metadata.get("action_items", "").split(",") if metadata.get("action_items") else [],
            outcome=metadata.get("outcome"),
            access_count=int(metadata.get("access_count", 0)),
            last_accessed=datetime.fromisoformat(metadata["last_accessed"]) if metadata.get("last_accessed") else None,
            parent_memory_id=metadata.get("parent_memory_id"),
            related_memory_ids=metadata.get("related_memory_ids", "").split(",") if metadata.get("related_memory_ids") else [],
            sequence_num=int(metadata.get("sequence_num", 0)),
            tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
            timestamp=datetime.fromisoformat(metadata["timestamp"]) if metadata.get("timestamp") else datetime.now(),
            short_id=metadata.get("short_id"),
        )


class SessionMetadata(BaseModel):
    """Metadata for conversation sessions tracking lifecycle and participants."""

    # Identity
    session_id: str
    name: str | None = None

    # Context
    project: str | None = None
    topic: str | None = None
    participants: list[str] = []

    # Lifecycle
    created_at: datetime = datetime.now()
    last_activity: datetime = datetime.now()
    status: Literal["active", "completed", "archived"] = "active"

    # Statistics
    total_memories: int = 0
    avg_importance: float = 0.0

    model_config = ConfigDict(frozen=True)

    @field_validator('total_memories')
    @classmethod
    def validate_total_memories(cls, v: int) -> int:
        if v < 0:
            raise ValueError(
                f"Invalid total_memories: expected non-negative integer, got {v}. "
                f"Memory count cannot be negative."
            )
        return v

    @field_validator('avg_importance')
    @classmethod
    def validate_avg_importance(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(
                f"Invalid avg_importance: expected value in range [0.0, 1.0], got {v}. "
                f"Average importance must be between 0 and 1."
            )
        return v

    @field_validator('participants')
    @classmethod
    def validate_participants(cls, v: list[str]) -> list[str]:
        if not isinstance(v, list):
            raise ValueError(
                f"Invalid participants: expected list of strings, got {type(v).__name__}. "
                f"Provide participant names like ['Rahul', 'Claude']."
            )
        return v

    def __str__(self) -> str:
        participants_str: str = ", ".join(self.participants) if self.participants else "None"
        return f"Session({self.name or self.session_id[:8]}, {self.status}, {self.total_memories} memories, participants: {participants_str})"

    def __repr__(self) -> str:
        return (
            f"SessionMetadata(session_id='{self.session_id}', name='{self.name}', "
            f"status='{self.status}', total_memories={self.total_memories}, "
            f"participants={self.participants}, avg_importance={self.avg_importance:.2f})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "project": self.project,
            "topic": self.topic,
            "participants": self.participants,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "status": self.status,
            "total_memories": self.total_memories,
            "avg_importance": self.avg_importance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMetadata:
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            name=data.get("name"),
            project=data.get("project"),
            topic=data.get("topic"),
            participants=data.get("participants", []),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            last_activity=datetime.fromisoformat(data["last_activity"]) if data.get("last_activity") else datetime.now(),
            status=data.get("status", "active"),
            total_memories=data.get("total_memories", 0),
            avg_importance=float(data.get("avg_importance", 0.0)),
        )


class SearchResult(BaseModel):
    rank: int
    content: str
    rerank_score: float
    cosine_distance: float
    metadata: dict[str, Any]
    memory_id: str | None = None
    short_id: str | None = None
    enhanced_metadata: EnhancedMemoryMetadata | None = None
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
    enhanced_metadata: EnhancedMemoryMetadata | None = None
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
