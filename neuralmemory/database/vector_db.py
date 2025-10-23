from __future__ import annotations
import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import chromadb
from torch import Tensor

from neuralmemory.core.exceptions import (
    VectorDatabaseError,
    MemoryValidationError,
    BatchValidationError
)
from neuralmemory.core.models import (
    EnhancedMemoryMetadata,
    SessionMetadata,
    SearchResult,
    MemoryContent,
    StorageResult,
    MemoryResult
)
from neuralmemory.core.config import EmbeddingConfig, RerankerConfig
from neuralmemory.core.logging_setup import LoggerSetup
from neuralmemory.engines.embedding import Qwen3EmbeddingEngine
from neuralmemory.engines.reranker import Qwen3RerankerEngine

class NeuralVector:
    def __init__(self, db_path: str, enable_session_tracking: bool = True) -> None:
        self._db_path: Path = Path(db_path)
        self._client: chromadb.PersistentClient | None = None
        self._collection: Any | None = None
        self._embedding_engine: Qwen3EmbeddingEngine | None = None
        self._reranker_engine: Qwen3RerankerEngine | None = None

        # Session tracking for conversation threading
        self._enable_session_tracking: bool = enable_session_tracking
        self._current_session_id: str | None = None
        self._session_sequence_num: int = 0

        # Session metadata storage (Feature 2 & 3: Named Sessions + Session Metadata)
        self._sessions_file: Path = self._db_path / "sessions.json"
        self._sessions: dict[str, SessionMetadata] = {}
        self._session_name_to_id: dict[str, str] = {}  # name -> session_id mapping

        log_path: Path = Path(__file__).parent / "logs" / "neuralvector.log"
        self._logger: logging.Logger = LoggerSetup.get_logger("NeuralVector", log_path)
        self._logger.info(f"Initializing NeuralVector with database path: {db_path}")

        self._initialize_components()
        self._load_sessions()
    
    def _initialize_components(self) -> None:
        self._initialize_database()
        self._initialize_models()
    
    def _initialize_database(self) -> None:
        try:
            self._logger.info(f"Connecting to ChromaDB at: {self._db_path}")
            self._client = chromadb.PersistentClient(path=str(self._db_path))
            self._collection = self._client.get_collection("memory_collection")
            self._logger.info("Successfully connected to ChromaDB collection: memory_collection")
        except Exception as e:
            self._logger.error(f"Failed to connect to database: {e}")
            raise VectorDatabaseError(f"Failed to connect to database: {e}")
    
    def _initialize_models(self) -> None:
        embedding_config: EmbeddingConfig = EmbeddingConfig.create_qwen3_mps_config()
        reranker_config: RerankerConfig = RerankerConfig.create_qwen3_mps_config()
        
        self._embedding_engine = Qwen3EmbeddingEngine(embedding_config)
        self._reranker_engine = Qwen3RerankerEngine(reranker_config)

    # ==================== SESSION MANAGEMENT (Features 2 & 3) ====================

    def _load_sessions(self) -> None:
        """Load session metadata from JSON file."""
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

    def _save_sessions(self) -> None:
        """Save session metadata to JSON file."""
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

    def start_new_session(
        self,
        name: str | None = None,
        project: str | None = None,
        topic: str | None = None,
        participants: list[str] | None = None
    ) -> str:
        """Start a new conversation session with optional name and metadata."""
        import uuid

        # Validate and handle session name
        if name:
            # Validate name format
            if not re.match(r'^[a-zA-Z0-9_-]+$', name):
                raise MemoryValidationError(
                    f"Invalid session name '{name}': only alphanumeric, dash, and underscore allowed"
                )
            # Check uniqueness
            if name in self._session_name_to_id:
                raise MemoryValidationError(f"Session name '{name}' already exists")

        session_id: str = str(uuid.uuid4())
        self._current_session_id = session_id
        self._session_sequence_num = 0

        # Create session metadata
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

    def list_sessions(self) -> dict[str, SessionMetadata]:
        """List all sessions with their metadata."""
        return dict(self._sessions)

    def get_session_by_name(self, name: str) -> SessionMetadata | None:
        """Get session metadata by name."""
        session_id: str | None = self._session_name_to_id.get(name)
        if session_id:
            return self._sessions.get(session_id)
        return None

    # ==================== AUTO-IMPORTANCE CALCULATION (Feature 6) ====================

    def _calculate_importance(self, content: str, tags: list[str], action_items: list[str]) -> float:
        """
        Automatically calculate importance score (0.0-1.0) based on content analysis.
        Uses decision keywords, entity mentions, action items, and content length.
        """
        score: float = 0.5  # Start at medium

        content_lower: str = content.lower()

        # Decision keywords boost (+0.3)
        decision_keywords: list[str] = [
            "decided", "chose", "will implement", "selected", "determined",
            "concluded", "agreed", "committed", "finalized"
        ]
        for keyword in decision_keywords:
            if keyword in content_lower:
                score += 0.3
                break

        # Entity mentions boost (+0.2)
        important_entities: list[str] = ["rahul", "claude", "neuralmemory", "pydantic"]
        entity_count: int = sum(1 for entity in important_entities if entity in content_lower)
        if entity_count >= 2:
            score += 0.2

        # Action items presence (+0.2)
        if action_items and len(action_items) > 0:
            score += 0.2

        # Content length scoring
        word_count: int = len(content.split())
        if word_count > 100:  # Detailed content
            score += 0.1

        # Normalize to 0.0-1.0
        return min(1.0, max(0.0, score))

    # ==================== AUTO-TAG SUGGESTION (Feature 8) ====================

    def _suggest_tags(self, content: str) -> list[str]:
        """Suggest tags based on content analysis."""
        suggested_tags: list[str] = []
        content_lower: str = content.lower()

        # Technical keywords
        tech_keywords: list[str] = [
            "refactoring", "pydantic", "architecture", "validation", "metadata",
            "vector", "database", "embedding", "search", "memory", "consolidation",
            "threading", "session", "query", "preprocessing", "importance",
            "python", "code", "guidelines", "model", "config", "api", "cli"
        ]
        for keyword in tech_keywords:
            if keyword in content_lower:
                suggested_tags.append(keyword)

        # Programming concepts
        if "class" in content_lower or "def " in content_lower:
            suggested_tags.append("code")
        if "bug" in content_lower or "fix" in content_lower or "error" in content_lower:
            suggested_tags.append("bugfix")
        if "implement" in content_lower or "add" in content_lower or "create" in content_lower:
            suggested_tags.append("feature")

        # Deduplicate and limit
        return list(set(suggested_tags))[:10]

    def _preprocess_query(self, query: str) -> str:
        self._logger.debug(f"Preprocessing query: {query}")
        processed_query: str = query
        
        current_year: int = datetime.now().year
        current_date: datetime = datetime.now()
        
        month_names: dict[str, str] = {
            'january': '01', 'jan': '01',
            'february': '02', 'feb': '02',
            'march': '03', 'mar': '03',
            'april': '04', 'apr': '04',
            'may': '05',
            'june': '06', 'jun': '06',
            'july': '07', 'jul': '07',
            'august': '08', 'aug': '08',
            'september': '09', 'sep': '09', 'sept': '09',
            'october': '10', 'oct': '10',
            'november': '11', 'nov': '11',
            'december': '12', 'dec': '12'
        }
        
        for month_name, month_num in month_names.items():
            pattern: str = rf'\b{month_name}\s+(\d{{1,2}})\b(?!\d)'
            
            def replace_func(match: re.Match[str]) -> str:
                day: int = int(match.group(1))
                return f'{day:02d}/{month_num}/{current_year}'
            
            processed_query = re.sub(pattern, replace_func, processed_query, flags=re.IGNORECASE)
        
        time_pattern_12hr: re.Pattern[str] = re.compile(r'\b(\d{1,2})\s*(AM|PM|am|pm)\b')
        
        def replace_time_12hr(match: re.Match[str]) -> str:
            hour: int = int(match.group(1))
            period: str = match.group(2).upper()
            
            if hour < 1 or hour > 12:
                return match.group(0)
            
            return f'{hour:02d}:00 {period}'
        
        processed_query = time_pattern_12hr.sub(replace_time_12hr, processed_query)
        
        relative_date_replacements: dict[str, str] = {
            r'\byesterday\b': (current_date - timedelta(days=1)).strftime('%d/%m/%Y'),
            r'\btoday\b': current_date.strftime('%d/%m/%Y'),
            r'\btomorrow\b': (current_date + timedelta(days=1)).strftime('%d/%m/%Y'),
            r'\blast week\b': (current_date - timedelta(weeks=1)).strftime('%d/%m/%Y'),
            r'\blast month\b': (current_date - timedelta(days=30)).strftime('%d/%m/%Y'),
        }
        
        for pattern, replacement in relative_date_replacements.items():
            processed_query = re.sub(pattern, replacement, processed_query, flags=re.IGNORECASE)
        
        time_expansions: dict[str, str] = {
            r'\bmorning\b': '06:00 AM OR 07:00 AM OR 08:00 AM OR 09:00 AM OR 10:00 AM OR 11:00 AM',
            r'\bafternoon\b': '12:00 PM OR 01:00 PM OR 02:00 PM OR 03:00 PM OR 04:00 PM OR 05:00 PM',
            r'\bevening\b': '06:00 PM OR 07:00 PM OR 08:00 PM OR 09:00 PM',
            r'\bnight\b': '10:00 PM OR 11:00 PM OR 12:00 AM OR 01:00 AM OR 02:00 AM',
        }
        
        for pattern, expansion in time_expansions.items():
            processed_query = re.sub(pattern, expansion, processed_query, flags=re.IGNORECASE)
        
        date_slash_pattern: re.Pattern[str] = re.compile(r'\b(\d{1,2})/(\d{1,2})(?!/\d{4})\b')
        
        def replace_date_slash(match: re.Match[str]) -> str:
            month: int = int(match.group(1))
            day: int = int(match.group(2))
            
            if 1 <= month <= 12 and 1 <= day <= 31:
                return f'{day:02d}/{month:02d}/{current_year}'
            return match.group(0)
        
        processed_query = date_slash_pattern.sub(replace_date_slash, processed_query)
        
        if processed_query != query:
            self._logger.info(f"Query preprocessed: '{query}' -> '{processed_query}'")

        return processed_query

    def _extract_entities(self, content: str, tags: list[str]) -> list[str]:
        """Extract entity names from content and tags."""
        entities: list[str] = []

        # Common entity names for this memory system
        common_entities: list[str] = ["Rahul", "Claude", "NeuralMemory", "Pydantic", "ChromaDB", "Qwen3"]

        content_lower: str = content.lower()
        for entity in common_entities:
            if entity.lower() in content_lower:
                entities.append(entity)

        # Extract entities from tags that look like names (capitalized)
        for tag in tags:
            if tag and tag[0].isupper() and tag not in entities:
                entities.append(tag)

        return list(set(entities))  # Remove duplicates

    def _extract_topics(self, content: str, tags: list[str]) -> list[str]:
        """Extract topic keywords from content and tags."""
        topics: list[str] = []

        # Common technical topics
        technical_keywords: list[str] = [
            "refactoring", "pydantic", "architecture", "validation", "metadata",
            "vector", "database", "embedding", "search", "memory", "consolidation",
            "threading", "session", "query", "preprocessing", "importance",
            "python", "code", "guidelines", "model", "config"
        ]

        content_lower: str = content.lower()
        for keyword in technical_keywords:
            if keyword in content_lower:
                topics.append(keyword)

        # Add all tags as topics (lowercase)
        for tag in tags:
            tag_lower: str = tag.lower()
            if tag_lower not in topics and not tag[0].isupper():  # Don't include entity-like tags
                topics.append(tag_lower)

        return list(set(topics))  # Remove duplicates

    def start_new_session(self, session_id: str | None = None) -> str:
        """Start a new conversation session for threading memories."""
        import uuid
        self._current_session_id = session_id if session_id else str(uuid.uuid4())
        self._session_sequence_num = 0
        self._logger.info(f"Started new session: {self._current_session_id}")
        return self._current_session_id

    def get_current_session_id(self) -> str | None:
        """Get the current session ID."""
        return self._current_session_id

    def _get_last_memory_in_session(self, session_id: str) -> str | None:
        """Get the most recent memory ID in the given session."""
        if self._collection is None:
            return None

        try:
            results = self._collection.get(
                where={"session_id": session_id},
                include=["metadatas"]
            )

            if results and results.get("ids") and results.get("metadatas"):
                # Find memory with highest sequence number
                max_seq: int = -1
                last_memory_id: str | None = None

                for idx, metadata in enumerate(results["metadatas"]):
                    seq_num: int = int(metadata.get("sequence_num", 0))
                    if seq_num > max_seq:
                        max_seq = seq_num
                        last_memory_id = results["ids"][idx]

                return last_memory_id

            return None

        except Exception as e:
            self._logger.warning(f"Failed to get last memory in session: {e}")
            return None

    def get_conversation_thread(self, memory_id: str) -> list[MemoryResult]:
        """Get the full conversation thread for a given memory by following parent links."""
        thread: list[MemoryResult] = []
        current_id: str | None = memory_id

        while current_id:
            memory: MemoryResult | None = self.read_memory(current_id)
            if not memory:
                break

            thread.insert(0, memory)  # Add to beginning for chronological order

            # Get parent memory ID from enhanced metadata
            if memory.enhanced_metadata and memory.enhanced_metadata.parent_memory_id:
                current_id = memory.enhanced_metadata.parent_memory_id
            else:
                break

        self._logger.info(f"Retrieved conversation thread with {len(thread)} memories")
        return thread

    def get_memory_with_context(
        self,
        memory_id: str,
        context_window: int = 3
    ) -> dict[str, list[MemoryResult]]:
        """
        Get a memory with surrounding context from the same session.
        Returns dict with 'before', 'target', and 'after' keys.
        """
        target_memory: MemoryResult | None = self.read_memory(memory_id)
        if not target_memory or not target_memory.enhanced_metadata:
            return {"before": [], "target": [], "after": []}

        session_id: str | None = target_memory.enhanced_metadata.session_id
        if not session_id or self._collection is None:
            return {"before": [], "target": [target_memory], "after": []}

        try:
            # Get all memories in the session
            results = self._collection.get(
                where={"session_id": session_id},
                include=["documents", "metadatas"]
            )

            if not results or not results.get("ids"):
                return {"before": [], "target": [target_memory], "after": []}

            # Parse all memories with sequence numbers
            session_memories: list[tuple[int, MemoryResult]] = []
            target_seq: int = target_memory.enhanced_metadata.access_count

            for idx in range(len(results["ids"])):
                metadata: dict[str, Any] = results["metadatas"][idx] if results.get("metadatas") else {}
                seq_num: int = int(metadata.get("sequence_num", 0))

                enhanced_meta: EnhancedMemoryMetadata | None = None
                if metadata:
                    try:
                        enhanced_meta = EnhancedMemoryMetadata.from_chromadb_dict(metadata)
                        if enhanced_meta.parent_memory_id:
                            target_seq = seq_num  # Update target sequence if this is our memory
                    except Exception:
                        pass

                tags: list[str] = metadata.get("tags", "").split(",") if metadata.get("tags") else []
                timestamp: datetime = datetime.fromisoformat(metadata["timestamp"]) if metadata.get("timestamp") else datetime.now()

                mem_result: MemoryResult = MemoryResult(
                    memory_id=results["ids"][idx],
                    short_id=metadata.get("short_id"),
                    content=results["documents"][idx] if results.get("documents") else "",
                    tags=tags,
                    memory_type=metadata.get("memory_type"),
                    timestamp=timestamp,
                    metadata=metadata,
                    enhanced_metadata=enhanced_meta,
                    success=True
                )

                session_memories.append((seq_num, mem_result))

            # Sort by sequence number
            session_memories.sort(key=lambda x: x[0])

            # Find target index
            target_idx: int = -1
            for idx, (seq, mem) in enumerate(session_memories):
                if mem.memory_id == memory_id:
                    target_idx = idx
                    break

            if target_idx == -1:
                return {"before": [], "target": [target_memory], "after": []}

            # Get context window
            before_memories: list[MemoryResult] = [mem for _, mem in session_memories[max(0, target_idx - context_window):target_idx]]
            after_memories: list[MemoryResult] = [mem for _, mem in session_memories[target_idx + 1:min(len(session_memories), target_idx + context_window + 1)]]

            self._logger.info(
                f"Retrieved context for memory {memory_id}: "
                f"{len(before_memories)} before, {len(after_memories)} after"
            )

            return {
                "before": before_memories,
                "target": [target_memory],
                "after": after_memories
            }

        except Exception as e:
            self._logger.error(f"Failed to get memory with context: {e}")
            return {"before": [], "target": [target_memory], "after": []}

    def read_memory(self, identifier: str) -> MemoryResult | None:
        if not identifier:
            error_msg: str = "Identifier cannot be empty"
            self._logger.error(error_msg)
            raise MemoryValidationError(error_msg)
        
        self._logger.info(f"Reading memory with identifier: {identifier}")
        
        if self._collection is None:
            error_msg: str = "Database not initialized"
            self._logger.error(error_msg)
            raise VectorDatabaseError(error_msg)
        
        try:
            if self._is_valid_uuid(identifier):
                self._logger.debug(f"Reading by UUID: {identifier}")
                results = self._collection.get(
                    ids=[identifier],
                    include=["documents", "metadatas"]
                )
            else:
                self._logger.debug(f"Reading by short_id: {identifier}")
                results = self._collection.get(
                    where={"short_id": identifier},
                    include=["documents", "metadatas"]
                )
            
            if results and results["documents"] and len(results["documents"]) > 0:
                document: str = results["documents"][0]
                metadata: dict[str, Any] = results["metadatas"][0] if results["metadatas"] else {}
                memory_id: str = results["ids"][0] if results.get("ids") else identifier
                
                tags: list[str] = []
                if metadata.get("tags"):
                    tags = metadata["tags"].split(",") if isinstance(metadata["tags"], str) else metadata["tags"]
                
                timestamp: datetime = datetime.now()
                if metadata.get("timestamp"):
                    try:
                        timestamp = datetime.fromisoformat(metadata["timestamp"])
                    except (ValueError, TypeError):
                        self._logger.warning(f"Invalid timestamp format: {metadata.get('timestamp')}")

                # Parse enhanced metadata if available
                enhanced_meta: EnhancedMemoryMetadata | None = None
                if metadata:
                    try:
                        enhanced_meta = EnhancedMemoryMetadata.from_chromadb_dict(metadata)
                    except Exception as e:
                        self._logger.warning(f"Failed to parse enhanced metadata: {e}")

                memory_result: MemoryResult = MemoryResult(
                    memory_id=memory_id,
                    short_id=metadata.get("short_id"),
                    content=document,
                    tags=tags,
                    memory_type=metadata.get("memory_type"),
                    timestamp=timestamp,
                    metadata=metadata,
                    enhanced_metadata=enhanced_meta,
                    success=True
                )
                
                self._logger.info(f"Successfully read memory: {memory_result}")
                return memory_result
            
            self._logger.info(f"No memory found with identifier: {identifier}")
            return None
            
        except Exception as e:
            self._logger.error(f"Failed to read memory: {e}")
            raise VectorDatabaseError(f"Failed to read memory: {e}")
    
    def batch_read_memories(self, identifiers: list[str]) -> list[MemoryResult]:
        if not identifiers:
            error_msg: str = "No identifiers provided"
            self._logger.error(error_msg)
            raise MemoryValidationError(error_msg)
        
        self._logger.info(f"Batch reading {len(identifiers)} memories")
        
        if self._collection is None:
            error_msg: str = "Database not initialized"
            self._logger.error(error_msg)
            raise VectorDatabaseError(error_msg)
        
        memory_results: list[MemoryResult] = []
        
        uuid_ids: list[str] = []
        short_ids: list[str] = []
        
        for identifier in identifiers:
            if self._is_valid_uuid(identifier):
                uuid_ids.append(identifier)
            else:
                short_ids.append(identifier)
        
        try:
            if uuid_ids:
                self._logger.debug(f"Reading {len(uuid_ids)} memories by UUID")
                uuid_results = self._collection.get(
                    ids=uuid_ids,
                    include=["documents", "metadatas"]
                )
                
                for idx in range(len(uuid_results.get("documents", []))):
                    if uuid_results["documents"][idx]:
                        document: str = uuid_results["documents"][idx]
                        metadata: dict[str, Any] = uuid_results["metadatas"][idx] if uuid_results.get("metadatas") else {}
                        memory_id: str = uuid_results["ids"][idx]
                        
                        tags: list[str] = []
                        if metadata.get("tags"):
                            tags = metadata["tags"].split(",") if isinstance(metadata["tags"], str) else metadata["tags"]
                        
                        timestamp: datetime = datetime.now()
                        if metadata.get("timestamp"):
                            try:
                                timestamp = datetime.fromisoformat(metadata["timestamp"])
                            except (ValueError, TypeError):
                                pass

                        # Parse enhanced metadata if available
                        enhanced_meta: EnhancedMemoryMetadata | None = None
                        if metadata:
                            try:
                                enhanced_meta = EnhancedMemoryMetadata.from_chromadb_dict(metadata)
                            except Exception as e:
                                self._logger.warning(f"Failed to parse enhanced metadata: {e}")

                        memory_results.append(MemoryResult(
                            memory_id=memory_id,
                            short_id=metadata.get("short_id"),
                            content=document,
                            tags=tags,
                            memory_type=metadata.get("memory_type"),
                            timestamp=timestamp,
                            metadata=metadata,
                            enhanced_metadata=enhanced_meta,
                            success=True
                        ))
            
            for short_id in short_ids:
                self._logger.debug(f"Reading memory by short_id: {short_id}")
                short_id_results = self._collection.get(
                    where={"short_id": short_id},
                    include=["documents", "metadatas"]
                )
                
                if short_id_results and short_id_results.get("documents"):
                    for idx in range(len(short_id_results["documents"])):
                        if short_id_results["documents"][idx]:
                            document: str = short_id_results["documents"][idx]
                            metadata: dict[str, Any] = short_id_results["metadatas"][idx] if short_id_results.get("metadatas") else {}
                            memory_id: str = short_id_results["ids"][idx] if short_id_results.get("ids") else short_id
                            
                            tags: list[str] = []
                            if metadata.get("tags"):
                                tags = metadata["tags"].split(",") if isinstance(metadata["tags"], str) else metadata["tags"]
                            
                            timestamp: datetime = datetime.now()
                            if metadata.get("timestamp"):
                                try:
                                    timestamp = datetime.fromisoformat(metadata["timestamp"])
                                except (ValueError, TypeError):
                                    pass

                            # Parse enhanced metadata if available
                            enhanced_meta: EnhancedMemoryMetadata | None = None
                            if metadata:
                                try:
                                    enhanced_meta = EnhancedMemoryMetadata.from_chromadb_dict(metadata)
                                except Exception as e:
                                    self._logger.warning(f"Failed to parse enhanced metadata: {e}")

                            memory_results.append(MemoryResult(
                                memory_id=memory_id,
                                short_id=metadata.get("short_id"),
                                content=document,
                                tags=tags,
                                memory_type=metadata.get("memory_type"),
                                timestamp=timestamp,
                                metadata=metadata,
                                enhanced_metadata=enhanced_meta,
                                success=True
                            ))
            
            self._logger.info(f"Successfully read {len(memory_results)} memories")
            return memory_results
            
        except Exception as e:
            self._logger.error(f"Batch read failed: {e}")
            raise VectorDatabaseError(f"Batch read failed: {e}")
    
    def update_memory(self, identifier: str, content: str | None = None, tags: list[str] | None = None, timestamp: str | None = None) -> MemoryResult:
        if not identifier:
            error_msg: str = "Identifier cannot be empty"
            self._logger.error(error_msg)
            raise MemoryValidationError(error_msg)
        
        if content is not None and not content.strip():
            error_msg: str = "Content cannot be empty if provided"
            self._logger.error(error_msg)
            raise MemoryValidationError(error_msg)
        
        if self._collection is None:
            error_msg: str = "Database not initialized"
            self._logger.error(error_msg)
            raise VectorDatabaseError(error_msg)
        
        self._logger.info(f"Updating memory: {identifier}")
        
        existing_memory: MemoryResult | None = self.read_memory(identifier)
        if not existing_memory:
            error_msg: str = f"Memory not found: {identifier}"
            self._logger.error(error_msg)
            raise MemoryValidationError(error_msg)
        
        updated_content: str = content if content is not None else existing_memory.content
        updated_tags: list[str] = tags if tags is not None else existing_memory.tags
        
        updated_timestamp: datetime = existing_memory.timestamp
        if timestamp is not None:
            updated_timestamp = self._parse_timestamp(timestamp)
        
        try:
            new_metadata: dict[str, Any] = dict(existing_memory.metadata) if existing_memory.metadata else {}
            new_metadata["tags"] = ",".join(updated_tags) if updated_tags else ""
            new_metadata["timestamp"] = updated_timestamp.isoformat()
            
            if content is not None and self._embedding_engine is not None:
                self._logger.debug("Regenerating embeddings for updated content")
                new_embedding: Tensor = self._embedding_engine.encode(updated_content, is_query=False)
                
                self._collection.update(
                    ids=[existing_memory.memory_id],
                    embeddings=new_embedding.cpu().numpy().tolist(),
                    documents=[updated_content],
                    metadatas=[new_metadata]
                )
            else:
                self._collection.update(
                    ids=[existing_memory.memory_id],
                    documents=[updated_content],
                    metadatas=[new_metadata]
                )
            
            self._logger.info(f"Successfully updated memory: {existing_memory.memory_id}")
            
            return MemoryResult(
                memory_id=existing_memory.memory_id,
                short_id=existing_memory.short_id,
                content=updated_content,
                tags=updated_tags,
                memory_type=existing_memory.memory_type,
                timestamp=updated_timestamp,
                metadata=new_metadata,
                success=True
            )
            
        except Exception as e:
            self._logger.error(f"Failed to update memory: {e}")
            raise VectorDatabaseError(f"Failed to update memory: {e}")
    
    def batch_update_memories(
        self, 
        identifiers: list[str], 
        contents: list[str] | str | None = None,
        tags_list: list[list[str]] | list[str] | None = None,
        timestamps: list[str] | str | None = None
    ) -> list[MemoryResult]:
        if not identifiers:
            error_msg: str = "No identifiers provided"
            self._logger.error(error_msg)
            raise MemoryValidationError(error_msg)
        
        self._logger.info(f"Batch updating {len(identifiers)} memories")
        
        n_memories: int = len(identifiers)
        
        update_contents: list[str | None] = [None] * n_memories
        if contents is not None:
            if isinstance(contents, str):
                update_contents = [contents] * n_memories
            elif isinstance(contents, list):
                if len(contents) == 1:
                    update_contents = [contents[0]] * n_memories
                elif len(contents) == n_memories:
                    update_contents = contents
                else:
                    raise MemoryValidationError(f"Contents count {len(contents)} doesn't match identifiers count {n_memories}")
        
        update_tags: list[list[str] | None] = [None] * n_memories
        if tags_list is not None:
            if isinstance(tags_list, list):
                if tags_list and isinstance(tags_list[0], str):
                    update_tags = [tags_list] * n_memories
                elif len(tags_list) == 1:
                    update_tags = [tags_list[0]] * n_memories
                elif len(tags_list) == n_memories:
                    update_tags = tags_list
                else:
                    raise MemoryValidationError(f"Tags count {len(tags_list)} doesn't match identifiers count {n_memories}")
        
        update_timestamps: list[str | None] = [None] * n_memories
        if timestamps is not None:
            if isinstance(timestamps, str):
                update_timestamps = [timestamps] * n_memories
            elif isinstance(timestamps, list):
                if len(timestamps) == 1:
                    update_timestamps = [timestamps[0]] * n_memories
                elif len(timestamps) == n_memories:
                    update_timestamps = timestamps
                else:
                    raise MemoryValidationError(f"Timestamps count {len(timestamps)} doesn't match identifiers count {n_memories}")
        
        update_results: list[MemoryResult] = []
        for idx, identifier in enumerate(identifiers):
            try:
                result: MemoryResult = self.update_memory(
                    identifier,
                    content=update_contents[idx],
                    tags=update_tags[idx],
                    timestamp=update_timestamps[idx]
                )
                update_results.append(result)
            except (MemoryValidationError, VectorDatabaseError) as e:
                self._logger.warning(f"Failed to update {identifier}: {e}")
                update_results.append(MemoryResult(
                    memory_id=identifier,
                    short_id=None,
                    content="",
                    tags=[],
                    memory_type=None,
                    timestamp=datetime.now(),
                    metadata={},
                    success=False
                ))
        
        self._logger.info(f"Batch update completed: {sum(1 for r in update_results if r.success)}/{len(identifiers)} successful")
        return update_results
    
    def delete_memory(self, identifier: str, soft_delete: bool = False) -> bool:
        if not identifier:
            error_msg: str = "Identifier cannot be empty"
            self._logger.error(error_msg)
            raise MemoryValidationError(error_msg)
        
        if self._collection is None:
            error_msg: str = "Database not initialized"
            self._logger.error(error_msg)
            raise VectorDatabaseError(error_msg)
        
        self._logger.info(f"{'Soft' if soft_delete else 'Hard'} deleting memory: {identifier}")
        
        existing_memory: MemoryResult | None = self.read_memory(identifier)
        if not existing_memory:
            error_msg: str = f"Memory not found: {identifier}"
            self._logger.error(error_msg)
            raise MemoryValidationError(error_msg)
        
        try:
            if soft_delete:
                new_metadata: dict[str, Any] = dict(existing_memory.metadata) if existing_memory.metadata else {}
                new_metadata["deleted"] = True
                new_metadata["deleted_at"] = datetime.now().isoformat()
                
                self._collection.update(
                    ids=[existing_memory.memory_id],
                    metadatas=[new_metadata]
                )
                self._logger.info(f"Soft deleted memory: {existing_memory.memory_id}")
            else:
                self._collection.delete(ids=[existing_memory.memory_id])
                self._logger.info(f"Hard deleted memory: {existing_memory.memory_id}")
            
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to delete memory: {e}")
            raise VectorDatabaseError(f"Failed to delete memory: {e}")
    
    def batch_delete_memories(self, identifiers: list[str], soft_delete: bool = False) -> dict[str, bool]:
        if not identifiers:
            error_msg: str = "No identifiers provided"
            self._logger.error(error_msg)
            raise MemoryValidationError(error_msg)
        
        self._logger.info(f"Batch {'soft' if soft_delete else 'hard'} deleting {len(identifiers)} memories")
        
        delete_results: dict[str, bool] = {}
        for identifier in identifiers:
            try:
                success: bool = self.delete_memory(identifier, soft_delete)
                delete_results[identifier] = success
            except (MemoryValidationError, VectorDatabaseError) as e:
                self._logger.warning(f"Failed to delete {identifier}: {e}")
                delete_results[identifier] = False
        
        successful_deletes: int = sum(1 for success in delete_results.values() if success)
        self._logger.info(f"Batch delete completed: {successful_deletes}/{len(identifiers)} successful")
        return delete_results
    
    def retrieve_memory(self, query: str, n_results: int = 3) -> list[SearchResult]:
        if not query.strip():
            error_msg: str = "Query cannot be empty"
            self._logger.error(error_msg)
            raise MemoryValidationError(error_msg)
        if n_results < 1:
            error_msg: str = "n_results must be positive"
            self._logger.error(error_msg)
            raise ValueError(error_msg)
        
        self._logger.info(f"Retrieving memories for query: '{query[:50]}...' with n_results={n_results}")
        
        if self._embedding_engine is None or self._reranker_engine is None or self._collection is None:
            error_msg: str = "Components not initialized"
            self._logger.error(error_msg)
            raise VectorDatabaseError(error_msg)
        
        start_time: float = time.time()
        n_candidates: int = min(n_results * 3, 15)
        
        preprocessed_query: str = self._preprocess_query(query)
        
        try:
            self._logger.debug(f"Generating query embedding and fetching {n_candidates} candidates")
            query_embedding: Tensor = self._embedding_engine.encode(preprocessed_query, is_query=True)
            
            results = self._collection.query(
                query_embeddings=query_embedding.cpu().numpy().tolist(),
                n_results=n_candidates,
                include=["documents", "metadatas", "distances"]
            )
            
            documents: list[str] = results["documents"][0]
            distances: list[float] = results["distances"][0]
            metadatas: list[dict[str, Any]] = results["metadatas"][0] or [{} for _ in documents]
            ids: list[str] = results["ids"][0]
            
            self._logger.debug(f"Retrieved {len(documents)} candidates, reranking...")
            reranked_indices: list[tuple[int, float]] = self._reranker_engine.rerank(
                preprocessed_query, documents, top_k=n_results
            )
            
            search_results: list[SearchResult] = []
            for rank, (idx, score) in enumerate(reranked_indices, 1):
                if idx < len(documents):
                    # Parse enhanced metadata if available
                    enhanced_meta: EnhancedMemoryMetadata | None = None
                    if idx < len(metadatas) and metadatas[idx]:
                        try:
                            enhanced_meta = EnhancedMemoryMetadata.from_chromadb_dict(metadatas[idx])
                        except Exception as e:
                            self._logger.warning(f"Failed to parse enhanced metadata: {e}")

                    result: SearchResult = SearchResult(
                        rank=rank,
                        content=documents[idx],
                        rerank_score=score,
                        cosine_distance=distances[idx],
                        metadata=metadatas[idx],
                        memory_id=ids[idx],
                        short_id=metadatas[idx].get("short_id") if idx < len(metadatas) else None,
                        enhanced_metadata=enhanced_meta
                    )
                    search_results.append(result)
                    self._logger.debug(
                        f"Result {rank}: score={score:.3f}, distance={distances[idx]:.3f}, "
                        f"type={enhanced_meta.memory_type if enhanced_meta else 'N/A'}, "
                        f"importance={enhanced_meta.importance if enhanced_meta else 'N/A'}"
                    )
            
            total_time: float = time.time() - start_time
            self._logger.info(f"Retrieved {len(search_results)} results in {total_time:.3f} seconds")
            
            return search_results
            
        except Exception as e:
            self._logger.error(f"Memory retrieval failed: {e}")
            raise VectorDatabaseError(f"Memory retrieval failed: {e}")

    def smart_search(
        self,
        query: str,
        n_results: int = 3,
        importance_weight: float = 0.3,
        recency_weight: float = 0.2
    ) -> list[SearchResult]:
        """
        Enhanced search with importance and recency reranking.
        Combines semantic similarity with metadata-based scoring.
        """
        # Get base results from semantic search
        results: list[SearchResult] = self.retrieve_memory(query, n_results=n_results * 2)

        if not results:
            return results

        # Re-rank with importance and recency
        scored_results: list[tuple[float, SearchResult]] = []

        for result in results:
            # Base score from reranker
            score: float = result.rerank_score

            # Add importance bonus if available
            if result.enhanced_metadata:
                importance_bonus: float = result.enhanced_metadata.importance * importance_weight
                score += importance_bonus

                # Add recency bonus (memories from last 7 days get boost)
                days_old: int = (datetime.now() - result.enhanced_metadata.timestamp).days
                if days_old < 7:
                    recency_bonus: float = (7 - days_old) / 7 * recency_weight
                    score += recency_bonus

            scored_results.append((score, result))

        # Sort by new score and return top N
        scored_results.sort(key=lambda x: x[0], reverse=True)

        final_results: list[SearchResult] = []
        for rank, (score, result) in enumerate(scored_results[:n_results], 1):
            # Create new SearchResult with updated rank
            final_result: SearchResult = SearchResult(
                rank=rank,
                content=result.content,
                rerank_score=score,  # Use combined score
                cosine_distance=result.cosine_distance,
                metadata=result.metadata,
                memory_id=result.memory_id,
                short_id=result.short_id,
                enhanced_metadata=result.enhanced_metadata
            )
            final_results.append(final_result)

        self._logger.info(f"Smart search completed with importance/recency reranking: {len(final_results)} results")
        return final_results

    def consolidate_memories(
        self,
        similarity_threshold: float = 0.95,
        dry_run: bool = True
    ) -> dict[str, Any]:
        """
        Consolidate similar memories to reduce database bloat.
        Returns statistics about consolidation.
        """
        if self._collection is None:
            error_msg: str = "Database not initialized"
            self._logger.error(error_msg)
            raise VectorDatabaseError(error_msg)

        self._logger.info(f"Starting memory consolidation (dry_run={dry_run}, threshold={similarity_threshold})")

        try:
            # Get all memories
            all_memories = self._collection.get(include=["embeddings", "metadatas", "documents"])

            if not all_memories or not all_memories.get("ids"):
                return {"consolidated": 0, "kept": 0, "total": 0}

            total: int = len(all_memories["ids"])
            to_archive: list[str] = []

            # Simple consolidation: find very similar memories and keep most recent
            embeddings = all_memories.get("embeddings", [])
            metadatas = all_memories.get("metadatas", [])
            ids = all_memories["ids"]

            import numpy as np

            for i in range(len(embeddings)):
                if ids[i] in to_archive:
                    continue

                for j in range(i + 1, len(embeddings)):
                    if ids[j] in to_archive:
                        continue

                    # Calculate similarity
                    emb_i = np.array(embeddings[i])
                    emb_j = np.array(embeddings[j])
                    similarity: float = float(np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j)))

                    if similarity > similarity_threshold:
                        # Keep the one with higher importance or more recent
                        importance_i: float = float(metadatas[i].get("importance", 0.5))
                        importance_j: float = float(metadatas[j].get("importance", 0.5))

                        timestamp_i = metadatas[i].get("timestamp", "")
                        timestamp_j = metadatas[j].get("timestamp", "")

                        # Archive the less important or older one
                        if importance_i > importance_j:
                            to_archive.append(ids[j])
                        elif importance_j > importance_i:
                            to_archive.append(ids[i])
                            break  # i is archived, move to next i
                        elif timestamp_j > timestamp_i:
                            to_archive.append(ids[i])
                            break
                        else:
                            to_archive.append(ids[j])

            # Perform archival if not dry run
            if not dry_run and to_archive:
                # Mark as archived in metadata
                for memory_id in to_archive:
                    try:
                        memory: MemoryResult | None = self.read_memory(memory_id)
                        if memory and memory.metadata:
                            updated_meta: dict[str, Any] = dict(memory.metadata)
                            updated_meta["archived"] = "true"
                            self._collection.update(
                                ids=[memory_id],
                                metadatas=[updated_meta]
                            )
                    except Exception as e:
                        self._logger.warning(f"Failed to archive memory {memory_id}: {e}")

            stats: dict[str, Any] = {
                "total": total,
                "consolidated": len(to_archive),
                "kept": total - len(to_archive),
                "dry_run": dry_run
            }

            self._logger.info(f"Consolidation complete: {stats}")
            return stats

        except Exception as e:
            self._logger.error(f"Consolidation failed: {e}")
            raise VectorDatabaseError(f"Consolidation failed: {e}")

    # ==================== ADVANCED SEARCH FILTERS (Feature 7) ====================

    def filtered_search(
        self,
        query: str,
        n_results: int = 3,
        memory_type: str | None = None,
        importance_min: float | None = None,
        importance_max: float | None = None,
        project: str | None = None,
        session_id: str | None = None,
        entities: list[str] | None = None,
        topics: list[str] | None = None,
        has_action_items: bool | None = None,
        outcome: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None
    ) -> list[SearchResult]:
        """Advanced search with multiple metadata filters."""
        # Build ChromaDB where clause
        where_filters: dict[str, Any] = {}

        if memory_type:
            where_filters["memory_type"] = memory_type
        if importance_min is not None:
            where_filters["importance"] = {"$gte": importance_min}
        if importance_max is not None:
            if "importance" in where_filters:
                where_filters["importance"]["$lte"] = importance_max
            else:
                where_filters["importance"] = {"$lte": importance_max}
        if project:
            where_filters["project"] = project
        if session_id:
            where_filters["session_id"] = session_id
        if outcome:
            where_filters["outcome"] = outcome

        # Perform search with filters
        results: list[SearchResult] = self.smart_search(query, n_results=n_results * 2)

        # Post-filter for complex criteria
        filtered_results: list[SearchResult] = []
        for result in results:
            if not result.enhanced_metadata:
                continue

            meta: EnhancedMemoryMetadata = result.enhanced_metadata

            # Entity filter
            if entities and not any(e in meta.entities for e in entities):
                continue

            # Topic filter
            if topics and not any(t in meta.topics for t in topics):
                continue

            # Action items filter
            if has_action_items is not None:
                has_items: bool = len(meta.action_items) > 0
                if has_items != has_action_items:
                    continue

            # Date range filter
            if start_date and meta.timestamp < start_date:
                continue
            if end_date and meta.timestamp > end_date:
                continue

            filtered_results.append(result)

        self._logger.info(f"Filtered search returned {len(filtered_results)} results from {len(results)} candidates")
        return filtered_results[:n_results]

    # ==================== TEMPORAL QUERIES (Feature 10) ====================

    def search_by_time(
        self,
        query: str,
        start_date: datetime,
        end_date: datetime,
        n_results: int = 3
    ) -> list[SearchResult]:
        """Search memories within a specific time range."""
        return self.filtered_search(
            query=query,
            n_results=n_results,
            start_date=start_date,
            end_date=end_date
        )

    def search_recent(
        self,
        query: str,
        last_hours: int | None = None,
        last_days: int | None = None,
        n_results: int = 3
    ) -> list[SearchResult]:
        """Search recent memories (last N hours or days)."""
        now: datetime = datetime.now()
        start_date: datetime

        if last_hours is not None:
            start_date = now - timedelta(hours=last_hours)
        elif last_days is not None:
            start_date = now - timedelta(days=last_days)
        else:
            start_date = now - timedelta(days=7)  # Default: last week

        return self.search_by_time(query, start_date, now, n_results)

    # ==================== SESSION SUMMARIZATION (Feature 5) ====================

    def end_session(self, summarize: bool = True) -> str | None:
        """End the current session, optionally creating a summary."""
        if not self._current_session_id:
            self._logger.warning("No active session to end")
            return None

        session_id: str = self._current_session_id

        # Update session status to completed
        if session_id in self._sessions:
            session: SessionMetadata = self._sessions[session_id]
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
            self._sessions[session_id] = completed_session
            self._save_sessions()

        # Create summary if requested
        summary_text: str | None = None
        if summarize:
            summary_text = self._generate_session_summary(session_id)
            if summary_text:
                # Store summary as high-importance memory
                self.store_memory(
                    content=summary_text,
                    tags=["summary", "session"],
                    memory_type="semantic",
                    importance=0.9,
                    session_id=session_id
                )

        self._current_session_id = None
        self._session_sequence_num = 0
        self._logger.info(f"Ended session: {session_id}")

        return summary_text

    def _generate_session_summary(self, session_id: str) -> str:
        """Generate summary of session memories."""
        if self._collection is None:
            return ""

        try:
            # Get all memories from session
            results = self._collection.get(
                where={"session_id": session_id},
                include=["documents", "metadatas"]
            )

            if not results or not results.get("documents"):
                return ""

            # Extract key information
            decisions: list[str] = []
            all_action_items: list[str] = []
            outcomes: list[str] = []

            for idx, doc in enumerate(results["documents"]):
                content_lower: str = doc.lower()
                metadata: dict[str, Any] = results["metadatas"][idx] if results.get("metadatas") else {}

                # Extract decisions
                if any(keyword in content_lower for keyword in ["decided", "chose", "will implement"]):
                    decisions.append(doc[:200])

                # Extract action items
                action_items_str: str = metadata.get("action_items", "")
                if action_items_str:
                    items: list[str] = action_items_str.split(",")
                    all_action_items.extend(items)

                # Extract outcomes
                if metadata.get("outcome") == "completed":
                    outcomes.append(doc[:100])

            # Create summary
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

    # ==================== SESSION ANALYTICS (Feature 9) ====================

    def get_session_stats(self, session_id: str | None = None) -> dict[str, Any]:
        """Get comprehensive statistics for a session."""
        target_session_id: str | None = session_id or self._current_session_id
        if not target_session_id:
            return {}

        if self._collection is None:
            return {}

        try:
            # Get session metadata
            session_meta: SessionMetadata | None = self._sessions.get(target_session_id)

            # Get all memories from session
            results = self._collection.get(
                where={"session_id": target_session_id},
                include=["documents", "metadatas"]
            )

            if not results or not results.get("documents"):
                return {
                    "session_id": target_session_id,
                    "total_memories": 0
                }

            # Calculate statistics
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

                # Importance
                importance: float = float(metadata.get("importance", 0.5))
                importances.append(importance)

                # Topics
                topics_str: str = metadata.get("topics", "")
                if topics_str:
                    for topic in topics_str.split(","):
                        topic = topic.strip()
                        topics_count[topic] = topics_count.get(topic, 0) + 1

                # Entities
                entities_str: str = metadata.get("entities", "")
                if entities_str:
                    for entity in entities_str.split(","):
                        entity = entity.strip()
                        entities_count[entity] = entities_count.get(entity, 0) + 1

                # Memory types
                memory_type: str = metadata.get("memory_type", "episodic")
                memory_types[memory_type] = memory_types.get(memory_type, 0) + 1

                # Action items
                action_items_str: str = metadata.get("action_items", "")
                if action_items_str:
                    items: list[str] = [i.strip() for i in action_items_str.split(",") if i.strip()]
                    action_items_total += len(items)

                if metadata.get("outcome") == "completed":
                    action_items_completed += 1

                # Timestamps
                timestamp_str: str = metadata.get("timestamp")
                if timestamp_str:
                    timestamps.append(datetime.fromisoformat(timestamp_str))

            # Calculate duration
            duration_str: str = "N/A"
            if len(timestamps) >= 2:
                timestamps.sort()
                duration: timedelta = timestamps[-1] - timestamps[0]
                hours: int = int(duration.total_seconds() // 3600)
                minutes: int = int((duration.total_seconds() % 3600) // 60)
                duration_str = f"{hours}h {minutes}m"

            # Compile stats
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

            return stats

        except Exception as e:
            self._logger.error(f"Failed to get session stats: {e}")
            return {}

    # ==================== CROSS-SESSION RELATIONSHIPS (Feature 4) ====================

    def add_related_memory(self, memory_id: str, related_memory_id: str, bidirectional: bool = True) -> bool:
        """Add a relationship between two memories, optionally bidirectional."""
        try:
            # Update first memory
            memory1: MemoryResult | None = self.read_memory(memory_id)
            if not memory1 or not memory1.enhanced_metadata:
                return False

            # Add to related_memory_ids
            related_ids: list[str] = list(memory1.enhanced_metadata.related_memory_ids)
            if related_memory_id not in related_ids:
                related_ids.append(related_memory_id)

            # Update metadata in ChromaDB
            if self._collection:
                current_meta: dict[str, Any] = dict(memory1.metadata)
                current_meta["related_memory_ids"] = ",".join(related_ids)
                self._collection.update(
                    ids=[memory_id],
                    metadatas=[current_meta]
                )

            # Bidirectional linking
            if bidirectional:
                memory2: MemoryResult | None = self.read_memory(related_memory_id)
                if memory2 and memory2.enhanced_metadata:
                    related_ids_2: list[str] = list(memory2.enhanced_metadata.related_memory_ids)
                    if memory_id not in related_ids_2:
                        related_ids_2.append(memory_id)
                        current_meta_2: dict[str, Any] = dict(memory2.metadata)
                        current_meta_2["related_memory_ids"] = ",".join(related_ids_2)
                        self._collection.update(
                            ids=[related_memory_id],
                            metadatas=[current_meta_2]
                        )

            self._logger.info(f"Added relationship: {memory_id} <-> {related_memory_id}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to add related memory: {e}")
            return False

    def get_related_memories(self, memory_id: str, max_depth: int = 2) -> list[MemoryResult]:
        """Get all related memories following relationship links up to max_depth."""
        visited: set[str] = set()
        related: list[MemoryResult] = []

        def traverse(current_id: str, depth: int) -> None:
            if depth > max_depth or current_id in visited:
                return

            visited.add(current_id)
            memory: MemoryResult | None = self.read_memory(current_id)

            if memory and memory.enhanced_metadata:
                related.append(memory)
                for related_id in memory.enhanced_metadata.related_memory_ids:
                    if related_id and related_id not in visited:
                        traverse(related_id, depth + 1)

        traverse(memory_id, 0)
        return related[1:]  # Exclude the original memory

    def store_memory(
        self,
        content: str,
        tags: list[str] | None = None,
        timestamp: str | None = None,
        memory_type: str | None = None,
        importance: float = 0.5,
        session_id: str | None = None,
        project: str | None = None,
        action_items: list[str] | None = None,
        outcome: str | None = None,
        parent_memory_id: str | None = None,
        related_memory_ids: list[str] | None = None,
        auto_importance: bool = False,
        auto_tags: bool = False
    ) -> StorageResult:
        if not content.strip():
            error_msg: str = "Content cannot be empty"
            self._logger.error(error_msg)
            raise MemoryValidationError(error_msg)

        # Auto-tag suggestion (Feature 8)
        memory_tags: list[str] = tags if tags else []
        if auto_tags:
            suggested: list[str] = self._suggest_tags(content)
            memory_tags = list(set(memory_tags + suggested))
            self._logger.info(f"Auto-suggested tags: {suggested}")

        self._logger.info(f"Storing memory with {len(memory_tags)} tags")
        self._logger.debug(f"Memory content preview: {content[:100]}...")

        memory_date: datetime = self._parse_timestamp(timestamp) if timestamp else datetime.now()

        # Auto-importance calculation (Feature 6)
        final_importance: float = importance
        if auto_importance:
            action_items_list: list[str] = action_items if action_items else []
            final_importance = self._calculate_importance(content, memory_tags, action_items_list)
            self._logger.info(f"Auto-calculated importance: {final_importance:.2f}")

        short_id: str = self._generate_short_id(content, memory_type)

        # Extract entities and topics automatically
        entities: list[str] = self._extract_entities(content, memory_tags)
        topics: list[str] = self._extract_topics(content, memory_tags)

        # Handle session tracking and automatic parent linking
        actual_session_id: str | None = session_id
        actual_parent_id: str | None = parent_memory_id
        sequence_num: int = 0

        if self._enable_session_tracking:
            # Use current session if no explicit session_id provided
            if actual_session_id is None and self._current_session_id:
                actual_session_id = self._current_session_id

            # Auto-link to previous memory in session if no explicit parent
            if actual_parent_id is None and actual_session_id:
                actual_parent_id = self._get_last_memory_in_session(actual_session_id)

            # Increment sequence number
            if actual_session_id == self._current_session_id:
                self._session_sequence_num += 1
                sequence_num = self._session_sequence_num

        # Create enhanced metadata
        enhanced_metadata: EnhancedMemoryMetadata = EnhancedMemoryMetadata(
            memory_type=memory_type if memory_type in ["episodic", "semantic", "procedural", "working"] else "episodic",
            importance=final_importance,  # Use auto-calculated or manual importance
            session_id=actual_session_id,
            project=project,
            entities=entities,
            topics=topics,
            action_items=action_items if action_items else [],
            outcome=outcome if outcome in ["completed", "pending", "failed", "cancelled"] else None,
            access_count=0,
            last_accessed=None,
            parent_memory_id=actual_parent_id,
            related_memory_ids=related_memory_ids if related_memory_ids else [],
            sequence_num=sequence_num,
            tags=memory_tags,
            timestamp=memory_date,
            short_id=short_id
        )

        if self._embedding_engine is None or self._collection is None:
            error_msg: str = "Components not initialized"
            self._logger.error(error_msg)
            raise VectorDatabaseError(error_msg)

        try:
            content_embedding: Tensor = self._embedding_engine.encode(content, is_query=False)

            import uuid
            memory_id: str = str(uuid.uuid4())

            # Convert enhanced metadata to ChromaDB format
            chromadb_metadata: dict[str, Any] = enhanced_metadata.to_chromadb_dict()

            self._collection.add(
                ids=[memory_id],
                embeddings=content_embedding.cpu().numpy().tolist(),
                documents=[content],
                metadatas=[chromadb_metadata]
            )

            self._logger.info(
                f"Successfully stored memory with ID: {memory_id}, "
                f"entities: {entities}, topics: {topics}, importance: {final_importance:.2f}"
            )

            # Update session metadata (Feature 3)
            if actual_session_id and actual_session_id in self._sessions:
                session: SessionMetadata = self._sessions[actual_session_id]
                # Create updated session with new values
                updated_session: SessionMetadata = SessionMetadata(
                    session_id=session.session_id,
                    name=session.name,
                    project=session.project,
                    topic=session.topic,
                    participants=session.participants,
                    created_at=session.created_at,
                    last_activity=datetime.now(),
                    status=session.status,
                    total_memories=session.total_memories + 1,
                    avg_importance=((session.avg_importance * session.total_memories) + final_importance) / (session.total_memories + 1)
                )
                self._sessions[actual_session_id] = updated_session
                self._save_sessions()

            return StorageResult(
                memory_id=memory_id,
                short_id=short_id,
                success=True,
                message=f"Memory stored with {len(memory_tags)} tags, {len(entities)} entities, {len(topics)} topics - {memory_date.strftime('%d/%m/%Y %I:%M %p')}"
            )

        except Exception as e:
            self._logger.error(f"Memory storage failed: {e}")
            raise VectorDatabaseError(f"Memory storage failed: {e}")
    
    def batch_store_memories(
        self, 
        contents: list[str], 
        tags_list: list[list[str]], 
        timestamps: list[str] | str | None = None,
        memory_types: list[str] | str | None = None
    ) -> list[StorageResult]:
        self._logger.info(f"Starting batch storage of {len(contents)} memories")
        
        if not contents:
            error_msg: str = "No memories provided for batch storage"
            self._logger.error(error_msg)
            raise BatchValidationError(error_msg)
        
        if len(contents) != len(tags_list):
            error_msg: str = f"{len(contents)} memories provided but only {len(tags_list)} tag sets. Each memory requires tags."
            self._logger.error(error_msg)
            raise BatchValidationError(error_msg)
        
        parsed_timestamps: list[datetime] = self._process_batch_timestamps(contents, timestamps)
        
        parsed_memory_types: list[str | None] = []
        if memory_types is None:
            parsed_memory_types = [None] * len(contents)
        elif isinstance(memory_types, str):
            parsed_memory_types = [memory_types] * len(contents)
        elif isinstance(memory_types, list):
            if len(memory_types) == 1:
                parsed_memory_types = [memory_types[0]] * len(contents)
            elif len(memory_types) == len(contents):
                parsed_memory_types = memory_types
            else:
                error_msg: str = f"Memory types count ({len(memory_types)}) doesn't match contents count ({len(contents)})"
                self._logger.error(error_msg)
                raise BatchValidationError(error_msg)
        
        for idx, content in enumerate(contents):
            if not content.strip():
                error_msg: str = f"Memory at index {idx} is empty"
                self._logger.error(error_msg)
                raise MemoryValidationError(error_msg)
        
        if self._embedding_engine is None or self._collection is None:
            error_msg: str = "Components not initialized"
            self._logger.error(error_msg)
            raise VectorDatabaseError(error_msg)
        
        try:
            self._logger.info("Generating embeddings for all memories")
            all_embeddings: Tensor = self._embedding_engine.encode(contents, is_query=False)
            
            import uuid
            memory_ids: list[str] = []
            short_ids: list[str] = []
            documents: list[str] = []
            metadatas: list[dict[str, Any]] = []
            embeddings_list: list[list[float]] = []
            
            for idx in range(len(contents)):
                short_id: str = self._generate_short_id(contents[idx], parsed_memory_types[idx])
                short_ids.append(short_id)
                
                memory: MemoryContent = MemoryContent(
                    content=contents[idx],
                    tags=tags_list[idx],
                    timestamp=parsed_timestamps[idx],
                    memory_type=parsed_memory_types[idx],
                    short_id=short_id
                )
                
                memory_id: str = str(uuid.uuid4())
                memory_ids.append(memory_id)
                documents.append(memory.content)
                metadatas.append(memory.metadata)
                
                embedding: list[float] = all_embeddings[idx].cpu().numpy().tolist()
                embeddings_list.append(embedding)
            
            self._logger.info(f"Storing {len(memory_ids)} memories in ChromaDB")
            self._collection.add(
                ids=memory_ids,
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas
            )
            
            results: list[StorageResult] = []
            for idx, memory_id in enumerate(memory_ids):
                result: StorageResult = StorageResult(
                    memory_id=memory_id,
                    short_id=short_ids[idx],
                    success=True,
                    message=f"Memory {idx + 1} stored with {len(tags_list[idx])} tags - {parsed_timestamps[idx].strftime('%d/%m/%Y %I:%M %p')}"
                )
                results.append(result)
                self._logger.debug(f"Stored memory {idx + 1} with ID: {memory_id}")
            
            self._logger.info(f"Successfully stored all {len(contents)} memories")
            return results
            
        except Exception as e:
            self._logger.error(f"Batch memory storage failed: {e}")
            raise VectorDatabaseError(f"Batch memory storage failed: {e}")
    
    def _process_batch_timestamps(self, contents: list[str], timestamps: list[str] | str | None) -> list[datetime]:
        if timestamps is None:
            self._logger.debug("No timestamps provided, using current time for all memories")
            current_time: datetime = datetime.now()
            return [current_time for _ in contents]
        
        if isinstance(timestamps, str):
            self._logger.debug(f"Single timestamp provided: {timestamps}, applying to all memories")
            parsed_date: datetime = self._parse_timestamp(timestamps)
            return [parsed_date for _ in contents]
        
        if isinstance(timestamps, list):
            if len(timestamps) == 1:
                self._logger.debug(f"Single timestamp in list: {timestamps[0]}, applying to all memories")
                parsed_date: datetime = self._parse_timestamp(timestamps[0])
                return [parsed_date for _ in contents]
            
            if len(timestamps) != len(contents):
                error_msg: str = f"{len(contents)} memories with {len(timestamps)} dates provided. Expected 1 date for all or {len(contents)} individual dates."
                self._logger.error(error_msg)
                raise BatchValidationError(error_msg)
            
            self._logger.debug("Individual timestamps provided for each memory")
            return [self._parse_timestamp(ts) for ts in timestamps]
        
        error_msg: str = f"Invalid timestamps type: {type(timestamps)}"
        self._logger.error(error_msg)
        raise BatchValidationError(error_msg)
    
    def _generate_short_id(self, content: str, memory_type: str | None = None) -> str:
        words: list[str] = content.lower().split()[:5]
        
        words = [w for w in words if w not in {'the', 'a', 'an', 'is', 'are', 'was', 'were', '|'}]
        
        if memory_type:
            short_id: str = f"{memory_type.lower()}-{'-'.join(words[:3])}"
        else:
            short_id: str = f"memory-{'-'.join(words[:3])}"
        
        short_id = re.sub(r'[^a-z0-9-]', '', short_id)
        
        short_id = re.sub(r'-+', '-', short_id).strip('-')
        
        if len(short_id) > 50:
            short_id = short_id[:50]
        
        return short_id
    
    def _is_valid_uuid(self, identifier: str) -> bool:
        try:
            import uuid
            uuid.UUID(identifier)
            return True
        except (ValueError, AttributeError):
            return False
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        timestamp_formats: list[str] = [
            "%I:%M %p, %d/%m/%Y",
            "%H:%M, %d/%m/%Y",
            "%d/%m/%Y %I:%M %p",
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y",
            "%d-%m-%Y", 
            "%Y-%m-%d",
            "%d.%m.%Y"
        ]
        
        for fmt in timestamp_formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        raise ValueError(f"Invalid timestamp format: {timestamp_str}. Supported formats: 'DD/MM/YYYY', 'HH:MM PM, DD/MM/YYYY', 'DD/MM/YYYY HH:MM PM'")

