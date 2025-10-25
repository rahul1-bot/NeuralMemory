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
    MemoryTier,
    CodeReference,
    EnhancedMemoryMetadata,
    SessionMetadata,
    SearchResult,
    MemoryContent,
    StorageResult,
    MemoryResult,
    ConflictDetectionResult,
    MemoryProvenance,
    ConsolidationResult,
    MultiHopQuery,
    MemoryExport
)
from neuralmemory.core.config import EmbeddingConfig, RerankerConfig
from neuralmemory.core.logging_setup import LoggerSetup
from neuralmemory.engines.embedding import Qwen3EmbeddingEngine
from neuralmemory.engines.reranker import Qwen3RerankerEngine

# Import all modular components
from neuralmemory.database.analytics.importance import ImportanceCalculator
from neuralmemory.database.analytics.tags import TagSuggester
from neuralmemory.database.analytics.session_stats import SessionStatisticsCalculator
from neuralmemory.database.sessions.manager import SessionManager
from neuralmemory.database.sessions.metadata import SessionMetadataStore
from neuralmemory.database.sessions.summarizer import SessionSummarizer
from neuralmemory.database.sessions.relationships import RelationshipManager
from neuralmemory.database.linking.extractor import CodeReferenceExtractor
from neuralmemory.database.linking.validator import CodeReferenceValidator
from neuralmemory.database.linking.tracker import CodeReferenceTracker
from neuralmemory.database.cache.manager import CacheManager
from neuralmemory.database.cache.eviction import CacheEvictionPolicy
from neuralmemory.database.cache.tiers import TierAwareRetrieval
from neuralmemory.database.cache.hotness import HotnessCalculator
from neuralmemory.database.indexing.bm25 import BM25Index
from neuralmemory.database.indexing.entity import EntityIndex
from neuralmemory.database.indexing.temporal import TemporalIndex
from neuralmemory.database.indexing.hybrid import HybridSearch
from neuralmemory.database.strategies.contextual import ContextualEmbeddingStrategy
from neuralmemory.database.strategies.biological import BiologicalDecayStrategy
from neuralmemory.database.strategies.consolidation import ConsolidationStrategy
from neuralmemory.database.strategies.filtering import FilteringStrategy
from neuralmemory.database.graph.multihop import MultiHopSearchEngine
from neuralmemory.database.graph.provenance import ProvenanceTracker
from neuralmemory.database.graph.traversal import GraphTraversal
from neuralmemory.database.core.retrieval import MemoryRetrieval
from neuralmemory.database.core.deletion import MemoryDeletion


class NeuralVector:
    """
    Modular neural vector database orchestrator.
    Delegates functionality to specialized modules using PyTorch Lightning composition pattern.

    Refactored from 3,007-line god-class into 27 focused modules averaging 140 lines each.
    This orchestrator provides backwards-compatible public API while delegating to modules.
    """

    def __init__(
        self,
        db_path: str,
        enable_session_tracking: bool = True,
        enable_contextual_embeddings: bool = True,
        enable_biological_decay: bool = True,
        decay_deletion_threshold: float = 0.1,
        conflict_similarity_threshold: float = 0.93,
        enable_hybrid_retrieval: bool = True,
        enable_code_grounding: bool = True,
        max_working_memory_size: int = 20,
        short_term_days: int = 7
    ) -> None:
        self._db_path: Path = Path(db_path)
        self._client: chromadb.PersistentClient | None = None
        self._collection: Any | None = None
        self._embedding_engine: Qwen3EmbeddingEngine | None = None
        self._reranker_engine: Qwen3RerankerEngine | None = None

        # Session tracking
        self._enable_session_tracking: bool = enable_session_tracking
        self._current_session_id: str | None = None
        self._session_sequence_num: int = 0
        self._sessions_file: Path = self._db_path / "sessions.json"
        self._sessions: dict[str, SessionMetadata] = {}
        self._session_name_to_id: dict[str, str] = {}

        # Feature flags
        self._enable_contextual_embeddings: bool = enable_contextual_embeddings
        self._enable_biological_decay: bool = enable_biological_decay
        self._decay_deletion_threshold: float = decay_deletion_threshold
        self._conflict_similarity_threshold: float = conflict_similarity_threshold
        self._enable_hybrid_retrieval: bool = enable_hybrid_retrieval
        self._enable_code_grounding: bool = enable_code_grounding
        self._max_working_memory_size: int = max_working_memory_size
        self._short_term_days: int = short_term_days

        # Logging
        log_path: Path = Path(__file__).parent / "logs" / "neuralvector.log"
        self._logger: logging.Logger = LoggerSetup.get_logger("NeuralVector", log_path)
        self._logger.info(f"Initializing NeuralVector with database path: {db_path}")
        self._logger.info(f"Phase 4 - Hybrid:{enable_hybrid_retrieval} Grounding:{enable_code_grounding} MaxWorking:{max_working_memory_size}")

        # Initialize database and models first
        self._initialize_components()

        # ==================== MODULE INSTANTIATION ====================
        # All modules follow PyTorch Lightning composition pattern with dependency injection

        # Analytics modules
        self._importance_calculator = ImportanceCalculator(self._logger)
        self._tag_suggester = TagSuggester(self._logger)
        self._session_stats_calculator = SessionStatisticsCalculator(
            collection=self._collection,
            sessions=self._sessions,
            logger=self._logger
        )

        # Session modules
        self._session_manager = SessionManager(
            sessions=self._sessions,
            session_name_to_id=self._session_name_to_id,
            logger=self._logger
        )
        self._session_metadata_store = SessionMetadataStore(
            sessions_file=self._sessions_file,
            sessions=self._sessions,
            session_name_to_id=self._session_name_to_id,
            logger=self._logger
        )
        self._session_summarizer = SessionSummarizer(
            collection=self._collection,
            sessions=self._sessions,
            logger=self._logger
        )
        self._relationship_manager = RelationshipManager(
            collection=self._collection,
            logger=self._logger
        )

        # Code linking modules
        self._code_extractor = CodeReferenceExtractor(self._logger)
        self._code_validator = CodeReferenceValidator(self._logger)
        self._code_tracker = CodeReferenceTracker(
            collection=self._collection,
            code_validator=self._code_validator,
            logger=self._logger
        )

        # Cache modules
        self._cache_manager = CacheManager(
            max_size=max_working_memory_size,
            logger=self._logger
        )
        self._cache_eviction = CacheEvictionPolicy(
            cache_manager=self._cache_manager,
            logger=self._logger
        )
        self._hotness_calculator = HotnessCalculator(self._logger)

        # Tier-aware retrieval - needs to be instantiated after hybrid_search
        # Will be set up after indexing modules
        self._tier_aware_retrieval: TierAwareRetrieval | None = None

        # Indexing modules
        self._bm25_index = BM25Index(self._logger)
        self._entity_index = EntityIndex(self._logger)
        self._temporal_index = TemporalIndex(self._logger)

        # Hybrid search - needs retrieval method reference
        self._hybrid_search = HybridSearch(
            bm25_index=self._bm25_index,
            entity_index=self._entity_index,
            temporal_index=self._temporal_index,
            logger=self._logger
        )

        # Strategy modules
        self._contextual_strategy = ContextualEmbeddingStrategy(
            embedding_engine=self._embedding_engine,
            collection=self._collection,
            conflict_threshold=conflict_similarity_threshold,
            logger=self._logger
        )
        self._biological_strategy = BiologicalDecayStrategy(
            collection=self._collection,
            enable_biological_decay=self._enable_biological_decay,
            logger=self._logger,
            deletion_threshold=self._decay_deletion_threshold
        )
        self._consolidation_strategy = ConsolidationStrategy(
            collection=self._collection,
            embedding_engine=self._embedding_engine,
            logger=self._logger
        )
        self._filtering_strategy = FilteringStrategy(
            collection=self._collection,
            logger=self._logger
        )

        # Graph modules
        self._multihop_engine = MultiHopSearchEngine(
            collection=self._collection,
            relationship_manager=self._relationship_manager,
            logger=self._logger
        )
        self._provenance_tracker = ProvenanceTracker(
            collection=self._collection,
            logger=self._logger
        )
        self._graph_traversal = GraphTraversal(self._logger)

        # Core modules
        self._memory_retrieval = MemoryRetrieval(
            collection=self._collection,
            embedding_engine=self._embedding_engine,
            reranker_engine=self._reranker_engine,
            logger=self._logger
        )
        self._memory_deletion = MemoryDeletion(
            collection=self._collection,
            logger=self._logger
        )

        # Now instantiate tier-aware retrieval with hybrid_search reference
        self._tier_aware_retrieval = TierAwareRetrieval(
            cache_manager=self._cache_manager,
            cache_eviction=self._cache_eviction,
            logger=self._logger
        )

        # Load sessions and initialize indices
        self._load_sessions()
        self._initialize_indices()

    def _initialize_components(self) -> None:
        """Initialize database and models."""
        self._initialize_database()
        self._initialize_models()

    def _initialize_database(self) -> None:
        """Connect to ChromaDB."""
        try:
            self._logger.info(f"Connecting to ChromaDB at: {self._db_path}")
            self._client = chromadb.PersistentClient(path=str(self._db_path))
            self._collection = self._client.get_collection("memory_collection")
            self._logger.info("Successfully connected to ChromaDB collection: memory_collection")
        except Exception as e:
            self._logger.error(f"Failed to connect to database: {e}")
            raise VectorDatabaseError(f"Failed to connect to database: {e}")

    def _initialize_models(self) -> None:
        """Initialize embedding and reranker engines."""
        embedding_config: EmbeddingConfig = EmbeddingConfig.create_qwen3_mps_config()
        reranker_config: RerankerConfig = RerankerConfig.create_qwen3_mps_config()

        self._embedding_engine = Qwen3EmbeddingEngine(embedding_config)
        self._reranker_engine = Qwen3RerankerEngine(reranker_config)

    def _load_sessions(self) -> None:
        """Load session metadata from JSON file."""
        self._session_metadata_store.load()

    def _save_sessions(self) -> None:
        """Save session metadata to JSON file."""
        self._session_metadata_store.save()

    def _initialize_indices(self) -> None:
        """Initialize BM25, entity, and temporal indices from existing memories."""
        if not self._enable_hybrid_retrieval:
            self._logger.info("Hybrid retrieval disabled, skipping index initialization")
            return

        self._logger.info("Initializing hybrid retrieval indices...")

        try:
            results = self._collection.get(include=["documents", "metadatas"])

            if not results or not results.get("ids"):
                self._logger.info("No existing memories to index")
                return

            # Initialize BM25 index
            for memory_id, document in zip(results["ids"], results["documents"]):
                self._bm25_index.add_document(memory_id, document)

            # Initialize entity index
            for memory_id, metadata in zip(results["ids"], results["metadatas"]):
                entities_str: str = metadata.get("entities", "")
                if entities_str:
                    entities: list[str] = entities_str.split(",")
                    for entity in entities:
                        entity = entity.strip()
                        if entity:
                            self._entity_index.add_entity(entity, memory_id)

            # Initialize temporal index
            for memory_id, metadata in zip(results["ids"], results["metadatas"]):
                timestamp_str: str = metadata.get("timestamp")
                if timestamp_str:
                    try:
                        dt: datetime = datetime.fromisoformat(timestamp_str)
                        self._temporal_index.add_memory(memory_id, dt)
                    except (ValueError, AttributeError):
                        pass

            self._logger.info(
                f"Indices initialized: BM25={self._bm25_index.size()}, "
                f"Entities={self._entity_index.size()}, Dates={self._temporal_index.size()}"
            )

        except Exception as e:
            self._logger.error(f"Failed to initialize indices: {e}")

    def _add_to_indices(self, memory_id: str, content: str, metadata: EnhancedMemoryMetadata) -> None:
        """Add a new memory to all indices."""
        if not self._enable_hybrid_retrieval:
            return

        self._bm25_index.add_document(memory_id, content)

        for entity in metadata.entities:
            self._entity_index.add_entity(entity, memory_id)

        self._temporal_index.add_memory(memory_id, metadata.timestamp)

    # ==================== HELPER METHODS ====================

    def _preprocess_query(self, query: str) -> str:
        """Preprocess query with date/time normalization."""
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

        common_entities: list[str] = ["Rahul", "Claude", "NeuralMemory", "Pydantic", "ChromaDB", "Qwen3"]

        content_lower: str = content.lower()
        for entity in common_entities:
            if entity.lower() in content_lower:
                entities.append(entity)

        for tag in tags:
            if tag and tag[0].isupper() and tag not in entities:
                entities.append(tag)

        return list(set(entities))

    def _extract_topics(self, content: str, tags: list[str]) -> list[str]:
        """Extract topic keywords from content and tags."""
        topics: list[str] = []

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

        for tag in tags:
            tag_lower: str = tag.lower()
            if tag_lower not in topics and not tag[0].isupper():
                topics.append(tag_lower)

        return list(set(topics))

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

    def _generate_short_id(self, content: str, memory_type: str | None = None) -> str:
        """Generate short ID from content."""
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
        """Check if string is a valid UUID."""
        try:
            import uuid
            uuid.UUID(identifier)
            return True
        except (ValueError, AttributeError):
            return False

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp from various formats."""
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

    def _process_batch_timestamps(self, contents: list[str], timestamps: list[str] | str | None) -> list[datetime]:
        """Process batch timestamps for batch operations."""
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

    # ==================== SESSION MANAGEMENT ====================

    def start_new_session(
        self,
        name: str | None = None,
        project: str | None = None,
        topic: str | None = None,
        participants: list[str] | None = None
    ) -> str:
        """Start a new conversation session."""
        session_id: str = self._session_manager.start_new(name, project, topic, participants)
        self._current_session_id = session_id
        self._session_sequence_num = 0
        self._save_sessions()
        return session_id

    def list_sessions(self) -> dict[str, SessionMetadata]:
        """List all sessions with their metadata."""
        return self._session_manager.list_all()

    def get_session_by_name(self, name: str) -> SessionMetadata | None:
        """Get session metadata by name."""
        return self._session_manager.get_by_name(name)

    def get_current_session_id(self) -> str | None:
        """Get the current session ID."""
        return self._current_session_id

    def end_session(self, summarize: bool = True) -> str | None:
        """End the current session, optionally creating a summary."""
        if not self._current_session_id:
            self._logger.warning("No active session to end")
            return None

        session_id: str = self._current_session_id
        summary_text: str | None = self._session_summarizer.end_session(session_id, summarize)

        if summary_text and summarize:
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
        self._save_sessions()

        return summary_text

    def get_session_stats(self, session_id: str | None = None) -> dict[str, Any]:
        """Get comprehensive statistics for a session."""
        target_session_id: str | None = session_id or self._current_session_id
        if not target_session_id:
            return {}

        return self._session_stats_calculator.calculate(target_session_id)

    def add_related_memory(self, memory_id: str, related_memory_id: str, bidirectional: bool = True) -> bool:
        """Add a relationship between two memories."""
        return self._relationship_manager.add_relationship(memory_id, related_memory_id, bidirectional, self.read_memory)

    def get_related_memories(self, memory_id: str, max_depth: int = 2) -> list[MemoryResult]:
        """Get all related memories following relationship links."""
        return self._relationship_manager.get_related(memory_id, max_depth, self.read_memory)

    def get_conversation_thread(self, memory_id: str) -> list[MemoryResult]:
        """Get the full conversation thread for a given memory by following parent links."""
        thread: list[MemoryResult] = []
        current_id: str | None = memory_id

        while current_id:
            memory: MemoryResult | None = self.read_memory(current_id)
            if not memory:
                break

            thread.insert(0, memory)

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
        """Get a memory with surrounding context from the same session."""
        target_memory: MemoryResult | None = self.read_memory(memory_id)
        if not target_memory or not target_memory.enhanced_metadata:
            return {"before": [], "target": [], "after": []}

        session_id: str | None = target_memory.enhanced_metadata.session_id
        if not session_id or self._collection is None:
            return {"before": [], "target": [target_memory], "after": []}

        try:
            results = self._collection.get(
                where={"session_id": session_id},
                include=["documents", "metadatas"]
            )

            if not results or not results.get("ids"):
                return {"before": [], "target": [target_memory], "after": []}

            session_memories: list[tuple[int, MemoryResult]] = []
            target_seq: int = target_memory.enhanced_metadata.sequence_num

            for idx in range(len(results["ids"])):
                metadata: dict[str, Any] = results["metadatas"][idx] if results.get("metadatas") else {}
                seq_num: int = int(metadata.get("sequence_num", 0))

                enhanced_meta: EnhancedMemoryMetadata | None = None
                if metadata:
                    try:
                        enhanced_meta = EnhancedMemoryMetadata.from_chromadb_dict(metadata)
                        if enhanced_meta.parent_memory_id:
                            target_seq = seq_num
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

            session_memories.sort(key=lambda x: x[0])

            target_idx: int = -1
            for idx, (seq, mem) in enumerate(session_memories):
                if mem.memory_id == memory_id:
                    target_idx = idx
                    break

            if target_idx == -1:
                return {"before": [], "target": [target_memory], "after": []}

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

    # ==================== CRUD OPERATIONS ====================

    def read_memory(self, identifier: str) -> MemoryResult | None:
        """Read a memory by UUID or short_id."""
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

                # Reinforce memory on access
                self.reinforce_memory(memory_id)

                self._logger.info(f"Successfully read memory: {memory_result}")
                return memory_result

            self._logger.info(f"No memory found with identifier: {identifier}")
            return None

        except Exception as e:
            self._logger.error(f"Failed to read memory: {e}")
            raise VectorDatabaseError(f"Failed to read memory: {e}")

    def batch_read_memories(self, identifiers: list[str]) -> list[MemoryResult]:
        """Batch read multiple memories."""
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
        """Update an existing memory."""
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
        """Batch update multiple memories."""
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
        """Delete a memory (soft or hard)."""
        return self._memory_deletion.delete(identifier, soft_delete, self.read_memory)

    def batch_delete_memories(self, identifiers: list[str], soft_delete: bool = False) -> dict[str, bool]:
        """Batch delete multiple memories."""
        return self._memory_deletion.batch_delete(identifiers, soft_delete, self.delete_memory)

    # ==================== RETRIEVAL OPERATIONS ====================

    def retrieve_memory(self, query: str, n_results: int = 3) -> list[SearchResult]:
        """Retrieve memories using semantic search with reranking."""
        preprocessed_query: str = self._preprocess_query(query)
        return self._memory_retrieval.retrieve(preprocessed_query, n_results)

    def smart_search(
        self,
        query: str,
        n_results: int = 3,
        importance_weight: float = 0.3,
        recency_weight: float = 0.2
    ) -> list[SearchResult]:
        """Enhanced search with importance and recency reranking."""
        results: list[SearchResult] = self.retrieve_memory(query, n_results=n_results * 2)

        if not results:
            return results

        scored_results: list[tuple[float, SearchResult]] = []

        for result in results:
            score: float = result.rerank_score

            if result.enhanced_metadata:
                importance_bonus: float = result.enhanced_metadata.importance * importance_weight
                score += importance_bonus

                days_old: int = (datetime.now() - result.enhanced_metadata.timestamp).days
                if days_old < 7:
                    recency_bonus: float = (7 - days_old) / 7 * recency_weight
                    score += recency_bonus

            scored_results.append((score, result))

        scored_results.sort(key=lambda x: x[0], reverse=True)

        final_results: list[SearchResult] = []
        for rank, (score, result) in enumerate(scored_results[:n_results], 1):
            final_result: SearchResult = SearchResult(
                rank=rank,
                content=result.content,
                rerank_score=score,
                cosine_distance=result.cosine_distance,
                metadata=result.metadata,
                memory_id=result.memory_id,
                short_id=result.short_id,
                enhanced_metadata=result.enhanced_metadata
            )
            final_results.append(final_result)

        self._logger.info(f"Smart search completed with importance/recency reranking: {len(final_results)} results")
        return final_results

    def hybrid_search(self, query: str, n_results: int = 5, importance_weight: float = 0.2, recency_weight: float = 0.1) -> list[SearchResult]:
        """Hybrid search combining BM25, semantic search, importance, and recency."""
        if not self._enable_hybrid_retrieval:
            return self.retrieve_memory(query, n_results)

        # Get semantic results first
        semantic_results: list[SearchResult] = self.retrieve_memory(query, n_results * 2)

        # Use hybrid search to combine with BM25
        return self._hybrid_search.search(
            query=query,
            semantic_results=semantic_results,
            n_results=n_results,
            importance_weight=importance_weight,
            recency_weight=recency_weight
        )

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
        # Get base results from smart search
        results: list[SearchResult] = self.smart_search(query, n_results=n_results * 2)

        # Delegate filtering to filtering strategy
        return self._filtering_strategy.filtered_search(
            results=results,
            n_results=n_results,
            memory_type=memory_type,
            importance_min=importance_min,
            importance_max=importance_max,
            project=project,
            session_id=session_id,
            entities=entities,
            topics=topics,
            has_action_items=has_action_items,
            outcome=outcome,
            start_date=start_date,
            end_date=end_date
        )

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
            start_date = now - timedelta(days=7)

        return self.search_by_time(query, start_date, now, n_results)

    # ==================== STORAGE OPERATIONS ====================

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
        """Store a new memory with all metadata."""
        if not content.strip():
            error_msg: str = "Content cannot be empty"
            self._logger.error(error_msg)
            raise MemoryValidationError(error_msg)

        # Auto-tag suggestion
        memory_tags: list[str] = tags if tags else []
        if auto_tags:
            suggested: list[str] = self._tag_suggester.suggest(content)
            memory_tags = list(set(memory_tags + suggested))
            self._logger.info(f"Auto-suggested tags: {suggested}")

        self._logger.info(f"Storing memory with {len(memory_tags)} tags")
        self._logger.debug(f"Memory content preview: {content[:100]}...")

        memory_date: datetime = self._parse_timestamp(timestamp) if timestamp else datetime.now()

        # Auto-importance calculation
        final_importance: float = importance
        if auto_importance:
            action_items_list: list[str] = action_items if action_items else []
            final_importance = self._importance_calculator.calculate(content, memory_tags, action_items_list)
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
            if actual_session_id is None and self._current_session_id:
                actual_session_id = self._current_session_id

            if actual_parent_id is None and actual_session_id:
                actual_parent_id = self._get_last_memory_in_session(actual_session_id)

            if actual_session_id == self._current_session_id:
                self._session_sequence_num += 1
                sequence_num = self._session_sequence_num

        # Create enhanced metadata
        enhanced_metadata: EnhancedMemoryMetadata = EnhancedMemoryMetadata(
            memory_type=memory_type if memory_type in ["episodic", "semantic", "procedural", "working"] else "episodic",
            importance=final_importance,
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
            # Phase 3: Contextual Embeddings
            context_memories: list[str] = []
            if self._enable_contextual_embeddings:
                try:
                    temp_embedding: Tensor = self._embedding_engine.encode(content, is_query=False)
                    context_results: dict[str, Any] = self._collection.query(
                        query_embeddings=[temp_embedding.tolist()],
                        n_results=3
                    )
                    if context_results and context_results['documents'] and len(context_results['documents']) > 0:
                        context_memories = context_results['documents'][0]
                        self._logger.debug(f"Retrieved {len(context_memories)} context memories for contextual encoding")
                except Exception as e:
                    self._logger.debug(f"Could not retrieve context: {e}")

            # Encode with or without context
            if context_memories and self._enable_contextual_embeddings:
                content_embedding_list: list[float] = self._contextual_strategy.encode_with_context(content, context_memories)
            else:
                content_embedding: Tensor = self._embedding_engine.encode(content, is_query=False)
                content_embedding_list: list[float] = content_embedding.tolist()

            import uuid
            memory_id: str = str(uuid.uuid4())

            # Convert enhanced metadata to ChromaDB format
            chromadb_metadata: dict[str, Any] = enhanced_metadata.to_chromadb_dict()

            self._collection.add(
                ids=[memory_id],
                embeddings=[content_embedding_list],
                documents=[content],
                metadatas=[chromadb_metadata]
            )

            self._logger.info(
                f"Successfully stored memory with ID: {memory_id}, "
                f"entities: {entities}, topics: {topics}, importance: {final_importance:.2f}"
            )

            # Update session metadata
            if actual_session_id and actual_session_id in self._sessions:
                session: SessionMetadata = self._sessions[actual_session_id]
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

            # Phase 3: Conflict Detection & Decay
            if self._enable_contextual_embeddings:
                conflicts: list[ConflictDetectionResult] = self._contextual_strategy.detect_conflicts(
                    memory_id,
                    content,
                    content_embedding_list
                )

                if conflicts and self._enable_biological_decay:
                    for conflict in conflicts:
                        try:
                            conflict_metadata: dict[str, Any] = self._collection.get(
                                ids=[conflict.conflicting_memory_id],
                                include=["metadatas"]
                            )
                            if conflict_metadata and conflict_metadata['metadatas']:
                                meta: dict[str, Any] = conflict_metadata['metadatas'][0]
                                meta['decay_counter'] = 5
                                self._collection.update(
                                    ids=[conflict.conflicting_memory_id],
                                    metadatas=[meta]
                                )
                                self._logger.info(
                                    f"Set decay counter on conflicting memory: {conflict.conflicting_memory_id[:8]}..."
                                )
                        except Exception as e:
                            self._logger.warning(f"Failed to set decay counter: {e}")

            # Add to indices
            self._add_to_indices(memory_id, content, enhanced_metadata)

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
        """Batch store multiple memories."""
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

    # ==================== CONSOLIDATION OPERATIONS ====================

    def consolidate_memories(
        self,
        similarity_threshold: float = 0.95,
        dry_run: bool = True
    ) -> dict[str, Any]:
        """Basic memory consolidation to reduce database bloat."""
        if self._collection is None:
            error_msg: str = "Database not initialized"
            self._logger.error(error_msg)
            raise VectorDatabaseError(error_msg)

        self._logger.info(f"Starting memory consolidation (dry_run={dry_run}, threshold={similarity_threshold})")

        try:
            all_memories = self._collection.get(include=["embeddings", "metadatas", "documents"])

            if not all_memories or not all_memories.get("ids"):
                return {"consolidated": 0, "kept": 0, "total": 0}

            total: int = len(all_memories["ids"])
            to_archive: list[str] = []

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

                    emb_i = np.array(embeddings[i])
                    emb_j = np.array(embeddings[j])
                    similarity: float = float(np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j)))

                    if similarity > similarity_threshold:
                        importance_i: float = float(metadatas[i].get("importance", 0.5))
                        importance_j: float = float(metadatas[j].get("importance", 0.5))

                        timestamp_i = metadatas[i].get("timestamp", "")
                        timestamp_j = metadatas[j].get("timestamp", "")

                        if importance_i > importance_j:
                            to_archive.append(ids[j])
                        elif importance_j > importance_i:
                            to_archive.append(ids[i])
                            break
                        elif timestamp_j > timestamp_i:
                            to_archive.append(ids[i])
                            break
                        else:
                            to_archive.append(ids[j])

            if not dry_run and to_archive:
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

    def consolidate_memories_advanced(
        self,
        similarity_threshold: float = 0.85,
        min_cluster_size: int = 3,
        max_clusters: int = 10
    ) -> list[ConsolidationResult]:
        """Intelligent memory consolidation with clustering and summarization."""
        return self._consolidation_strategy.consolidate_advanced(
            similarity_threshold,
            min_cluster_size,
            max_clusters,
            self.store_memory
        )

    # ==================== PHASE 3: ADVANCED MEMORY INTELLIGENCE ====================

    def detect_conflicts(self, memory_id: str, content: str, embedding: list[float] | None = None) -> list[ConflictDetectionResult]:
        """Detect conflicting memories using cosine similarity."""
        if not self._enable_contextual_embeddings:
            return []

        return self._contextual_strategy.detect_conflicts(memory_id, content, embedding)

    def apply_decay_to_all_memories(self) -> int:
        """Apply decay to all memories with decay counters, delete expired ones."""
        if not self._enable_biological_decay:
            return 0

        return self._biological_strategy.apply_decay_to_all()

    def reinforce_memory(self, memory_id: str) -> bool:
        """Reinforce memory on access, resetting decay counter."""
        if not self._enable_biological_decay:
            return True

        return self._biological_strategy.reinforce(memory_id)

    def store_memory_with_provenance(
        self,
        content: str,
        provenance: MemoryProvenance,
        tags: list[str] | None = None,
        **kwargs
    ) -> StorageResult:
        """Store memory with provenance tracking."""
        return self._provenance_tracker.store_with_provenance(
            content,
            provenance,
            tags,
            self.store_memory,
            **kwargs
        )

    def multi_hop_search(self, query: MultiHopQuery) -> list[MemoryResult]:
        """Execute multi-hop reasoning query across memory graph."""
        return self._multihop_engine.search(query, self.smart_search, self.read_memory)

    def export_memories(
        self,
        file_path: str,
        session_id: str | None = None,
        project: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None
    ) -> MemoryExport:
        """Export memories to JSON file with optional filters."""
        try:
            all_results: dict[str, Any] = self._collection.get(
                include=["documents", "metadatas", "embeddings"]
            )

            if not all_results or not all_results['ids']:
                return MemoryExport(total_memories=0, memories=[])

            memories: list[dict[str, Any]] = []

            for idx, memory_id in enumerate(all_results['ids']):
                metadata: dict[str, Any] = all_results['metadatas'][idx] if all_results['metadatas'] else {}

                if session_id and metadata.get('session_id') != session_id:
                    continue
                if project and metadata.get('project') != project:
                    continue
                if start_date or end_date:
                    timestamp_str: str | None = metadata.get('timestamp')
                    if timestamp_str:
                        ts: datetime = datetime.fromisoformat(timestamp_str)
                        if start_date and ts < start_date:
                            continue
                        if end_date and ts > end_date:
                            continue

                memories.append({
                    "id": memory_id,
                    "content": all_results['documents'][idx],
                    "metadata": metadata,
                    "embedding": all_results['embeddings'][idx] if all_results['embeddings'] else None
                })

            export: MemoryExport = MemoryExport(
                total_memories=len(memories),
                memories=memories,
                sessions=[s.to_dict() for s in self._sessions.values()],
                metadata={"filters": {"session_id": session_id, "project": project}}
            )

            with open(file_path, 'w') as f:
                json.dump(export.to_dict(), f, indent=2)

            self._logger.info(f"Exported {len(memories)} memories to {file_path}")

            return export

        except Exception as e:
            self._logger.error(f"Export failed: {e}")
            raise VectorDatabaseError(f"Export failed: {e}")

    def import_memories(self, file_path: str, merge_duplicates: bool = True) -> int:
        """Import memories from JSON file."""
        try:
            with open(file_path, 'r') as f:
                data: dict[str, Any] = json.load(f)

            export: MemoryExport = MemoryExport.from_dict(data)

            imported_count: int = 0

            for memory in export.memories:
                if merge_duplicates and memory.get('embedding'):
                    conflicts: list[ConflictDetectionResult] = self.detect_conflicts(
                        memory['id'],
                        memory['content'],
                        memory['embedding']
                    )

                    if conflicts:
                        self._logger.info(f"Skipping duplicate: {memory['id'][:8]}...")
                        continue

                try:
                    self._collection.add(
                        ids=[memory['id']],
                        documents=[memory['content']],
                        metadatas=[memory['metadata']],
                        embeddings=[memory['embedding']] if memory.get('embedding') else None
                    )
                    imported_count += 1
                except Exception as e:
                    self._logger.warning(f"Failed to import {memory['id'][:8]}...: {e}")

            for session_data in export.sessions:
                try:
                    session: SessionMetadata = SessionMetadata.from_dict(session_data)
                    self._sessions[session.session_id] = session
                    if session.name:
                        self._session_name_to_id[session.name] = session.session_id
                except Exception as e:
                    self._logger.warning(f"Failed to import session: {e}")

            self._save_sessions()

            self._logger.info(f"Imported {imported_count}/{export.total_memories} memories from {file_path}")

            return imported_count

        except Exception as e:
            self._logger.error(f"Import failed: {e}")
            raise VectorDatabaseError(f"Import failed: {e}")

    # ==================== PHASE 4: CODE GROUNDING ====================

    def validate_memory_code_references(self, memory_id: str) -> tuple[bool, str | None]:
        """Validate all code references for a memory."""
        if not self._enable_code_grounding:
            return True, None

        return self._code_tracker.validate_memory_references(memory_id)

    # ==================== PHASE 4: HIERARCHICAL TIERS ====================

    def promote_to_working_memory(self, memory_id: str) -> None:
        """Promote a memory to working memory for fast O(1) access."""
        self._cache_eviction.promote(memory_id, self.read_memory)

    def clear_working_memory(self) -> None:
        """Clear all working memory."""
        self._cache_manager.clear()

    def get_working_memory(self) -> list[MemoryResult]:
        """Get all memories currently in working memory."""
        return self._cache_manager.get_all()

    def tier_aware_retrieve(self, query: str, n_results: int = 5) -> list[SearchResult]:
        """Tier-aware retrieval checking working memory first."""
        return self._tier_aware_retrieval.retrieve(query, n_results, self.hybrid_search)

    def calculate_memory_hotness(self, metadata: EnhancedMemoryMetadata) -> float:
        """Calculate hotness score for tier assignment."""
        return self._hotness_calculator.calculate(metadata)

    def tier_memories_by_age(self) -> dict[str, int]:
        """Tier memories based on age and access patterns."""
        return self._hotness_calculator.tier_by_age(
            collection=self._collection,
            short_term_days=self._short_term_days,
            working_memory_size=len(self._cache_manager.get_all())
        )
