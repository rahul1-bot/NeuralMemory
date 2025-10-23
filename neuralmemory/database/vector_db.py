from __future__ import annotations
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
    def __init__(self, db_path: str) -> None:
        self._db_path: Path = Path(db_path)
        self._client: chromadb.PersistentClient | None = None
        self._collection: Any | None = None
        self._embedding_engine: Qwen3EmbeddingEngine | None = None
        self._reranker_engine: Qwen3RerankerEngine | None = None
        
        log_path: Path = Path(__file__).parent / "logs" / "neuralvector.log"
        self._logger: logging.Logger = LoggerSetup.get_logger("NeuralVector", log_path)
        self._logger.info(f"Initializing NeuralVector with database path: {db_path}")
        
        self._initialize_components()
    
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
                
                memory_result: MemoryResult = MemoryResult(
                    memory_id=memory_id,
                    short_id=metadata.get("short_id"),
                    content=document,
                    tags=tags,
                    memory_type=metadata.get("memory_type"),
                    timestamp=timestamp,
                    metadata=metadata,
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
                        
                        memory_results.append(MemoryResult(
                            memory_id=memory_id,
                            short_id=metadata.get("short_id"),
                            content=document,
                            tags=tags,
                            memory_type=metadata.get("memory_type"),
                            timestamp=timestamp,
                            metadata=metadata,
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
                            
                            memory_results.append(MemoryResult(
                                memory_id=memory_id,
                                short_id=metadata.get("short_id"),
                                content=document,
                                tags=tags,
                                memory_type=metadata.get("memory_type"),
                                timestamp=timestamp,
                                metadata=metadata,
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
                    result: SearchResult = SearchResult(
                        rank=rank,
                        content=documents[idx],
                        rerank_score=score,
                        cosine_distance=distances[idx],
                        metadata=metadatas[idx],
                        memory_id=ids[idx],
                        short_id=metadatas[idx].get("short_id") if idx < len(metadatas) else None
                    )
                    search_results.append(result)
                    self._logger.debug(f"Result {rank}: score={score:.3f}, distance={distances[idx]:.3f}")
            
            total_time: float = time.time() - start_time
            self._logger.info(f"Retrieved {len(search_results)} results in {total_time:.3f} seconds")
            
            return search_results
            
        except Exception as e:
            self._logger.error(f"Memory retrieval failed: {e}")
            raise VectorDatabaseError(f"Memory retrieval failed: {e}")
    
    def store_memory(self, content: str, tags: list[str] | None = None, timestamp: str | None = None, memory_type: str | None = None) -> StorageResult:
        if not content.strip():
            error_msg: str = "Content cannot be empty"
            self._logger.error(error_msg)
            raise MemoryValidationError(error_msg)
        
        self._logger.info(f"Storing memory with {len(tags) if tags else 0} tags")
        self._logger.debug(f"Memory content preview: {content[:100]}...")
        
        memory_date: datetime = self._parse_timestamp(timestamp) if timestamp else datetime.now()
        memory_tags: list[str] = tags if tags else []
        
        short_id: str = self._generate_short_id(content, memory_type)
        
        memory: MemoryContent = MemoryContent(
            content=content,
            tags=memory_tags,
            timestamp=memory_date,
            memory_type=memory_type,
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
            
            self._collection.add(
                ids=[memory_id],
                embeddings=content_embedding.cpu().numpy().tolist(),
                documents=[memory.content],
                metadatas=[memory.metadata]
            )
            
            self._logger.info(f"Successfully stored memory with ID: {memory_id}")
            
            return StorageResult(
                memory_id=memory_id,
                short_id=short_id,
                success=True,
                message=f"Memory stored with {len(memory_tags)} tags - {memory_date.strftime('%d/%m/%Y %I:%M %p')}"
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

