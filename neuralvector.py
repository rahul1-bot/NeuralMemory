from __future__ import annotations
import logging
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from unittest.mock import MagicMock

triton_mock = MagicMock()
triton_mock.__spec__ = MagicMock()
triton_mock.__spec__.name = 'triton'
sys.modules['triton'] = triton_mock

triton_lang_mock = MagicMock()
triton_lang_mock.__spec__ = MagicMock()
triton_lang_mock.__spec__.name = 'triton.language'
sys.modules['triton.language'] = triton_lang_mock

import chromadb
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


class LoggerSetup:
    @staticmethod
    def get_logger(name: str, log_file: Path) -> logging.Logger:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger: logging.Logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        
        if logger.handlers:
            return logger
        
        file_handler: logging.FileHandler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.DEBUG)
        
        formatter: logging.Formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger


class EmbeddingEngineError(Exception):
    pass


class RerankerEngineError(Exception):
    pass


class VectorDatabaseError(Exception):
    pass


class BatchValidationError(Exception):
    pass


class MemoryValidationError(Exception):
    pass


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


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    model_path: str
    max_length: int
    instruction: str
    device: str
    
    def __post_init__(self) -> None:
        if self.max_length < 1:
            raise ValueError("Max length must be positive")
        if not self.instruction.strip():
            raise ValueError("Instruction cannot be empty")
        if self.device not in {"mps", "cpu", "cuda"}:
            raise ValueError("Device must be mps, cpu, or cuda")
    
    def __str__(self) -> str:
        return f"EmbeddingConfig({self.device}, {self.max_length})"
    
    @classmethod
    def create_qwen3_mps_config(cls) -> EmbeddingConfig:
        device: str = "mps" if torch.backends.mps.is_available() else "cpu"
        return cls(
            model_path="Qwen/Qwen3-Embedding-8B",
            max_length=8192,
            instruction="Given a web search query, retrieve relevant passages that answer the query",
            device=device
        )


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    model_path: str
    max_length: int
    instruction: str
    device: str
    
    def __post_init__(self) -> None:
        if self.max_length < 1:
            raise ValueError("Max length must be positive")
        if not self.instruction.strip():
            raise ValueError("Instruction cannot be empty")
        if self.device not in {"mps", "cpu", "cuda"}:
            raise ValueError("Device must be mps, cpu, or cuda")
    
    def __str__(self) -> str:
        return f"RerankerConfig({self.device}, {self.max_length})"
    
    @classmethod
    def create_qwen3_mps_config(cls) -> RerankerConfig:
        device: str = "mps" if torch.backends.mps.is_available() else "cpu"
        return cls(
            model_path="/Users/rahulsawhney/LocalCode/Models/Qwen3-Reranker-8B",
            max_length=2048,
            instruction="Given a web search query, retrieve relevant passages that answer the query",
            device=device
        )


class Qwen3EmbeddingEngine:
    def __init__(self, config: EmbeddingConfig) -> None:
        self._config: EmbeddingConfig = config
        self._model: AutoModel | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._initialize_environment()
        self._load_models()
    
    def _initialize_environment(self) -> None:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    def _load_models(self) -> None:
        try:
            self._model = AutoModel.from_pretrained(
                self._config.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                attn_implementation="eager"
            ).to(self._config.device)
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._config.model_path,
                trust_remote_code=True,
                padding_side="left"
            )
        except Exception as e:
            raise EmbeddingEngineError(f"Failed to load embedding model: {e}")
    
    def _last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding: bool = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        
        sequence_lengths: Tensor = attention_mask.sum(dim=1) - 1
        batch_size: int = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def _get_detailed_instruct(self, query: str) -> str:
        return f"Instruct: {self._config.instruction}\nQuery:{query}"
    
    def encode(self, sentences: list[str] | str, is_query: bool = False) -> Tensor:
        if self._model is None or self._tokenizer is None:
            raise EmbeddingEngineError("Models not loaded")
        
        if isinstance(sentences, str):
            sentences = [sentences]
        
        processed_sentences: list[str] = sentences
        if is_query:
            processed_sentences = [self._get_detailed_instruct(sent) for sent in sentences]
        
        try:
            inputs = self._tokenizer(
                processed_sentences,
                padding=True,
                truncation=True,
                max_length=self._config.max_length,
                return_tensors="pt"
            ).to(self._config.device)
            
            with torch.no_grad():
                model_outputs = self._model(**inputs)
                output: Tensor = self._last_token_pool(
                    model_outputs.last_hidden_state, 
                    inputs["attention_mask"]
                )
                output = F.normalize(output, p=2, dim=1)
            
            return output
            
        except Exception as e:
            raise EmbeddingEngineError(f"Encoding failed: {e}")


class Qwen3RerankerEngine:
    def __init__(self, config: RerankerConfig) -> None:
        self._config: RerankerConfig = config
        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._token_false_id: int = 0
        self._token_true_id: int = 0
        self._prefix_tokens: list[int] = []
        self._suffix_tokens: list[int] = []
        self._initialize_environment()
        self._load_models()
    
    def _initialize_environment(self) -> None:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    def _load_models(self) -> None:
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._config.model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self._config.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                attn_implementation="eager"
            ).to(self._config.device).eval()
            
            self._token_false_id = self._tokenizer.convert_tokens_to_ids("no")
            self._token_true_id = self._tokenizer.convert_tokens_to_ids("yes")
            
            prefix: str = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            suffix: str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            
            self._prefix_tokens = self._tokenizer.encode(prefix, add_special_tokens=False)
            self._suffix_tokens = self._tokenizer.encode(suffix, add_special_tokens=False)
            
        except Exception as e:
            raise RerankerEngineError(f"Failed to load reranker model: {e}")
    
    def _format_instruction(self, query: str, document: str) -> str:
        return f"<Instruct>: {self._config.instruction}\n<Query>: {query}\n<Document>: {document}"
    
    def _process_inputs(self, pairs: list[str]) -> dict[str, Tensor]:
        if self._tokenizer is None:
            raise RerankerEngineError("Tokenizer not loaded")
        
        try:
            out = self._tokenizer(
                pairs,
                padding=False,
                truncation="longest_first",
                return_attention_mask=False,
                max_length=self._config.max_length - len(self._prefix_tokens) - len(self._suffix_tokens)
            )
            
            for i, tokens in enumerate(out["input_ids"]):
                out["input_ids"][i] = self._prefix_tokens + tokens + self._suffix_tokens
            
            out = self._tokenizer.pad(
                out, 
                padding=True, 
                return_tensors="pt", 
                max_length=self._config.max_length
            )
            
            for key in out:
                out[key] = out[key].to(self._config.device)
            
            return out
            
        except Exception as e:
            raise RerankerEngineError(f"Input processing failed: {e}")
    
    def compute_scores(self, query: str, documents: list[str]) -> list[float]:
        if self._model is None:
            raise RerankerEngineError("Model not loaded")
        
        try:
            pairs: list[str] = [self._format_instruction(query, doc) for doc in documents]
            inputs: dict[str, Tensor] = self._process_inputs(pairs)
            
            with torch.no_grad():
                batch_scores: Tensor = self._model(**inputs).logits[:, -1, :]
                true_vector: Tensor = batch_scores[:, self._token_true_id]
                false_vector: Tensor = batch_scores[:, self._token_false_id]
                combined_scores: Tensor = torch.stack([false_vector, true_vector], dim=1)
                log_softmax_scores: Tensor = torch.nn.functional.log_softmax(combined_scores, dim=1)
                scores: list[float] = log_softmax_scores[:, 1].exp().tolist()
            
            return scores
            
        except Exception as e:
            raise RerankerEngineError(f"Score computation failed: {e}")
    
    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        scores: list[float] = self.compute_scores(query, documents)
        ranked_results: list[tuple[int, float]] = [(i, score) for i, score in enumerate(scores)]
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        return ranked_results[:top_k]


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


class MemoryTextProcessor:
    def __init__(self) -> None:
        self._logger: logging.Logger = LoggerSetup.get_logger(
            "MemoryTextProcessor",
            Path(__file__).parent / "logs" / "memory_processor.log"
        )
    
    def parse_comma_separated(self, value: str) -> list[str]:
        self._logger.debug(f"Parsing comma-separated value: {value[:50]}...")
        
        items: list[str] = []
        current: str = ""
        in_quotes: bool = False
        quote_char: str | None = None
        
        for char in value:
            if char == '"' and not in_quotes:
                in_quotes = True
                quote_char = char
                current += char
            elif char == quote_char and in_quotes:
                in_quotes = False
                current += char
                quote_char = None
            elif char == "," and not in_quotes:
                items.append(current.strip())
                current = ""
            else:
                current += char
        
        if current:
            items.append(current.strip())
        
        cleaned_items: list[str] = []
        for item in items:
            if item.startswith('"') and item.endswith('"'):
                processed_item: str = self._process_escape_sequences(item[1:-1])
                cleaned_items.append(processed_item)
            else:
                cleaned_items.append(item)
        
        self._logger.info(f"Parsed {len(cleaned_items)} items from comma-separated input")
        return cleaned_items
    
    def _process_escape_sequences(self, text: str) -> str:
        self._logger.debug("Processing escape sequences")
        
        processed: str = text
        processed = processed.replace("\\n", "\n")
        processed = processed.replace("\\t", "\t")
        processed = processed.replace("\\\\", "\\")
        processed = processed.replace('\\"', '"')
        
        return processed


class MemoryArgumentParser:
    def __init__(self) -> None:
        import argparse
        
        self._parser: argparse.ArgumentParser = argparse.ArgumentParser(
            description=self._get_description(),
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self._get_epilog()
        )
        self._setup_arguments()
    
    def _get_description(self) -> str:
        return "Neural Memory System - Vector Search CLI"
    
    def _get_epilog(self) -> str:
        return "Use --help for more information"
    
    def _setup_arguments(self) -> None:
        self._parser.add_argument(
            "query",
            type=str,
            nargs="?",
            help="Search query (required unless --store is used)"
        )
        
        self._parser.add_argument(
            "--store",
            type=str,
            nargs='+',
            metavar="CONTENT",
            help="Store memory content(s) - can provide multiple quoted strings"
        )
        
        self._parser.add_argument(
            "--read",
            type=str,
            nargs='+',
            metavar="ID",
            help="Read memory by ID (UUID or short_id). Can read multiple."
        )
        
        self._parser.add_argument(
            "--update",
            type=str,
            nargs='+',
            metavar="ID",
            help="Update memory by ID. Can update multiple memories."
        )
        
        self._parser.add_argument(
            "--delete",
            type=str,
            nargs='+',
            metavar="ID",
            help="Delete memory by ID. Can delete multiple memories."
        )
        
        self._parser.add_argument(
            "--content",
            type=str,
            nargs='+',
            metavar="CONTENT",
            help="New content for update - one per memory or one for all"
        )
        
        self._parser.add_argument(
            "--tags", 
            type=str,
            nargs='+',
            metavar="TAGS",
            help="Tags for memories - provide one quoted set per memory"
        )
        
        self._parser.add_argument(
            "--timestamp",
            type=str,
            nargs='+',
            metavar="TIMESTAMP",
            help="Timestamp for memory (e.g. '04/08/2025' or '05:22 PM, 04/08/2025')"
        )
        
        self._parser.add_argument(
            "--when",
            type=str,
            nargs='+',
            metavar="WHEN",
            help="When the memory occurred (e.g. '04/08/2025' or '05:22 PM, 04/08/2025')"
        )
        
        self._parser.add_argument(
            "--memory-date",
            type=str,
            nargs='+',
            metavar="MEMORY_DATE",
            dest="memory_timestamp",
            help="Date/time of the memory (e.g. '04/08/2025' or '05:22 PM, 04/08/2025')"
        )
        
        self._parser.add_argument(
            "--n_results",
            type=int,
            default=3,
            metavar="N",
            help="Number of results to return (default: 3, max: 50)"
        )
        
        self._parser.add_argument(
            "--db_path",
            type=str,
            default="/Users/rahulsawhney/.mcp_memory/chroma_db",
            metavar="PATH",
            help="Database path (default: ~/.mcp_memory/chroma_db)"
        )
    
    def parse_arguments(self) -> Any:
        return self._parser.parse_args()
    
    def print_help(self) -> None:
        self._parser.print_help()


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
        
        # First check if content has actual newlines
        if '\n' in result.content:
            content_lines: list[str] = result.content.split('\n')
            formatted_content: str = "Content:\n"
            for line in content_lines:
                formatted_content += f"  {line}\n"
            formatted_content = formatted_content.rstrip()
        # Check if content has literal \n that needs processing
        elif '\\n' in result.content:
            # Process escape sequences that weren't processed during storage
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


class MemoryCLI:
    def __init__(self) -> None:
        self._arg_parser: MemoryArgumentParser = self._create_argument_parser()
        self._formatter: MemoryFormatter = self._create_formatter()
        self._text_processor: MemoryTextProcessor = MemoryTextProcessor()
        self._neural_vector: NeuralVector | None = None
        
        log_path: Path = Path(__file__).parent / "logs" / self._get_log_filename()
        self._logger: logging.Logger = LoggerSetup.get_logger(self._get_logger_name(), log_path)
        self._original_cwd: str = os.getcwd()
    
    def _create_argument_parser(self) -> MemoryArgumentParser:
        return MemoryArgumentParser()
    
    def _create_formatter(self) -> MemoryFormatter:
        return MemoryFormatter()
    
    def _get_log_filename(self) -> str:
        return "memory_cli.log"
    
    def _get_logger_name(self) -> str:
        return "MemoryCLI"
    
    def _is_store_mode(self, args: Any) -> bool:
        return args.store is not None
    
    def _is_read_mode(self, args: Any) -> bool:
        return args.read is not None
    
    def _is_update_mode(self, args: Any) -> bool:
        return args.update is not None
    
    def _is_delete_mode(self, args: Any) -> bool:
        return args.delete is not None
    
    def _get_timestamp_value(self, args: Any) -> list[str] | None:
        if args.timestamp:
            return args.timestamp
        elif args.when:
            return args.when
        elif hasattr(args, 'memory_timestamp') and getattr(args, 'memory_timestamp', None):
            return getattr(args, 'memory_timestamp')
        return None
    
    def _validate_store_arguments(self, args: Any) -> None:
        if not args.store:
            raise ValueError("Store content cannot be empty")
        
        if isinstance(args.store, list):
            for memory in args.store:
                if not memory.strip():
                    raise ValueError("Store content cannot be empty")
        else:
            if not args.store.strip():
                raise ValueError("Store content cannot be empty")
        
        if not Path(args.db_path).exists():
            raise FileNotFoundError(f"Database path does not exist: {args.db_path}")
    
    def _validate_search_arguments(self, args: Any) -> None:
        if not args.query:
            raise ValueError("Query is required for search mode")
        
        if not args.query.strip():
            raise ValueError("Query cannot be empty")
        
        if args.n_results < 1:
            raise ValueError("n_results must be positive")
        
        if args.n_results > 50:
            raise ValueError("n_results cannot exceed 50")
        
        if not Path(args.db_path).exists():
            raise FileNotFoundError(f"Database path does not exist: {args.db_path}")
    
    def _initialize_neural_vector(self, db_path: str) -> None:
        try:
            print("[INFO] Initializing Neural Vector with Qwen3 models...")
            self._neural_vector = NeuralVector(db_path)
            print("[SUCCESS] Neural Vector initialized successfully")
        except (VectorDatabaseError, EmbeddingEngineError, RerankerEngineError) as e:
            raise RuntimeError(f"Failed to initialize Neural Vector: {e}")
    
    def _execute_search(self, query: str, n_results: int) -> list[SearchResult]:
        if self._neural_vector is None:
            raise RuntimeError("Neural Vector not initialized")
        
        try:
            return self._neural_vector.retrieve_memory(query=query, n_results=n_results)
        except (VectorDatabaseError, EmbeddingEngineError, RerankerEngineError) as e:
            raise RuntimeError(f"Search execution failed: {e}")
    
    def _execute_store(self, content: str, tags: str | None, timestamp: str | None) -> StorageResult:
        if self._neural_vector is None:
            raise RuntimeError("Neural Vector not initialized")
        
        tag_list: list[str] = []
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        
        try:
            return self._neural_vector.store_memory(
                content=content,
                tags=tag_list,
                timestamp=timestamp
            )
        except (VectorDatabaseError, EmbeddingEngineError) as e:
            raise RuntimeError(f"Store execution failed: {e}")
    
    def _execute_batch_store(self, contents: list[str], tags_list: list[list[str]], timestamps: list[str] | str | None) -> list[StorageResult]:
        if self._neural_vector is None:
            raise RuntimeError("Neural Vector not initialized")
        
        try:
            return self._neural_vector.batch_store_memories(
                contents=contents,
                tags_list=tags_list,
                timestamps=timestamps
            )
        except (VectorDatabaseError, BatchValidationError, MemoryValidationError) as e:
            raise RuntimeError(f"Batch store execution failed: {e}")
    
    def _display_results(self, query: str, results: list[SearchResult], db_path: str, execution_time: float) -> None:
        print(self._formatter.format_header(query, len(results), db_path))
        
        for result in results:
            print(self._formatter.format_result(result))
        
        print(self._formatter.format_footer(execution_time))
    
    def _display_store_result(self, result: StorageResult, execution_time: float) -> None:
        print(self._formatter.format_store_header())
        print(self._formatter.format_store_result(result))
        print(self._formatter.format_store_footer(execution_time))
    
    def _display_batch_store_results(self, results: list[StorageResult], execution_time: float) -> None:
        print(self._formatter.format_store_header())
        print(self._formatter.format_batch_store_results(results))
        print(self._formatter.format_store_footer(execution_time))
    
    def _display_read_result(self, result: MemoryResult, execution_time: float) -> None:
        print(self._formatter.format_read_header())
        print(self._formatter.format_memory_result(result))
        print(self._formatter.format_store_footer(execution_time))
    
    def _display_batch_read_results(self, results: list[MemoryResult], execution_time: float) -> None:
        print(self._formatter.format_read_header())
        if len(results) <= 3:
            for result in results:
                print(self._formatter.format_memory_result(result))
        else:
            print(self._formatter.format_batch_read_results(results))
        print(self._formatter.format_store_footer(execution_time))
    
    def run(self) -> None:
        try:
            args: Any = self._arg_parser.parse_arguments()
            
            import time
            start_time: float = time.time()
            
            self._initialize_neural_vector(args.db_path)
            
            if self._is_read_mode(args):
                identifiers: list[str] = args.read
                
                if len(identifiers) == 1:
                    result: MemoryResult | None = self._neural_vector.read_memory(identifiers[0])
                    if result:
                        execution_time: float = time.time() - start_time
                        self._display_read_result(result, execution_time)
                    else:
                        print(f"[INFO] No memory found with identifier: {identifiers[0]}")
                else:
                    results: list[MemoryResult] = self._neural_vector.batch_read_memories(identifiers)
                    execution_time: float = time.time() - start_time
                    if results:
                        self._display_batch_read_results(results, execution_time)
                    else:
                        print(f"[INFO] No memories found for the provided identifiers")
            elif self._is_update_mode(args):
                identifiers: list[str] = args.update
                
                contents: list[str] | str | None = args.content if hasattr(args, 'content') else None
                timestamps: list[str] | str | None = self._get_timestamp_value(args)
                
                tags_sets: list[list[str]] | list[str] | None = None
                if args.tags:
                    if len(args.tags) == 1 and "," in args.tags[0]:
                        tags_sets = [tag.strip() for tag in args.tags[0].split(",") if tag.strip()]
                    else:
                        tags_list: list[list[str]] = []
                        for tag_string in args.tags:
                            if "," in tag_string:
                                tag_list: list[str] = [tag.strip() for tag in tag_string.split(",") if tag.strip()]
                            else:
                                tag_list: list[str] = [tag_string.strip()]
                            tags_list.append(tag_list)
                        tags_sets = tags_list
                
                results: list[MemoryResult] = self._neural_vector.batch_update_memories(
                    identifiers, contents, tags_sets, timestamps
                )
                execution_time: float = time.time() - start_time
                
                successful: int = sum(1 for r in results if r.success)
                print(f"[SUCCESS] Updated {successful}/{len(identifiers)} memories in {execution_time:.3f} seconds")
                for result in results:
                    if result.success:
                        identifier: str = result.short_id if result.short_id else result.memory_id[:8]
                        print(f"  ✓ [{identifier}] Updated successfully")
                    else:
                        print(f"  ✗ [{result.memory_id[:8]}] Update failed")
            elif self._is_delete_mode(args):
                identifiers: list[str] = args.delete
                
                delete_results: dict[str, bool] = self._neural_vector.batch_delete_memories(
                    identifiers, soft_delete=False
                )
                execution_time: float = time.time() - start_time
                
                successful: int = sum(1 for success in delete_results.values() if success)
                print(f"[SUCCESS] Deleted {successful}/{len(identifiers)} memories in {execution_time:.3f} seconds")
                for identifier, success in delete_results.items():
                    if success:
                        print(f"  ✓ [{identifier[:8]}...] Deleted")
                    else:
                        print(f"  ✗ [{identifier[:8]}...] Failed")
            elif self._is_store_mode(args):
                self._validate_store_arguments(args)
                
                # args.store is now a list of memories
                memories: list[str] = args.store
                
                if len(memories) == 1:
                    # Single memory storage
                    timestamp_values: list[str] | None = self._get_timestamp_value(args)
                    timestamp: str | None = timestamp_values[0] if timestamp_values else None
                    
                    # CRITICAL: Timestamp validation
                    if not timestamp:
                        self._logger.warning("No timestamp provided for single memory - using current time")
                        print("[WARNING] No timestamp provided! Using current time. "
                              "Consider using --when for proper memory tracking.")
                    
                    # Handle tags - could be a list or single string
                    tags_str: str | None = None
                    if args.tags:
                        tags_str = args.tags[0] if isinstance(args.tags, list) else args.tags
                    
                    if not tags_str:
                        print("[WARNING] No tags provided! Tags help with memory retrieval.")
                    
                    result: StorageResult = self._execute_store(
                        memories[0], tags_str, timestamp
                    )
                    execution_time: float = time.time() - start_time
                    self._display_store_result(result, execution_time)
                else:
                    # Batch memory storage
                    # CRITICAL: Timestamps are REQUIRED for batch storage
                    timestamps: list[str] | str | None = self._get_timestamp_value(args)
                    if not timestamps:
                        raise ValueError(
                            "[CRITICAL ERROR] Timestamps are REQUIRED for batch storage! "
                            "Use --when or --timestamp to provide dates/times. "
                            "Without timestamps, memories are USELESS! "
                            "Example: --when \"09:44 PM, 03/04/2025\" \"08:44 AM, 09/11/2025\""
                        )
                    
                    tags_sets: list[list[str]] = []
                    if args.tags:
                        # Validate tag count matches memory count
                        if len(args.tags) != len(memories):
                            raise ValueError(
                                f"[ERROR] {len(memories)} memories provided but {len(args.tags)} tag sets. "
                                f"Each memory MUST have corresponding tags!"
                            )
                        
                        # args.tags is now a list of tag strings
                        for tag_string in args.tags:
                            if "," in tag_string:
                                tag_list: list[str] = [tag.strip() for tag in tag_string.split(",") if tag.strip()]
                            else:
                                tag_list: list[str] = [tag_string.strip()]
                            tags_sets.append(tag_list)
                    else:
                        raise ValueError(
                            "[ERROR] Tags are REQUIRED for batch storage! "
                            "Each memory needs tags for retrieval."
                        )
                    
                    results: list[StorageResult] = self._execute_batch_store(
                        memories, tags_sets, timestamps
                    )
                    execution_time: float = time.time() - start_time
                    self._display_batch_store_results(results, execution_time)
            else:
                self._validate_search_arguments(args)
                results: list[SearchResult] = self._execute_search(
                    args.query, args.n_results
                )
                execution_time: float = time.time() - start_time
                self._display_results(args.query, results, args.db_path, execution_time)
            
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            self._logger.error(f"CLI error: {e}")
            print(self._formatter.format_error(e))
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n[INFO] Operation interrupted by user")
            sys.exit(1)
        except Exception as e:
            self._logger.error(f"Unexpected error: {e}")
            print(self._formatter.format_error(e))
            sys.exit(1)
        finally:
            os.chdir(self._original_cwd)


class NeuralVectorTester:
    def __init__(self) -> None:
        db_path: str = "/Users/rahulsawhney/.mcp_memory/chroma_db"
        self._vector_engine: NeuralVector = NeuralVector(db_path)
    
    def test_single_memory_storage(self) -> None:
        print("\n" + "="*60)
        print("TEST 1: Single Memory Storage")
        print("="*60)
        
        content: str = "| Memory | Test Single Storage | Date: 04/08/2025 | Time: 09:00 PM | Name: Lyra | Testing single memory storage functionality"
        tags: list[str] = ["test", "single", "neuralvector"]
        timestamp: str = "04/08/2025"
        
        try:
            result: StorageResult = self._vector_engine.store_memory(content, tags, timestamp)
            print(f"[SUCCESS] Stored memory ID: {result.memory_id}")
            print(f"Message: {result.message}")
        except Exception as e:
            print(f"[ERROR] Failed to store single memory: {e}")
    
    def test_batch_memory_storage_single_date(self) -> None:
        print("\n" + "="*60)
        print("TEST 2: Batch Storage with Single Date")
        print("="*60)
        
        contents: list[str] = [
            "| Memory | Batch Test 1 | Date: 04/08/2025 | Time: 09:00 PM | Name: Lyra | First batch memory",
            "| Memory | Batch Test 2 | Date: 04/08/2025 | Time: 09:00 PM | Name: Lyra | Second batch memory",
            "| Memory | Batch Test 3 | Date: 04/08/2025 | Time: 09:00 PM | Name: Lyra | Third batch memory"
        ]
        tags_list: list[list[str]] = [
            ["batch", "test1", "memory"],
            ["batch", "test2", "neural"],
            ["batch", "test3", "vector"]
        ]
        single_date: str = "04/08/2025"
        
        try:
            results: list[StorageResult] = self._vector_engine.batch_store_memories(
                contents, tags_list, single_date
            )
            print(f"[SUCCESS] Stored {len(results)} memories with single date")
            for idx, result in enumerate(results):
                print(f"  Memory {idx+1}: ID={result.memory_id[:8]}... - {result.message}")
        except Exception as e:
            print(f"[ERROR] Batch storage failed: {e}")
    
    def test_batch_memory_storage_multiple_dates(self) -> None:
        print("\n" + "="*60)
        print("TEST 3: Batch Storage with Individual Dates")
        print("="*60)
        
        contents: list[str] = [
            "| Memory | Past Event | Date: 01/08/2025 | Time: 10:00 AM | Name: Lyra | Memory from the past",
            "| Memory | Current Event | Date: 04/08/2025 | Time: 09:00 PM | Name: Lyra | Current memory",
        ]
        tags_list: list[list[str]] = [
            ["past", "history", "event"],
            ["current", "present", "now"]
        ]
        dates: list[str] = ["01/08/2025", "04/08/2025"]
        
        try:
            results: list[StorageResult] = self._vector_engine.batch_store_memories(
                contents, tags_list, dates
            )
            print(f"[SUCCESS] Stored {len(results)} memories with individual dates")
            for idx, result in enumerate(results):
                print(f"  Memory {idx+1}: ID={result.memory_id[:8]}... - {result.message}")
        except Exception as e:
            print(f"[ERROR] Batch storage with multiple dates failed: {e}")
    
    def test_validation_errors(self) -> None:
        print("\n" + "="*60)
        print("TEST 4: Error Handling Validation")
        print("="*60)
        
        print("\nTest 4.1: Mismatched memory and tag counts")
        contents: list[str] = ["Memory 1", "Memory 2", "Memory 3"]
        tags_list: list[list[str]] = [["tag1"], ["tag2"]]  # Only 2 tag sets for 3 memories
        
        try:
            self._vector_engine.batch_store_memories(contents, tags_list)
            print("[ERROR] Should have failed but didn't!")
        except BatchValidationError as e:
            print(f"[EXPECTED ERROR] {e}")
        
        print("\nTest 4.2: Invalid date count")
        contents: list[str] = ["Memory 1", "Memory 2"]
        tags_list: list[list[str]] = [["tag1"], ["tag2"]]
        dates: list[str] = ["01/08/2025", "02/08/2025", "03/08/2025"]  # 3 dates for 2 memories
        
        try:
            self._vector_engine.batch_store_memories(contents, tags_list, dates)
            print("[ERROR] Should have failed but didn't!")
        except BatchValidationError as e:
            print(f"[EXPECTED ERROR] {e}")
        
        print("\nTest 4.3: Empty memory content")
        try:
            self._vector_engine.store_memory("", ["tag"])
            print("[ERROR] Should have failed but didn't!")
        except MemoryValidationError as e:
            print(f"[EXPECTED ERROR] {e}")
    
    def test_retrieval_after_batch_storage(self) -> None:
        print("\n" + "="*60)
        print("TEST 5: Retrieval After Batch Storage")
        print("="*60)
        
        contents: list[str] = [
            "| Memory | NeuralGraph Testing | Date: 04/08/2025 | Time: 09:00 PM | Name: Lyra | Testing batch storage for NeuralGraph system",
            "| Memory | Qwen3 Models | Date: 04/08/2025 | Time: 09:00 PM | Name: Lyra | Using Qwen3-Embedding-8B and Qwen3-Reranker-8B models"
        ]
        tags_list: list[list[str]] = [
            ["neuralgraph", "testing", "batch"],
            ["qwen3", "models", "embedding"]
        ]
        
        try:
            print("Storing test memories...")
            results: list[StorageResult] = self._vector_engine.batch_store_memories(
                contents, tags_list, "04/08/2025"
            )
            print(f"Stored {len(results)} memories")
            
            print("\nSearching for 'NeuralGraph'...")
            search_results: list[SearchResult] = self._vector_engine.retrieve_memory(
                "NeuralGraph", n_results=2
            )
            print(f"Found {len(search_results)} results:")
            for result in search_results:
                print(f"  Rank {result.rank}: Score={result.rerank_score:.3f}")
                print(f"    Content preview: {result.content[:80]}...")
        except Exception as e:
            print(f"[ERROR] Retrieval test failed: {e}")
    
    def run_all_tests(self) -> None:
        print("\nNEURAL VECTOR BATCH STORAGE TEST SUITE")
        print("="*60)
        
        self.test_single_memory_storage()
        self.test_batch_memory_storage_single_date()
        self.test_batch_memory_storage_multiple_dates()
        self.test_validation_errors()
        self.test_retrieval_after_batch_storage()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)


if __name__ == "__main__":
    tester: NeuralVectorTester = NeuralVectorTester()
    tester.run_all_tests()