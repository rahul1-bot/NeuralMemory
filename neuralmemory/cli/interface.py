from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any

from neuralmemory.core.exceptions import (
    VectorDatabaseError,
    EmbeddingEngineError,
    RerankerEngineError,
    MemoryValidationError,
    BatchValidationError
)
from neuralmemory.core.models import SearchResult, StorageResult, MemoryResult
from neuralmemory.core.logging_setup import LoggerSetup
from neuralmemory.database.vector_db import NeuralVector
from neuralmemory.cli.parser import MemoryArgumentParser
from neuralmemory.cli.formatter import MemoryFormatter
from neuralmemory.cli.processor import MemoryTextProcessor

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

