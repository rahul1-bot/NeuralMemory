from __future__ import annotations

from neuralmemory.database.vector_db import NeuralVector
from neuralmemory.core.models import StorageResult, SearchResult

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

