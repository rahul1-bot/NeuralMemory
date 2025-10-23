from __future__ import annotations
from dataclasses import dataclass
import torch


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
