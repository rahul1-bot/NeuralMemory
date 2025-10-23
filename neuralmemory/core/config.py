from __future__ import annotations
import torch
from pydantic import BaseModel, ConfigDict, field_validator


class EmbeddingConfig(BaseModel):
    model_path: str
    max_length: int
    instruction: str
    device: str
    model_config = ConfigDict(frozen=True)

    @field_validator('max_length')
    @classmethod
    def validate_max_length(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                f"Invalid max_length: expected positive integer (>= 1), got {v}. "
                f"Check embedding configuration and model requirements."
            )
        return v

    @field_validator('instruction')
    @classmethod
    def validate_instruction(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                f"Invalid instruction: expected non-empty string, got empty or whitespace-only string. "
                f"Provide a meaningful instruction for embedding queries."
            )
        return v

    @field_validator('device')
    @classmethod
    def validate_device(cls, v: str) -> str:
        valid_devices: set[str] = {"mps", "cpu", "cuda"}
        if v not in valid_devices:
            raise ValueError(
                f"Invalid device: expected one of {valid_devices}, got '{v}'. "
                f"Use 'mps' for Apple Silicon, 'cuda' for NVIDIA GPU, or 'cpu' for CPU-only."
            )
        return v

    def __str__(self) -> str:
        return f"EmbeddingConfig(device={self.device}, max_length={self.max_length})"

    def __repr__(self) -> str:
        return (
            f"EmbeddingConfig(model_path='{self.model_path}', max_length={self.max_length}, "
            f"device='{self.device}')"
        )

    @classmethod
    def create_qwen3_mps_config(cls) -> EmbeddingConfig:
        device: str = "mps" if torch.backends.mps.is_available() else "cpu"
        return cls(
            model_path="Qwen/Qwen3-Embedding-8B",
            max_length=8192,
            instruction="Given a web search query, retrieve relevant passages that answer the query",
            device=device
        )


class RerankerConfig(BaseModel):
    model_path: str
    max_length: int
    instruction: str
    device: str
    model_config = ConfigDict(frozen=True)

    @field_validator('max_length')
    @classmethod
    def validate_max_length(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                f"Invalid max_length: expected positive integer (>= 1), got {v}. "
                f"Check reranker configuration and model requirements."
            )
        return v

    @field_validator('instruction')
    @classmethod
    def validate_instruction(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                f"Invalid instruction: expected non-empty string, got empty or whitespace-only string. "
                f"Provide a meaningful instruction for reranking queries."
            )
        return v

    @field_validator('device')
    @classmethod
    def validate_device(cls, v: str) -> str:
        valid_devices: set[str] = {"mps", "cpu", "cuda"}
        if v not in valid_devices:
            raise ValueError(
                f"Invalid device: expected one of {valid_devices}, got '{v}'. "
                f"Use 'mps' for Apple Silicon, 'cuda' for NVIDIA GPU, or 'cpu' for CPU-only."
            )
        return v

    def __str__(self) -> str:
        return f"RerankerConfig(device={self.device}, max_length={self.max_length})"

    def __repr__(self) -> str:
        return (
            f"RerankerConfig(model_path='{self.model_path}', max_length={self.max_length}, "
            f"device='{self.device}')"
        )

    @classmethod
    def create_qwen3_mps_config(cls) -> RerankerConfig:
        device: str = "mps" if torch.backends.mps.is_available() else "cpu"
        return cls(
            model_path="/Users/rahulsawhney/LocalCode/Models/Qwen3-Reranker-8B",
            max_length=2048,
            instruction="Given a web search query, retrieve relevant passages that answer the query",
            device=device
        )
