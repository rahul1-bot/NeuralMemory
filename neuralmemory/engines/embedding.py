from __future__ import annotations
import os
from torch import Tensor
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from neuralmemory.core.config import EmbeddingConfig
from neuralmemory.core.exceptions import EmbeddingEngineError


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
