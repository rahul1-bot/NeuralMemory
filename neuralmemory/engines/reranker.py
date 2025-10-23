from __future__ import annotations
import os
from torch import Tensor
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from neuralmemory.core.config import RerankerConfig
from neuralmemory.core.exceptions import RerankerEngineError


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
