"""
LLM client wrapper with retry, rate limiting, and robust batch processing.
Only supports OpenAI and OpenAI-compatible APIs (e.g. vLLM, ollama, lm-studio, etc.).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Iterable

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from llm_eval.config import LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract LLM client."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: Iterable[ChatCompletionMessageParam], **kwargs) -> str:
        pass

    @abstractmethod
    def _raw_generate(self, prompt: Iterable[ChatCompletionMessageParam], temperature=None, **kwargs) -> str:
        pass

    def generate_batch(self, prompts: Iterable[Iterable[ChatCompletionMessageParam]], temperature=None, **kwargs) -> List[str]:
        prompts_list = list(prompts)
        results = ["" for _ in prompts_list]

        def _task(idx: int, prompt: Iterable[ChatCompletionMessageParam]):
            return idx, self._raw_generate(prompt, temperature, **kwargs)

        with ThreadPoolExecutor(max_workers=self.config.batch_size) as executor:
            future_to_idx = {
                executor.submit(_task, i, p): i
                for i, p in enumerate(prompts_list)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    _, result = future.result()
                    results[idx] = result
                except Exception as exc:
                    logger.error(f"Batch generate failed for index {idx}: {exc}")
                    results[idx] = ""
        return results


class OpenAIClient(LLMClient):
    """支持官方 OpenAI 以及所有 OpenAI-Compatible 服务（vLLM、ollama、lm-studio、text-generation-webui 等）"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)

        api_key = config.api_key or "sk-no-key-required"
        base_url = config.api_base.rstrip("/") if config.api_base else None

        self.client: OpenAI = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=3,
        )

    def generate(self, prompt: Iterable[ChatCompletionMessageParam], **kwargs) -> str:
        return self._raw_generate(prompt, **kwargs)

    def _raw_generate(self, prompt: Iterable[ChatCompletionMessageParam], temperature=None, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=prompt,
                temperature=temperature if temperature is not None else self.config.temperature,
                **kwargs,
            )
            content = response.choices[0].message.content
            return content or ""
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit hit: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Factory function."""
    if config.provider in (LLMProvider.OPENAI, LLMProvider.OPENAI_COMPATIBLE):
        return OpenAIClient(config)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")
