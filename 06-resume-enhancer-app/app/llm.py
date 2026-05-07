"""
llm.py — LLM backend abstraction
==================================

Two implementations behind the same `complete()` interface:

  HuggingFaceClient   — default; uses `huggingface_hub.InferenceClient`
                        with a hosted open-weights model. Free tier
                        works for low-volume.
  AnthropicClient     — swap-in if user has an Anthropic API key.
                        Uses Claude Opus 4.7 with adaptive thinking
                        and prompt caching on the rules system prompt
                        (the same rules text is reused across every
                        section call, so cache reads keep cost low).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from .config import settings

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────────────────────────────
class LLMClient(ABC):
    """Common surface: complete(system, user) -> str."""

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


# ──────────────────────────────────────────────────────────────────────
# Hugging Face — default backend
# ──────────────────────────────────────────────────────────────────────
class HuggingFaceClient(LLMClient):
    """
    Uses the HF Inference API (serverless). Requires HF_API_KEY for
    private models; many open models work without a key but with stricter
    quotas.
    """

    def __init__(self) -> None:
        from huggingface_hub import InferenceClient
        self._client = InferenceClient(
            model=settings.hf_model,
            token=settings.hf_api_key,
        )
        self._model = settings.hf_model

    @property
    def name(self) -> str:
        return f"huggingface:{self._model}"

    def complete(self, system: str, user: str) -> str:
        log.info("[LLM:HF] %s — calling chat_completion", self._model)
        out = self._client.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=600,
            temperature=0.2,    # low for editing tasks
        )
        text = out.choices[0].message.content
        return text.strip() if text else ""


# ──────────────────────────────────────────────────────────────────────
# Anthropic Claude — swap-in backend
# ──────────────────────────────────────────────────────────────────────
class AnthropicClient(LLMClient):
    """
    Uses Claude Opus 4.7 by default. The system prompt (rules text) is
    cached via cache_control: every subsequent section call within the
    5-minute TTL window pays the cheap cache-read price for the rules
    portion of the request.
    """

    def __init__(self) -> None:
        import anthropic
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.anthropic_model

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    def complete(self, system: str, user: str) -> str:
        log.info("[LLM:Claude] %s — calling messages.create", self._model)
        # Cache the system rules (reused across every section call)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=2000,
            thinking={"type": "adaptive"},
            output_config={"effort": "medium"},
            system=[
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
        )
        # Concatenate text blocks; ignore thinking blocks
        out = "".join(
            b.text for b in response.content if getattr(b, "type", None) == "text"
        )
        cache_hits = getattr(response.usage, "cache_read_input_tokens", 0)
        log.info(
            "[LLM:Claude] tokens in=%d cache_read=%d out=%d",
            response.usage.input_tokens,
            cache_hits,
            response.usage.output_tokens,
        )
        return out.strip()


# ──────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────
def get_llm() -> LLMClient:
    """Return the LLM client per the RESUME_LLM_BACKEND env var."""
    if settings.llm_backend == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError(
                "RESUME_LLM_BACKEND=anthropic but ANTHROPIC_API_KEY is not set. "
                "Set the key, or switch to RESUME_LLM_BACKEND=huggingface."
            )
        return AnthropicClient()
    return HuggingFaceClient()
