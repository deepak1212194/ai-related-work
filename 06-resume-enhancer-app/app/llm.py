"""
llm.py — LLM backend abstraction
==================================

Two implementations behind the same `complete()` interface:

  HuggingFaceClient   — default; uses `huggingface_hub.InferenceClient`.
  AnthropicClient     — swap-in; uses Claude Opus 4.7 with adaptive
                        thinking and prompt caching on the rules
                        system prompt.

Both clients enforce a hard per-call timeout (settings.llm_call_timeout
_seconds). Any failure inside `complete()` raises an exception that the
caller catches; the caller then keeps the original input. This keeps
the pipeline deterministic and crash-proof.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from .config import settings

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Anything that goes wrong inside an LLM call. Caller always catches."""


# ──────────────────────────────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────────────────────────────
class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, user: str) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


# ──────────────────────────────────────────────────────────────────────
# Hugging Face — default backend
# ──────────────────────────────────────────────────────────────────────
class HuggingFaceClient(LLMClient):
    """Uses HF Inference API serverless. Fully timeout-bounded."""

    def __init__(self) -> None:
        from huggingface_hub import InferenceClient
        self._client = InferenceClient(
            model=settings.hf_model,
            token=settings.hf_api_key,
            timeout=settings.llm_call_timeout_seconds,
        )
        self._model = settings.hf_model

    @property
    def name(self) -> str:
        return f"huggingface:{self._model}"

    def complete(self, system: str, user: str) -> str:
        try:
            out = self._client.chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=600,
                temperature=0.2,
            )
        except Exception as e:                              # noqa: BLE001
            raise LLMError(f"HF call failed: {e!s}") from e
        text = out.choices[0].message.content if out.choices else ""
        return text.strip() if text else ""


# ──────────────────────────────────────────────────────────────────────
# Anthropic Claude — swap-in backend
# ──────────────────────────────────────────────────────────────────────
class AnthropicClient(LLMClient):
    """
    Uses Claude Opus 4.7 by default. The system prompt is cached
    (cache_control ephemeral) so every section call within the 5-minute
    TTL pays the cheap cache-read rate for the rules portion.
    """

    def __init__(self) -> None:
        import anthropic
        self._client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=float(settings.llm_call_timeout_seconds),
        )
        self._model = settings.anthropic_model

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    def complete(self, system: str, user: str) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=2000,
                thinking={"type": "adaptive"},
                output_config={"effort": "medium"},
                system=[{
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user}],
            )
        except Exception as e:                              # noqa: BLE001
            raise LLMError(f"Claude call failed: {e!s}") from e

        out = "".join(
            b.text for b in response.content if getattr(b, "type", None) == "text"
        )
        cache_hits = getattr(response.usage, "cache_read_input_tokens", 0)
        log.debug(
            "[LLM:Claude] in=%d cache_read=%d out=%d",
            response.usage.input_tokens, cache_hits, response.usage.output_tokens,
        )
        return out.strip()


# ──────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────
def get_llm() -> LLMClient:
    """Return the configured client. Raises LLMError if backend isn't usable."""
    try:
        if settings.llm_backend == "anthropic":
            if not settings.anthropic_api_key:
                raise LLMError(
                    "RESUME_LLM_BACKEND=anthropic but ANTHROPIC_API_KEY is not "
                    "set. Set the key, or switch to RESUME_LLM_BACKEND="
                    "huggingface."
                )
            return AnthropicClient()
        return HuggingFaceClient()
    except LLMError:
        raise
    except Exception as e:                                  # noqa: BLE001
        raise LLMError(f"Could not initialize LLM client: {e!s}") from e
