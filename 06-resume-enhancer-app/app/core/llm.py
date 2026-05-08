"""
llm.py - tri-backend LLM client (Claude Code subscription / Anthropic API / HF).

A single `LLMClient` interface so every agent is backend-agnostic.

Backends supported:

  1. `claude_code` - uses your existing Claude Code login (no API key needed).
       Requires the `claude-agent-sdk` Python package and the `claude` CLI to
       be installed and authenticated. Zero-config when those are present.

  2. `anthropic` - direct Anthropic API. Requires ANTHROPIC_API_KEY. Uses
       prompt caching on the system prompt so the critic / role-reviewer
       can be invoked many times per job cheaply.

  3. `huggingface` - free fallback. Lower quality but no paid API needed.

`build_llm()` is the public factory; `detect_available_backends()` lists
which backends are usable in the current environment.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from typing import List, Optional, Protocol

from .config import settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Backend-agnostic LLM failure."""


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    elapsed_s: float = 0.0


class LLMClient(Protocol):
    name: str

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        cache_system: bool = True,
    ) -> str: ...

    def last_usage(self) -> LLMUsage: ...


# ----------------------------------------------------------------------
# Claude Code SDK backend (uses your local Claude Code login)
# ----------------------------------------------------------------------
class ClaudeCodeLLM:
    """
    Uses the `claude-agent-sdk` Python package, which spawns the local
    `claude` CLI under the hood. As long as you've already logged into
    Claude Code (via `claude /login`), no API key is needed.

    The SDK is async-only; we run a fresh event loop per call to keep the
    rest of the codebase synchronous and thread-safe.
    """

    name = "claude_code"

    def __init__(self, model: Optional[str] = None) -> None:
        try:
            from claude_agent_sdk import query, ClaudeAgentOptions  # noqa: F401
        except ImportError as e:
            raise LLMError(
                "`claude-agent-sdk` is not installed. Either:\n"
                "  - pip install claude-agent-sdk   (uses your Claude Code login)\n"
                "  - OR switch backend to 'anthropic' (with ANTHROPIC_API_KEY)\n"
                "  - OR switch backend to 'huggingface' (with HF_API_KEY)"
            ) from e
        if not shutil.which("claude"):
            raise LLMError(
                "Claude Code CLI (`claude`) is not on PATH. Install Claude Code "
                "from https://claude.com/code, run `claude /login`, then try again."
            )
        self.model = model or "claude-sonnet-4-5"
        self._usage = LLMUsage()

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        cache_system: bool = True,
    ) -> str:
        import asyncio
        from claude_agent_sdk import query, ClaudeAgentOptions

        started = time.perf_counter()
        opts = ClaudeAgentOptions(
            system_prompt=system,
            model=self.model,
            max_turns=4,
            permission_mode="bypassPermissions",
            allowed_tools=[],
        )

        async def _run() -> str:
            chunks: list[str] = []
            try:
                async for msg in query(prompt=user, options=opts):
                    # The SDK yields heterogeneous messages; we want
                    # AssistantMessage text blocks only.
                    if hasattr(msg, "content"):
                        for block in getattr(msg, "content", []) or []:
                            text = getattr(block, "text", None)
                            if isinstance(text, str):
                                chunks.append(text)
            except Exception as e:                          # noqa: BLE001
                raise LLMError(f"Claude Code SDK call failed: {e}") from e
            return "".join(chunks).strip()

        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    raise RuntimeError("running loop")
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            text = loop.run_until_complete(_run())
        except LLMError:
            raise
        except Exception as e:                              # noqa: BLE001
            raise LLMError(f"Claude Code SDK call failed: {e}") from e

        elapsed = time.perf_counter() - started
        self._usage = LLMUsage(
            input_tokens=len(system + user) // 4,
            output_tokens=len(text) // 4,
            elapsed_s=elapsed,
        )
        return text

    def last_usage(self) -> LLMUsage:
        return self._usage


# ----------------------------------------------------------------------
# Anthropic backend with prompt caching
# ----------------------------------------------------------------------
class AnthropicLLM:
    name = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise LLMError(
                "ANTHROPIC_API_KEY is not set. Either set the key, switch "
                "to the Claude Code backend (uses your Claude Code login), "
                "or switch to the Hugging Face backend (free)."
            )
        try:
            import anthropic                           # noqa: F401
        except ImportError as e:
            raise LLMError(
                f"`anthropic` package is not installed: {e}. "
                "Run `pip install anthropic`."
            ) from e
        from anthropic import Anthropic
        self._client = Anthropic(api_key=api_key, timeout=settings.llm_call_timeout_s)
        self.model = model
        self._usage = LLMUsage()

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        cache_system: bool = True,
    ) -> str:
        started = time.perf_counter()
        sys_block: List[dict] = [{"type": "text", "text": system}]
        if cache_system and len(system) > 1024:
            sys_block[0]["cache_control"] = {"type": "ephemeral"}
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_block,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as e:                          # noqa: BLE001
            raise LLMError(f"Anthropic call failed: {e}") from e
        elapsed = time.perf_counter() - started
        usage = getattr(resp, "usage", None)
        self._usage = LLMUsage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            elapsed_s=elapsed,
        )
        chunks: List[str] = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                chunks.append(block.text)
        return "".join(chunks).strip()

    def last_usage(self) -> LLMUsage:
        return self._usage


# ----------------------------------------------------------------------
# Hugging Face Inference backend
# ----------------------------------------------------------------------
class HuggingFaceLLM:
    name = "huggingface"

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise LLMError(
                "HF_API_KEY is not set. Either set the key (free at "
                "https://huggingface.co/settings/tokens), switch to the "
                "Claude Code backend (uses your Claude Code login), or set "
                "ANTHROPIC_API_KEY for the Anthropic backend."
            )
        try:
            import requests                             # noqa: F401
        except ImportError as e:
            raise LLMError(f"`requests` is not installed: {e}") from e
        self._token = api_key
        self.model = model
        self._endpoint = f"https://api-inference.huggingface.co/models/{model}"
        self._usage = LLMUsage()

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        cache_system: bool = True,
    ) -> str:
        import requests
        started = time.perf_counter()
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": max(0.01, temperature),
                "return_full_text": False,
            },
            "options": {"wait_for_model": True},
        }
        try:
            r = requests.post(
                self._endpoint,
                headers={"Authorization": f"Bearer {self._token}"},
                json=payload,
                timeout=settings.llm_call_timeout_s,
            )
        except Exception as e:                          # noqa: BLE001
            raise LLMError(f"Hugging Face request failed: {e}") from e
        elapsed = time.perf_counter() - started
        if r.status_code >= 400:
            raise LLMError(
                f"Hugging Face returned {r.status_code}: {r.text[:200]}"
            )
        try:
            data = r.json()
        except Exception as e:                          # noqa: BLE001
            raise LLMError(f"Hugging Face returned non-JSON: {e}") from e
        text = ""
        if isinstance(data, list) and data:
            text = data[0].get("generated_text", "") or ""
        elif isinstance(data, dict):
            text = data.get("generated_text", "") or ""
        self._usage = LLMUsage(
            input_tokens=len(prompt) // 4,
            output_tokens=len(text) // 4,
            elapsed_s=elapsed,
        )
        return text.strip()

    def last_usage(self) -> LLMUsage:
        return self._usage


# ----------------------------------------------------------------------
# Detection + factory
# ----------------------------------------------------------------------
def _has_claude_code_sdk() -> bool:
    try:
        import claude_agent_sdk                          # noqa: F401
    except ImportError:
        return False
    return shutil.which("claude") is not None


def is_backend_configured(backend: str) -> bool:
    backend = backend.strip().lower()
    if backend in ("claude_code", "claudecode", "code"):
        return _has_claude_code_sdk()
    if backend in ("anthropic", "claude"):
        return bool(settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"))
    if backend in ("huggingface", "hf"):
        return bool(settings.hf_api_key or os.environ.get("HF_API_KEY"))
    return False


def detect_available_backends() -> dict[str, dict]:
    """Return a status dict for the UI's setup helper."""
    return {
        "claude_code": {
            "name": "Claude Code (your existing login)",
            "ready": is_backend_configured("claude_code"),
            "needs": "pip install claude-agent-sdk + Claude Code CLI logged in",
            "cost": "Included in your Claude subscription",
        },
        "anthropic": {
            "name": "Anthropic API (highest quality)",
            "ready": is_backend_configured("anthropic"),
            "needs": "ANTHROPIC_API_KEY from console.anthropic.com",
            "cost": "Pay-per-token (Claude Sonnet ~$3/MTok in)",
        },
        "huggingface": {
            "name": "Hugging Face (free)",
            "ready": is_backend_configured("huggingface"),
            "needs": "HF_API_KEY from huggingface.co/settings/tokens",
            "cost": "Free tier (rate-limited, slower)",
        },
    }


def best_available_backend() -> Optional[str]:
    """Pick the highest-quality backend that's actually usable right now."""
    for backend in ("claude_code", "anthropic", "huggingface"):
        if is_backend_configured(backend):
            return backend
    return None


def build_llm(backend: Optional[str] = None) -> LLMClient:
    """Create an LLM client for the requested backend.

    `backend=None` selects the best available backend automatically.
    """
    if backend in (None, "auto", ""):
        chosen = best_available_backend()
        if chosen is None:
            raise LLMError(
                "No LLM backend is configured. See the Setup tab in the UI "
                "or pick one of: Claude Code (zero-config if you're logged "
                "into Claude Code), Anthropic API (key from console.anthropic.com), "
                "or Hugging Face (free token from huggingface.co/settings/tokens)."
            )
        backend = chosen
    backend = backend.strip().lower()
    if backend in ("claude_code", "claudecode", "code"):
        return ClaudeCodeLLM(model=settings.anthropic_model or "claude-sonnet-4-5")
    if backend in ("anthropic", "claude"):
        return AnthropicLLM(
            api_key=settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            model=settings.anthropic_model,
        )
    if backend in ("huggingface", "hf"):
        return HuggingFaceLLM(
            api_key=settings.hf_api_key or os.environ.get("HF_API_KEY", ""),
            model=settings.hf_model,
        )
    raise LLMError(
        f"Unknown backend '{backend}'. Use 'claude_code', 'anthropic', or 'huggingface'."
    )
