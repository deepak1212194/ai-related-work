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

  3. `huggingface` - free fallback via HF Inference API (chat completions).

All backends include automatic retry with exponential backoff for transient
failures (rate limits, server errors, timeouts).

`build_llm()` is the public factory; `detect_available_backends()` lists
which backends are usable in the current environment.
"""

from __future__ import annotations

import logging
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from typing import List, Optional, Protocol

from .config import settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Backend-agnostic LLM failure."""


_SERVERLESS_FALLBACKS: List[str] = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "google/gemma-2-2b-it",
]

# Groq free-tier models ordered by TPM (highest first) — updated May 2026.
# llama-4-scout: 30,000 TPM | llama-3.3-70b / llama-3.1-8b: 6,000 TPM each.
# Decommissioned: gemma2-9b-it, gemma-7b-it, mixtral-8x7b-32768,
#                 llama3-8b-8192, llama3-70b-8192
_GROQ_FALLBACKS: List[str] = [
    "meta-llama/llama-4-scout-17b-16e-instruct",  # 30K TPM — best headroom
    "llama-3.3-70b-versatile",                    # 6K TPM — highest quality
    "llama-3.1-8b-instant",                       # 6K TPM — fastest
]


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
# Retry wrapper — shared by all backends
# ----------------------------------------------------------------------
def _retry_call(fn, *, max_retries: int = 0, base_delay: float = 1.0, label: str = "LLM"):
    """Call `fn()` with exponential backoff + jitter on transient failures.

    `fn` should raise `LLMError` on failure. Retries up to `max_retries`
    times (so total attempts = max_retries + 1).
    """
    if max_retries <= 0:
        max_retries = settings.llm_max_retries
    if base_delay <= 0:
        base_delay = settings.llm_retry_base_delay

    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except LLMError as e:
            last_err = e
            err_str = str(e).lower()
            # Don't retry on auth errors or clearly permanent failures
            is_permanent = any(k in err_str for k in (
                "api key", "authentication", "unauthorized", "forbidden",
                "not installed", "not on path", "unknown backend",
                "decommissioned",   # model removed — no point retrying, let caller rotate
            ))
            if is_permanent or attempt >= max_retries:
                raise
            # For rate-limit errors, honor the server's suggested retry-after time.
            # Groq and many other APIs embed "Please try again in X.Xs" in the message.
            if "429" in str(e):
                m = re.search(r"try again in (\d+(?:\.\d+)?)s", err_str)
                if m:
                    delay = float(m.group(1)) + 1.5   # exact wait + small buffer
                else:
                    delay = 12.0 + random.uniform(0, 2.0)  # safe fallback for 429
            else:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            log.warning(
                "[%s] attempt %d/%d failed (%s), retrying in %.1fs…",
                label, attempt + 1, max_retries + 1, e, delay,
            )
            time.sleep(delay)
    # Should not reach here, but just in case
    raise last_err or LLMError(f"{label} failed after {max_retries + 1} attempts")


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
        def _call() -> str:
            import asyncio
            from claude_agent_sdk import query, ClaudeAgentOptions

            started = time.perf_counter()
            opts = ClaudeAgentOptions(
                system_prompt=system,
                model=self.model,
                max_turns=1,        # single prompt→response, no tool use
                permission_mode="bypassPermissions",
                allowed_tools=[],
            )

            async def _run() -> str:
                chunks: list[str] = []
                try:
                    async for msg in query(prompt=user, options=opts):
                        if hasattr(msg, "content"):
                            for block in getattr(msg, "content", []) or []:
                                text = getattr(block, "text", None)
                                if isinstance(text, str):
                                    chunks.append(text)
                except Exception as e:                          # noqa: BLE001
                    raise LLMError(f"Claude Code SDK call failed: {e}") from e
                return "".join(chunks).strip()

            # Create a new event loop for this call to avoid conflicts
            # with Gradio's own async loop
            loop = asyncio.new_event_loop()
            try:
                text = loop.run_until_complete(_run())
            except LLMError:
                raise
            except Exception as e:                              # noqa: BLE001
                raise LLMError(f"Claude Code SDK call failed: {e}") from e
            finally:
                loop.close()

            elapsed = time.perf_counter() - started
            self._usage = LLMUsage(
                input_tokens=len(system + user) // 4,
                output_tokens=len(text) // 4,
                elapsed_s=elapsed,
            )
            return text

        return _retry_call(_call, label="ClaudeCode")

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
        def _call() -> str:
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

        return _retry_call(_call, label="Anthropic")

    def last_usage(self) -> LLMUsage:
        return self._usage


# ----------------------------------------------------------------------
# Hugging Face Inference backend (OpenAI-compatible chat completions)
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
        # Primary endpoint is HF router (OpenAI-compatible).
        self._router_endpoint = "https://router.huggingface.co/v1/chat/completions"
        # Additional fallbacks for broader model compatibility.
        self._chat_endpoint = "https://api-inference.huggingface.co/v1/chat/completions"
        self._model_chat_endpoint = f"https://api-inference.huggingface.co/models/{model}/v1/chat/completions"
        self._textgen_endpoint = f"https://api-inference.huggingface.co/models/{model}"
        self._usage = LLMUsage()
        self._active = model  # tracks which model actually works; updated on 404 fallback

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        cache_system: bool = True,
    ) -> str:
        def _call() -> str:
            import requests
            started = time.perf_counter()
            chat_payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": max(0.01, temperature),
                "stream": False,
            }
            textgen_prompt = (
                "<SYSTEM>\n" + system.strip() + "\n</SYSTEM>\n\n"
                "<USER>\n" + user.strip() + "\n</USER>\n\n"
                "<ASSISTANT>\n"
            )
            textgen_payload = {
                "inputs": textgen_prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": max(0.01, temperature),
                    "return_full_text": False,
                },
                "options": {
                    "wait_for_model": True,
                    "use_cache": True,
                },
            }
            endpoints = [
                (self._router_endpoint, chat_payload),
                (self._chat_endpoint, chat_payload),
                (self._model_chat_endpoint, chat_payload),
                (self._textgen_endpoint, textgen_payload),
            ]
            last_err = ""
            r = None
            for endpoint, payload in endpoints:
                try:
                    r = requests.post(
                        endpoint,
                        headers={
                            "Authorization": f"Bearer {self._token}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                        timeout=settings.llm_call_timeout_s,
                    )
                except Exception as e:                      # noqa: BLE001
                    last_err = f"{endpoint} request failed: {e}"
                    continue
                if r.status_code < 400:
                    break
                # Fast-fail only on permanent auth errors — 429 is transient,
                # try next endpoint then let outer retry handle backoff.
                if r.status_code in (401, 403):
                    raise LLMError(f"Hugging Face returned {r.status_code}: {r.text[:300]}")
                last_err = f"{endpoint} -> {r.status_code}: {r.text[:220]}"
            if r is None:
                raise LLMError(f"Hugging Face request failed: {last_err}")
            elapsed = time.perf_counter() - started
            if r.status_code >= 400:
                raise LLMError(
                    f"Hugging Face returned {r.status_code}: {r.text[:300]}"
                )
            try:
                data = r.json()
            except Exception as e:                          # noqa: BLE001
                raise LLMError(f"Hugging Face returned non-JSON: {e}") from e
            # OpenAI-compatible response format
            text = ""
            if isinstance(data, dict):
                choices = data.get("choices", [])
                if choices and isinstance(choices, list):
                    msg = choices[0].get("message", {})
                    text = msg.get("content", "") or ""
                # Fallback for legacy text-generation format
                elif "generated_text" in data:
                    text = data["generated_text"] or ""
                elif "error" in data:
                    raise LLMError(f"Hugging Face error: {str(data.get('error'))[:240]}")
                # Extract usage if available
                usage = data.get("usage", {})
                self._usage = LLMUsage(
                    input_tokens=usage.get("prompt_tokens", 0) or 0,
                    output_tokens=usage.get("completion_tokens", 0) or 0,
                    elapsed_s=elapsed,
                )
            elif isinstance(data, list) and data:
                text = data[0].get("generated_text", "") or ""
                self._usage = LLMUsage(
                    input_tokens=len(system + user) // 4,
                    output_tokens=len(text) // 4,
                    elapsed_s=elapsed,
                )
            else:
                log.warning("[HuggingFace] unexpected response format: %s", type(data).__name__)
                self._usage = LLMUsage(elapsed_s=elapsed)
            return text.strip()

        # Build candidate list: active model first, then fallbacks
        candidates = [self._active] + [m for m in _SERVERLESS_FALLBACKS if m != self._active]
        last_404_err: Optional[LLMError] = None
        for cand in candidates:
            # Update model references so _call() closure uses the candidate
            self.model = cand
            self._model_chat_endpoint = f"https://api-inference.huggingface.co/models/{cand}/v1/chat/completions"
            self._textgen_endpoint = f"https://api-inference.huggingface.co/models/{cand}"
            n_retries = settings.llm_max_retries if cand == self._active else 1
            try:
                out = _retry_call(_call, max_retries=n_retries, label=f"HF/{cand.split('/')[-1]}")
                if cand != self._active:
                    log.warning("[HuggingFace] auto-switched to %s (original %s unavailable)", cand, self._active)
                    self._active = cand
                return out
            except LLMError as exc:
                if "404" in str(exc):
                    last_404_err = exc
                    continue
                raise
        raise last_404_err or LLMError("All fallback models returned 404; check HF_API_KEY and model availability")

    def last_usage(self) -> LLMUsage:
        return self._usage


# ----------------------------------------------------------------------
# Groq backend (OpenAI-compatible; free tier 14 400 req/day)
# ----------------------------------------------------------------------
class GroqLLM:
    """Groq Inference API — free, fast, OpenAI-compatible.

    Get a free API key at https://console.groq.com/keys.
    Good models for this use-case: llama-3.1-8b-instant, gemma2-9b-it,
    llama-3.3-70b-versatile, mixtral-8x7b-32768.
    """

    name = "groq"
    _ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise LLMError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys then paste it into the UI."
            )
        try:
            import requests  # noqa: F401
        except ImportError as e:
            raise LLMError(f"`requests` is not installed: {e}") from e
        self._token = api_key
        self.model = model
        self._active = model   # updated when model rotation kicks in
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
        def _call() -> str:
            import requests
            started = time.perf_counter()
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "max_tokens": max_tokens,
                "temperature": max(0.01, temperature),
            }
            try:
                r = requests.post(
                    self._ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {self._token}",
                        "Content-Type":  "application/json",
                    },
                    json=payload,
                    timeout=settings.llm_call_timeout_s,
                )
            except Exception as e:                          # noqa: BLE001
                raise LLMError(f"Groq request failed: {e}") from e
            elapsed = time.perf_counter() - started
            if r.status_code in (401, 403):
                raise LLMError(
                    f"Groq returned {r.status_code}: invalid API key. "
                    "Check your key at https://console.groq.com/keys"
                )
            if r.status_code >= 400:
                raise LLMError(f"Groq returned {r.status_code}: {r.text[:400]}")
            try:
                data = r.json()
            except Exception as e:                          # noqa: BLE001
                raise LLMError(f"Groq non-JSON response: {e}") from e
            choices = data.get("choices", [])
            text = choices[0].get("message", {}).get("content", "") if choices else ""
            usage = data.get("usage", {})
            self._usage = LLMUsage(
                input_tokens=usage.get("prompt_tokens", 0) or 0,
                output_tokens=usage.get("completion_tokens", 0) or 0,
                elapsed_s=elapsed,
            )
            return (text or "").strip()

        # Try active model with full retries; rotate on persistent 429 or
        # decommissioned-model 400 errors (each model has its own TPM bucket).
        candidates = [self._active] + [m for m in _GROQ_FALLBACKS if m != self._active]
        last_err: Optional[LLMError] = None
        for cand in candidates:
            self.model = cand
            # Decommissioned models fail immediately (1 attempt), others use full retries.
            n_retries = settings.llm_max_retries if cand == self._active else 2
            try:
                out = _retry_call(_call, max_retries=n_retries, label=f"Groq/{cand}")
                if cand != self._active:
                    log.warning("[Groq] switched to %s (original %s unavailable)", cand, self._active)
                    self._active = cand
                return out
            except LLMError as exc:
                exc_str = str(exc)
                # Rotate on rate-limit or decommissioned model — both are recoverable
                # by trying a different model in the fallback list.
                if "429" in exc_str or "decommissioned" in exc_str.lower():
                    last_err = exc
                    log.warning("[Groq] rotating away from %s: %s", cand, exc_str[:120])
                    continue
                raise
        raise last_err or LLMError(
            "All Groq models unavailable (rate-limited or decommissioned). "
            "Try again in a minute or choose a different model."
        )

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
    if backend in ("groq",):
        return bool(settings.groq_api_key or os.environ.get("GROQ_API_KEY"))
    if backend in ("huggingface", "hf"):
        return bool(settings.hf_api_key or os.environ.get("HF_API_KEY"))
    return False


def detect_available_backends() -> dict[str, dict]:
    """Return a status dict for the UI's setup helper."""
    return {
        "groq": {
            "name": "Groq (recommended)",
            "ready": is_backend_configured("groq"),
            "needs": "GROQ_API_KEY — free at console.groq.com/keys",
            "cost": "Free tier: 14,400 req/day · very fast",
        },
        "huggingface": {
            "name": "Hugging Face",
            "ready": is_backend_configured("huggingface"),
            "needs": "HF_API_KEY from huggingface.co/settings/tokens",
            "cost": "Free but model availability varies",
        },
    }


def best_available_backend() -> Optional[str]:
    """Pick the best available backend. Groq preferred over HF (more reliable)."""
    if is_backend_configured("groq"):
        return "groq"
    if is_backend_configured("huggingface"):
        return "huggingface"
    return None


def build_llm(backend: Optional[str] = None, *, model_override: Optional[str] = None) -> LLMClient:
    """Create an LLM client for the requested backend.

    `backend=None` selects the best available backend automatically.
    Groq is preferred over HuggingFace when both keys are present.
    """
    if backend in (None, "auto", ""):
        chosen = best_available_backend()
        if chosen is None:
            raise LLMError(
                "No LLM backend is configured.\n\n"
                "Recommended (free): Groq — get a free API key at "
                "https://console.groq.com/keys and paste it into the UI.\n\n"
                "Alternative (free): HuggingFace — get a token at "
                "https://huggingface.co/settings/tokens"
            )
        backend = chosen
    backend = backend.strip().lower()
    if backend in ("claude_code", "claudecode", "code", "anthropic", "claude"):
        raise LLMError(
            f"Backend '{backend}' is disabled in this build. Use 'groq' or 'huggingface'."
        )
    if backend in ("groq",):
        model = model_override or os.environ.get("GROQ_MODEL") or settings.groq_model
        return GroqLLM(
            api_key=settings.groq_api_key or os.environ.get("GROQ_API_KEY", ""),
            model=model,
        )
    if backend in ("huggingface", "hf"):
        model = model_override or os.environ.get("HF_MODEL") or settings.hf_model
        return HuggingFaceLLM(
            api_key=settings.hf_api_key or os.environ.get("HF_API_KEY", ""),
            model=model,
        )
    raise LLMError(
        f"Unknown backend '{backend}'. Use 'groq' or 'huggingface'."
    )
