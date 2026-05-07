"""
agent.py — ReAct Tool-Calling Agent
=====================================
ReAct Weather Agent — Core Module

Implements a pure ReAct (Reasoning + Acting) loop:
  1. Send user query + tool definitions to LLM
  2. LLM reasons about what to do next
  3. If LLM calls a tool → execute it, feed result back
  4. If LLM returns text → that's the final answer
  5. Repeat until done or MAX_ITERATIONS reached

Supports two backends:
  - OpenAI-compatible API (GPT models via OpenAI/OpenRouter)
  - Local LLM via LM Studio (any model with tool-call support)

The agent is domain-gated: only answers weather-related queries.
"""

import json
import os
import sys
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, field

from .tools import get_weather, TOOL_SCHEMA


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "6"))

# LLM backend configuration
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

WEATHER_KEYWORDS = [
    "weather", "temperature", "forecast", "humidity", "rain",
    "snow", "sunny", "cloudy", "wind", "storm", "climate",
    "hot", "cold", "warm", "cool", "fog", "hail", "heat",
]


# ──────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────
@dataclass
class ReActStep:
    """A single step in the ReAct trace."""
    iteration: int
    step_type: str          # "thought" | "action" | "observation" | "answer"
    content: str
    tool_name: str = ""
    tool_args: Dict = field(default_factory=dict)
    tool_result: str = ""


@dataclass
class AgentResult:
    """Full result of an agent execution."""
    query: str
    answer: str
    status: str             # "success" | "rejected" | "max_iterations" | "error"
    steps: List[ReActStep] = field(default_factory=list)
    total_iterations: int = 0


# ──────────────────────────────────────────────────────────────────────
# Domain gate
# ──────────────────────────────────────────────────────────────────────
def is_weather_query(question: str) -> bool:
    """Check if the user question is related to weather."""
    q = question.lower()
    return any(kw in q for kw in WEATHER_KEYWORDS)


# ──────────────────────────────────────────────────────────────────────
# LLM client
# ──────────────────────────────────────────────────────────────────────
def _get_client():
    """Create an OpenAI client with configured base URL and key."""
    from openai import OpenAI

    if not LLM_API_KEY:
        raise ValueError(
            "No API key configured. Set OPENAI_API_KEY or OPENROUTER_API_KEY "
            "environment variable. Alternatively, use offline mode."
        )

    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


# ──────────────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a weather assistant. You answer ONLY weather-related questions.\n"
    "If the user asks something unrelated to weather, respond with:\n"
    "'I can only help with weather-related queries.'\n\n"
    "For weather queries:\n"
    "- Call the get_weather tool ONCE per city.\n"
    "- For multiple cities, call the tool separately for each.\n"
    "- Only provide the final answer after ALL cities are covered.\n"
    "- Format results clearly with temperature, humidity, and conditions.\n"
)


# ──────────────────────────────────────────────────────────────────────
# ReAct loop (streaming trace)
# ──────────────────────────────────────────────────────────────────────
def react_loop(request: str) -> Generator[ReActStep, None, AgentResult]:
    """
    Execute the ReAct agent loop, yielding steps as they occur.

    This is a generator that yields ReActStep objects for real-time
    streaming to the UI.  The final return value is the AgentResult.
    """
    # Gate: reject non-weather questions
    if not is_weather_query(request):
        step = ReActStep(
            iteration=0,
            step_type="answer",
            content="I can only help with weather-related queries "
                    "(temperature, forecasts, conditions, etc.).",
        )
        yield step
        return AgentResult(
            query=request,
            answer=step.content,
            status="rejected",
            steps=[step],
        )

    # Check for API key — use offline mode if not set
    if not LLM_API_KEY:
        return _offline_react(request)

    client = _get_client()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request},
    ]

    steps: List[ReActStep] = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        # Yield a "thinking" step
        thought = ReActStep(
            iteration=iteration,
            step_type="thought",
            content=f"Iteration {iteration}/{MAX_ITERATIONS} — reasoning...",
        )
        yield thought
        steps.append(thought)

        # Call LLM with tools
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=[TOOL_SCHEMA],
                tool_choice="auto",
            )
        except Exception as e:
            error_step = ReActStep(
                iteration=iteration,
                step_type="answer",
                content=f"LLM call failed: {e}",
            )
            yield error_step
            steps.append(error_step)
            return AgentResult(
                query=request, answer=error_step.content,
                status="error", steps=steps, total_iterations=iteration,
            )

        assistant_msg = response.choices[0].message
        messages.append(assistant_msg)

        # Check for tool calls
        if assistant_msg.tool_calls:
            for tool_call in assistant_msg.tool_calls:
                fn_name = tool_call.function.name

                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                # Action step
                action = ReActStep(
                    iteration=iteration,
                    step_type="action",
                    content=f"Calling {fn_name}({json.dumps(args)})",
                    tool_name=fn_name,
                    tool_args=args,
                )
                yield action
                steps.append(action)

                # Execute tool
                if fn_name == "get_weather":
                    location = args.get("location", "Unknown")
                    result = get_weather(location)
                    tool_output = result.model_dump_json()
                else:
                    tool_output = json.dumps({"error": f"Unknown tool: {fn_name}"})

                # Observation step
                obs = ReActStep(
                    iteration=iteration,
                    step_type="observation",
                    content=tool_output,
                    tool_name=fn_name,
                    tool_result=tool_output,
                )
                yield obs
                steps.append(obs)

                # Feed result back to LLM
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output,
                })
        else:
            # No tool call → final answer
            answer_text = assistant_msg.content or ""
            answer_step = ReActStep(
                iteration=iteration,
                step_type="answer",
                content=answer_text,
            )
            yield answer_step
            steps.append(answer_step)
            return AgentResult(
                query=request, answer=answer_text,
                status="success", steps=steps, total_iterations=iteration,
            )

    # Max iterations reached
    fallback = "Could not complete the request within the iteration limit."
    final = ReActStep(iteration=MAX_ITERATIONS, step_type="answer", content=fallback)
    yield final
    steps.append(final)
    return AgentResult(
        query=request, answer=fallback,
        status="max_iterations", steps=steps, total_iterations=MAX_ITERATIONS,
    )


# ──────────────────────────────────────────────────────────────────────
# Offline mode — works without any API key
# ──────────────────────────────────────────────────────────────────────
def _offline_react(request: str) -> Generator[ReActStep, None, AgentResult]:
    """
    Offline mode: executes the tool directly for detected city names
    without an LLM.  Useful for demos without API keys.
    """
    import re

    # Simple city extraction from the query
    # Remove common words and split
    cleaned = re.sub(r'\b(weather|in|the|what|is|for|and|of|how|today|current|tell|me|about)\b', '', request, flags=re.I)
    cities = [c.strip().strip(',').strip('?').strip() for c in re.split(r'[,\s]+', cleaned) if len(c.strip()) > 2]
    cities = [c for c in cities if c]

    if not cities:
        cities = ["Unknown"]

    steps = []
    results = []

    for i, city in enumerate(cities, 1):
        # Thought
        thought = ReActStep(iteration=i, step_type="thought", content=f"[Offline] Looking up weather for {city}")
        yield thought
        steps.append(thought)

        # Action
        action = ReActStep(iteration=i, step_type="action", content=f"get_weather({city})", tool_name="get_weather", tool_args={"location": city})
        yield action
        steps.append(action)

        # Execute
        result = get_weather(city)
        tool_output = result.model_dump_json()
        results.append(result)

        # Observation
        obs = ReActStep(iteration=i, step_type="observation", content=tool_output, tool_name="get_weather", tool_result=tool_output)
        yield obs
        steps.append(obs)

    # Final answer
    lines = []
    for r in results:
        lines.append(f"**{r.location}**: {r.temperature}°C, {r.humidity}% humidity, {r.description}")
    answer = "Here's the weather information:\n\n" + "\n".join(lines)

    final = ReActStep(iteration=len(cities) + 1, step_type="answer", content=answer)
    yield final
    steps.append(final)

    return AgentResult(
        query=request, answer=answer,
        status="success", steps=steps, total_iterations=len(cities),
    )


# ──────────────────────────────────────────────────────────────────────
# Synchronous wrapper (for CLI / non-streaming use)
# ──────────────────────────────────────────────────────────────────────
def run_agent(request: str) -> AgentResult:
    """Run the agent and collect all steps into a result."""
    steps = []
    result = None
    gen = react_loop(request)
    try:
        while True:
            step = next(gen)
            steps.append(step)
    except StopIteration as e:
        result = e.value

    if result is None:
        # Generator exhausted without return — use last step as answer
        answer = steps[-1].content if steps else "No answer generated."
        result = AgentResult(query=request, answer=answer, status="success", steps=steps)

    return result
