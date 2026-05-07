# 07 · ReAct Weather Agent

A **pure ReAct (Reasoning + Acting)** agent that answers weather queries through iterative tool calling. The agent reasons about what information it needs, calls tools to get that information, observes the results, and repeats until it can formulate a complete answer.

## What it demonstrates

- **ReAct loop architecture**: Thought → Action → Observation → repeat
- **LLM tool calling**: OpenAI-compatible function calling with structured schemas
- **Domain gating**: Rejects non-weather queries gracefully
- **Multi-city orchestration**: Handles "Compare weather in X, Y, Z" by calling tools iteratively
- **Streaming trace**: SSE endpoint shows reasoning steps in real-time
- **Offline mode**: Works without any API key using mock data

## Architecture

```
User Query → Domain Gate → ReAct Loop ─┬─→ LLM Reasoning
                                        │   ↓
                                        │   Tool Call Decision
                                        │   ↓
                                        ├─→ Tool Execution (get_weather)
                                        │   ↓
                                        │   Observation
                                        │   ↓
                                        └─→ Loop or Final Answer
```

## Quick start

```bash
cd 07-react-weather-agent
pip install -r requirements.txt

# Offline mode (no API key needed)
uvicorn app.main:app --reload --port 8007

# Online mode
export OPENAI_API_KEY=sk-...
uvicorn app.main:app --reload --port 8007

# Open http://localhost:8007
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Status + mode (online/offline) |
| `POST` | `/api/query` | Full agent execution with trace |
| `GET` | `/api/stream?question=...` | SSE stream of reasoning steps |

## How it maps to production

| This demo | Production pattern |
|-----------|-------------------|
| Weather tool calling | Multi-tool agent calling compliance APIs, database queries, document parsers |
| Domain gate | Input validation layer that routes queries to appropriate agent |
| ReAct loop | Iterative reasoning for complex multi-step tasks |
| SSE trace streaming | Real-time observability dashboard for agent debugging |

## Stack

FastAPI · OpenAI · SSE · Pydantic · Docker
