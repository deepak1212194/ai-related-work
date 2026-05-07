"""
tools.py — Weather Tool
=========================
ReAct Weather Agent — Tool Registry

Defines the tools available to the ReAct agent.  Each tool has:
  - A Python implementation
  - An OpenAI-compatible JSON schema for the LLM

Currently implements one tool: get_weather.
The architecture supports adding more tools by extending TOOL_REGISTRY.
"""

import random
from pydantic import BaseModel


class Weather(BaseModel):
    """Weather data for a single location."""
    location: str
    temperature: float
    humidity: float
    description: str


def get_weather(location: str) -> Weather:
    """
    Get current weather for a location.

    In production, this would call a real weather API (OpenWeatherMap,
    WeatherAPI, etc.).  For this demo, returns realistic mock data
    so the agent loop is always demonstrable without external deps.
    """
    city = location.split(",")[0].strip() if "," in location else location.strip()
    return Weather(
        location=city,
        temperature=round(random.uniform(-10, 45), 1),
        humidity=round(random.uniform(10, 100), 1),
        description=random.choice(
            ["sunny", "cloudy", "rainy", "snowy", "windy", "foggy",
             "partly cloudy", "thunderstorm", "clear sky", "overcast"]
        ),
    )


# ── OpenAI-compatible tool schema ────────────────────────────────────

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Get the current weather for a single city/location. "
            "Returns temperature (°C), humidity (%), and conditions. "
            "Call once per city. For multiple cities, call separately."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "A single city name, e.g. 'Hyderabad', 'London'.",
                }
            },
            "required": ["location"],
        },
    },
}


# ── Tool registry (extensible) ──────────────────────────────────────

TOOL_REGISTRY = {
    "get_weather": get_weather,
}
