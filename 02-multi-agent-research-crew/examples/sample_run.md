# Sample run — offline stub mode

```
$ python -m src.main --topic "What are the key trade-offs of MoE LLMs vs dense models?"

[CREW] → planner (Research Planner)
[CREW] → researcher (Researcher)
[CREW] → critic (Critic)
[CREW] → writer (Writer)

================================================================
FINAL BRIEF
================================================================
## Summary
Offline-mode brief based on stub research.

- MoE LLMs reduce per-token cost but raise memory cost.
- RAG reduces hallucination at the price of retrieval latency.
- Multi-agent crews scale composition better than single-agent loops.

## Caveats
Generated in offline-stub mode; all evidence is canned.
```

Setting `OPENAI_API_KEY` switches every agent to the live LLM and the brief
becomes specific to the topic. The orchestration trace below the brief is
unchanged in shape — useful for debugging prompt regressions.
