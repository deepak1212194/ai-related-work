"""
main.py — CLI Entry
====================
Multi-Agent Research Crew — Module 4

Usage:
    python -m src.main --topic "What are MoE LLMs?"
"""

import argparse

from .orchestrator import run_crew


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the research crew on a topic.")
    parser.add_argument("--topic", "-t", required=True)
    args = parser.parse_args()

    ctx = run_crew(args.topic)

    print("\n" + "=" * 64)
    print("FINAL BRIEF")
    print("=" * 64)
    print(ctx.final_brief)

    print("\n" + "-" * 64)
    print("ORCHESTRATION TRACE")
    print("-" * 64)
    print("\n".join(ctx.trace))


if __name__ == "__main__":
    main()
