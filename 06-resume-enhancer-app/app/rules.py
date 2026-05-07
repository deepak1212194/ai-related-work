"""
rules.py — Embedded enhancement rules + 5 role profiles
=========================================================

Two layers of rules:

1. SYSTEM_RULES                — universal "enhancement only" rules,
                                  applied to every section call.
2. ROLE_PROFILES[role].emphasis — role-specific guidance appended to
                                  the system prompt at request time
                                  so the LLM lifts the resume toward
                                  the target audience.
"""

from dataclasses import dataclass, field

# ──────────────────────────────────────────────────────────────────────
# Universal system prompt — the non-negotiables
# ──────────────────────────────────────────────────────────────────────
SYSTEM_RULES = """\
You are a strict, senior-grade resume editor. Your job is to ENHANCE the
candidate's existing resume content per the rules below. You must NEVER
weaken, remove, or fabricate.

ENHANCEMENT-ONLY RULES (non-negotiable):

1.  Never remove a specific fact, number, model name, library, or tech
    keyword that is present in the input.
2.  Never weaken a claim. Only strengthen.
3.  Never invent metrics, numbers, scale claims, or achievements that
    are not already in the input.
4.  Replace passive openers and weak verbs with senior-tier action
    verbs: Architected, Engineered, Owned, Designed, Co-invented,
    Productionized, Shipped, Migrated, Built. Avoid filler like
    "Built and shipped" (redundant), "Worked on", "Helped with",
    "Responsible for".
5.  Lead with WHAT in plain English (one sentence), then HOW with
    tech keywords (one sentence).
6.  Preserve italic em-dash scope phrases when supported by the input.
    Do NOT invent a scope phrase if the input does not support one.
7.  Maintain ATS keyword density.
8.  Tighten bloat without losing information.
9.  Do NOT change company / product / employer / dates / institutions.
10. Output ONLY the rewritten section text — no preamble, no
    explanation, no markdown fences.

If you are unsure how to enhance a section, return the input unchanged.
NEVER OUTPUT a worse version than the input.
"""


# ──────────────────────────────────────────────────────────────────────
# Role profiles — the top 5 LinkedIn role categories
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RoleProfile:
    id: str
    name: str
    description: str
    emphasis: str
    keywords: list[str] = field(default_factory=list)


ROLE_PROFILES: dict[str, RoleProfile] = {
    "ai_ml_engineer": RoleProfile(
        id="ai_ml_engineer",
        name="AI / ML Engineer",
        description="LLMs, RAG, multi-agent, fine-tuning, computer vision, recommendation",
        emphasis=(
            "TARGET ROLE: AI / ML Engineer (4+ years).\n"
            "Emphasize end-to-end ML lifecycle ownership, model evaluation "
            "metrics (R^2, MAE, held-out testing), production deployment "
            "(autoscaling, managed endpoints), and modern 2025-2026 "
            "stack: LLMs, RAG, multi-agent systems (CrewAI, LangChain), "
            "fine-tuning, vector search (FAISS, pgvector), agentic AI."
        ),
        keywords=[
            "LLM", "RAG", "fine-tuning", "multi-agent", "CrewAI", "LangChain",
            "vector search", "FAISS", "PyTorch", "Hugging Face",
            "model evaluation", "production ML", "MLOps",
        ],
    ),
    "software_engineer": RoleProfile(
        id="software_engineer",
        name="Software Engineer",
        description="Backend, frontend, full-stack, distributed systems",
        emphasis=(
            "TARGET ROLE: Software Engineer (4+ years).\n"
            "Emphasize systems design, scale (requests/sec, p99 latency), "
            "languages mastered, architectural decisions, performance "
            "improvements, code-quality practices (testing, CI/CD), "
            "and modern stack: distributed systems, microservices, REST/"
            "GraphQL APIs, observability, container orchestration."
        ),
        keywords=[
            "distributed systems", "microservices", "REST", "GraphQL",
            "Kubernetes", "Docker", "PostgreSQL", "Redis", "Kafka",
            "scalability", "performance", "p99 latency", "CI/CD",
        ],
    ),
    "data_scientist": RoleProfile(
        id="data_scientist",
        name="Data Scientist",
        description="Analytics, statistics, ML modeling, business impact",
        emphasis=(
            "TARGET ROLE: Data Scientist (4+ years).\n"
            "Emphasize business impact (revenue, conversion lift, churn "
            "reduction, cost savings), hypothesis-driven approach, "
            "statistical rigor (A/B testing, causal inference), "
            "modeling techniques, and modern stack: SQL, Python, "
            "pandas, scikit-learn, BI tools (Tableau, Looker, Power BI)."
        ),
        keywords=[
            "SQL", "Python", "pandas", "scikit-learn", "A/B testing",
            "causal inference", "statistical modeling", "Tableau",
            "Looker", "BigQuery", "Snowflake", "experimentation",
        ],
    ),
    "product_manager": RoleProfile(
        id="product_manager",
        name="Product Manager",
        description="Roadmaps, launches, stakeholder management, metrics",
        emphasis=(
            "TARGET ROLE: Product Manager (4+ years).\n"
            "Emphasize launches with quantified outcomes (DAU growth, "
            "retention lift, NPS, revenue), cross-functional leadership "
            "(eng / design / data), product decisions backed by data, "
            "user research, OKR ownership, and discovery-to-delivery "
            "lifecycle."
        ),
        keywords=[
            "roadmap", "OKR", "PRD", "user research", "stakeholders",
            "cross-functional", "GTM", "discovery", "product strategy",
            "metrics", "KPIs", "experimentation", "user retention",
        ],
    ),
    "devops_cloud_engineer": RoleProfile(
        id="devops_cloud_engineer",
        name="DevOps / Cloud Engineer",
        description="CI/CD, infrastructure, observability, automation",
        emphasis=(
            "TARGET ROLE: DevOps / Cloud Engineer (4+ years).\n"
            "Emphasize automation impact (deploy frequency, lead time, "
            "MTTR, change-failure rate), reliability metrics (uptime, "
            "SLO adherence), infrastructure scale (nodes, regions, $/"
            "month savings), and modern stack: Kubernetes, Terraform, "
            "AWS / Azure / GCP, IaC, GitOps, observability platforms."
        ),
        keywords=[
            "Kubernetes", "Terraform", "AWS", "Azure", "GCP",
            "CI/CD", "GitOps", "ArgoCD", "Helm", "observability",
            "Prometheus", "Grafana", "SLO", "MTTR", "uptime",
        ],
    ),
}


def list_roles() -> list[dict]:
    """JSON-friendly listing for the UI to render the role picker."""
    return [
        {"id": p.id, "name": p.name, "description": p.description}
        for p in ROLE_PROFILES.values()
    ]


def compose_system_prompt(role_id: str) -> str:
    """Universal rules + role-specific emphasis."""
    profile = ROLE_PROFILES.get(role_id) or ROLE_PROFILES["ai_ml_engineer"]
    return f"{SYSTEM_RULES}\n\n{profile.emphasis}\n"


# ──────────────────────────────────────────────────────────────────────
# Per-section task instructions
# ──────────────────────────────────────────────────────────────────────
SUMMARY_TASK = """\
Task: rewrite the Professional Summary below.

Constraints:
- 3-4 sentences, total 60-110 words.
- First sentence: avoid template openers. Lead with action or
  differentiator.
- Mention years-of-experience, top specialty areas, and the strongest
  1-2 signals (named conferences, patents, employers).
- Preserve any open-source portfolio URL or live link from the input.

Input summary:
\"\"\"
{content}
\"\"\"

Output the enhanced summary as plain text, no quotes, no preamble.
"""

BULLET_TASK = """\
Task: rewrite the experience bullet below.

Constraints:
- Lead with a senior past-tense action verb.
- One sentence WHAT, one sentence HOW.
- Preserve every tech keyword and number from the input.
- Maximum 3 lines when rendered at 10.5pt LaTeX.
- If the bullet contains team-effort language, preserve "the team's
  ... achieved ..." attribution. Do NOT promote team work to solo work.

Input bullet:
\"\"\"
{content}
\"\"\"

Output the enhanced bullet as plain text, no quotes, no preamble.
"""

SKILLS_TASK = """\
Task: lightly polish the skills line below.

Constraints:
- Do NOT remove any technology, library, or framework.
- You may reorder for better grouping.
- You may add at most ONE high-impact 2025-2026 keyword if STRONGLY
  IMPLIED by the input (no fabrication).
- If you cannot improve, return the input unchanged.

Input skills line:
\"\"\"
{content}
\"\"\"

Output the enhanced skills line as plain text, no quotes, no preamble.
"""

ACHIEVEMENT_TASK = """\
Task: lightly polish the achievement line below.

Constraints:
- Tighten phrasing only.
- Do NOT change credential name, year, or issuing body.
- If you cannot improve, return the input unchanged.

Input achievement line:
\"\"\"
{content}
\"\"\"

Output the enhanced achievement line as plain text, no quotes,
no preamble.
"""
