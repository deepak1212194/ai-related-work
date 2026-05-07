# Skill: Software Engineer Role

> Role-specific emphasis loaded by the agent when the user selects this role.
> Applies to: Backend Engineer, Frontend Engineer, Full-Stack, Systems
> Engineer, Platform Engineer, Distributed-Systems Engineer.

---

## Target Role

Software Engineer (4+ years senior-grade)

Variants this role covers:
- Backend Engineer · Frontend Engineer · Full-Stack Engineer
- Senior SWE · Staff SWE candidate (entry)
- Platform Engineer · Distributed-Systems Engineer
- API Engineer · Mobile Engineer (iOS/Android)
- Systems Engineer · Performance Engineer

---

## What to Emphasize

- **Systems thinking and design ownership**: emphasize architectural
  decisions, trade-off rationale, and the boundaries the candidate
  drew between services / modules / layers.
- **Scale and performance numbers** (ONLY if in the input): requests
  per second, p50 / p95 / p99 latency, throughput, concurrent users,
  database size, query times.
- **Reliability and incident reduction**: uptime improvements,
  on-call burden reduction, MTTR improvements, error-budget wins.
- **Concrete language and framework expertise**: name specific
  versions and stacks when input supports it (Go 1.22, TypeScript 5,
  React 19, Postgres 16) — vague "modern stack" is weak.
- **Code-quality practice**: testing strategies, coverage targets,
  linting / type-checking, code-review culture, refactoring wins.
- **Cross-team scope at senior level**: design docs reviewed,
  cross-team API contracts owned, downstream consumer count.
- **Production deployment and observability**: CI/CD ownership,
  feature flags, blue-green / canary, structured logging, tracing.

---

## Priority Keywords

Python, Go, Java, Kotlin, TypeScript, JavaScript, Rust, C++,
Node.js, Spring Boot, Django, FastAPI, Express, React, Next.js,
Vue, Angular, distributed systems, microservices, event-driven,
REST APIs, GraphQL, gRPC, OpenAPI, async, concurrency,
PostgreSQL, MySQL, MongoDB, Redis, Elasticsearch, Kafka,
RabbitMQ, message queue, pub/sub, idempotency, eventual consistency,
sharding, partitioning, replication, scalability, p99 latency,
throughput, requests per second, SLO, SLA, error budget,
Kubernetes, Docker, Helm, Terraform, AWS, Azure, GCP,
S3, EC2, Lambda, Cloud Functions, ECS, EKS, AKS, GKE,
Datadog, Prometheus, Grafana, OpenTelemetry, Jaeger, Sentry,
CI/CD, GitHub Actions, GitLab CI, Jenkins, blue-green, canary,
feature flags, A/B testing, observability, structured logging,
distributed tracing, monorepo, Bazel, Nx, Lerna, Vite, Webpack

---

## Summary

Target shape (3-4 sentences, ~70-110 words):

1. **Sentence 1** — Senior framing + years + dominant stack
   (backend / frontend / full-stack / mobile / systems).
2. **Sentence 2** — Marquee shipped system with scale or impact
   number (only if in input). E.g., "shipped the order-routing
   service handling 30K req/s during peak."
3. **Sentence 3** — Cross-cutting strength (reliability, perf,
   developer experience, mentorship).
4. **Optional 4** — Notable artifact (open-source project, talk,
   patent, conference paper).

Avoid:
- "Full-stack developer with experience in modern technologies"
- "Strong understanding of computer science fundamentals"
- "Excellent problem-solving skills"
- Listing every language ever touched

Prefer:
- Specific dominant stack ("Go + Postgres + Kafka")
- Specific scale ("services serving 50K req/s")
- Specific reliability outcome ("cut p99 from 1.2s to 240ms" — only
  if input has those numbers)

---

## Experience Bullets

Per-bullet template:

```
**System / Project Name** — *italic em-dash scope phrase if input
supports one.* [Architect-tier verb] [WHAT in plain English].
[HOW with stack keywords]; [scale / latency / reliability outcome
ONLY if from input].
```

Concrete examples to mirror:

```
Order Routing Service — the path every checkout takes through the
fulfillment pipeline. Architected the rewrite from a monolithic Java
endpoint to an event-driven Go service; introduced idempotency keys,
exactly-once semantics over Kafka, and circuit-breakers around four
downstream APIs. Cut p99 from 1.2s to 240ms.
```

```
Search Frontend Refactor — the discovery surface seen by all logged-in
users. Migrated from a monolithic React 16 SPA to a Next.js 14 app
router build with edge-rendered listings. Improved Lighthouse perf
from 62 to 91 and reduced TTFB by 380ms.
```

What earns full marks:
- Action verb at sentence-1 start
- Stack named at version when input gives the version
- A latency / throughput / reliability number when the input has one
- 3 lines max in 10.5pt LaTeX

What loses marks:
- "Worked on the order routing service"
- "Improved performance" (without a number)
- "Used various tools and frameworks"
- Listing every library imported

---

## Skills

Recommended skill bucket order (6-8 buckets):

1. **Languages** — Go, Python, TypeScript, Java, Kotlin, Rust, C++,
   SQL, Bash
2. **Backend & APIs** — FastAPI, Spring Boot, Express, gRPC, REST,
   GraphQL, OpenAPI, async/await, concurrency primitives
3. **Frontend** *(if applicable)* — React, Next.js, Vue, TypeScript,
   Tailwind, state management (Redux, Zustand, Pinia)
4. **Data & Storage** — PostgreSQL, MySQL, MongoDB, Redis,
   Elasticsearch, Kafka, RabbitMQ, S3
5. **Cloud & Infra** — AWS / Azure / GCP, Kubernetes, Docker,
   Terraform, Helm, CI/CD (GitHub Actions / GitLab CI)
6. **Observability** — Datadog, Prometheus, Grafana,
   OpenTelemetry, Jaeger, Sentry, structured logging
7. **Testing & Quality** — Pytest, Jest, Go testing, integration
   testing, contract testing, load testing (k6, Locust)
8. **Architecture & Patterns** — distributed systems, microservices,
   event-driven, idempotency, circuit breakers, bulkheads, CQRS

Rules for the skills section:
- 6-8 buckets max
- Frontend bucket only if the candidate is full-stack or frontend-leaning
- No "Modern technologies" or "Open source software" — be specific

---

## Education

- B.E. / B.Tech / B.S. in CS / SE / EE — sufficient for senior SWE
- M.S. / M.Tech only if relevant to the role
- Notable specialization (Distributed Systems, OS, Compilers): mention
  in coursework line
- **No percentages / CGPA** for 4+ year senior

---

## Achievements

High-signal items:
- Open-source project with stars / contributors (with numbers if input
  supports them)
- Conference talks (KubeCon, GopherCon, JSConf, Strange Loop, etc.)
- Patents
- Top-level competitive programming (ICPC finalist, Google KickStart top N,
  Codeforces grandmaster)
- Internal architecture awards (when verifiable)

Low-signal items (drop at senior level):
- Hackathon wins from undergrad
- Online course completions
- Class Representative
- Sports wins

---

## Common Anti-Patterns for Software Engineer

- Listing 20+ languages — credibility falls; pick the top 5-7
- Claiming "performance optimization" without a before/after number
- "Built REST API" with no scale or domain context
- Mixing ML keywords (LLM, RAG) into a backend SWE resume — only if
  the candidate genuinely shipped ML systems
- Putting Education above Experience for someone with 4+ years
- Including a CGPA / percentage
- "Familiar with" / "Exposure to" — strip these; use Skills only
  for things the candidate has actually shipped
