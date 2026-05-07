# Skill: DevOps / Cloud Engineer Role

> Role-specific emphasis loaded by the agent when the user selects this role.
> Applies to: DevOps, SRE, Platform Engineer, Cloud Engineer, Infrastructure
> Engineer, Cloud Architect.

---

## Target Role

DevOps / Cloud Engineer (4+ years senior-grade)

Variants this role covers:
- DevOps Engineer · Senior DevOps · Lead DevOps
- Site Reliability Engineer (SRE) · Senior SRE
- Platform Engineer · Cloud Engineer · Infrastructure Engineer
- Cloud Architect · Solutions Architect (cloud-leaning)
- Production Engineer · Build / Release Engineer

---

## What to Emphasize

- **DORA metrics** (deploy frequency, lead time for changes, MTTR,
  change-failure rate) — when input supports them. These are the
  industry-standard outcomes recruiters look for.
- **Reliability metrics**: SLO adherence, error budget burn, uptime
  percentage, p99 latency, incident reduction percentage.
- **Infrastructure scale**: number of services / nodes / clusters /
  regions managed; size of orgs supported (engineers / teams).
- **Cost optimization**: $ saved per month, % savings on cloud bill,
  rightsizing wins, spot/reserved-instance migrations.
- **Automation impact**: hours of toil eliminated, manual deploys
  removed, time-to-onboard-new-service improvements.
- **Security and compliance**: SOC 2 / ISO 27001 / HIPAA / PCI-DSS
  audits passed, secrets management, IAM hygiene, network policies.
- **Tooling depth**: IaC (Terraform / Pulumi), GitOps (ArgoCD / Flux),
  CI/CD systems, observability stacks (Datadog / Prometheus / ELK).
- **Production incident leadership**: incidents led / postmortems
  authored / runbooks created / on-call cadence improved.

---

## Priority Keywords

Kubernetes, K8s, Helm, Docker, container, containerization,
Terraform, Pulumi, CloudFormation, Ansible, Chef, Puppet, IaC,
infrastructure as code, GitOps, ArgoCD, Flux, FluxCD,
AWS, Azure, GCP, multi-cloud, hybrid cloud,
EC2, Lambda, ECS, EKS, S3, RDS, Aurora, DynamoDB, CloudFront,
Azure VM, AKS, Functions, Cosmos DB, Blob Storage, Service Bus,
GCP Compute Engine, GKE, Cloud Functions, Cloud Run, BigQuery,
CI/CD, GitHub Actions, GitLab CI, Jenkins, CircleCI, Travis,
Spinnaker, Argo Workflows, Tekton,
observability, monitoring, alerting, on-call,
Datadog, Prometheus, Grafana, Loki, Tempo, Mimir,
ELK, Elastic Stack, Splunk, New Relic, Sentry, PagerDuty, Opsgenie,
SLO, SLA, SLI, error budget, MTTR, MTBF, RTO, RPO,
DORA, deploy frequency, lead time, change failure rate,
toil reduction, automation,
network, VPC, subnets, peering, transit gateway, service mesh,
Istio, Linkerd, Envoy, Consul, Nginx, HAProxy,
security, IAM, RBAC, OPA, secrets management, Vault, KMS,
SOC 2, ISO 27001, HIPAA, PCI-DSS, compliance, audit,
cost optimization, FinOps, rightsizing, reserved instances, spot

---

## Summary

Target shape (3-4 sentences, ~70-110 words):

1. **Sentence 1** — Senior framing + years + primary cloud(s) +
   primary stack (K8s / Terraform / GitOps).
2. **Sentence 2** — Marquee reliability or scale outcome (only if
   input has the number). E.g., "Cut deployment lead time from
   hours to minutes across 80+ services."
3. **Sentence 3** — Cross-cutting expertise (FinOps, security
   compliance, multi-region rollouts, incident management).
4. **Optional 4** — Notable artifact (open-source contribution to
   Terraform provider, Helm chart, K8s controller; conference talk).

Avoid:
- "DevOps engineer with experience in cloud technologies"
- "Strong knowledge of CI/CD pipelines"
- "Passionate about automation"
- Listing every cloud and tool ever touched

Prefer:
- Specific cloud + dominant stack ("AWS + Terraform + ArgoCD on EKS")
- Specific scale ("80+ services across 4 regions")
- Specific reliability outcome ("cut MTTR from 47 to 12 minutes")

---

## Experience Bullets

Per-bullet template:

```
**Initiative / System Name** — *italic em-dash scope phrase if input
supports one.* [Architect-tier verb] [WHAT was built / migrated].
[HOW with stack + IaC]; [reliability / cost / DORA outcome ONLY if
from input].
```

Concrete examples to mirror:

```
Multi-Region GitOps Platform — the deployment system every backend
service uses. Architected the rollout to ArgoCD-driven GitOps across
3 AWS regions; introduced Helm-based progressive delivery, automated
canary rollback on SLO breach, and a self-service onboarding flow
for new services. Cut deploy lead time from 4 hrs to 8 min across
80+ services.
```

```
Cost Optimization Program — the FinOps initiative across the data
platform. Owned rightsizing of 200+ EC2 instances, migration of
batch jobs to Spot, and S3 lifecycle policies for cold data.
Achieved $42K/month in savings with no SLA degradation.
```

What earns full marks:
- Architect-tier verb at sentence-1 start
- Stack named ("Terraform + ArgoCD + Helm + Datadog")
- Scale stated ("80+ services", "200+ EC2 instances")
- Outcome with concrete reliability / cost number (only if input has it)
- 3 lines max

What loses marks:
- "Managed cloud infrastructure"
- "Implemented CI/CD pipelines"
- "Worked with Kubernetes"
- "Improved deployment process" without numbers
- Listing every tool the team uses without saying who owned what

---

## Skills

Recommended skill bucket order (6-8 buckets):

1. **Cloud Platforms** — AWS (EC2, EKS, Lambda, S3, RDS, IAM),
   Azure (AKS, Functions, Cosmos DB), GCP (GKE, Cloud Run, BigQuery)
2. **Container & Orchestration** — Kubernetes, Helm, Docker,
   service mesh (Istio / Linkerd), KEDA
3. **Infrastructure as Code** — Terraform, Pulumi, CloudFormation,
   Ansible, Crossplane
4. **CI/CD & GitOps** — GitHub Actions, GitLab CI, Jenkins,
   ArgoCD, Flux, Spinnaker, Argo Workflows
5. **Observability** — Datadog, Prometheus, Grafana, Loki, Tempo,
   ELK / Splunk, OpenTelemetry, distributed tracing
6. **Reliability & Incident** — SLO/SLI design, error-budget
   policy, PagerDuty / Opsgenie, postmortem culture, runbook authoring
7. **Security & Compliance** — IAM, RBAC, OPA / Gatekeeper,
   Vault / KMS, SOC 2 / ISO 27001 / HIPAA experience
8. **Languages & Scripting** — Python, Go, Bash, YAML, HCL

Rules for the skills section:
- Lead with cloud platforms (recruiters filter by cloud first)
- Don't list every minor tool; list dominant ones
- "Languages" bucket lower since DevOps is less language-centric

---

## Education

- B.E. / B.Tech / B.S. in CS / EE / Networking — sufficient
- Cloud certifications matter at this level (mention in Achievements):
  AWS Solutions Architect / DevOps Pro / SRE; Azure Administrator /
  DevOps Engineer; CKA / CKAD / CKS for Kubernetes
- **No percentages / CGPA** at senior level

---

## Achievements

High-signal items:
- Active certifications (AWS Pro / Azure Expert / CKA / CKAD / CKS,
  HashiCorp Certified)
- Open-source contributions to Terraform providers, Helm charts,
  Kubernetes operators, Prometheus exporters
- Conference talks (KubeCon, AWS re:Invent, HashiConf, SREcon)
- Patents on infra / cost-optimization / scheduling algorithms
- Internal awards for incident response or cost wins (with numbers)

Low-signal items (drop at senior level):
- Online course completions
- "Best Project" from undergrad
- Sports / cultural wins

---

## Common Anti-Patterns for DevOps / Cloud Engineer

- Listing every cloud service the candidate ever clicked on —
  recruiters only believe the top 5 used in production
- "Implemented CI/CD" with no DORA improvement number
- Conflating SRE and DevOps without clear emphasis: SRE is
  reliability-first; DevOps is delivery-velocity-first; pick the lane
- Keyword-stuffing tools (15+ in one bucket) — looks like a study
  list, not production experience
- "Familiar with Kubernetes" — strip; if you ran it in production,
  describe what you ran and at what scale
- Inventing cost-savings numbers — only if the input has them
- Vague reliability claims ("improved reliability") without an
  uptime number, MTTR delta, or incident-reduction stat
- Putting Education above Experience for 4+ year senior
