# Skill: Data Scientist Role

> Role-specific emphasis loaded by the agent when the user selects this role.
> Applies to: Data Scientist, Senior Data Scientist, Applied Scientist
> (analytics-leaning), Marketing Analytics, Growth DS, Product DS.

---

## Target Role

Data Scientist (4+ years senior-grade)

Variants this role covers:
- Data Scientist · Senior Data Scientist · Lead Data Scientist
- Applied Scientist (analytics / measurement)
- Product Data Scientist · Growth Data Scientist
- Marketing Analytics Manager · Quant Analyst
- Data Analyst (senior, model-building) · Decision Scientist

---

## What to Emphasize

- **Business impact metrics first, technique second**: revenue lift,
  conversion rate change, churn reduction, cost savings, time-to-ship,
  risk reduction. Senior reviewers grade on impact narrative.
- **Hypothesis-driven approach**: explicit problem framing, hypothesis
  registration, experiment design, conclusion + recommendation.
- **Statistical rigor**: power analysis, sample-size justification,
  confidence intervals, p-values where appropriate, causal inference
  techniques (DiD, IV, RDD, propensity scoring, matched cohorts).
- **A/B testing and experimentation**: experiment platforms used,
  number of experiments shipped, average uplift, holdout group design.
- **Modeling depth**: regression, classification, time series, survival,
  uplift, recommendation, clustering — name techniques and the
  business question each addressed.
- **End-to-end ownership**: SQL data extraction → feature engineering
  → modeling → deployment / handoff → measurement.
- **Stakeholder fluency**: cross-functional partnerships with PMs /
  Eng / Marketing; written deliverables (PRDs, post-mortems, decision
  memos) the candidate authored.

---

## Priority Keywords

SQL, Python, R, pandas, NumPy, scikit-learn, statsmodels, scipy,
PySpark, Spark, Dask, A/B testing, experimentation, causal inference,
difference-in-differences, propensity scoring, instrumental variables,
regression discontinuity, uplift modeling, churn prediction,
recommender systems, time series, forecasting, ARIMA, Prophet,
hypothesis testing, p-value, confidence interval, power analysis,
Bayesian inference, MCMC, PyMC, XGBoost, LightGBM, CatBoost,
Random Forest, logistic regression, linear regression, GLM, GAM,
SHAP, feature importance, model interpretability,
ETL, dbt, Airflow, Dagster, Snowflake, BigQuery, Redshift, Databricks,
Tableau, Looker, Power BI, Mode, Metabase, Plotly, Streamlit,
Jupyter, MLflow, model registry, A/B test platform, experimentation
framework, holdout group, uplift, lift, retention curve, cohort
analysis, funnel analysis, attribution modeling

---

## Summary

Target shape (3-4 sentences, ~70-110 words):

1. **Sentence 1** — Senior framing + years + dominant analytical
   surface (growth / fraud / pricing / risk / supply chain / etc.).
2. **Sentence 2** — Marquee impact: a number + the business it
   moved (only if input has it). E.g., "delivered an uplift model
   that lifted retention by 4.2 pts."
3. **Sentence 3** — Methodological strength (causal inference,
   experimentation framework ownership, modeling stack).
4. **Optional 4** — Notable artifact (Kaggle finalist, paper,
   public dashboard, MLflow model registry ownership).

Avoid:
- "Data-driven decision maker"
- "Passionate about data"
- "Strong analytical skills"
- "Translates business problems into data problems" (everyone says
  this — show, don't tell)

Prefer:
- "Owned the experimentation platform serving N teams"
- "Shipped N causal-inference studies that informed pricing decisions"
- "Built the retention model used by Growth weekly"

---

## Experience Bullets

Per-bullet template:

```
**Project / Model Name** — *italic em-dash scope phrase if input supports
one.* [Architect-tier verb] [WHAT business question]. [HOW with technique
+ stack]; [outcome with number ONLY if input had one].
```

Concrete examples to mirror:

```
Churn Prediction Model — the model the Retention team uses to prioritize
weekly outreach. Designed a gradient-boosted churn classifier on 6 months
of behavioral features; calibrated using Platt scaling and validated on a
12-week held-out window. Drove a 4.2-pt retention lift on the top-decile
cohort.
```

```
Pricing Causal Inference Study — measured the effect of a new pricing
tier on conversion. Used a difference-in-differences design across
matched market pairs; reported pre-registered effect size with 95%
confidence interval. Decision adopted in the Q3 pricing review.
```

What earns full marks:
- Action verb leads
- Business question is clear in sentence 1
- Technique is named (DiD, GBM, ARIMA, etc.)
- Outcome with number when input has one
- Decision / handoff / adoption mentioned

What loses marks:
- "Analyzed data" / "Performed analyses"
- "Used machine learning" (which model? which task?)
- "Improved metrics" without a number
- Tutorial-style technique listing without a business hook

---

## Skills

Recommended skill bucket order (6-8 buckets):

1. **Languages** — Python, R, SQL, Bash
2. **Stats & Modeling** — regression, classification, time series,
   causal inference (DiD, IV, RDD, matching), uplift modeling, A/B
   testing, hypothesis testing
3. **ML Libraries** — scikit-learn, statsmodels, XGBoost, LightGBM,
   PyMC / Bayesian, Prophet
4. **Data Stack** — Snowflake, BigQuery, Redshift, Databricks, dbt,
   Airflow, Spark / PySpark, Dask
5. **Visualization & BI** — Tableau, Looker, Power BI, Plotly,
   Streamlit, Mode, Metabase
6. **Experimentation** — internal A/B platforms, holdout design,
   power analysis, sample-size calculation
7. **MLOps / Productionization** — MLflow, model registry, Docker,
   batch scoring, CI/CD for models
8. **Communication & Tooling** — Jupyter, GitHub, Confluence /
   Notion, decision memos, PRDs

Rules for the skills section:
- Lead with stats and modeling depth, not with "Python"
- Specific causal-inference techniques score higher than "statistics"
- Don't list every BI tool ever opened; the top 3 max

---

## Education

- M.S. / Ph.D. in Statistics / Applied Math / Operations Research /
  Economics / CS strongly signal Data Scientist
- B.S. + 4-6 years experience also senior-eligible
- Specialization in causal inference, time series, or experimentation
  is worth a coursework line
- **No percentages / CGPA** for senior 4+ year resumes

---

## Achievements

High-signal items:
- Kaggle Master / Grandmaster, top N% in named competitions
- Publications in journals (Stat / OR / Marketing / etc.)
- Conference talks (Strata, ODSC, JSM, CHI)
- Patents on modeling / measurement techniques
- Cross-team awards for shipped impact

Low-signal items (drop):
- Coursera completions
- "Best Project" from undergrad
- Hackathon participations from undergrad
- "Excellent communication skills" (show in summary, don't list)

---

## Common Anti-Patterns for Data Scientist

- All-method, no-business: "Built XGBoost classifier on 1M rows"
  with no business hook. Add the business question.
- Inventing metrics: "drove $5M in revenue" — only if input has it.
- Hyping tools over technique: listing 12 BI tools but no causal
  technique. Causal techniques are scarcer and rank higher.
- Confusing Data Scientist with Data Engineer: emphasis on Airflow /
  Spark plumbing without modeling impact pushes the resume toward DE
  recruiting; if the candidate is DS, lead with modeling.
- Confusing Data Scientist with Analyst: lots of dashboards but no
  hypothesis-driven analyses or models — for senior DS, lead with
  the latter.
- Stating soft skills as bullets ("strong communicator", "team
  player") — recruiters discount these.
