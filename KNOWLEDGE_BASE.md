# 📘 MAS-DQA: Project Knowledge Base for Agentic LLM Implementation

This document is structured as a **machine-readable knowledge base** optimized for agentic LLM tools (Cursor, Devin, GitHub Copilot Workspace, Claude Code, etc.). It contains architecture specs, component interfaces, constraints, validation methodology, and implementation directives derived directly from your presentation slides and feedback.

---

## 🎯 1. PROJECT METADATA & VISION
| Field | Value |
|-------|-------|
| **Project Name** | MAS-DQA (Multi-Agent System for Data Quality Assurance) |
| **Author** | Aroh Sunday Melitus |
| **Core Philosophy** | `"Bad Data = No Data"` → `"Trusted Data = Confident Decisions"` |
| **Key Insight** | Quality ≠ Acceptance. Acceptance = Quality + Agreement + Verifiability. |
| **Target Domains** | Smart Mobility (BusPas), Healthcare IoT (MIMIC-III), Industrial IoT, Autonomous Systems |
| **Primary Innovation** | First framework combining autonomous multi-agent coordination, real-time streaming, LLM semantic reasoning, verifiable governance, and conflict adjudication. |

---

## 🏗️ 2. SYSTEM ARCHITECTURE & DATA FLOW
### 6-Step Pipeline (Slides 7, 8)
```
[Input] → ① Event Detection → ② Bootstrap → ③ Parallel Validation → ④ Trust / ⑤ Judge → ⑥ Output → [Downstream ML/Control]
```
| Step | Name | Layer | Core Function |
|------|------|-------|---------------|
| ① | Context-Aware Event Detection | Gateway | Classify producer (New/Known/Drift) & route |
| ② | Bootstrap | Monitoring (Cold-Start) | Learn baseline without ground truth (SFDA) |
| ③ | Parallel Validation | Monitoring | Profiler (stats) + Validator (semantic) cross-check |
| ④ | Trust Scoring | Intervention | Aggregate signals → SRI (0.0–1.0) + explanation |
| ⑤ | Judge Agent | Reasoning | Resolve conflicts using LLM + precedent database |
| ⑥ | Verifiable Outputs | Output Plane | Clean / Quarantine / Fallback / Verdict + XAI Log |

### Cross-Cutting Layers
- **Orchestrator:** Lifecycle coordination (trigger Bootstrap, route flows, activate Fallback). Does NOT synchronize real-time messages.
- **XAI Log:** Parallel audit trail. Records SRI/verdict, natural-language explanation, audit ID, and precedent cited for EVERY decision.

---

## 🔌 3. COMPONENT SPECIFICATIONS (I/O & LOGIC)

### ① Event Detector + Producer Registry
- **Input:** Raw stream + metadata (`producer_id`, `schema_hash`, `timestamp`)
- **Logic:** Hash-based registry lookup. Routes to `Bootstrap` if new/drift, `Validation` if known.
- **Output:** Routing directive (`NEW`, `KNOWN`, `DRIFT`)
- **Tech:** Redis/In-memory dict for O(1) lookups. <100ms latency.

### ② Bootstrap Agent (SFDA)
- **Input:** Schema + event trigger
- **Logic:** Source-Free Domain Adaptation (clustering, distribution fitting, feature scaling). No labeled data required.
- **Output:** Baseline model + confidence score (0.0–1.0)
- **Tech:** `scikit-learn` / `river` for online stats. Confidence ≥0.85 required to enter steady-state.

### ③ Profiler Agent (Statistical)
- **Input:** Record + baseline model
- **Logic:** Lightweight deterministic module. Computes Z-scores, PSI/KL-Divergence, KS-test.
- **Output:** `deviation_score` (0.0–1.0), `drift_detected` (bool), `confidence` (0.0–1.0)
- **Tech:** `numpy`, `scipy.stats`. **NOT an LLM agent.** Target: <10ms/record.

### ④ Semantic Validator Agent (Logical)
- **Input:** Record + domain context (contracts, rules, schedules)
- **Logic:** LLM-based reasoning with structured JSON output. Evaluates operational feasibility.
- **Output:** `verdict` (Valid/Invalid), `confidence` (0.0–1.0), `reason` (string)
- **Tech:** `litellm` / `langchain`, `temperature=0.0`. Autorater loop if confidence <0.70.
- **Optimization:** Cache identical records. Skip if Profiler flags severe drift.

### ⚖️ Decision Diamond (Agreement Logic)
```python
if profiler.deviation_score >= 0.85 and validator.confidence >= 0.85 and validator.verdict == "Valid":
    route = "TRUST"
elif profiler.deviation_score < 0.50 or validator.confidence < 0.50 or validator.verdict == "Invalid":
    route = "JUDGE"
else:
    route = "AMBIGUOUS" → JUDGE
```

### ⑤ Trust Agent
- **Input:** Deviation score + verdict + confidence
- **Logic:** Weighted aggregation → SRI (0.0–1.0). Generates natural-language explanation.
- **Output:** `SRI`, `explanation`, triggers `Low SRI` alert to Orchestrator if < threshold.

### ⑥ Judge Agent
- **Input:** Conflicting evidence + precedent database
- **Logic:** LLM reasoning with constraint checking. Outputs final verdict + cited precedent.
- **Output:** `verdict`, `rationale`, `precedent_id`
- **Fallback:** If unresolvable after max iterations → `Unresolvable` signal → Orchestrator triggers Fallback.

---

## ⚡ 4. TECHNICAL CONSTRAINTS & OPTIMIZATION DIRECTIVES
| Constraint | Specification | Implementation Rule |
|------------|---------------|---------------------|
| **Latency** | <2s end-to-end per record | Run Profiler/Validator in parallel (`asyncio`). Cache LLM calls. |
| **Cost** | <$0.01 per 1K records | Use `gpt-4o-mini` or local Llama 3 8B. Prune Validator if Profiler flags critical drift. |
| **Over-Sensitivity** | Prevent false alarms on transient noise | Hysteresis: Drift must persist ≥5 records. SRI decay over time. |
| **Stability** | No infinite loops | Max 1 re-eval per record. Orchestrator state machine prevents re-triggering. |
| **Modularity** | Avoid monolithic agents | Profiler = Python module. Validator/Judge = LLM agents. Orchestrator = routing FSM. |
| **DoS Protection** | Rate limit + circuit breaker | Max N evaluations/sec per producer. Disable non-critical checks under load. |

---

## 📊 5. VALIDATION & EVALUATION FRAMEWORK
### 4-Phase Validation (Slides 8–10)
| Phase | What | Method | Target Metric |
|-------|------|--------|---------------|
| **Phase 0** | Routing Accuracy | 1K connection events + simulated schema changes | Accuracy ≥95%, FPR <5%, Latency <100ms |
| **Phase I** | Detection Accuracy | Ground truth (2K labeled) + synthetic anomalies (5/10/20%) | Precision ≥0.85, Recall ≥0.80, F1 ≥0.82 |
| **Phase II** | Intervention Effectiveness | ML model: Raw vs MAS-DQA accepted data | Accuracy Gain ≥5%, Latency <2s |
| **Phase III** | Reasoning Quality | Expert panel (3+) on 100 contentious cases | Agreement ≥85%, Explanation Coherence ≥4/5 |

### Statistical Tests (Slide 12)
- `Paired t-test`: ML accuracy gain (Raw vs Cleaned) → p<0.05
- `Cohen's Kappa`: Inter-rater reliability (Judge vs Experts) → κ>0.60
- `ANOVA`: Cross-dataset generalizability (Buspas vs Montreal vs MIMIC) → p<0.05

### Baselines for Comparison
- Great Expectations (rule-based)
- Deequ (statistical)
- Cocoon (LLM batch cleaner)
- Apache Griffin (enterprise DQ)

---

## 📁 6. CODEBASE STRUCTURE & CONVENTIONS
```
mas_dqa/
├── data/                  # BusPas samples, synthetic injections, baselines, labeled subsets
├── config/                # thresholds.yaml, rules.json, llm_config.yaml, domain_contexts/
├── src/
│   ├── ingestion.py       # Stream loader, schema parser, metadata extractor
│   ├── profiler.py        # Lightweight stats engine (PSI, KS-test, Z-score)
│   ├── validator.py       # LLM semantic agent with structured JSON output
│   ├── agreement.py       # Decision diamond logic & routing FSM
│   ├── trust.py           # SRI calculation, explanation generation
│   ├── judge.py           # Conflict resolution, precedent DB integration
│   ├── orchestrator.py    # Lifecycle management, fallback triggers, caching
│   └── xai_log.py         # Append-only audit trail, export to JSON/CSV/DB
├── tests/                 # Unit tests, integration tests, latency/cost benchmarks
├── scripts/               # Synthetic anomaly injector, baseline trainer, metrics exporter
├── docs/                  # Architecture diagrams, API contracts, validation reports
└── main.py                # MVP pipeline runner & evaluation harness
```
**Conventions:**
- Use `pydantic` for all I/O schemas.
- All agents return typed `BaseModel` outputs.
- LLM calls use `litellm` for provider abstraction.
- Async processing for parallel Profiler/Validator execution.
- Strict logging: `INFO` for routing, `DEBUG` for metrics, `ERROR` for fallback triggers.

---

## 🤖 7. HOW TO USE THIS KB WITH AGENTIC LLM TOOLS

### Prompt Template for Task Assignment
```
You are an expert ML/Data Engineering AI agent. Use the MAS-DQA Knowledge Base to implement [TASK].
Follow these constraints:
1. Keep components modular and lightweight.
2. Use Pydantic for all data schemas.
3. Target latency <Xms for this component.
4. Return code + unit tests + usage example.
Reference slides: [Slide #]
Knowledge Base: [Paste relevant sections above]
```

### Common AI Agent Tasks & Directives
| Task | Directive |
|------|-----------|
| `Implement Profiler` | Use `numpy`/`scipy`. Return `deviation_score` 0-1. No LLM. <10ms target. |
| `Implement Validator` | Use `litellm` with `temperature=0.0`. Enforce JSON schema. Add Autorater retry if confidence <0.70. |
| `Implement Agreement Logic` | Use explicit thresholds (0.85). Return routing enum (`TRUST`, `JUDGE`, `QUARANTINE`). |
| `Add Hysteresis/Caching` | Implement LRU cache for identical records. Require N consecutive drift signals before action. |
| `Generate Evaluation Script` | Load labeled data. Compute Precision/Recall/F1. Run paired t-test. Export CSV report. |

---

## 📌 8. CURRENT STATUS & NEXT STEPS
| Status | Next Milestone |
|--------|----------------|
| ✅ Architecture defined, gaps validated, baselines selected | 🛠️ Implement `profiler.py` + `validator.py` MVP |
| ✅ 4-phase validation methodology defined | 📊 Run Phase 0 & I metrics on 1K BusPas records |
| ✅ Latency/cost constraints documented | ⚡ Add Redis caching + async parallel execution |
| ✅ Feedback integrated (avoid over-engineering) | 📝 Generate technical spec for Judge Agent + Precedent DB |

---

## 💡 9. KEY DESIGN PRINCIPLES (DO / DON'T)
| ✅ DO | ❌ DON'T |
|------|----------|
| Use agents only for reasoning/autonomy (Validator, Judge) | Wrap deterministic math (Profiler) in LLM agent loops |
| Cache identical records & LLM responses | Re-evaluate same record multiple times without decay |
| Log all decisions to XAI for auditability | Output black-box scores without explanations |
| Fail safely (Fallback/Quarantine) on ambiguity | Guess or block pipeline on unresolved conflicts |
| Tune thresholds per domain (healthcare vs transit) | Use hardcoded global thresholds for all use cases |

