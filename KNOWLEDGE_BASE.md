
## 🛠️ IMPLEMENTATION BLUEPRINT

### **1. Profiler Agent → Python Scripts (Deterministic Statistical Monitoring)**
**Why Python?** Fast, deterministic, zero external dependencies, ideal for real-time statistical computation.

| Component | Recommendation | Purpose |
|-----------|----------------|---------|
| **Streaming Engine** | `Bytewax` or `Faust` (Python-native) | Windowed stream processing, stateful aggregation |
| **Data Processing** | `Polars` or `Pandas` | High-performance metric computation |
| **Drift Detection** | `river` (online ML) + `scipy.stats` | KS-test, PSI, EWMA against Bootstrap baselines |
| **Baseline Store** | `SQLite` or `Redis` | Stores rolling stats from Phase 0/Bootstrap |

**Implementation Flow:**
1. Ingest streaming window (e.g., 1-min sliding window)
2. Compute: completeness rate, latency distribution, value ranges, variance from baseline
3. Run statistical tests → calculate deviation score & confidence
4. Apply thresholding → output `verdict` (`valid`/`invalid`/`uncertain`) + `confidence` (0.0–1.0)
5. Emit standardized JSON to message queue (Kafka/Redis Streams)

⚡ **Target Latency:** `<300ms` per window  
📊 **Validation Output:** Precision, Recall, F1 against 2,000 expert-labeled records

---

### **2. Semantic Validator Agent → LLM-Based (Contextual/Logical Monitoring)**
**Why LLM?** Handles ambiguous, domain-specific, cross-stream logic without rigid rule engines. Adapts to new schemas via prompt/context updates.

| Component | Recommendation | Purpose |
|-----------|----------------|---------|
| **LLM Runtime** | `vLLM` (local) or Mistral/OpenAI API | Fast inference, structured output support |
| **Context Retrieval** | `FAISS`/`Chroma` or rule cache | Fetch GTFS schedules, route polygons, agency contracts |
| **Output Structuring** | `Pydantic` + `Instructor` | Guarantees JSON schema compliance, prevents hallucination |
| **Confidence Calibration** | Temperature scaling + self-consistency (run 2-3 prompts, majority vote) | Stabilizes confidence scores |

**Implementation Flow:**
1. Fetch relevant context for producer (schedule, max speed, allowed detours, cross-stream rules)
2. Construct prompt: `"Given record {data} and context {rules}, is this operationally valid? Return JSON with verdict, confidence, explanation."`
3. Run LLM → parse structured output via `Instructor`
4. **Autorater Loop:** If `confidence < 0.7`, re-prompt with stricter constraints or fallback to deterministic rule check
5. Emit standardized JSON

⚡ **Target Latency:** `<500ms` (achieved via async calls, prompt caching, model quantization if local)  
📊 **Validation Output:** Precision, Recall, F1 + explanation coherence (Phase III)

---

### **3. Standardized Output Contract (Critical for Decision Diamond)**
Both agents **must** output identical JSON so the Decision Diamond can compare signals deterministically:

```json
{
  "producer_id": "agency_42",
  "record_id": "uuid-123",
  "timestamp": "2026-05-13T14:30:00Z",
  "agent_type": "profiler | semantic_validator",
  "verdict": "valid | invalid | uncertain",
  "confidence": 0.85,
  "metrics": {
    "completeness": 0.98,
    "latency_ms": 45,
    "deviation_sigma": 1.2,
    "domain_rule_violated": "speed_exceeds_limit"
  },
  "explanation": "GPS speed 180km/h exceeds route max 80km/h. Confirmed by schedule context."
}
```
✅ **Why this matters:** Enables clean agreement logic: `if profiler.verdict == validator.verdict and both.confidence >= 0.75 → AGREE`

---

### **4. Integration & Decision Diamond Logic**
```python
def check_agreement(profiler, validator):
    threshold = 0.75
    if profiler.verdict == validator.verdict and \
       profiler.confidence >= threshold and \
       validator.confidence >= threshold:
        return "AGREE"
    elif profiler.verdict != validator.verdict:
        return "CONFLICT"
    else:
        return "UNCERTAIN"  # Route to Judge or fallback
```
⚠️ **Key Design:** Use **confidence-weighted voting**, not simple boolean matching. Prevents low-confidence false agreements.

---

### **5. Phase I Validation Alignment**
| Validation Goal | How to Implement | Metric |
|----------------|------------------|--------|
| **Precision** | Compare agent flags vs 2,000 expert-labeled records | `TP / (TP + FP)` |
| **Recall** | Ensure injected synthetic anomalies are caught | `TP / (TP + FN)` |
| **F1-Score** | Harmonic mean for balanced evaluation | `2 * (P * R) / (P + R)` |
| **Latency** | Log end-to-end signal generation time per window | `<800ms` target |
| **Confidence Calibration** | Plot reliability diagram (predicted vs actual accuracy) | Brier score <0.1 |

🛠️ **Tooling:** `pytest` (unit tests), `locust`/`k6` (streaming load testing), `scikit-learn` (metrics)

---

### **⚠️ Engineering Risks & Mitigations**
| Risk | Mitigation |
|------|------------|
| LLM hallucination / invalid JSON | `Instructor`/`Pydantic` strict parsing + fallback to rule-only if parsing fails |
| High latency under load | Async processing, prompt caching, timeout fallback, local quantized model (Mistral-7B-GGUF) |
| Heterogeneous schema handling | Dynamic adapter layer + Bootstrap-generated schema mapping |
| False Positive Rate >5% | Tune confidence thresholds, use precision-recall optimization, negative sampling during validation |

---

### **📅 4-Week Implementation Plan**
| Week | Deliverable |
|------|-------------|
| **1** | Finalize JSON contract + streaming skeleton (Kafka/Redis → windowing → agent routing) |
| **2** | Profiler core (completeness, timeliness, drift detection) + unit tests |
| **3** | Validator pipeline (context fetch → LLM prompt → structured output + Autorater retry) |
| **4** | Decision Diamond integration + Phase I synthetic validation (Precision/Recall/F1 + latency benchmarks) |

---