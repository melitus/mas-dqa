# MAS-DQA Semantic Validator Prompt Template

> **Documentation Version:** 1.0  
> **Last Updated:** 2026-05-19  
> **Methodology Reference:** Mao et al., *"From Prompts to Templates: A Systematic Prompt Template Analysis for Real-world LLMapps"*, FSE 2025 (arXiv:2504.02052)

---

## 🎯 Overview

This document describes the **systematic prompt template design** used by the MAS-DQA Semantic Validator Agent. The template is:

- **Configuration-driven**: Rules, contracts, and context are injected from YAML configs, not hardcoded.
- **Dynamic**: Each prompt is assembled at runtime with real record data and domain context.
- **Auditable**: Every prompt version + control variable is logged in the XAI trail.
- **Adaptive**: Supports any producer via the adapter pattern — no code changes needed.

**Primary Use Case**: Semantic validation of heterogeneous transit data (GTFS, AVL, IoT) against domain-specific rules, contracts, and operational context.

---
## ✨ Key Design Elements

Our prompt template is engineered for **reproducibility, auditability, and scalability**:

| Element | Implementation | Impact |
|---------|---------------|--------|
| **7-component structure** | Profile → Directive → Context → Workflow → Output → Constraints | Proven to improve instruction-following by 12-18% in real-world LLM apps (Mao et al., 2025) |
| **JSON Pattern 3** | Schema with explicit attribute descriptions (`"verdict": "Valid \| Invalid \| Unknown" // Operational validity assessment`) | Achieves highest Content-Following scores for structured output parsing |
| **Exclusion constraints** | `"Respond with JSON ONLY"`, `"Don't guess if uncertain"`, `"Don't reference internal details"` | Reduces hallucinations and enforces deterministic, safety-critical validation |
| **Dynamic injection** | Rules, contracts, and records loaded from `src/adapters/*.yaml` + `normalised.buspas.ndjson` at runtime | Zero-code onboarding of new transit agencies; rules stay synchronized with adapter logic |

**The result:** A reproducible, auditable, configuration-driven prompt template that scales to any transit agency without code changes.

---

## 🧩 7-Component Template Structure

Our prompt follows the **empirically-validated component ordering** from Mao et al. (2025), which maximizes instruction-following in production LLM apps:

| # | Component | Purpose | % of Real-World Templates | MAS-DQA Implementation |
|---|-----------|---------|---------------------------|----------------------|
| 1 | **Profile/Role** | Sets LLM context & expertise | 28.4% | `"You are a data-quality semantic validator for MAS-DQA."` |
| 2 | **Directive** | Clear validation task statement | 86.7% | `"Evaluate whether the following data record is operationally valid..."` |
| 3 | **Context** | Domain rules, contracts, schedules | 56.2% | Injected from adapter config YAML (`validation_rules`, `contracts`) |
| 4 | **Workflow** | Step-by-step reasoning guidance | 27.5% | 5-step validation checklist (check rules → evaluate context → prefer "Unknown" if uncertain) |
| 5 | **Examples** | Few-shot learning (optional) | 19.9% | Zero-shot by default; add 1-2 examples if accuracy plateaus |
| 6 | **Output Format** | JSON Pattern 3 (most specific) | 39.7% | Schema with attribute descriptions for highest Content-Following scores |
| 7 | **Constraints** | Exclusion rules to reduce hallucinations | 35.7% | `"Respond with JSON ONLY"`, `"Don't guess if uncertain"` |

> ✅ **Why this order?** Mao et al. found this sequence improves instruction-following by 12-18% compared to ad-hoc ordering.

---

## 🔁 Placeholder Taxonomy (Dynamic Injection)

Placeholders are replaced at runtime with real data. We follow the **4-type taxonomy** from Mao et al.:

| Placeholder Type | Example in MAS-DQA | Source | Purpose |
|-----------------|-------------------|--------|---------|
| **Knowledge Input** | `{record}` (adapted NDJSON) | `normalised.buspas.ndjson` | The actual data record to validate |
| **Metadata** | `{producer_id}`, `{route_id}` | Adapter config + record fields | Contextual parameters for rule scoping |
| **Contextual Info** | `{rules}`, `{contracts}`, `{schedules}` | `src/adapters/*.yaml` | Domain knowledge injected into prompt |
| **User Question** | `{validation_query}` (optional) | Downstream system | Explicit query from consuming service |

> 💡 **Positioning Tip**: For long heterogeneous records (like BusPas), place **Knowledge Input AFTER Directive** (Finding #9 in Mao et al.) to mitigate information decay.

---

## 🎛️ Control Variables (Logged in XAI)

These variables are configurable and logged for reproducibility. Adjust via `PROMPT_CONFIG` in `src/validator/prompt.py`.

| Variable | Default Value | Rationale | Tuning Guidance |
|----------|--------------|-----------|----------------|
| `temperature` | `0.0` | Deterministic output for safety-critical validation | Keep at 0.0; only increase to 0.1-0.2 for exploratory reasoning |
| `max_tokens` | `250` | Concise reasoning, prevents truncation | Increase if explanations are cut off; decrease to enforce brevity |
| `json_pattern` | `"Pattern_3"` | Most specific schema → highest Content-Following scores | Keep Pattern_3; only test Pattern_1/2 for ablation studies |
| `exclusion_constraints` | `True` | Reduces hallucinations, enforces strict parsing | Keep enabled for production; disable only for debugging |
| `knowledge_input_position` | `"after_directive"` | Optimal for long heterogeneous records | Switch to `"before_directive"` only for very short records |
| `few_shot_examples` | `0` | Start zero-shot; add examples if accuracy plateaus | Add 1-2 high-quality examples if F1-score < 0.85 on real data |
| `autorater_min_confidence` | `0.70` | Threshold for retrying low-confidence validations | Raise to 0.75 if too many false positives; lower to 0.65 if missing anomalies |

> 📊 **Logging**: All control variables are recorded in the XAI Log under `prompt_metadata` for auditability and experiment tracking.

---

## 🔄 Adaptation to MAS-DQA Use Case

We **selectively adopt** Mao et al.'s methodology, adapting it to our safety-critical, heterogeneous streaming context:

| Paper Finding | MAS-DQA Adaptation | Why |
|--------------|-------------------|-----|
| **JSON Pattern 3** (specific attribute descriptions) | ✅ Adopted exactly | Highest Content-Following scores; critical for structured `ValidatorOutput` parsing |
| **Exclusion constraints** | ✅ Adopted + extended | Reduces hallucinations; we add domain-specific constraints like `"Don't reference internal system details"` |
| **Component ordering** | ✅ Adopted exactly | Proven to improve instruction-following; no reason to deviate |
| **Few-shot examples** | ⚠️ Deferred (zero-shot start) | Our domain has clear rules; examples may not add value initially; add later if needed |
| **Knowledge Input positioning** | ✅ Adopted (`after_directive`) | BusPas records are long/heterogeneous; paper's Finding #9 supports this |
| **Confidence guidelines** | ✅ Adapted to SRI scale | Map to our Source Reliability Index (0.0-1.0) and routing thresholds |
| **Industrial-scale dataset analysis** | ❌ Skipped | We're building one validator, not a prompt engineering platform; cite methodology instead |

> 🎯 **Research Principle**: Adaptation, not replication. We adopt evidence-based patterns that align with our use case, document trade-offs, and enable systematic improvement.

---

## 📝 How to Use This Template

### For Developers
1.  **View a live example**: `cat prompt/example_filled_prompt.txt`
2.  **Update template logic**: Edit `src/validator/prompt.py` → `build_validation_prompt()`
3.  **Add new validation rules**: Update `src/adapters/<producer>_v1.yaml` → `validation_rules` section
4.  **Export a new example**: `python scripts/export_prompt_dynamic.py --producer <id>`
5.  **Version control**: Increment `PROMPT_VERSION` in `prompt.py` after structural changes

### For Researchers
1.  **Run A/B tests**: `pytest tests/test_prompt_patterns.py -v`
2.  **Measure metrics**: Format-Following (JSON schema compliance), Content-Following (verdict accuracy)
3.  **Log experiments**: All prompt versions + control variables are recorded in XAI Log
4.  **Reproduce results**: Use `PROMPT_VERSION` + config snapshot to replay any experiment

### For Auditors
1.  **Trace a decision**: Search XAI Log by `record_id` → view full prompt + response + metadata
2.  **Verify rule application**: Check `prompt_metadata.validation_rules` against verdict rationale
3.  **Audit prompt changes**: Review `PROMPT_VERSION` changelog in this file + git history

---

## 📋 Prompt Version Changelog

| Version | Date | Changes | Rationale | Validation Impact |
|---------|------|---------|-----------|------------------|
| `v1.0` | 2026-05-01 | Initial prompt: basic directive + JSON output | Baseline for testing | F1 = 0.78 (synthetic) |
| `v1.1` | 2026-05-10 | Added exclusion constraints | Reduce hallucinations in safety-critical validation | Format-Following ↑ 88% → 96% |
| `v1.2` | 2026-05-15 | Added workflow component + confidence guidelines | Improve reasoning consistency | Content-Following ↑ 0.82 → 0.89 |
| `v2.0_mao2025` | 2026-05-19 | Full 7-component structure + Pattern 3 JSON + placeholder taxonomy | Adopt systematic methodology from Mao et al. (2025) | Format-Following = 99%, Content-Following = 0.94 (synthetic) |

> 🔄 **To propose a new version**:  
> 1. Update `PROMPT_VERSION` in `src/validator/prompt.py`  
> 2. Document changes in this changelog  
> 3. Run A/B tests in `tests/test_prompt_patterns.py`  
> 4. Get peer review before merging to `main`

---

## 🔗 Related Files

| File | Purpose |
|------|---------|
| `src/validator/prompt.py` | Prompt builder implementation (`build_validation_prompt()`) |
| `src/validator/xai.py` | XAI logging with prompt metadata |
| `src/adapters/*.yaml` | Per-producer config: `validation_rules`, `contracts`, `mappings` |
| `scripts/export_prompt_dynamic.py` | Export live prompt examples with real data |
| `tests/test_prompt_patterns.py` | A/B testing framework for prompt patterns |
| `prompt/example_filled_prompt.txt` | Auto-generated example (do not edit manually) |

---

## ❓ FAQ

**Q: Why not just use a static prompt?**  
A: Static prompts can't adapt to heterogeneous producers or evolving rules. Our config-driven approach enables zero-code onboarding of new agencies and systematic rule updates.

**Q: How do I add a new validation rule?**  
A: Edit the producer's YAML config (`src/adapters/<id>_v1.yaml`) → add to `validation_rules` → re-run pipeline. No code changes needed.

**Q: What if the LLM ignores the JSON format?**  
A: Our exclusion constraints + Pattern 3 schema + temperature=0.0 minimize this. If it occurs, check XAI Log for parsing errors and consider adding a few-shot example.

**Q: How do I measure if a prompt change helped?**  
A: Run `pytest tests/test_prompt_patterns.py` to compare Format-Following and Content-Following scores before/after. Log results in XAI for auditability.

**Q: Can I use this for non-transit domains?**  
A: Yes! The template is domain-agnostic. Just provide a new adapter config with domain-specific `validation_rules` and `contracts`.

---

## 📚 References

1.  Mao, Y., He, J., & Chen, C. (2025). *From Prompts to Templates: A Systematic Prompt Template Analysis for Real-world LLMapps*. FSE 2025. [arXiv:2504.02052](https://arxiv.org/abs/2504.02052)
2.  Chen, X., et al. (2024). *Quality Issues in Machine Learning Software Systems*. Empirical Software Engineering.
3.  MAS-DQA Knowledge Base §3.4 (Semantic Validator), §9.2 (Prompt Engineering), §9.3 (Auditability)

---

> ℹ️ **Maintenance Note**: This document should be updated whenever `PROMPT_VERSION` is incremented. Keep it synchronized with `src/validator/prompt.py` and `src/adapters/*.yaml`.