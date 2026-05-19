# MAS-DQA Profiler: Metrics Specification

> **Version:** 1.0  
> **Last Updated:** 2026-05-19  
> **Reference:** MAS-DQA Knowledge Base §3.2, §5.1 (Phase I Validation)

---

## 🎯 Purpose

The Profiler Agent evaluates incoming records against a precomputed baseline to detect **statistical anomalies** without requiring labeled ground truth. It outputs structured signals for the Decision Diamond.

---

## 📈 Core Metrics (Output in `ProfilerOutput`)

| Metric | Formula / Logic | Range | Purpose | Target |
|--------|----------------|-------|---------|--------|
| **`deviation_score`** | `1.0 - (mean_abs_z / 6.0)`, clamped [0,1] | 0.0 (bad) → 1.0 (normal) | Overall statistical normality | ≥0.70 for "Valid" |
| **`point_anomaly_detected`** | `deviation_score < ANOMALY_THRESHOLD` | bool | Binary flag for routing decisions | — |
| **`confidence`** | `deviation_score` if Valid, else `1.0 - deviation_score` | 0.0 → 1.0 | Confidence in verdict for Decision Diamond | ≥0.75 for TRUST routing |
| **`flagged_features`** | Features with `\|z\| > 3.0` | List[str] | Explainability: which fields are anomalous | — |
| **`feature_scores`** | Per-feature z-scores: `(val - mean) / std` | Dict[str, float] | Fine-grained explainability + debugging | — |
| **`verdict`** | `"Valid"` if `deviation_score >= threshold`, else `"Invalid"`/`"Unknown"` | str | Final decision for Decision Diamond | — |
| **`reason`** | Human-readable explanation | str | XAI Log + human debugging | — |
| **`metrics`** | Additional diagnostics (max_z, threshold, etc.) | Dict[str, Any] | Audit trail + threshold tuning | — |

---

## 🔢 Z-Score Calculation (Core Statistical Logic)

```python
def calculate_z_score(value: float, mean: float, std: float) -> float:
    """Compute z-score with std floor to avoid div-by-zero."""
    std = max(std, 1e-8)  # Prevent division by zero
    return (value - mean) / std

```

## Interpretation:
    - |z| ≤ 1: Within 1σ → very normal
    - 1 < |z| ≤ 3: Within 3σ → acceptable variation
    - |z| > 3: Beyond 3σ → potential anomaly (flagged)

## ❓ FAQ
Q: Why use z-scores instead of raw thresholds?
A: Z-scores normalize across features with different scales (speed vs. lat vs. passenger count), enabling fair comparison and automatic anomaly detection without manual per-feature tuning.

Q: What if baseline stats are stale?
A: The Profiler supports update_baseline() for adaptive re-onboarding when schema drift is detected (triggered by Orchestrator).

Q: How do I add a new feature to profile?
A: Ensure it's numeric in the normalized schema → Profiler auto-includes it in z-score computation. No code change needed.

Q: Can I use this for non-numeric fields?
A: Not directly. For categorical fields, use the Semantic Validator (LLM-based) for semantic rule checking.