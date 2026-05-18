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