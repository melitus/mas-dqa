"""Core Statistical Profiler for MAS-DQA.

Evaluates incoming records against a precomputed baseline.
Lightweight, deterministic, and optimized for <5ms latency.

Reference: MAS-DQA Knowledge Base §3.2
"""
from __future__ import annotations

import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

from src.schemas.profiler import ProfilerOutput
from src.profiler.stats import _AnalysisModule
from src.config.thresholds import ValidationThresholds, DEFAULT_THRESHOLDS

logger = logging.getLogger(__name__)


class Profiler:
    """Evaluate new records against a baseline.

    Parameters
    ----------
    baseline_df: pandas.DataFrame
        Dataframe containing historic values.  Mean and standard deviation are
        computed lazily for each column during evaluation.
    """

    def __init__(
        self,
        baseline_df: pd.DataFrame,
        thresholds: Optional[ValidationThresholds] = None
    ):
        self._baseline = baseline_df
        self._analysis = _AnalysisModule()
        self._thresholds = thresholds or DEFAULT_THRESHOLDS
        # Precompute & cache numeric stats for O(1) streaming evaluation
        self._baseline_stats = self._compute_stats(baseline_df)

    @staticmethod
    def _compute_stats(df: pd.DataFrame) -> Dict[str, tuple[float, float]]:
        """Precompute mean and std for all numeric columns."""
        numeric_df = df.select_dtypes(include="number")
        return {
            col: (float(numeric_df[col].mean()), float(numeric_df[col].std() + 1e-8))  # Avoid div-by-zero
            for col in numeric_df.columns
        }

    def update_baseline(self, new_baseline_df: pd.DataFrame):
        """Update the baseline stats (used during adaptive re-onboarding)."""
        self._baseline = new_baseline_df
        self._baseline_stats = self._compute_stats(new_baseline_df)
        logger.info("Profiler baseline updated successfully")

    def _compute_z_scores(self, record: Dict) -> Dict[str, float]:
        """Calculate Z‑scores for all numeric columns present in *record*.
        Returns dict of {feature: z_score} for traceability.
        """
        z_scores = {}
        for col, val in record.items():
            if col not in self._baseline_stats:
                continue
            try:
                mean, std = self._baseline_stats[col]
                z_scores[col] = self._analysis.calculate_z_score(float(val), mean, std)
            except (TypeError, ValueError):
                # Skip non‑numeric or unparsable values
                continue
        return z_scores

    def evaluate_record(self, record: Dict) -> ProfilerOutput:
        """Evaluate *record* and return a :class:`ProfilerOutput`.
        
        Schema: deviation_score 0.0 (bad) → 1.0 (normal)
        """
        z_scores_dict = self._compute_z_scores(record)

        # Convert to list for scoring, keep dict for explainability
        z_scores_list = list(z_scores_dict.values())
        
        # Compute deviation score: 0.0 = bad, 1.0 = normal
        if not z_scores_list:
            deviation = 1.0  # No numeric features → assume normal
        else:
            # Use mean absolute z-score inverted: higher z = lower deviation score
            mean_abs_z = np.mean([abs(z) for z in z_scores_list])
            # Map: z=0 → deviation=1.0, z=3 → deviation=0.5, z=6 → deviation=0.0
            deviation = max(0.0, min(1.0, 1.0 - (mean_abs_z / 6.0)))

        anomaly = deviation < self._thresholds.ANOMALY_PROFILER_THRESHOLD

        # Identify flagged features (3-sigma rule)
        flagged = [col for col, z in z_scores_dict.items() if abs(z) > 3.0]

        # Derive verdict and confidence from deviation score
        if deviation >= self._thresholds.ANOMALY_PROFILER_THRESHOLD:
            verdict = "Valid"
            reason = "Record is within normal statistical range"
            confidence = deviation  # Higher deviation score = higher confidence in "Valid"
            metrics = {
                "max_abs_z": max(abs(z) for z in z_scores_dict.values()) if z_scores_dict else 0.0,
                "deviation_score": round(deviation, 3)
            }
        elif deviation >= self._thresholds.ANOMALY_PROFILER_THRESHOLD * 0.5:
            # Buffer zone: uncertain
            verdict = "Unknown"
            reason = f"Uncertain: borderline deviation (score: {deviation:.2f})"
            confidence = 0.5  # Neutral confidence for uncertain cases
            metrics = {
                "deviation_score": round(deviation,  3),
                "uncertain_zone": True
            }
        else:
            # Clear anomaly
            verdict = "Invalid"
            reason = f"Statistical anomaly detected in features: {', '.join(flagged)}" if flagged else "Deviation exceeds baseline threshold"
            confidence = 1.0 - deviation  # Lower deviation score = higher confidence in "Invalid"
            metrics = {
                "flagged_features": flagged,
                "deviation_score": round(deviation, 3),
                "anomaly_threshold": self._thresholds.ANOMALY_PROFILER_THRESHOLD
            }

        return ProfilerOutput(
            deviation_score=round(deviation, 3),
            point_anomaly_detected=anomaly,
            confidence=round(confidence, 3),
            flagged_features=flagged,
            feature_scores={k: round(v, 3) for k, v in z_scores_dict.items()},
            verdict=verdict,
            reason=reason,
            metrics=metrics
        )