# Statistical deviation engine

# Profiler module – refactored for clarity, testability and type safety

"""Statistical profiling utilities.

The module provides a :class:`Profiler` that evaluates a new record against a
baseline pandas DataFrame.  It returns a :class:`ProfilerOutput` with a
normalised deviation score, a drift flag and a simple confidence proxy.

The implementation is deliberately modular: low‑level statistical helpers are
encapsulated in :class:`_AnalysisModule`.  This makes the code easier to unit
test and to extend with more sophisticated drift‑detection algorithms.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class ProfilerOutput(BaseModel):
    """Result of a profiling evaluation.

    Attributes
    ----------
    deviation_score: float
        Normalised score in the range ``0.0`` (bad) → ``1.0`` (normal).
    drift_detected: bool
        ``True`` when the deviation score falls below the configured
        threshold, indicating a potential drift.
    confidence: float
        Proxy for confidence in the detection – currently mirrors the
        ``deviation_score``.
    """

    deviation_score: float = Field(..., description="0.0 (bad) → 1.0 (normal)")
    drift_detected: bool = Field(..., description="Indicates statistical significance")
    confidence: float = Field(..., description="Proxy for statistical confidence")


class _AnalysisModule:
    """Pure statistical helpers – no internal state.

    Separating these helpers from the :class:`Profiler` keeps the core class
    focused on orchestration and makes the helpers straightforward to test in
    isolation.
    """

    _epsilon: float = 1e-6  # Guard against division‑by‑zero

    def calculate_z_score(self, value: float, mean: float, std: float) -> float:
        """Return the absolute Z‑score for *value*.

        The ``_epsilon`` term prevents catastrophic division when ``std`` is
        zero.
        """
        denominator = std + self._epsilon
        return abs((value - mean) / denominator)

    def compute_deviation_score(self, z_scores: Iterable[float]) -> float:
        """Convert a collection of Z‑scores to a normalised deviation score.

        The average Z‑score is scaled by ``3.0`` (roughly three standard
        deviations) and inverted so that higher values indicate a closer match to
        the baseline.  The result is clamped to the ``0.0‑1.0`` interval.
        """
        z_array = np.array(list(z_scores))
        avg_z = np.mean(z_array) if z_array.size else 0.0
        return max(0.0, 1.0 - (avg_z / 3.0))

    def determine_drift(self, deviation_score: float, threshold: float = 0.70) -> bool:
        """Return ``True`` when *deviation_score* indicates drift.

        The default threshold mirrors the original implementation.
        """
        return deviation_score < threshold


class Profiler:
    """Evaluate new records against a baseline.

    Parameters
    ----------
    baseline_df: pandas.DataFrame
        Dataframe containing historic values.  Mean and standard deviation are
        computed lazily for each column during evaluation.
    """

    def __init__(self, baseline_df: pd.DataFrame):
        self._baseline = baseline_df
        self._analysis = _AnalysisModule()

    def _column_stats(self, col: str) -> tuple[float, float]:
        """Return ``(mean, std)`` for *col* from the baseline.

        If the column is non‑numeric pandas will raise ``TypeError`` – callers
        should handle this and skip the column.
        """
        series = self._baseline[col]
        return float(series.mean()), float(series.std())

    def _compute_z_scores(self, record: Dict) -> np.ndarray:
        """Calculate Z‑scores for all numeric columns present in *record*.
        """
        z_scores = []
        for col, val in record.items():
            if col not in self._baseline.columns:
                continue
            try:
                mean, std = self._column_stats(col)
                z = self._analysis.calculate_z_score(float(val), mean, std)
                z_scores.append(z)
            except (TypeError, ValueError):
                # Skip non‑numeric or unparsable values
                continue
        return np.array(z_scores) if z_scores else np.array([0.0])

    def evaluate_record(self, record: Dict) -> ProfilerOutput:
        """Evaluate *record* and return a :class:`ProfilerOutput`.
        """
        z_scores = self._compute_z_scores(record)
        deviation = self._analysis.compute_deviation_score(z_scores)
        drift = self._analysis.determine_drift(deviation)
        return ProfilerOutput(
            deviation_score=round(deviation, 3),
            drift_detected=drift,
            confidence=round(deviation, 3),
        )
