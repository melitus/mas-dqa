"""Pure statistical helpers for MAS-DQA Profiler.

Contains low-level mathematical operations for deviation scoring.
Deliberately stateless to enable straightforward unit testing.

Reference: MAS-DQA Knowledge Base §3.2 (Statistical Profiling)
"""
from __future__ import annotations

from typing import Iterable
import numpy as np


class _AnalysisModule:
    """Pure statistical helpers – no internal state.

    Separating these helpers from the :class:`Profiler` keeps the core class
    focused on orchestration and makes the helpers straightforward to test in
    isolation.
    """

    _epsilon: float = 1e-6  # Guard against division‑by‑zero

    @staticmethod
    def calculate_z_score(value: float, mean: float, std: float) -> float:
        """Return the absolute Z‑score for *value*.

        The ``_epsilon`` term prevents catastrophic division when ``std`` is
        zero.
        """
        denominator = std + _AnalysisModule._epsilon
        return abs((value - mean) / denominator)

    @staticmethod
    def compute_deviation_score(z_scores: Iterable[float]) -> float:
        """Convert a collection of Z‑scores to a normalised deviation score.

        The average Z‑score is scaled by ``3.0`` (roughly three standard
        deviations) and inverted so that higher values indicate a closer match to
        the baseline.  The result is clamped to the ``0.0‑1.0`` interval.
        """
        z_array = np.array(list(z_scores))
        avg_z = np.mean(z_array) if z_array.size else 0.0
        return float(np.clip(1.0 - (avg_z / 3.0), 0.0, 1.0))

    @staticmethod
    def determine_drift(deviation_score: float, threshold: float = 0.70) -> bool:
        """Return ``True`` when *deviation_score* indicates drift.

        The default threshold mirrors the original implementation.
        """
        return deviation_score < threshold