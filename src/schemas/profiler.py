"""Pydantic schemas for Profiler I/O.

This module contains ONLY data schemas — no business logic.
All schemas are Pydantic v2 compatible.

Reference: MAS-DQA Knowledge Base §3 (Component Specifications)
"""
from __future__ import annotations

from typing import Dict, List
from pydantic import BaseModel, Field


class ProfilerOutput(BaseModel):
    """Result of a profiling evaluation.

    Attributes
    ----------
    deviation_score: float
        Normalised score in the range ``0.0`` (bad) → ``1.0`` (normal).
    point_anomaly_detected: bool
        ``True`` when the deviation score falls below the configured
        threshold, indicating a potential drift.
    confidence: float
        Proxy for confidence in the detection – currently mirrors the
        ``deviation_score``.
    """
    deviation_score: float = Field(..., ge=0.0, le=1.0, description="0.0 (bad) → 1.0 (normal)")
    point_anomaly_detected: bool = Field(..., description="Indicates statistical significance")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Proxy for statistical confidence")
    # XAI & debugging support (extends original structure)
    flagged_features: List[str] = Field(default_factory=list, description="Features exceeding 3-sigma threshold")
    feature_scores: Dict[str, float] = Field(default_factory=dict, description="Per-feature Z-scores for explainability")