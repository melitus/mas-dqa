"""Centralized configuration thresholds for MAS-DQA validation.

Reference: MAS-DQA Knowledge Base §4 (Technical Constraints)
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationThresholds:
    """Thresholds for validation logic and routing decisions."""
    
    # Trust routing: minimum confidence to route to Trust Agent
    TRUST_MIN_CONFIDENCE: float = 0.65
    
    # Autorater: minimum confidence to accept without retry
    AUTORATER_MIN_CONFIDENCE: float = 0.70
    
    # Profiler anomaly threshold: deviation_score < this → flag as anomalous
    # (Note: deviation_score range: 0.0 (bad) → 1.0 (normal))
    ANOMALY_PROFILER_THRESHOLD: float = 0.50

    # Anomaly skip: skip validation if profiler deviation < this (highly anomalous)
    ANOMALY_SKIP_THRESHOLD: float = 0.15  # deviation_score range: 0.0 (bad) → 1.0 (normal)
    
    # Judge escalation: confidence below this triggers escalation
    JUDGE_ESCALATION_CONFIDENCE: float = 0.50
    
    # LLM token limits
    MAX_LLM_TOKENS: int = 500
    
    # Cache settings
    DEFAULT_CACHE_SIZE: int = 1000
    CACHE_TTL_SECONDS: int = 300  # 5 minutes


# Default instance for easy import
DEFAULT_THRESHOLDS = ValidationThresholds()