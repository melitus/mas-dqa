"""Pydantic schemas for Semantic Validator I/O.

This module contains ONLY data schemas — no business logic.
All schemas are Pydantic v2 compatible.

Reference: MAS-DQA Knowledge Base §3 (Component Specifications)
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, field_validator


class ProfilerResult(BaseModel):
    """
    Output from the Profiler Agent.
    Passed to the validator to enable anomaly-based skipping.
    
    Attributes
    ----------
    deviation_score: float
        Normalised score in the range ``0.0`` (anomalous) → ``1.0`` (normal).
    point_anomaly_detected: bool
        ``True`` when the deviation score falls below the configured
        threshold, indicating a statistical outlier.
    confidence: float
        Proxy for confidence in the detection – currently mirrors the
        ``deviation_score``.
    """
    deviation_score: float = Field(..., ge=0.0, le=1.0, description="0.0 (anomalous) → 1.0 (normal)")
    point_anomaly_detected: bool = Field(..., description="Indicates statistical outlier")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Proxy for statistical confidence")


class ScheduleEntry(BaseModel):
    """Represents a temporal or conditional schedule rule."""
    name: str
    condition: str
    priority: int = Field(default=0, ge=0)


class DomainContext(BaseModel):
    """
    Domain context containing contracts, rules, and schedules
    that the validator evaluates against.
    """
    rules: Dict[str, str] = Field(default_factory=dict)
    contracts: Dict[str, str] = Field(default_factory=dict)
    schedules: List[ScheduleEntry] = Field(default_factory=list)

    @field_validator("rules", "contracts", mode="before")
    @classmethod
    def ensure_dict(cls, v):
        """Ensure rules/contracts are dicts (handle None input)."""
        if v is None:
            return {}
        return v

    @field_validator("schedules", mode="before")
    @classmethod
    def ensure_list(cls, v):
        """Ensure schedules is a list (handle None input)."""
        if v is None:
            return []
        return v

    def model_dump_compat(self) -> dict:
        """Pydantic v2-compatible dump for caching and serialization."""
        return {
            "rules": self.rules,
            "contracts": self.contracts,
            "schedules": [s.model_dump() for s in self.schedules],
        }


class ValidatorInput(BaseModel):
    """Input to the Semantic Validator."""
    record: Dict[str, Any]
    domain_context: DomainContext
    profiler_result: Optional[Any] = None  # Accept ProfilerOutput from profiler


class ValidatorOutput(BaseModel):
    """
    Output from the Semantic Validator.

    Fields:
        verdict: Valid | Invalid | Unknown
        confidence: 0.0 to 1.0
        reason: Natural-language explanation
        metadata: Additional diagnostic information
    """
    verdict: str = Field(default="Unknown", pattern="^(Valid|Invalid|Unknown)$")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = "No evaluation performed"
    metadata: Dict[str, Any] = Field(default_factory=dict)