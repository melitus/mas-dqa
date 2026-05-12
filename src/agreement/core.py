"""Decision Diamond logic for MAS-DQA.

Evaluates consensus between Profiler and Validator agents.
Returns a RoutingDecision that the Orchestrator will execute.

Note: The Decision Diamond *determines* the routing path.
      The Orchestrator *acts* on it (triggers agents, fallbacks, etc.).

Reference: MAS-DQA Knowledge Base §3 (Decision Diamond), §5 (Validation)
"""
from enum import Enum
from typing import Optional

from src.schemas.profiler import ProfilerOutput
from src.schemas.validator import ValidatorOutput
from src.config.thresholds import ValidationThresholds, DEFAULT_THRESHOLDS


class RoutingDecision(str, Enum):
    """Routing directives determined by the Decision Diamond."""
    TRUST = "TRUST"           # High confidence + valid → Route to Trust Agent
    JUDGE = "JUDGE"          # Conflict or low confidence → Route to Judge Agent
    QUARANTINE = "QUARANTINE" # Severe anomaly + invalid → Isolate
    AMBIGUOUS = "AMBIGUOUS"  # Fallback → escalate to Judge


def determine_routing_decision(
    prof_out: ProfilerOutput,
    val_out: ValidatorOutput,
    thresholds: Optional[ValidationThresholds] = None
) -> RoutingDecision:
    """
    Decision Diamond: Evaluate Profiler + Validator consensus.
    
    Args:
        prof_out: Statistical profiling result (deviation_score 0.0→1.0)
        val_out: Semantic validation result (verdict + confidence)
        thresholds: Optional custom thresholds (uses defaults if None)
        
    Returns:
        RoutingDecision enum value for the Orchestrator to execute
    """
    t = thresholds or DEFAULT_THRESHOLDS
    
    # Normalize signals
    p_normal = prof_out.deviation_score >= t.TRUST_MIN_CONFIDENCE
    v_valid = val_out.confidence >= t.TRUST_MIN_CONFIDENCE and val_out.verdict == "Valid"
    v_invalid = val_out.verdict == "Invalid"
    
    # Explicit TRUST path: both agents confident + valid + normal stats
    if p_normal and v_valid:
        return RoutingDecision.TRUST

    # Explicit QUARANTINE path: severe anomaly or clear invalid verdict
    if prof_out.deviation_score < t.ANOMALY_SKIP_THRESHOLD or v_invalid:
        return RoutingDecision.QUARANTINE

    # Explicit JUDGE path: low confidence or conflicting signals
    if (
        val_out.confidence < t.JUDGE_ESCALATION_CONFIDENCE or
        prof_out.deviation_score < t.JUDGE_ESCALATION_CONFIDENCE
    ):
        return RoutingDecision.JUDGE

    # Fallback: ambiguous but not severe → escalate to Judge
    return RoutingDecision.AMBIGUOUS