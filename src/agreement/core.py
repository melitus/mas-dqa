"""Decision Diamond logic for MAS-DQA.

Evaluates consensus between Profiler and Validator agents.
Returns a RoutingDecision that the Orchestrator will execute.

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
    QUARANTINE = "QUARANTINE" # Severe anomaly or invalid → Isolate
    AMBIGUOUS = "AMBIGUOUS"  # Fallback → escalate to Judge


def determine_routing_decision(
    prof_out: ProfilerOutput,
    val_out: ValidatorOutput,
    thresholds: Optional[ValidationThresholds] = None
) -> RoutingDecision:
    """
    Routing decision based on *verdict* agreement first, mirroring the contract
    defined in the Knowledge Base. Confidence thresholds are used only as a
    secondary filter when both agents agree on a "Valid" verdict.
    
    Key logic:
    - TRUST: Both Valid + high confidence
    - QUARANTINE: Both Invalid (severe, agreed-upon anomaly)
    - JUDGE: Any conflict or uncertainty (Profiler ≠ Validator, or low confidence)
    - AMBIGUOUS: Fallback for edge cases
    """
    t = thresholds or DEFAULT_THRESHOLDS

    # 1️⃣ Both agents agree on Valid + high confidence → TRUST
    if prof_out.verdict == "Valid" and val_out.verdict == "Valid":
        if (prof_out.confidence >= t.TRUST_MIN_CONFIDENCE and 
            val_out.confidence >= t.TRUST_MIN_CONFIDENCE):
            return RoutingDecision.TRUST
        # Low confidence despite agreement → human review
        return RoutingDecision.JUDGE

    # 2️⃣ Both agents agree on Invalid → QUARANTINE (only when BOTH agree)
    if prof_out.verdict == "Invalid" and val_out.verdict == "Invalid":
        return RoutingDecision.QUARANTINE

    # 3️⃣ Mismatched verdicts → conflict, route to Judge
    if prof_out.verdict != val_out.verdict:
        return RoutingDecision.JUDGE

    # 4️⃣ One Invalid + one Unknown → Judge decides severity
    if "Invalid" in [prof_out.verdict, val_out.verdict]:
        return RoutingDecision.JUDGE

    # 5️⃣ Fallback for Unknown/Unknown or other edge cases
    return RoutingDecision.AMBIGUOUS