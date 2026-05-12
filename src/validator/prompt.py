"""Prompt engineering utilities for Semantic Validator.

Contains deterministic prompt builders for LLM-based semantic validation.
No business logic — only string formatting and template management.

Reference: MAS-DQA Knowledge Base §3.4 (Prompt Engineering)
"""
import json
from typing import Dict, Any, Optional

from src.schemas.validator import DomainContext


def build_validation_prompt(
    record: Dict[str, Any],
    domain_context: DomainContext,
    attempt: int = 1,
    previous_reason: Optional[str] = None
) -> str:
    """
    Build a deterministic prompt for the validator LLM.

    Covers: contracts, rules, schedule constraints.
    Explicitly references KB §3.4 requirements.
    
    Args:
        record: The data record to validate
        domain_context: Domain rules, contracts, and schedules
        attempt: Retry attempt number (for enhanced re-evaluation)
        previous_reason: Reason from previous low-confidence attempt
        
    Returns:
        Formatted prompt string for LLM consumption
    """
    record_block = json.dumps(record, indent=2, default=str)

    rules_block = "\n".join(
        f"  - {name}: {desc}" for name, desc in domain_context.rules.items()
    ) or "  (none)"

    contracts_block = "\n".join(
        f"  - {name}: {desc}" for name, desc in domain_context.contracts.items()
    ) or "  (none)"

    schedules_block = "\n".join(
        f"  - [{e.priority}] {e.name}: {e.condition}"
        for e in sorted(domain_context.schedules, key=lambda s: s.priority)
    ) or "  (none)"

    prompt = (
        "Validate the following data record against the domain context.\n"
        "Consider operational feasibility, contractual obligations, "
        "temporal schedule constraints, and logical consistency.\n\n"
        f"=== DATA RECORD ===\n{record_block}\n\n"
        f"=== RULES ===\n{rules_block}\n\n"
        f"=== CONTRACTS ===\n{contracts_block}\n\n"
        f"=== SCHEDULES ===\n{schedules_block}\n\n"
        "Return JSON with exactly these fields:\n"
        '{"verdict": "Valid"|"Invalid"|"Unknown", '
        '"confidence": <float 0-1>, '
        '"reason": "<human-readable reason>"}\n'
    )

    # Enhance prompt for re-evaluation attempts (Autorater loop)
    if attempt > 1 and previous_reason:
        prompt += (
            f"\n\n[RE-EVALUATION CONTEXT] Previous attempt had low confidence. "
            f"Reason given: '{previous_reason}'. Please re-examine carefully "
            f"and provide a more decisive assessment."
        )

    return prompt