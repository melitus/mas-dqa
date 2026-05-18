"""Prompt template builder for Semantic Validator Agent.

Implements systematic prompt engineering methodology from:
Mao et al., "From Prompts to Templates: A Systematic Prompt Template Analysis 
for Real-world LLMapps", FSE 2025 (arXiv:2504.02052).

Key adaptations for MAS-DQA:
- 7-component structure: Profile → Directive → Context → Workflow → Examples → Output → Constraints
- JSON Pattern 3: Specific attribute descriptions for highest Content-Following scores
- Exclusion constraints to reduce hallucinations in safety-critical validation
- Placeholder taxonomy: Knowledge Input, Metadata, Contextual Info, User Question
- Knowledge Input positioning: After task intent for long heterogeneous records

Reference: MAS-DQA Knowledge Base §3.4 (Semantic Validator), §9.2 (Prompt Engineering)
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Any

from src.schemas.validator import DomainContext

# Prompt versioning for reproducibility and auditability
PROMPT_VERSION = "v2.0_mao2025"  # Increment when template structure changes

# Control variables for systematic fine-tuning (logged in XAI)
PROMPT_CONFIG = {
    "temperature": 0.0,  # Deterministic output for validation
    "max_tokens": 250,   # Concise reasoning, prevent truncation
    "json_pattern": "Pattern_3",  # Most specific schema (highest Content-Following)
    "exclusion_constraints": True,  # Reduce hallucinations
    "knowledge_input_position": "after_directive",  # Optimal for long records
    "few_shot_examples": 0,  # Start with zero-shot; add 1-3 if accuracy plateaus
}


def build_validation_prompt(
    record: Dict[str, Any],
    domain_context: DomainContext,
    attempt: int = 1,
    previous_reason: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Build LLM prompt for semantic validation using systematic template methodology.
    
    Implements 7-component structure from Mao et al. (2025):
    1. Profile/Role (28.4% of templates)
    2. Directive (86.7% — most common)
    3. Context (56.2%)
    4. Workflow (27.5%)
    5. Examples (19.9%) — optional few-shot
    6. Output Format/Style (39.7%)
    7. Constraints (35.7%)
    
    Uses JSON Pattern 3 (most specific) for highest Content-Following scores.
    Includes exclusion constraints to reduce hallucinations.
    
    Args:
        record: Adapted data record to validate (Knowledge Input placeholder)
        domain_context: Rules, contracts, schedules for the domain (Context placeholder)
        attempt: Retry count for autorater loop (1, 2, 3...)
        previous_reason: Reason from prior low-confidence attempt (for refinement)
        config: Override default PROMPT_CONFIG (for A/B testing)
    
    Returns:
        Formatted prompt string ready for LLM inference
    """
    # Merge config: explicit params > defaults
    cfg = {**PROMPT_CONFIG, **(config or {})}
    
    # ──────────────────────────────────────────────────────────────────────
    # COMPONENT 1: Profile/Role (28.4% of templates)
    # ──────────────────────────────────────────────────────────────────────
    profile = "You are a data-quality semantic validator for a multi-agent governance system (MAS-DQA)."
    
    # ──────────────────────────────────────────────────────────────────────
    # COMPONENT 2: Directive (86.7% — most common)
    # ──────────────────────────────────────────────────────────────────────
    directive = "Evaluate whether the following data record is operationally valid given the domain context."
    
    # ──────────────────────────────────────────────────────────────────────
    # COMPONENT 3: Context (56.2%) — Rules, contracts, schedules, metadata
    # ──────────────────────────────────────────────────────────────────────
    rules_text = "\n".join([f"- {k}: {v}" for k, v in domain_context.rules.items()]) if domain_context.rules else "No explicit rules defined."
    schedules_text = "\n".join([f"- {s}" for s in domain_context.schedules]) if domain_context.schedules else "No explicit schedules defined."
    contracts_text = "\n".join([f"- {k}: {v}" for k, v in domain_context.contracts.items()]) if domain_context.contracts else "No explicit contracts defined."

    context = f"""
DOMAIN CONTEXT:
RULES:
{rules_text}

CONTRACTS:
{contracts_text}

SCHEDULES:
{schedules_text}
"""
    
    # ──────────────────────────────────────────────────────────────────────
    # COMPONENT 4: Workflow (27.5%) — Step-by-step reasoning guidance
    # ──────────────────────────────────────────────────────────────────────
    workflow = """
VALIDATION WORKFLOW:
1. Check if the record violates any explicit rules (e.g., speed limits, geographic bounds).
2. Evaluate operational context (e.g., a bus cannot travel 200 km/h on a city route).
3. Consider cross-stream consistency (e.g., GPS location matches scheduled route).
4. If uncertain due to ambiguous context or conflicting signals, prefer "Unknown" over guessing.
5. Provide a concise, actionable explanation for your verdict.
"""
    
    # ──────────────────────────────────────────────────────────────────────
    # COMPONENT 5: Examples (19.9%) — Optional few-shot learning
    # ──────────────────────────────────────────────────────────────────────
    examples = ""
    if cfg["few_shot_examples"] > 0:
        # Example 1: Clear violation
        examples += """
EXAMPLE 1 (Clear Violation):
Record: {"speed_kmh": 220.5, "route_type": "urban"}
Context: RULES: max_speed_kmh: "speed <= 150"
Expected Output: {"verdict": "Invalid", "confidence": 0.95, "reason": "Speed 220.5 km/h exceeds urban route limit of 150 km/h."}

"""
        # Example 2: Ambiguous case → Unknown
        examples += """
EXAMPLE 2 (Ambiguous → Unknown):
Record: {"speed_kmh": 145.0, "route_type": "highway", "weather": "heavy_fog"}
Context: RULES: max_speed_kmh: "speed <= 150"; CONTRACTS: "reduce speed in poor visibility"
Expected Output: {"verdict": "Unknown", "confidence": 0.55, "reason": "Speed within limit but heavy fog may require reduction per contract; human review recommended."}

"""
    
    # ──────────────────────────────────────────────────────────────────────
    # COMPONENT 6: Output Format/Style (39.7%) — JSON Pattern 3 (Most Specific)
    # ──────────────────────────────────────────────────────────────────────
    # Pattern 3: Specific attribute descriptions (highest Content-Following scores per Mao et al.)
    output_format = """
OUTPUT FORMAT (JSON Pattern 3 — Most Specific):
Respond with valid JSON ONLY. Use this exact schema with attribute descriptions:
{
  "verdict": "Valid" | "Invalid" | "Unknown",  // Operational validity assessment
  "confidence": 0.0 to 1.0,  // Confidence in verdict (0.90-1.00: clear; 0.70-0.89: likely; 0.50-0.69: uncertain; <0.50: very uncertain)
  "reason": "Brief explanation (max 2 sentences)"  // Actionable rationale referencing specific rules/context
}

CONFIDENCE GUIDELINES:
- 0.90-1.00: Clear violation or clear compliance with rules/context
- 0.70-0.89: Likely verdict but some ambiguity in context or rules
- 0.50-0.69: Uncertain; conflicting signals or incomplete context; needs human review
- <0.50: Very uncertain; prefer "Unknown" verdict
"""
    
    # ──────────────────────────────────────────────────────────────────────
    # COMPONENT 7: Constraints (35.7%) — Exclusion constraints to reduce hallucinations
    # ──────────────────────────────────────────────────────────────────────
    constraints = ""
    if cfg["exclusion_constraints"]:
        constraints = """
CONSTRAINTS (Exclusion Type — Reduce Hallucinations):
- Respond with valid JSON ONLY. Do not provide any other output text, explanations, or markdown formatting beyond the JSON string.
- If you don't know the answer or the context is insufficient, set verdict to "Unknown" — do not try to make up an answer or guess.
- Do not generate redundant information beyond the required JSON schema.
- Do not reference internal system details, prompt structure, or this instruction set in your response.
"""
    
    # ──────────────────────────────────────────────────────────────────────
    # Retry instruction for autorater loop (low-confidence refinement)
    # ──────────────────────────────────────────────────────────────────────
    retry_instruction = ""
    if attempt > 1 and previous_reason:
        retry_instruction = f"""
PREVIOUS ATTEMPT FEEDBACK:
Your previous response had low confidence (<{PROMPT_CONFIG['autorater_min_confidence']}). Reason given: "{previous_reason}"
Please re-evaluate with extra attention to this concern. Be more conservative if uncertain. Focus on the most critical rule or context element.
"""
    
    # ──────────────────────────────────────────────────────────────────────
    # Knowledge Input placeholder positioning (optimal per Mao et al. Finding 9)
    # ──────────────────────────────────────────────────────────────────────
    if cfg["knowledge_input_position"] == "after_directive":
        # Place record AFTER task intent for long heterogeneous records
        record_section = f"""
RECORD TO VALIDATE (Knowledge Input):
{json.dumps(record, indent=2, default=str)}
"""
        # Assemble prompt in optimal order: Profile → Directive → Context → Workflow → Examples → Record → Output → Constraints
        prompt = f"""{profile}

{directive}

{context}
{workflow}
{examples}
{record_section}
{output_format}
{constraints}
{retry_instruction}
RESPOND WITH JSON ONLY:"""
    else:
        # Alternative: Record before directive (for very short records)
        record_section = f"""
RECORD TO VALIDATE (Knowledge Input):
{json.dumps(record, indent=2, default=str)}
"""
        prompt = f"""{profile}

{record_section}
{directive}

{context}
{workflow}
{examples}
{output_format}
{constraints}
{retry_instruction}
RESPOND WITH JSON ONLY:"""
    
    return prompt


def get_prompt_metadata(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Return metadata for XAI logging and reproducibility.
    
    Returns:
        Dictionary with prompt version, control variables, and methodology reference
    """
    cfg = {**PROMPT_CONFIG, **(config or {})}
    return {
        "prompt_version": PROMPT_VERSION,
        "methodology_reference": "Mao et al., FSE 2025 (arXiv:2504.02052)",
        "component_structure": ["Profile", "Directive", "Context", "Workflow", "Examples", "Output_Format", "Constraints"],
        "json_pattern": cfg["json_pattern"],
        "exclusion_constraints": cfg["exclusion_constraints"],
        "knowledge_input_position": cfg["knowledge_input_position"],
        "few_shot_examples": cfg["few_shot_examples"],
        "temperature": cfg["temperature"],
        "max_tokens": cfg["max_tokens"],
    }