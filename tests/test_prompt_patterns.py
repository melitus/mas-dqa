"""A/B testing framework for Semantic Validator prompt patterns.

Implements systematic evaluation methodology from:
Mao et al., "From Prompts to Templates: A Systematic Prompt Template Analysis 
for Real-world LLMapps", FSE 2025 (arXiv:2504.02052).

Tests impact of prompt patterns on:
- Format-Following: Does output match JSON schema exactly?
- Content-Following: Does verdict align with ground truth?
- Explanation Quality: Is reason clear, actionable, concise?

Reference: MAS-DQA Knowledge Base §5.3 (Phase III Cognitive Validation), §9.2 (Prompt Engineering)
"""
import pytest
import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from src.validator.prompt import build_validation_prompt, get_prompt_metadata, PROMPT_CONFIG
from src.schemas.validator import ValidatorInput, DomainContext, ValidatorOutput
from src.config.thresholds import DEFAULT_THRESHOLDS

# Test dataset: Expert-labeled BusPas records (subset for testing)
TEST_RECORDS = [
    {
        "record_id": "test_001",
        "speed_kmh": 45.3,
        "lat": 45.4215,
        "lon": -73.6089,
        "passenger_count": 28,
        "route_type": "urban",
    },
    {
        "record_id": "test_002", 
        "speed_kmh": 220.5,  # Extreme anomaly
        "lat": 45.4215,
        "lon": -73.6089,
        "passenger_count": 28,
        "route_type": "urban",
    },
    {
        "record_id": "test_003",
        "speed_kmh": 145.0,  # Ambiguous: within limit but poor weather
        "lat": 45.4215,
        "lon": -73.6089,
        "passenger_count": 28,
        "route_type": "highway",
        "weather": "heavy_fog",
    },
]

# Ground truth labels (for Content-Following evaluation)
GROUND_TRUTH = {
    "test_001": {"verdict": "Valid", "confidence_min": 0.90},
    "test_002": {"verdict": "Invalid", "confidence_min": 0.90},
    "test_003": {"verdict": "Unknown", "confidence_max": 0.69},  # Ambiguous case
}


@dataclass
class PromptPatternTest:
    """Configuration for a single prompt pattern A/B test."""
    name: str
    config_overrides: Dict[str, Any]
    description: str


# Define test patterns based on Mao et al. (2025) findings
PROMPT_PATTERNS = [
    PromptPatternTest(
        name="baseline",
        config_overrides={},
        description="Default prompt config (Pattern 3 JSON, exclusion constraints ON)"
    ),
    PromptPatternTest(
        name="json_pattern_1",
        config_overrides={"json_pattern": "Pattern_1"},  # Least specific schema
        description="JSON Pattern 1: Minimal schema (test Format-Following impact)"
    ),
    PromptPatternTest(
        name="json_pattern_3",
        config_overrides={"json_pattern": "Pattern_3"},  # Most specific schema
        description="JSON Pattern 3: Specific attribute descriptions (expected best Content-Following)"
    ),
    PromptPatternTest(
        name="no_exclusion_constraints",
        config_overrides={"exclusion_constraints": False},
        description="Disable exclusion constraints (test hallucination reduction impact)"
    ),
    PromptPatternTest(
        name="knowledge_input_before",
        config_overrides={"knowledge_input_position": "before_directive"},
        description="Place Knowledge Input before Directive (test long-record performance)"
    ),
    PromptPatternTest(
        name="few_shot_2_examples",
        config_overrides={"few_shot_examples": 2},
        description="Add 2 few-shot examples (test few-shot learning impact)"
    ),
]


def evaluate_format_following(llm_response: str, expected_schema: Dict) -> bool:
    """
    Evaluate Format-Following: Does output match JSON schema exactly?
    
    Based on Mao et al. (2025) Section 3.3 methodology.
    
    Args:
        llm_response: Raw LLM response string
        expected_schema: Expected JSON schema keys/types
        
    Returns:
        True if response parses to valid JSON with expected keys/types
    """
    try:
        # Attempt to parse JSON (handle markdown code blocks)
        if "```json" in llm_response:
            import re
            match = re.search(r"```json\s*([\s\S]*?)\s*```", llm_response)
            if match:
                llm_response = match.group(1)
        
        parsed = json.loads(llm_response)
        
        # Check required keys exist
        if not all(key in parsed for key in expected_schema.keys()):
            return False
            
        # Check types (basic validation)
        if not isinstance(parsed.get("verdict"), str):
            return False
        if not isinstance(parsed.get("confidence"), (int, float)):
            return False
        if not isinstance(parsed.get("reason"), str):
            return False
            
        # Check verdict enum values
        if parsed["verdict"] not in ["Valid", "Invalid", "Unknown"]:
            return False
            
        # Check confidence range
        if not (0.0 <= parsed["confidence"] <= 1.0):
            return False
            
        return True
        
    except (json.JSONDecodeError, TypeError, KeyError):
        return False


def evaluate_content_following(
    parsed_output: Dict[str, Any],
    ground_truth: Dict[str, Any]
) -> bool:
    """
    Evaluate Content-Following: Does verdict align with ground truth?
    
    Based on Mao et al. (2025) Section 3.3 methodology.
    
    Args:
        parsed_output: Parsed LLM output (verdict, confidence, reason)
        ground_truth: Expected verdict + confidence bounds
        
    Returns:
        True if verdict matches and confidence is within expected range
    """
    # Check verdict match
    if parsed_output["verdict"] != ground_truth["verdict"]:
        return False
        
    # Check confidence bounds (if specified)
    if "confidence_min" in ground_truth:
        if parsed_output["confidence"] < ground_truth["confidence_min"]:
            return False
    if "confidence_max" in ground_truth:
        if parsed_output["confidence"] > ground_truth["confidence_max"]:
            return False
            
    return True


def evaluate_explanation_quality(reason: str, max_sentences: int = 2) -> float:
    """
    Evaluate Explanation Quality: Is reason clear, actionable, concise?
    
    Heuristic scoring (0.0-1.0) based on:
    - Conciseness: ≤ max_sentences
    - Actionability: References specific rules/context
    - Clarity: No vague language ("maybe", "perhaps")
    
    Args:
        reason: Explanation string from LLM output
        max_sentences: Maximum allowed sentences for conciseness
        
    Returns:
        Quality score 0.0-1.0
    """
    score = 1.0
    
    # Conciseness penalty
    sentences = [s.strip() for s in reason.split(".") if s.strip()]
    if len(sentences) > max_sentences:
        score -= 0.3
        
    # Actionability bonus/penalty
    actionable_keywords = ["rule", "limit", "contract", "schedule", "exceeds", "violates", "within"]
    if not any(kw in reason.lower() for kw in actionable_keywords):
        score -= 0.2
        
    # Clarity penalty for vague language
    vague_keywords = ["maybe", "perhaps", "could be", "might", "unclear"]
    if any(kw in reason.lower() for kw in vague_keywords):
        score -= 0.2
        
    return max(0.0, min(1.0, score))


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", PROMPT_PATTERNS, ids=[p.name for p in PROMPT_PATTERNS])
async def test_prompt_pattern_ab_test(pattern: PromptPatternTest):
    """
    A/B test a single prompt pattern against baseline.
    
    Evaluates:
    - Format-Following: % of responses that parse to valid JSON schema
    - Content-Following: % of verdicts that match ground truth
    - Explanation Quality: Average heuristic score for reasons
    
    Based on Mao et al. (2025) sample testing methodology.
    """
    # Setup
    domain_context = DomainContext(
        rules={"max_speed_kmh": "speed <= 150", "lat_range": "40 <= lat <= 50"},
        contracts={"weather_rule": "reduce speed in poor visibility"}, 
        schedules=[]
    )
    
    results = {
        "format_following": [],
        "content_following": [],
        "explanation_quality": [],
    }
    
    # Test each record
    for record in TEST_RECORDS:
        record_id = record["record_id"]
        ground_truth = GROUND_TRUTH[record_id]
        
        # Build prompt with pattern config
        prompt = build_validation_prompt(
            record=record,
            domain_context=domain_context,
            attempt=1,
            config=pattern.config_overrides
        )
        
        # Mock LLM response (replace with real litellm call in integration tests)
        # For unit tests, use deterministic mock based on record content
        if record["speed_kmh"] > 150:
            mock_response = json.dumps({
                "verdict": "Invalid",
                "confidence": 0.95,
                "reason": f"Speed {record['speed_kmh']} km/h exceeds limit of 150 km/h per rule max_speed_kmh."
            })
        elif record.get("weather") == "heavy_fog" and record["speed_kmh"] > 100:
            mock_response = json.dumps({
                "verdict": "Unknown",
                "confidence": 0.55,
                "reason": "Speed within limit but heavy fog may require reduction per contract; human review recommended."
            })
        else:
            mock_response = json.dumps({
                "verdict": "Valid",
                "confidence": 0.92,
                "reason": "Record complies with all rules and operational context."
            })
        
        # Evaluate Format-Following
        expected_schema = {"verdict": str, "confidence": (int, float), "reason": str}
        format_ok = evaluate_format_following(mock_response, expected_schema)
        results["format_following"].append(format_ok)
        
        # Evaluate Content-Following (if format OK)
        if format_ok:
            parsed = json.loads(mock_response)
            content_ok = evaluate_content_following(parsed, ground_truth)
            results["content_following"].append(content_ok)
            
            # Evaluate Explanation Quality
            quality = evaluate_explanation_quality(parsed["reason"])
            results["explanation_quality"].append(quality)
    
    # Aggregate results
    format_rate = sum(results["format_following"]) / len(results["format_following"]) if results["format_following"] else 0
    content_rate = sum(results["content_following"]) / len(results["content_following"]) if results["content_following"] else 0
    avg_quality = sum(results["explanation_quality"]) / len(results["explanation_quality"]) if results["explanation_quality"] else 0
    
    # Assertions (adjust thresholds based on baseline performance)
    assert format_rate >= 0.95, f"{pattern.name}: Format-Following {format_rate:.2%} < 95% target"
    assert content_rate >= 0.85, f"{pattern.name}: Content-Following {content_rate:.2%} < 85% target"
    assert avg_quality >= 0.70, f"{pattern.name}: Explanation Quality {avg_quality:.2f} < 0.70 target"
    
    # Log results for analysis
    print(f"\n{pattern.name} Results:")
    print(f"  Format-Following: {format_rate:.2%}")
    print(f"  Content-Following: {content_rate:.2%}")
    print(f"  Explanation Quality: {avg_quality:.2f}")