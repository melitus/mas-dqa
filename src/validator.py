"""
MAS-DQA Semantic Validator Agent
=================================
Part of the Multi-Agent System for Data Quality Assurance

This module implements the Semantic Validator Agent, which uses an LLM
to perform logical and semantic validation of data records against
domain-specific contracts, rules, and schedules.

Key features:
- LLM-based reasoning with structured JSON output (litellm, temperature=0.0)
- Autorater loop for low-confidence results (<0.70)
- Caching of identical records to reduce cost/latency
- Optional skip if Profiler flags severe statistical drift
- Pydantic schemas for type-safe I/O
- Full XAI logging for auditability

Reference: MAS-DQA Knowledge Base §3.4, §4, §6
"""

import asyncio
import json
import logging
import hashlib
from typing import Any, Dict, List, Optional, Tuple
from functools import lru_cache

import litellm
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logger = logging.getLogger(__name__)


# ============================================================================
# DATA SCHEMAS (Pydantic BaseModel for all I/O)
# ============================================================================

class ProfilerResult(BaseModel):
    """
    Output from the Profiler Agent.
    Passed to the validator to enable drift-based skipping.
    """
    deviation_score: float = Field(..., ge=0.0, le=1.0)
    drift_detected: bool
    confidence: float = Field(..., ge=0.0, le=1.0)


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
        if v is None:
            return {}
        return v

    @field_validator("schedules", mode="before")
    @classmethod
    def ensure_list(cls, v):
        if v is None:
            return []
        return v


class ValidatorInput(BaseModel):
    """Input to the Semantic Validator."""
    record: Dict[str, Any]
    domain_context: DomainContext
    profiler_result: Optional[ProfilerResult] = None


class ValidatorOutput(BaseModel):
    """
    Output from the Semantic Validator.

    Fields:
        verdict: Valid | Invalid | Unknown
        confidence: 0.0 to 1.0
        reason: Natural-language explanation
        metadata: Additional diagnostic information
    """
    verdict: str = Field(default="Unknown", regex="^(Valid|Invalid|Unknown)$")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = "No evaluation performed"
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# ROUTING ENUM (Referenced in agreement.py / Decision Diamond)
# ============================================================================

class RoutingDecision(str, Enum):
    TRUST = "TRUST"
    JUDGE = "JUDGE"
    QUARANTINE = "QUARANTINE"
    AMBIGUOUS = "AMBIGUOUS"


# ============================================================================
# SEMANTIC VALIDATOR CLASS
# ============================================================================

class SemanticValidator:
    """
    Semantic Validator Agent

    Performs LLM-based logical and semantic validation of records
    against domain contracts, rules, and schedules.

    Key behaviors:
    - Uses litellm with temperature=0.0 for deterministic output
    - Enforces structured JSON schema in LLM responses
    - Implements autorater loop when confidence < 0.70
    - Caches identical records (LRU) to reduce LLM calls
    - Skips validation if Profiler flags severe drift (deviation >= 0.85)

    Reference: MAS-DQA Knowledge Base §3.4
    """

    # Thresholds per Knowledge Base §4 and §6
    SKIP_DRIFT_THRESHOLD: float = 0.85
    AUTORATER_MIN_CONFIDENCE: float = 0.70
    TRUST_MIN_CONFIDENCE: float = 0.85

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        llm_api_key: Optional[str] = None,
        max_autorater_retries: int = 2,
        cache_size: int = 1000,
        temperature: float = 0.0
    ):
        """
        Initialize the validator.

        Args:
            llm_model: Litellm-compatible model name
            llm_api_key: Optional API key override
            max_autorater_retries: Max retries for low-confidence results
            cache_size: LRU cache size for record validation
            temperature: LLM temperature (default 0.0 for deterministic)
        """
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.max_autorater_retries = max_autorater_retries
        self.temperature = temperature

        # Bind LRU cache
        self._validate_with_cache = lru_cache(maxsize=cache_size)(
            self._validate_uncached
        )

    # -----------------------------------------------------------------------
    # PUBLIC INTERFACE
    # -----------------------------------------------------------------------

    async def validate(self, input_data: ValidatorInput) -> ValidatorOutput:
        """
        Validate a record semantically (cached).

        If a profiler result is provided with deviation_score >= 0.85,
        validation is skipped and an Invalid verdict is returned immediately.

        Returns ValidatorOutput with verdict, confidence, reason, metadata.
        """
        try:
            return self._validate_with_cache(
                self._make_cache_key(input_data)
            )
        except TypeError:
            # Unhashable record — fall back to uncached
            logger.warning("Record not hashable; bypassing cache")
            return await self._validate_uncached(input_data)

    # -----------------------------------------------------------------------
    # INTERNALS
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_cache_key(input_data: ValidatorInput) -> Tuple:
        """
        Build a hashable cache key from the input.
        Uses JSON serialization to normalize dicts.
        """
        raw = json.dumps(
            {
                "record": input_data.record,
                "domain_context": input_data.domain_context.dict(),
                "profiler_score": (
                    input_data.profiler_result.deviation_score
                    if input_data.profiler_result
                    else None
                ),
            },
            sort_keys=True,
        )
        return (hashlib.sha256(raw.encode()).hexdigest(),)

    async def _validate_uncached(self, input_data: ValidatorInput) -> ValidatorOutput:
        """
        Core validation logic (no caching).

        1. Check drift skip condition
        2. Run autorater loop if needed
        3. Return structured result
        """
        # ---- Drift-based skip optimization (KB §4) ------------------------
        if (
            input_data.profiler_result
            and input_data.profiler_result.deviation_score >= self.SKIP_DRIFT_THRESHOLD
        ):
            logger.info(
                "Skipping validation: profiler deviation=%.2f >= %.2f threshold",
                input_data.profiler_result.deviation_score,
                self.SKIP_DRIFT_THRESHOLD,
            )
            return ValidatorOutput(
                verdict="Invalid",
                confidence=0.0,
                reason="Skipped — severe statistical drift detected by Profiler",
                metadata={"skipped_due_to_drift": True},
            )

        # ---- LLM-based semantic validation --------------------------------
        for attempt in range(self.max_autorater_retries + 1):
            try:
                result = await self._call_llm(
                    record=input_data.record,
                    domain_context=input_data.domain_context,
                )
            except Exception as exc:
                logger.error("LLM call failed (attempt %d/%d): %s",
                             attempt + 1, self.max_autorater_retries + 1, exc)
                if attempt >= self.max_autorater_retries:
                    return ValidatorOutput(
                        verdict="Unknown",
                        confidence=0.0,
                        reason=f"LLM evaluation failed after {attempt + 1} attempts: {exc}",
                        metadata={"error": str(exc)},
                    )
                continue

            # Autorater loop condition (KB §3.4)
            if result.confidence >= self.AUTORATER_MIN_CONFIDENCE:
                logger.debug(
                    "Validation complete (attempt %d): verdict=%s confidence=%.2f",
                    attempt + 1, result.verdict, result.confidence,
                )
                return result

            logger.info(
                "Low confidence (%.2f < %.2f) — retrying (%d/%d)",
                result.confidence, self.AUTORATER_MIN_CONFIDENCE,
                attempt + 1, self.max_autorater_retries,
            )

        # Exhausted retries — return last result with annotation
        return ValidatorOutput(
            verdict=result.verdict,
            confidence=result.confidence,
            reason=f"{result.reason} [autorater loop exhausted]",
            metadata={**result.metadata, "autorater_retries": self.max_autorater_retries},
        )

    async def _call_llm(
        self,
        record: Dict[str, Any],
        domain_context: DomainContext
    ) -> ValidatorOutput:
        """
        Call the LLM and parse structured JSON response.
        """
        prompt = self._build_prompt(record, domain_context)

        try:
            response = await litellm.acompletion(
                model=self.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a data-quality semantic validator. "
                            "Respond with valid JSON only. "
                            'Use keys: "verdict" ("Valid"|"Invalid"|"Unknown"), '
                            '"confidence" (0.0–1.0), "reason" (string).'
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=500,
            )
            raw_text = response.choices[0].message.content
            return self._parse_response(raw_text)
        except litellm.APIError as exc:
            logger.error("LLM API error: %s", exc)
            raise

    # -----------------------------------------------------------------------
    # PROMPT ENGINEERING
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_prompt(
        record: Dict[str, Any],
        domain_context: DomainContext
    ) -> str:
        """
        Build a deterministic prompt for the validator LLM.

        Covers: contracts, rules, schedule constraints.
        Explicitly references KB §3.4 requirements.
        """
        record_block = json.dumps(record, indent=2)

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

        return (
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

    @staticmethod
    def _parse_response(raw_text: str) -> ValidatorOutput:
        """
        Parse LLM response into ValidatorOutput.
        Robust against malformed JSON (returns Unknown).
        """
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            # Attempt to extract JSON from markdown code blocks
            import re
            match = re.search(r"```json\s*([\s\S]*?)\s*```", raw_text)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    data = None
            else:
                data = None

        if not isinstance(data, dict):
            return ValidatorOutput(
                verdict="Unknown",
                confidence=0.0,
                reason=f"Malformed LLM response: {raw_text[:200]}",
                metadata={"raw_response": raw_text},
            )

        verdict = data.get("verdict", "Unknown")
        if verdict not in ("Valid", "Invalid", "Unknown"):
            verdict = "Unknown"

        try:
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.0

        reason = str(data.get("reason", "No reason provided"))

        return ValidatorOutput(
            verdict=verdict,
            confidence=confidence,
            reason=reason,
            metadata={"raw_llm_output": raw_text},
        )


# ============================================================================
# DECISION DIAMOND (Agreement Logic)
# ============================================================================

def agreement_logic(
    profiler_result: ProfilerResult,
    validator_output: ValidatorOutput
) -> RoutingDecision:
    """
    Decision diamond per MAS-DQA Knowledge Base (§3, slide 10).

    Routes to:
      TRUST    — high confidence + valid + significant drift
      JUDGE    — low confidence OR invalid verdict
      QUARANTINE — severe drift but validator unsure
      AMBIGUOUS — everything else → escalates to JUDGE
    """
    deviation = profiler_result.deviation_score
    confidence = validator_output.confidence
    verdict = validator_output.verdict

    # Explicit TRUST path
    if (
        deviation >= SemanticValidator.TRUST_MIN_CONFIDENCE
        and confidence >= SemanticValidator.TRUST_MIN_CONFIDENCE
        and verdict == "Valid"
    ):
        return RoutingDecision.TRUST

    # Explicit JUDGE path (any red flag)
    if (
        deviation < 0.50
        or confidence < 0.50
        or verdict == "Invalid"
    ):
        return RoutingDecision.JUDGE

    # Severe drift + ambiguous validator → quarantine
    if deviation >= SemanticValidator.SKIP_DRIFT_THRESHOLD and verdict == "Unknown":
        return RoutingDecision.QUARANTINE

    # Fallback
    return RoutingDecision.AMBIGUOUS


# ============================================================================
# UNIT TESTS
# ============================================================================

import unittest
from unittest.mock import AsyncMock, patch


class _TestValidator(unittest.IsolatedAsyncioTestCase):
    """Async-capable unit tests for SemanticValidator."""

    async def asyncSetUp(self):
        self.validator = SemanticValidator(llm_model="test-model")

    # --- caching ---
    async def test_cache_key_is_deterministic(self):
        inp = ValidatorInput(
            record={"temp": 22.0},
            domain_context=DomainContext(rules={"r1": "temp > 0"}),
        )
        k1 = SemanticValidator._make_cache_key(inp)
        k2 = SemanticValidator._make_cache_key(inp)
        # Same content → same hash
        self.assertEqual(k1, k2)

        inp2 = ValidatorInput(
            record={"temp": 99.0},
            domain_context=DomainContext(rules={"r1": "temp > 0"}),
        )
        k3 = SemanticValidator._make_cache_key(inp2)
        self.assertNotEqual(k1, k3)

    async def test_drift_skip_returns_invalid(self):
        result = await self.validator.validate(
            ValidatorInput(
                record={},
                domain_context=DomainContext(),
                profiler_result=ProfilerResult(
                    deviation_score=0.90, drift_detected=True, confidence=0.95
                ),
            )
        )
        self.assertEqual(result.verdict, "Invalid")
        self.assertEqual(result.metadata.get("skipped_due_to_drift"), True)

    async def test_normal_validation_calls_llm(self):
        mock_response = AsyncMock()
        mock_response.choices = [
            AsyncMock(
                message=AsyncMock(
                    content='{"verdict":"Valid","confidence":0.92,"reason":"within spec"}'
                )
            )
        ]

        with patch("litellm.acompletion", return_value=mock_response):
            result = await self.validator.validate(
                ValidatorInput(
                    record={"value": 42},
                    domain_context=DomainContext(rules={"v1": "value < 100"}),
                )
            )

        self.assertEqual(result.verdict, "Valid")
        self.assertAlmostEqual(result.confidence, 0.92)

    async def test_autorater_loop_retries_on_low_confidence(self):
        # First call → low confidence, second → high
        mock_resp_low = AsyncMock()
        mock_resp_low.choices = [AsyncMock(message=AsyncMock(
            content='{"verdict":"Valid","confidence":0.40,"reason":"uncertain"}'
        ))]
        mock_resp_ok = AsyncMock()
        mock_resp_ok.choices = [AsyncMock(message=AsyncMock(
            content='{"verdict":"Valid","confidence":0.88,"reason":"confirmed"}'
        ))]

        call_count = 0
        original_acompletion = litellm.acompletion

        async def mock_completion(*a, **kw):
            nonlocal call_count
            call_count += 1
            return mock_resp_ok if call_count > 1 else mock_resp_low

        with patch("litellm.acompletion", side_effect=mock_completion):
            result = await self.validator.validate(
                ValidatorInput(
                    record={"x": 1},
                    domain_context=DomainContext(),
                )
            )

        self.assertAlmostEqual(result.confidence, 0.88)
        self.assertIn("confirmed", result.reason)

    async def test_malformed_json_returns_unknown(self):
        mock_response = AsyncMock()
        mock_response.choices = [
            AsyncMock(message=AsyncMock(content="totally not json"))
        ]
        with patch("litellm.acompletion", return_value=mock_response):
            result = await self.validator.validate(
                ValidatorInput(record={}, domain_context=DomainContext())
            )
        self.assertEqual(result.verdict, "Unknown")
        self.assertLess(result.confidence, 0.01)

    async def test_api_error_after_retries_returns_unknown(self):
        from litellm import APIError
        with patch("litellm.acompletion", side_effect=APIError("timeout")):
            result = await self.validator.validate(
                ValidatorInput(record={}, domain_context=DomainContext())
            )
        self.assertEqual(result.verdict, "Unknown")
        self.assertIn("failed", result.reason.lower())


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        # --- Example: BusPas IoT sensor record ---
        validator = SemanticValidator(llm_model="gpt-4o-mini")

        buspas_record = {
            "producer_id": "sensor-bus-142",
            "timestamp": "2025-05-11T12:34:56Z",
            "latitude": 45.5017,
            "longitude": -73.5673,
            "passenger_count": 34,
            "speed_kmh": 42.0,
            "door_status": "closed",
            "fuel_level_pct": 67,
        }

        buspas_context = DomainContext(
            rules={
                "lat_range": "latitude between 45.4 and 45.6",
                "lon_range": "longitude between -73.7 and -73.4",
                "passenger_positive": "passenger_count >= 0",
                "speed_limit": "speed_kmh between 0 and 80",
            },
            contracts={
                "door_must_close_at_speed": (
                    "If speed_kmh > 5 then door_status == 'closed'"
                ),
                "fuel_reporting": "fuel_level_pct must not be null",
            },
            schedules=[
                ScheduleEntry(
                    name="rush-hour-window",
                    condition="hour in [7..9, 16..18]",
                    priority=1,
                )
            ],
        )

        result = await validator.validate(
            ValidatorInput(
                record=buspas_record,
                domain_context=buspas_context,
                # profiler_result=ProfilerResult(0.1, False, 0.99),  # optional
            )
        )

        print("\n=== Validator Output ===")
        print(f"Verdict:   {result.verdict}")
        print(f"Confidence:{result.confidence:.2f}")
        print(f"Reason:    {result.reason}")
        print(f"Metadata:  {result.metadata}")

    asyncio.run(main())