"""Core Semantic Validator Agent for MAS-DQA.

Implements LLM-based logical and semantic validation with:
- Deterministic output (temperature=0.0)
- Autorater loop for low-confidence results
- Async-safe caching
- Anomaly-based skip optimization
- XAI audit logging

Reference: MAS-DQA Knowledge Base §3.4
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from src.schemas.validator import ValidatorInput, ValidatorOutput

from src.config.thresholds import ValidationThresholds, DEFAULT_THRESHOLDS
from src.config.llm import LLMConfig, DEFAULT_LLM_CONFIG
from src.validator.prompt import build_validation_prompt
from src.validator.cache import AsyncCache
from src.validator.xai import XAILogger
from src.utils.sanitization import sanitize_error_message

logger = logging.getLogger(__name__)


class SemanticValidator:
    """
    Semantic Validator Agent

    Performs LLM-based logical and semantic validation of records
    against domain contracts, rules, and schedules.

    Key behaviors:
    - Uses litellm with temperature=0.0 for deterministic output
    - Enforces structured JSON schema in LLM responses
    - Implements autorater loop when confidence < threshold
    - Caches identical records (LRU) to reduce LLM calls
    - Skips validation if Profiler flags severe anomaly

    Reference: MAS-DQA Knowledge Base §3.4
    """

    def __init__(
        self,
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        max_autorater_retries: Optional[int] = None,
        cache_size: Optional[int] = None,
        temperature: Optional[float] = None,
        thresholds: Optional[ValidationThresholds] = None,
        llm_config: Optional[LLMConfig] = None
    ):
        """
        Initialize the validator.

        Args:
            llm_model: Litellm-compatible model name (overrides config)
            llm_api_key: Optional API key override
            max_autorater_retries: Max retries for low-confidence results
            cache_size: LRU cache size for record validation
            temperature: LLM temperature (default 0.0 for deterministic)
            thresholds: Custom validation thresholds (uses defaults if None)
            llm_config: Full LLM config object (overrides individual params)
        """
        # Merge config: explicit params > llm_config > defaults
        config = llm_config or DEFAULT_LLM_CONFIG
        self.llm_model = llm_model or config.DEFAULT_MODEL
        self.llm_api_key = llm_api_key or config.API_KEY
        self.max_autorater_retries = max_autorater_retries or config.MAX_RETRIES
        self.temperature = temperature if temperature is not None else config.DEFAULT_TEMPERATURE
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        
        # Initialize components
        self._cache = AsyncCache(max_size=cache_size or self.thresholds.DEFAULT_CACHE_SIZE)
        self._xai_logger = XAILogger()

    def set_xai_logger(self, callback: Callable[['ValidatorOutput', 'ValidatorInput'], None]):
        """Register callback for XAI audit logging."""
        self._xai_logger.set_callback(callback)

    # -----------------------------------------------------------------------
    # PUBLIC INTERFACE
    # -----------------------------------------------------------------------

    async def validate(self, input_data: ValidatorInput) -> ValidatorOutput:
        """
        Validate a record semantically (cached).

        If a profiler result is provided with deviation_score < threshold,
        validation is skipped and an Invalid verdict is returned immediately.

        Returns ValidatorOutput with verdict, confidence, reason, metadata.
        """
        from src.schemas.validator import ValidatorOutput
        
        cache_key = AsyncCache.make_key(
            input_data.record,
            input_data.domain_context.model_dump_compat(),
            input_data.profiler_result.deviation_score if input_data.profiler_result else None
        )

        # Check cache first
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit for key %s", cache_key[:16])
            return cached

        # Compute and cache result
        result = await self._validate_uncached(input_data)
        self._cache.set(cache_key, result)

        return result

    # -----------------------------------------------------------------------
    # INTERNALS
    # -----------------------------------------------------------------------

    async def _validate_uncached(self, input_data: ValidatorInput) -> ValidatorOutput:
        """Core validation logic (no caching)."""
        from src.schemas.validator import ValidatorOutput
        
        # ---- Anomaly-based skip optimization (KB §4) ------------------------
        if (
            input_data.profiler_result
            and input_data.profiler_result.deviation_score < self.thresholds.ANOMALY_SKIP_THRESHOLD
        ):
            logger.info(
                "Skipping validation: profiler deviation=%.2f < %.2f threshold (severe anomaly)",
                input_data.profiler_result.deviation_score,
                self.thresholds.ANOMALY_SKIP_THRESHOLD,
            )
            result = ValidatorOutput(
                verdict="Invalid",
                confidence=0.0,
                reason="Skipped — severe statistical anomaly detected by Profiler",
                metadata={
                    "skipped_due_to_anomaly": True,
                    "profiler_deviation": input_data.profiler_result.deviation_score,
                },
            )
            self._xai_logger.log(result, input_data)
            return result

        # ---- LLM-based semantic validation with autorater loop ------------
        last_result: Optional[ValidatorOutput] = None
        for attempt in range(self.max_autorater_retries + 1):
            try:
                result = await self._call_llm(
                    record=input_data.record,
                    domain_context=input_data.domain_context,
                    attempt=attempt + 1,
                    previous_reason=last_result.reason if last_result and last_result.confidence < self.thresholds.AUTORATER_MIN_CONFIDENCE else None,
                )
            except Exception as exc:
                safe_msg = sanitize_error_message(str(exc))
                logger.error("LLM call failed (attempt %d/%d): %s",
                             attempt + 1, self.max_autorater_retries + 1, safe_msg)
                if attempt >= self.max_autorater_retries:
                    result = ValidatorOutput(
                        verdict="Unknown",
                        confidence=0.0,
                        reason=f"LLM evaluation failed after {attempt + 1} attempts: {safe_msg}",
                        metadata={"error": safe_msg},
                    )
                    break
                continue

            if result.confidence >= self.thresholds.AUTORATER_MIN_CONFIDENCE:
                logger.debug(
                    "Validation complete (attempt %d): verdict=%s confidence=%.2f",
                    attempt + 1, result.verdict, result.confidence,
                )
                last_result = result
                break

            logger.info(
                "Low confidence (%.2f < %.2f) — retrying with enhanced context (%d/%d)",
                result.confidence, self.thresholds.AUTORATER_MIN_CONFIDENCE,
                attempt + 1, self.max_autorater_retries,
            )
            last_result = result

        # Exhausted retries — return last result with annotation
        if last_result:
            result = ValidatorOutput(
                verdict=last_result.verdict,
                confidence=last_result.confidence,
                reason=f"{last_result.reason} [autorater loop exhausted]",
                metadata={**last_result.metadata, "autorater_retries": self.max_autorater_retries},
            )

        # XAI logging
        self._xai_logger.log(result, input_data)
        return result

    async def _call_llm(
        self,
        record: Dict[str, Any],
        domain_context: 'DomainContext',
        attempt: int = 1,
        previous_reason: Optional[str] = None
    ) -> 'ValidatorOutput':
        """Call the LLM and parse structured JSON response."""
        from src.schemas.validator import ValidatorOutput, DomainContext
        import litellm
        import time
        import json
        import re
        
        prompt = build_validation_prompt(record, domain_context, attempt, previous_reason)
        start_time = time.time()

        try:
            response = await litellm.acompletion(
                model=self.llm_model,
                api_key=self.llm_api_key,
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
                max_tokens=self.thresholds.MAX_LLM_TOKENS,
            )
            latency_ms = (time.time() - start_time) * 1000
            raw_text = response.choices[0].message.content

            # Extract token usage if available
            usage = getattr(response, "usage", None)
            token_info = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            } if usage else {}

            result = self._parse_response(raw_text)

            # Enrich metadata with performance metrics
            result.metadata.update({
                "llm_latency_ms": round(latency_ms, 2),
                "llm_model": self.llm_model,
                "token_usage": token_info,
                "attempt": attempt,
            })
            return result

        except ImportError:
            logger.error("litellm not installed. Install with: pip install litellm")
            return ValidatorOutput(
                verdict="Unknown",
                confidence=0.0,
                reason="litellm not installed",
                metadata={"error": "missing_dependency"},
            )
        except Exception as exc:
            safe_msg = sanitize_error_message(str(exc))
            logger.error("LLM API error: %s", safe_msg)
            raise

    @staticmethod
    def _parse_response(raw_text: str) -> 'ValidatorOutput':
        """Parse LLM response into ValidatorOutput."""
        from src.schemas.validator import ValidatorOutput
        import json
        import re
        
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
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