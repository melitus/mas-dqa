"""Core Semantic Validator Agent for MAS-DQA.

Implements LLM-based logical and semantic validation with:
- Deterministic output (temperature=0.0)
- Autorater loop for low-confidence results
- Async-safe caching
- Anomaly-based skip optimization
- XAI audit logging with prompt versioning

Reference: MAS-DQA Knowledge Base §3.4; Mao et al. (2025) prompt methodology
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Optional, Callable, TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from src.schemas.validator import ValidatorInput, ValidatorOutput, DomainContext

from src.config.thresholds import ValidationThresholds, DEFAULT_THRESHOLDS
from src.config.llm import LLMConfig, DEFAULT_LLM_CONFIG
from src.validator.prompt import build_validation_prompt, PROMPT_VERSION
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
    - Enforces structured JSON schema in LLM responses (Pattern 3)
    - Implements autorater loop when confidence < threshold
    - Caches identical records (LRU) to reduce LLM calls
    - Skips validation if Profiler flags severe anomaly
    - Logs all calls to XAI with prompt version + control variables

    Reference: MAS-DQA Knowledge Base §3.4
    """

    def __init__(
        self,
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_api_base: Optional[str] = None,  # For local vLLM/Ollama
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
            llm_api_base: Optional API base URL for local models (vLLM, Ollama)
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
        self.llm_api_base = llm_api_base or config.API_BASE
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
            input_data.domain_context.model_dump_compat() if hasattr(input_data.domain_context, 'model_dump_compat') else {},
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
            self._xai_logger.log(result, input_data, prompt_metadata={"prompt_version": PROMPT_VERSION})
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
                
                # Exponential backoff for rate limits
                if "rate limit" in safe_msg.lower() or "429" in safe_msg:
                    backoff = min(2 ** attempt, 30)  # Cap at 30s
                    logger.warning(f"⚠️  Rate limit hit; backing off for {backoff}s")
                    await asyncio.sleep(backoff)
                    if attempt < self.max_autorater_retries:
                        continue  # Retry after backoff
                
                if attempt >= self.max_autorater_retries:
                    result = ValidatorOutput(
                        verdict="Unknown",
                        confidence=0.0,
                        reason=f"LLM evaluation failed after {attempt + 1} attempts: {safe_msg}",
                        metadata={"error": safe_msg, "prompt_version": PROMPT_VERSION},
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
                metadata={**last_result.metadata, "autorater_retries": self.max_autorater_retries, "prompt_version": PROMPT_VERSION},
            )

        # XAI logging with prompt metadata
        self._xai_logger.log(
            result, 
            input_data, 
            prompt_metadata={"prompt_version": PROMPT_VERSION, "llm_model": self.llm_model}
        )
        return result

    async def _call_llm(
        self,
        record: Dict[str, Any],
        domain_context: DomainContext,
        attempt: int = 1,
        previous_reason: Optional[str] = None
    ) -> ValidatorOutput:
        """Call the LLM and parse structured JSON response."""
        from src.schemas.validator import ValidatorOutput
        import litellm
        
        # Build prompt using Mao et al. (2025) methodology
        prompt = build_validation_prompt(
            record=record,
            domain_context=domain_context,
            attempt=attempt,
            previous_reason=previous_reason,
            config={"temperature": self.temperature}
        )
        
        start_time = time.time()
        
        try:
            # Call LLM via litellm (supports OpenAI, Anthropic, Mistral, local vLLM, etc.)
            response = await litellm.acompletion(
                model=self.llm_model,
                api_key=self.llm_api_key,
                api_base=self.llm_api_base,  # For local models
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a data-quality semantic validator for MAS-DQA. "
                            "Respond with valid JSON only. "
                            'Use keys: "verdict" ("Valid"|"Invalid"|"Unknown"), '
                            '"confidence" (0.0–1.0), "reason" (string).'
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.thresholds.MAX_LLM_TOKENS,
                timeout=self.thresholds.LLM_TIMEOUT_SECONDS if hasattr(self.thresholds, 'LLM_TIMEOUT_SECONDS') else 30,
            )
            
            latency_ms = (time.time() - start_time) * 1000
            raw_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in raw_text:
                match = re.search(r"```json\s*([\s\S]*?)\s*```", raw_text)
                if match:
                    raw_text = match.group(1).strip()
            
            # Parse JSON with robust error handling
            try:
                data = json.loads(raw_text)
            except json.JSONDecodeError as e:
                # Fallback: try to extract JSON-like structure
                json_match = re.search(r'\{[^{}]*"verdict"[^{}]*\}', raw_text, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                    except:
                        data = None
                else:
                    data = None
            
            if not isinstance(data, dict):
                return ValidatorOutput(
                    verdict="Unknown",
                    confidence=0.0,
                    reason=f"Failed to parse LLM response as JSON: {raw_text[:100]}...",
                    metadata={"raw_response": raw_text[:200], "parse_error": True, "prompt_version": PROMPT_VERSION}
                )
            
            # Extract and validate fields
            verdict = data.get("verdict", "Unknown")
            if verdict not in ("Valid", "Invalid", "Unknown"):
                verdict = "Unknown"
            
            try:
                confidence = float(data.get("confidence", 0.0))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
            except (TypeError, ValueError):
                confidence = 0.0
            
            reason = str(data.get("reason", "No reason provided"))[:200]  # Truncate long explanations
            
            # Extract token usage if available
            usage = getattr(response, "usage", None)
            token_info = {}
            if usage:
                token_info = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                }
            
            # Build result with enriched metadata
            metadata = {
                "llm_latency_ms": round(latency_ms, 2),
                "llm_model": self.llm_model,
                "token_usage": token_info,
                "attempt": attempt,
                "prompt_version": PROMPT_VERSION,
            }
            
            return ValidatorOutput(
                verdict=verdict,
                confidence=round(confidence, 3),
                reason=reason,
                metadata=metadata
            )
            
        except litellm.exceptions.RateLimitError:
            logger.warning(f"⚠️  Rate limit hit on attempt {attempt}")
            raise  # Re-raise to trigger exponential backoff in caller
            
        except litellm.exceptions.AuthenticationError:
            logger.error("❌ LLM authentication failed — check API key")
            raise
            
        except litellm.exceptions.Timeout:
            logger.error(f"❌ LLM call timed out after {self.thresholds.LLM_TIMEOUT_SECONDS if hasattr(self.thresholds, 'LLM_TIMEOUT_SECONDS') else 30}s")
            raise
            
        except ImportError:
            logger.error("litellm not installed. Install with: pip install litellm")
            return ValidatorOutput(
                verdict="Unknown",
                confidence=0.0,
                reason="litellm not installed",
                metadata={"error": "missing_dependency", "prompt_version": PROMPT_VERSION}
            )
            
        except Exception as exc:
            safe_msg = sanitize_error_message(str(exc))
            logger.error(f"❌ LLM call failed (attempt {attempt}): {type(exc).__name__}: {safe_msg}")
            raise

    @staticmethod
    def _parse_response(raw_text: str) -> ValidatorOutput:
        """
        Parse LLM response into ValidatorOutput.
        
        Deprecated: Use inline parsing in _call_llm for better error context.
        Kept for backward compatibility.
        """
        from src.schemas.validator import ValidatorOutput
        
        # Try direct JSON parse
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code block
            match = re.search(r"```json\s*([\s\S]*?)\s*```", raw_text)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    data = None
            else:
                # Try loose JSON extraction
                json_match = re.search(r'\{[^{}]*"verdict"[^{}]*\}', raw_text, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                    except:
                        data = None
                else:
                    data = None
        
        if not isinstance(data, dict):
            return ValidatorOutput(
                verdict="Unknown",
                confidence=0.0,
                reason=f"Malformed LLM response: {raw_text[:200]}",
                metadata={"raw_response": raw_text, "parse_error": True}
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
            metadata={"raw_llm_output": raw_text}
        )