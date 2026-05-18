"""XAI (Explainable AI) logger for MAS-DQA Semantic Validator.

Logs validation decisions with full audit trail, including:
- Prompt version and control variables (for reproducibility)
- Input record snapshot (sanitized)
- LLM response raw text + parsed output
- Confidence calibration metrics
- Autorater loop history

Reference: MAS-DQA Knowledge Base §3.4 (Semantic Validator), §9.3 (Auditability)
"""
from __future__ import annotations

import json
import logging
import time
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from src.schemas.validator import ValidatorInput, ValidatorOutput

logger = logging.getLogger(__name__)


class XAILogger:
    """
    Explainable AI logger for Semantic Validator decisions.
    
    Logs every validation decision with:
    - Full input context (sanitized for privacy)
    - Prompt metadata (version, control variables)
    - LLM raw response + parsed output
    - Confidence calibration data
    - Autorater retry history
    - Timestamps for latency tracking
    
    Designed for compliance auditing, debugging, and prompt fine-tuning.
    """
    
    def __init__(self, log_file: Optional[str] = None, callback: Optional[Callable] = None):
        """
        Initialize XAI logger.
        
        Args:
            log_file: Optional file path for persistent logging (JSONL format)
            callback: Optional callback function for real-time logging (e.g., to monitoring system)
        """
        self.log_file = log_file
        self.callback = callback
        self._log_count = 0
        
    def set_callback(self, callback: Callable[['ValidatorOutput', 'ValidatorInput'], None]):
        """Register callback for real-time XAI logging."""
        self.callback = callback
        
    def log(
        self,
        output: 'ValidatorOutput',
        input_: 'ValidatorInput',
        prompt_metadata: Optional[Dict[str, Any]] = None,
        llm_raw_response: Optional[str] = None,
        autorater_history: Optional[list] = None,
        latency_ms: Optional[float] = None
    ):
        """
        Log a validation decision with full audit trail.
        
        Args:
            output: ValidatorOutput with verdict, confidence, reason
            input_: ValidatorInput with record, domain_context, profiler_result
            prompt_metadata: Metadata from get_prompt_metadata() (version, control variables)
            llm_raw_response: Raw LLM response text before parsing
            autorater_history: List of previous attempts if autorater loop was used
            latency_ms: End-to-end validation latency in milliseconds
        """
        log_entry = {
            "timestamp": time.time(),
            "record_id": input_.record.get("record_id", "unknown"),
            "producer_id": input_.record.get("producer_id", "unknown"),
            
            # Input snapshot (sanitized — remove PII if needed)
            "input_record": self._sanitize_record(input_.record),
            "domain_context_summary": {
                "rules_count": len(input_.domain_context.rules) if input_.domain_context else 0,
                "contracts_count": len(input_.domain_context.contracts) if input_.domain_context else 0,
                "schedules_count": len(input_.domain_context.schedules) if input_.domain_context else 0,
            },
            "profiler_result": {
                "deviation_score": input_.profiler_result.deviation_score if input_.profiler_result else None,
                "verdict": input_.profiler_result.verdict if input_.profiler_result else None,
            } if input_.profiler_result else None,
            
            # Prompt metadata (critical for reproducibility + fine-tuning)
            "prompt_metadata": prompt_metadata or {},
            
            # LLM response details
            "llm_raw_response": llm_raw_response[:500] + "..." if llm_raw_response and len(llm_raw_response) > 500 else llm_raw_response,
            "parsed_output": {
                "verdict": output.verdict,
                "confidence": output.confidence,
                "reason": output.reason,
            },
            
            # Autorater loop history (if applicable)
            "autorater_history": autorater_history or [],
            
            # Performance metrics
            "latency_ms": latency_ms,
            
            # Output metadata (from ValidatorOutput)
            "output_metadata": output.metadata,
        }
        
        # Log to file if configured
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, default=str) + "\n")
            except Exception as e:
                logger.error(f"Failed to write XAI log to {self.log_file}: {e}")
        
        # Log to console for debugging
        logger.debug(f"XAI Log #{self._log_count + 1}: {input_.record.get('record_id')} → {output.verdict} (conf: {output.confidence:.2f})")
        
        # Invoke callback if registered (for real-time monitoring)
        if self.callback:
            try:
                self.callback(output, input_)
            except Exception as e:
                logger.error(f"XAI callback failed: {e}")
        
        self._log_count += 1
        
    def _sanitize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize record for logging (remove PII, sensitive fields).
        
        Override this method for domain-specific sanitization rules.
        """
        # Default: keep all fields (customize for production)
        # Example sanitization:
        # sanitized = {k: v for k, v in record.items() if k not in ["user_id", "device_id"]}
        # sanitized["record_id"] = hash(record.get("record_id", ""))  # Anonymize ID
        # return sanitized
        return record  # No sanitization for research prototype
        
    def get_recent_logs(self, n: int = 10) -> list:
        """
        Retrieve recent log entries from file (for debugging/analysis).
        
        Args:
            n: Number of recent entries to return
            
        Returns:
            List of parsed log entries (most recent first)
        """
        if not self.log_file:
            return []
            
        logs = []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
        except FileNotFoundError:
            return []
            
        return logs[-n:][::-1]  # Return most recent first