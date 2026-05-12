"""XAI logging utilities for Semantic Validator.

Provides callback-based audit logging for verifiable decisions.
Does not implement storage — only defines the interface.

Reference: MAS-DQA Knowledge Base §3 (XAI Log Layer)
"""
import logging
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.schemas.validator import ValidatorOutput, ValidatorInput

logger = logging.getLogger(__name__)


class XAILogger:
    """Callback-based XAI audit logger."""
    
    def __init__(self, callback: Optional[Callable[['ValidatorOutput', 'ValidatorInput'], None]] = None):
        """
        Initialize XAI logger.
        
        Args:
            callback: Function to call with (output, input) for each decision
        """
        self.callback = callback

    def log(self, output: 'ValidatorOutput', input_data: 'ValidatorInput'):
        """Log a validation decision to the audit trail."""
        if not self.callback:
            logger.debug("XAI logger not configured; skipping log")
            return
        
        try:
            self.callback(output, input_data)
        except Exception as e:
            # Never fail validation due to logging errors
            logger.error("XAI log callback failed: %s", str(e)[:100])

    def set_callback(self, callback: Callable[['ValidatorOutput', 'ValidatorInput'], None]):
        """Register or update the logging callback."""
        self.callback = callback