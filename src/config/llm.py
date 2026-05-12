"""LLM configuration for MAS-DQA semantic validation.

Reference: MAS-DQA Knowledge Base §3.4 (Semantic Validator)
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for LLM-based validation."""
    
    # Default model (litellm-compatible)
    DEFAULT_MODEL: str = "gpt-4o-mini"
    
    # Deterministic output for validation
    DEFAULT_TEMPERATURE: float = 0.0
    
    # Retry settings
    MAX_RETRIES: int = 3
    RETRY_BACKOFF_SECONDS: float = 1.0
    
    # Response constraints
    MAX_TOKENS: int = 500
    RESPONSE_FORMAT: str = "json"
    
    # Optional API key override (for testing)
    API_KEY: Optional[str] = None


# Default instance
DEFAULT_LLM_CONFIG = LLMConfig()