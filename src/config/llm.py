"""LLM configuration for MAS-DQA Semantic Validator.

Centralized config for model selection, API keys, and inference parameters.
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMConfig:
    """Configuration for LLM-based semantic validation."""
    
    # Model selection
    DEFAULT_MODEL: str = "gpt-4o-mini"  # or "mistral-small", "llama-3-8b-instruct"
    
    # API configuration
    API_KEY: Optional[str] = os.getenv("LITELLM_API_KEY")
    API_BASE: Optional[str] = os.getenv("LITELLM_API_BASE")  # For local vLLM/Ollama
    
    # Inference parameters
    DEFAULT_TEMPERATURE: float = 0.0  # Deterministic output for validation
    MAX_TOKENS: int = 250  # Concise reasoning, prevent truncation
    MAX_RETRIES: int = 2  # Autorater loop max attempts
    
    # Performance tuning
    TIMEOUT_SECONDS: int = 30  # Prevent hanging on slow responses
    CACHE_PROMPTS: bool = True  # Cache identical prompts to reduce API calls
    
    # Safety/quality
    REQUIRE_JSON_OUTPUT: bool = True  # Enforce structured output
    EXCLUSION_CONSTRAINTS: bool = True  # Reduce hallucinations


# Default instance for easy import
DEFAULT_LLM_CONFIG = LLMConfig()