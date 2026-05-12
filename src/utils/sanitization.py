"""Utility functions for sanitizing sensitive information.

Prevents accidental leakage of API keys, tokens, or endpoints in logs.

Reference: MAS-DQA Knowledge Base §4 (Security Constraints)
"""
import re


def sanitize_error_message(message: str, max_length: int = 200) -> str:
    """
    Sanitize an error message to prevent leaking sensitive information.
    
    Removes or truncates:
    - API keys
    - Authorization headers
    - Full URLs with tokens
    - Long stack traces
    
    Args:
        message: Raw error message string
        max_length: Maximum length of sanitized output
        
    Returns:
        Safe, truncated error message suitable for logging
    """
    # Remove common sensitive patterns
    sanitized = re.sub(r'(?i)(api[_-]?key|token|secret|authorization)\s*[=:]\s*\S+', r'\1=[REDACTED]', message)
    
    # Truncate to safe length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."
    
    return sanitized