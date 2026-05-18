"""Pytest configuration for MAS-DQA tests.

Adds project root to sys.path so imports like `from src.validator.prompt` work.
"""
import sys
import os

# Add project root to sys.path for all tests
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Optional: Set asyncio mode for all async tests
pytest_plugins = ["pytest_asyncio"]