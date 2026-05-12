# src/agreement/__init__.py
"""Agreement module for MAS-DQA. Re-exports core logic for clean imports."""
from src.agreement.core import determine_routing_decision, RoutingDecision

__all__ = ["determine_routing_decision", "RoutingDecision"]