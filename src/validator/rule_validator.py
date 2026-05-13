"""Simple rule‑based validator for MAS‑DQA.

Implements the same async ``validate`` interface as the mock/LMM validators
so that ``main.py`` can swap it in transparently.

The validator checks a few deterministic business rules that mimic the
semantic checks described in the Knowledge Base:

* ``speed_kmh`` must be ≤ 150 km/h (allows highway speeds)
* ``lat`` must lie in the geographic bounds 40 ≤ lat ≤ 50
* ``lon`` must lie in –80 ≤ lon ≤ –70
* ``passenger_count`` must be non‑negative

If any rule fails the validator returns ``verdict='Invalid'`` with an
explanatory ``reason`` and a ``metadata`` dictionary that pinpoints the
failed rule.  When all checks pass the verdict is ``'Valid'``.
"""

import json
from typing import Dict

from src.schemas.validator import ValidatorInput, ValidatorOutput


class SimpleRuleValidator:
    """Deterministic, rule‑based validator.

    The class mirrors the ``MockValidator`` API (an ``async validate``
    method) but provides meaningful explanations for each failure.
    """

    async def validate(self, input_: ValidatorInput) -> ValidatorOutput:
        rec = input_.record
        # Default to a clean record
        valid = True
        reason = "All checks passed"
        confidence = 0.99  # High confidence for deterministic checks
        failed_rule = None

        # Rule 1: speed limit (tuned for transit: allow highway speeds)
        speed = rec.get("speed_kmh")
        if speed is not None and speed > 150:  # ← Increased from 100 to 150
            valid = False
            reason = f"Speed exceeds limit ({speed} km/h > 150)"  # ← Fixed f-string
            failed_rule = "speed_limit"
            confidence = 0.80

        # Rule 2: latitude bounds
        lat = rec.get("lat")
        if valid and lat is not None and not (40.0 <= lat <= 50.0):
            valid = False
            reason = f"Latitude out of bounds ({lat} not in [40, 50])"
            failed_rule = "lat_bounds"
            confidence = 0.80

        # Rule 3: longitude bounds
        lon = rec.get("lon")
        if valid and lon is not None and not (-80.0 <= lon <= -70.0):
            valid = False
            reason = f"Longitude out of bounds ({lon} not in [-80, -70])"
            failed_rule = "lon_bounds"
            confidence = 0.80

        # Rule 4: passenger count non‑negative
        pax = rec.get("passenger_count")
        if valid and pax is not None and pax < 0:
            valid = False
            reason = f"Negative passenger count ({pax})"
            failed_rule = "passenger_negative"
            confidence = 0.80

        # Assemble metadata for XAI / audit trail
        metadata: Dict[str, str] = {"validator": "simple_rule"}
        if failed_rule:
            metadata["failed_rule"] = failed_rule

        return ValidatorOutput(
            verdict="Valid" if valid else "Invalid",
            confidence=confidence,
            reason=reason,
            metadata=metadata,
        )