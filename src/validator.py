# LLM-based semantic reasoning

import json
import litellm
from pydantic import BaseModel

class ValidatorOutput(BaseModel):
    verdict: str          # "Valid" | "Invalid"
    confidence: float     # 0.0 - 1.0
    reason: str

class SemanticValidator:
    def __init__(self, rules: dict):
        self.rules = rules
        self.system_prompt = f"""You are a data quality validator for transit systems.
        Check if the record violates these operational constraints:
        {json.dumps(rules, indent=2)}
        Respond ONLY in JSON format with keys: verdict, confidence (0-1), reason."""

    def evaluate_record(self, record: dict) -> ValidatorOutput:
        user_prompt = f"Record: {json.dumps(record)}\nValidate and return JSON."
        
        response = litellm.completion(
            model="gpt-4o-mini",  # Start cheap & fast
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,  # Deterministic for validation
            max_tokens=150
        )
        
        raw = json.loads(response.choices[0].message.content)
        return ValidatorOutput(**raw)