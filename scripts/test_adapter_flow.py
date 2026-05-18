"""Test: raw NDJSON → Dynamic Adapter → normalized NDJSON → schema verification."""
import json
import os
import sys
sys.path.insert(0, os.path.abspath("src"))

from src.adapters.buspas_adapter import DynamicAdapter

RAW_PATH = "data/raw.buspas.ndjson"
OUTPUT_PATH = "data/normalised.buspas.ndjson"

def run_test():
    if not os.path.exists(RAW_PATH):
        print(f"❌ Raw file not found: {RAW_PATH}")
        return

    adapter = DynamicAdapter()
    os.makedirs("data", exist_ok=True)

    with open(RAW_PATH, "r") as f_in, open(OUTPUT_PATH, "w") as f_out:
        for i, line in enumerate(f_in, 1):
            if not line.strip(): continue
            try:
                raw = json.loads(line)
                normalized = adapter.adapt(raw)
                f_out.write(json.dumps(normalized) + "\n")
                if i == 1:
                    print("✅ First record adapted successfully:")
                    print(json.dumps(normalized, indent=2))
            except Exception as e:
                print(f"⚠️ Line {i} failed: {e}")

    print(f"\n📄 Adapter flow complete. Output saved to: {OUTPUT_PATH}")
    
    # Quick schema verification
    with open(OUTPUT_PATH, "r") as f:
        first = json.loads(f.readline())
    required = ["record_id", "producer_id", "timestamp", "record_type", "lat", "lon"]
    missing = [k for k in required if k not in first]
    if missing:
        print(f"❌ Schema violation. Missing fields: {missing}")
    else:
        print("✅ Schema verification PASSED. All required fields present.")

if __name__ == "__main__":
    run_test()