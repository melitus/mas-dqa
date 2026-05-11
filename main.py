# MVP pipeline runner

import time, json
from src.profiler import Profiler
from src.validator import SemanticValidator
from src.agreement import check_agreement
import pandas as pd

def run_mvp():
    # Load baseline & stream
    baseline = pd.read_csv("data/buspas_baseline.csv")
    stream = pd.read_csv("data/buspas_stream.csv")[:1000]  # Test slice
    
    profiler = Profiler(baseline)
    validator = SemanticValidator(rules={"max_speed_kmh": 100})
    
    results = {"agree": 0, "conflict": 0, "latency_ms": []}
    
    for _, record in stream.iterrows():
        start = time.time()
        p_out = profiler.evaluate_record(record.to_dict())
        v_out = validator.evaluate_record(record.to_dict())
        decision = check_agreement(p_out, v_out)
        
        results["latency_ms"].append((time.time() - start) * 1000)
        if decision == "AGREE_VALID": results["agree"] += 1
        else: results["conflict"] += 1

    print(f"✅ Agree: {results['agree']} | ❌ Conflict: {results['conflict']}")
    print(f"⏱️ Avg Latency: {sum(results['latency_ms'])/len(results['latency_ms']):.2f}ms")

if __name__ == "__main__":
    run_mvp()