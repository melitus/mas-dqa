import json
import sys
from collections import defaultdict

def analyze_xai_logs(log_path="data/xai_log.jsonl", n=100):
    stats = defaultdict(list)
    
    with open(log_path, "r") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            try:
                entry = json.loads(line.strip())
                metadata = entry.get("output_metadata", {})
                
                if "llm_latency_ms" in metadata:
                    stats["latency"].append(metadata["llm_latency_ms"])
                if metadata.get("parse_error"):
                    stats["parse_errors"].append(1)
                if "prompt_version" in metadata:
                    stats["prompt_versions"].append(metadata["prompt_version"])
            except:
                continue
    
    print(f"📊 LLM Performance Analysis (n={min(n, len(stats.get('latency', [])))}):")
    
    if stats["latency"]:
        import numpy as np
        latencies = stats["latency"]
        print(f"   Mean latency: {np.mean(latencies):.0f} ms")
        print(f"   p95 latency:  {np.percentile(latencies, 95):.0f} ms")
        print(f"   Max latency:  {max(latencies):.0f} ms")
    
    if stats["parse_errors"]:
        error_rate = sum(stats["parse_errors"]) / n * 100
        print(f"   JSON parse error rate: {error_rate:.1f}%")
    
    if stats["prompt_versions"]:
        versions = set(stats["prompt_versions"])
        print(f"   Prompt versions used: {versions}")

if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "data/xai_log.jsonl"
    analyze_xai_logs(log_path)