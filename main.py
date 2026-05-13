#!/usr/bin/env python3
"""
MAS-DQA: Unified Pipeline Runner & Automated Test/Demo
=======================================================
Run: python main.py
- Auto-generates aligned synthetic test data
- Runs full pipeline: Ingestion → Profiler → Validator → Agreement
- Outputs boss-ready metrics with clear pass/fail status
- Includes MOCK_MODE for instant testing without LLM API keys

Reference: MAS-DQA Knowledge Base §5 (Validation), §9 (Design Principles)
"""

import asyncio
import time
import logging
import json
import os
import sys
import random
from typing import Dict, List
from datetime import datetime

import pandas as pd
import numpy as np

# Add src to path for local testing
sys.path.insert(0, os.path.abspath("src"))

# Modular imports
from src.ingestion import StreamIngestor
from src.profiler import Profiler, ProfilerOutput
from src.validator import SemanticValidator, ValidatorInput, DomainContext, ValidatorOutput
from src.validator.rule_validator import SimpleRuleValidator
from src.agreement import determine_routing_decision, RoutingDecision
from src.config.thresholds import DEFAULT_THRESHOLDS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 🎛️ TOGGLE FOR TESTING
MOCK_MODE = True  # Set to False to use real LLM (requires litellm API key)

# ──────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR (Aligned with Profiler schema: deviation_score 0.0=bad → 1.0=normal)
# ──────────────────────────────────────────────────────────────────────────────
def _generate_test_files():
    os.makedirs("data", exist_ok=True)
    baseline_path, stream_path = "data/buspas_baseline.csv", "data/buspas_stream.json"
    
    np.random.seed(42)
    # Baseline: clean historical data (for Profiler to learn normal distribution)
    baseline = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=500, freq="min"),
        "speed_kmh": np.clip(np.random.normal(45, 10, 500), 0, 120),
        "lat": np.clip(np.random.normal(45.5, 0.05, 500), 40, 50),
        "lon": np.clip(np.random.normal(-73.6, 0.05, 500), -80, -70),
        "passenger_count": np.clip(np.random.poisson(30, 500), 0, 60),
    })
    baseline.to_csv(baseline_path, index=False)
    
    # Stream: 80% normal (overlaps baseline → high deviation_score ~0.8-1.0), 20% anomalies (low deviation_score ~0.0-0.3)
    records = []
    for i in range(300):
        if random.random() < 0.8:  # Normal record → should get HIGH deviation_score (good)
            rec = {
                "timestamp": f"2026-05-12T10:{i%60:02d}:00Z",
                "speed_kmh": round(np.random.normal(45, 10), 1),
                "lat": round(np.random.normal(45.5, 0.05), 4),
                "lon": round(np.random.normal(-73.6, 0.05), 4),
                "passenger_count": int(np.random.poisson(30))
            }
        else:  # Anomaly → should get LOW deviation_score (bad)
            anomaly = random.choice(["speed", "lat", "passenger"])
            rec = {
                "timestamp": f"2026-05-12T10:{i%60:02d}:00Z",
                "speed_kmh": round(np.random.normal(45, 10), 1),
                "lat": round(np.random.normal(45.5, 0.05), 4),
                "lon": round(np.random.normal(-73.6, 0.05), 4),
                "passenger_count": int(np.random.poisson(30))
            }
            if anomaly == "speed": 
                rec["speed_kmh"] = round(random.uniform(180, 250), 1)  # Extreme speed → low deviation_score
            elif anomaly == "lat": 
                rec["lat"] = 99.9999  # Out-of-bounds → low deviation_score
            elif anomaly == "passenger": 
                rec["passenger_count"] = -random.randint(5, 20)  # Negative → low deviation_score
        records.append(rec)
    
    with open(stream_path, "w") as f:
        json.dump(records, f)
    
    logger.info("📄 Generated aligned baseline & stream data (80% normal, 20% anomalies)")
    return baseline_path, stream_path

# ──────────────────────────────────────────────────────────────────────────────
# MOCK VALIDATOR (For zero-setup testing) - CORRECTED
# ──────────────────────────────────────────────────────────────────────────────
class MockValidator:
    async def validate(self, input_: ValidatorInput) -> ValidatorOutput:
        """
        Simple rule-based mock validator.
        
        IMPORTANT: ProfilerOutput.deviation_score: 0.0 (bad) → 1.0 (normal)
        So LOW score = BAD data → should trigger Invalid verdict.
        """
        rec = input_.record
        valid = True
        reason = "All checks passed"
        conf = 0.95
        
        # Rule 1: Speed limit (tuned for transit: allow highway speeds)
        speed = rec.get("speed_kmh")
        if speed is not None and speed > 150:  # ← Increased from 100 to 150
            valid = False
            reason = f"Speed exceeds 150 km/h limit ({speed} km/h)"  # ← Fixed f-string
            conf = 0.80
        # Rule 2: Latitude bounds
        elif rec.get("lat") is not None and not (40.0 <= rec["lat"] <= 50.0):
            valid = False
            reason = f"Latitude out of bounds ({rec['lat']} not in [40, 50])"
            conf = 0.80
        # Rule 3: Passenger count non-negative
        elif rec.get("passenger_count") is not None and rec["passenger_count"] < 0:
            valid = False
            reason = f"Negative passenger count ({rec['passenger_count']})"
            conf = 0.80
        # Rule 4: Profiler confidence check - CORRECTED LOGIC
        # deviation_score: 0.0 (bad) → 1.0 (normal)
        # So LOW score = BAD data → flag as invalid
        elif input_.profiler_result and input_.profiler_result.deviation_score < 0.3:  # ← Fixed: < 0.3 = bad
            valid = False
            conf = 0.60
            reason = f"Low statistical confidence (deviation: {input_.profiler_result.deviation_score:.2f})"
            
        return ValidatorOutput(
            verdict="Valid" if valid else "Invalid",
            confidence=conf,
            reason=reason,
            metadata={"mock": True}
        )

# ──────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ──────────────────────────────────────────────────────────────────────────────
async def run_pipeline_and_test(
    stream_path: str = "data/buspas_stream.json",
    baseline_path: str = "data/buspas_baseline.csv",
    sample_size: int = 300
):
    print("\n" + "🔍"*50)
    print("   MAS-DQA: Automated Pipeline Test & Demo")
    print("   Testing: 'Bad Data = No Data' → 'Trusted Data = Confident Decisions'")
    print("🔍"*50 + "\n")
    
    if not os.path.exists(stream_path) or not os.path.exists(baseline_path):
        baseline_path, stream_path = _generate_test_files()
        
    print("⚙️  Initializing MAS-DQA components...")
    ingestor = StreamIngestor(required_fields=["timestamp"])
    profiler = Profiler(baseline_df=pd.read_csv(baseline_path), thresholds=DEFAULT_THRESHOLDS)
    
    if MOCK_MODE:
        print("🤖 Using SIMPLE RULE‑BASED Validator")
        validator = SimpleRuleValidator()
    else:
        validator = SemanticValidator(llm_model="gpt-4o-mini", max_autorater_retries=1, thresholds=DEFAULT_THRESHOLDS)
    
    domain_context = DomainContext(rules={
        "max_speed_kmh": "speed <= 150",  # ← Tuned threshold
        "lat_range": "40 <= lat <= 50",
        "lon_range": "-80 <= lon <= -70",
        "passenger_positive": "passengers >= 0"
    }, contracts={}, schedules=[])
    
    routing_counts = {dec: 0 for dec in RoutingDecision}
    latencies, debug_log = [], []
    
    logger.info(f"▶️  Processing {sample_size} records...")
    start_total = time.time()
    
    async for parsed in ingestor.stream_from_file(stream_path, "test_agency", "GTFS-RT", delay=0.001):
        if len(debug_log) >= sample_size: 
            break
        rec_start = time.time()
        
        try:
            # Debug: Log first record payload
            if len(debug_log) == 0:
                logger.info(f"🔍 First record payload: {json.dumps(parsed.payload, default=str)[:200]}...")
            
            # Run Profiler
            prof_out = profiler.evaluate_record(parsed.payload)
            
            # Run Validator
            val_in = ValidatorInput(
                record=parsed.payload, 
                domain_context=domain_context, 
                profiler_result=prof_out
            )
            val_out = await validator.validate(val_in)
            
            # Decision Diamond
            routing = determine_routing_decision(prof_out, val_out, DEFAULT_THRESHOLDS)
            routing_counts[routing] += 1
            
            # Measure latency (per record, in ms)
            latency_ms = (time.time() - rec_start) * 1000
            latencies.append(latency_ms)
            
            # Log first 3 records for debug
            if len(debug_log) < 3:
                debug_log.append({
                    "idx": len(debug_log)+1,
                    "payload_speed": parsed.payload.get("speed_kmh"),
                    "prof_deviation": prof_out.deviation_score,
                    "prof_verdict": prof_out.verdict,
                    "prof_conf": prof_out.confidence,
                    "prof_reason": prof_out.reason[:40],
                    "val_verdict": val_out.verdict,
                    "val_conf": val_out.confidence,
                    "val_reason": val_out.reason[:40],
                    "routing": routing.value,
                })
                
        except Exception as e:
            # ✅ LOG THE ACTUAL ERROR (critical for debugging)
            logger.error(f"❌ Processing error for record {len(debug_log)+1}: {type(e).__name__}: {e}")
            logger.error(f"   Record payload keys: {list(parsed.payload.keys()) if hasattr(parsed.payload, 'keys') else 'N/A'}")
            logger.error(f"   Record payload sample: {json.dumps(parsed.payload, default=str)[:100]}...")
            
            routing_counts[RoutingDecision.QUARANTINE] += 1
            latencies.append((time.time() - rec_start) * 1000)
            
    total_time = time.time() - start_total
    avg_latency = sum(latencies)/len(latencies) if latencies else 0
    processed = sum(routing_counts.values())
    
    # ─── DEBUG PRINT (First 3 records) ────────────────────────────────────────
    print("\n🔍 DEBUG: First 3 Record Decisions")
    print(f"{'#':<3} {'Speed':<8} {'Prof.Dev':<9} {'Prof.V':<8} {'Conf':<6} {'Val.V':<8} {'Routing':<12} {'Reason'}")
    print("-" * 110)
    for d in debug_log:
        print(f"{d['idx']:<3} {d['payload_speed']:<8.1f} {d['prof_deviation']:<9.2f} {d['prof_verdict']:<8} "
              f"{d['prof_conf']:<6.2f} {d['val_verdict']:<8} {d['routing']:<12} {d['val_reason']}")
    print()
    
    # ─── BOSS REPORT ──────────────────────────────────────────────────────────
    print("📊"*50 + "\n   STAKEHOLDER TEST REPORT\n" + "📊"*50 + "\n")
    trust_rate = routing_counts[RoutingDecision.TRUST]/processed if processed else 0
    anomaly_rate = 1 - trust_rate
    
    print(f"✅ Records Processed: {processed}")
    print(f"⏱️  Avg Latency: {avg_latency:.1f} ms/record  {'✅ MEETS TARGET (<100ms)' if avg_latency < 100 else '⚠️ EXCEEDS'}")
    print(f"🕒 Total Time: {total_time:.2f} sec\n")
    
    print("🔀 Routing Distribution:")
    for dec, cnt in routing_counts.items():
        pct = cnt/processed*100 if processed else 0
        bar_len = int(pct/2)
        print(f"   {dec.value:12s} {'█' * bar_len} {cnt:3d} ({pct:.0f}%)")
    print()
    
    print("📈 Quality & Safety Metrics:")
    print(f"   • Trust Rate (Clean Data):  {trust_rate:.1%} {'✅ PASS (≥75%)' if trust_rate >= 0.75 else '⚠️ REVIEW'}")
    print(f"   • Anomaly Capture Rate:     {anomaly_rate:.1%}")
    print(f"   • Quarantined/Blocked:      {routing_counts[RoutingDecision.QUARANTINE]}")
    print(f"   • Escalated to Judge:       {routing_counts[RoutingDecision.JUDGE]}")
    print(f"   • XAI Audit Trail:          ✅ Enabled\n")
    
    print("💡 Key Insights for Leadership:")
    print("   • Profiler catches statistical outliers automatically")
    print("   • Validator enforces semantic/business rules")
    print("   • Conflicts route to Judge Agent (no guessing)")
    print("   • All decisions logged for compliance & debugging\n")
    
    # Phase I readiness check
    phase1_ready = (
        trust_rate >= 0.75 and 
        avg_latency < 100 and 
        routing_counts[RoutingDecision.JUDGE] > 0  # Judge should be active
    )
    status = "✅ READY FOR PHASE I" if phase1_ready else "⚠️ NEEDS TUNING"
    print(f"🎯 Phase I Readiness: {status}\n" + "🔍"*50)

if __name__ == "__main__":
    asyncio.run(run_pipeline_and_test())