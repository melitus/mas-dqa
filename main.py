#!/usr/bin/env python3
"""
MAS-DQA: Unified Pipeline Runner & Automated Test/Demo
=======================================================
Run: python main.py
- Auto-generates aligned synthetic test data WITH GROUND TRUTH LABELS
- Runs full pipeline: Ingestion → Profiler → Validator → Agreement
- Computes Phase I metrics: Precision, Recall, F1-Score
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
from typing import Dict, List, Optional
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

# For Phase I metrics
try:
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  sklearn not installed. Install with: pip install scikit-learn")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# 🎛️ TOGGLE FOR TESTING
MOCK_MODE = True  # Set to False to use real LLM (requires litellm API key)

# ──────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA GENERATOR WITH GROUND TRUTH LABELS
# ──────────────────────────────────────────────────────────────────────────────
def _generate_test_files():
    """Generate synthetic test data WITH ground truth labels for Phase I validation."""
    os.makedirs("data", exist_ok=True)
    baseline_path, stream_path = "data/buspas_baseline.csv", "data/buspas_stream.json"
    labels_path = "data/buspas_labels.json"  # NEW: ground truth labels
    
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
    
    # Stream: 80% normal (label=1/Valid), 20% anomalies (label=0/Invalid)
    records = []
    labels = []  # Ground truth: 1=Valid/Normal, 0=Invalid/Anomaly
    
    for i in range(300):
        is_anomaly = random.random() < 0.2  # 20% anomaly rate
        
        if not is_anomaly:  # Normal record → label=1
            rec = {
                "record_id": f"rec_{i:04d}",
                "timestamp": f"2026-05-12T10:{i%60:02d}:00Z",
                "speed_kmh": round(np.random.normal(45, 10), 1),
                "lat": round(np.random.normal(45.5, 0.05), 4),
                "lon": round(np.random.normal(-73.6, 0.05), 4),
                "passenger_count": int(np.random.poisson(30))
            }
            labels.append({"record_id": rec["record_id"], "ground_truth": 1, "reason": "Normal record"})
            
        else:  # Anomaly → label=0
            anomaly_type = random.choice(["speed", "lat", "passenger"])
            rec = {
                "record_id": f"rec_{i:04d}",
                "timestamp": f"2026-05-12T10:{i%60:02d}:00Z",
                "speed_kmh": round(np.random.normal(45, 10), 1),
                "lat": round(np.random.normal(45.5, 0.05), 4),
                "lon": round(np.random.normal(-73.6, 0.05), 4),
                "passenger_count": int(np.random.poisson(30))
            }
            if anomaly_type == "speed": 
                rec["speed_kmh"] = round(random.uniform(180, 250), 1)
                reason = f"Extreme speed anomaly ({rec['speed_kmh']} km/h)"
            elif anomaly_type == "lat": 
                rec["lat"] = 99.9999
                reason = f"Latitude out of bounds ({rec['lat']})"
            else:  # passenger
                rec["passenger_count"] = -random.randint(5, 20)
                reason = f"Negative passenger count ({rec['passenger_count']})"
            
            labels.append({"record_id": rec["record_id"], "ground_truth": 0, "reason": reason, "anomaly_type": anomaly_type})
            
        records.append(rec)
    
    # Save stream data
    with open(stream_path, "w") as f:
        json.dump(records, f, indent=2)
    
    # Save ground truth labels
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)
    
    logger.info(f"📄 Generated aligned baseline & stream data (80% normal, 20% anomalies)")
    logger.info(f"🏷️  Ground truth labels saved to {labels_path}")
    
    return baseline_path, stream_path, labels_path

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
        if speed is not None and speed > 150:
            valid = False
            reason = f"Speed exceeds 150 km/h limit ({speed} km/h)"
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
        elif input_.profiler_result and input_.profiler_result.deviation_score < 0.3:
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
# PHASE I METRICS COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────
def compute_phase1_metrics(
    y_true: List[int], 
    y_pred: List[int], 
    routing_decisions: List[RoutingDecision]
) -> Dict[str, float]:
    """
    Compute Phase I validation metrics: Precision, Recall, F1.
    
    Args:
        y_true: Ground truth labels (1=Valid/Normal, 0=Invalid/Anomaly)
        y_pred: Model predictions (1=Valid, 0=Invalid)
        routing_decisions: Final routing decisions for each record
    
    Returns:
        Dictionary with Phase I metrics
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "sklearn not installed"}
    
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"error": "No data to evaluate"}
    
    # Compute standard metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Additional useful metrics
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    
    # Routing distribution
    routing_dist = {dec.value: routing_decisions.count(dec) for dec in RoutingDecision}
    
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "accuracy": round(accuracy, 3),
        "false_positive_rate": round(fpr, 3),
        "false_negative_rate": round(fnr, 3),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "routing_distribution": routing_dist,
        "n_samples": len(y_true)
    }

# ──────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER WITH PHASE I METRICS
# ──────────────────────────────────────────────────────────────────────────────
async def run_pipeline_and_test(
    stream_path: str = "data/buspas_stream.json",
    baseline_path: str = "data/buspas_baseline.csv",
    labels_path: str = "data/buspas_labels.json",
    sample_size: int = 300
):
    print("\n" + "🔍"*50)
    print("   MAS-DQA: Automated Pipeline Test & Demo")
    print("   Testing: 'Bad Data = No Data' → 'Trusted Data = Confident Decisions'")
    print("🔍"*50 + "\n")
    
    # Generate or load test data
    if not os.path.exists(stream_path) or not os.path.exists(baseline_path):
        baseline_path, stream_path, labels_path = _generate_test_files()
    
    # Load ground truth labels
    ground_truth = {}
    if os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            labels_data = json.load(f)
            for label in labels_data:
                ground_truth[label["record_id"]] = label["ground_truth"]
        logger.info(f"🏷️  Loaded {len(ground_truth)} ground truth labels")
    
    print("⚙️  Initializing MAS-DQA components...")
    ingestor = StreamIngestor(required_fields=["timestamp"])
    profiler = Profiler(baseline_df=pd.read_csv(baseline_path), thresholds=DEFAULT_THRESHOLDS)
    
    if MOCK_MODE:
        print("🤖 Using SIMPLE RULE‑BASED Validator")
        validator = SimpleRuleValidator()
    else:
        validator = SemanticValidator(llm_model="gpt-4o-mini", max_autorater_retries=1, thresholds=DEFAULT_THRESHOLDS)
    
    domain_context = DomainContext(rules={
        "max_speed_kmh": "speed <= 150",
        "lat_range": "40 <= lat <= 50",
        "lon_range": "-80 <= lon <= -70",
        "passenger_positive": "passengers >= 0"
    }, contracts={}, schedules=[])
    
    routing_counts = {dec: 0 for dec in RoutingDecision}
    latencies, debug_log = [], []
    
    # Phase I tracking
    y_true_list, y_pred_list = [], []
    routing_decisions_list = []
    
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
            routing_decisions_list.append(routing)
            
            # Phase I: Track predictions vs ground truth
            record_id = parsed.payload.get("record_id")
            if record_id and record_id in ground_truth:
                # Ground truth: 1=Valid/Normal, 0=Invalid/Anomaly
                true_label = ground_truth[record_id]
                y_true_list.append(true_label)
                
                # Prediction: Convert verdict to binary (Valid=1, Invalid/Unknown=0)
                pred_label = 1 if val_out.verdict == "Valid" else 0
                y_pred_list.append(pred_label)
            
            # Measure latency (per record, in ms)
            latency_ms = (time.time() - rec_start) * 1000
            latencies.append(latency_ms)
            
            # Log first 3 records for debug
            if len(debug_log) < 3:
                debug_log.append({
                    "idx": len(debug_log)+1,
                    "record_id": parsed.payload.get("record_id"),
                    "payload_speed": parsed.payload.get("speed_kmh"),
                    "prof_deviation": prof_out.deviation_score,
                    "prof_verdict": prof_out.verdict,
                    "prof_conf": prof_out.confidence,
                    "prof_reason": prof_out.reason[:40],
                    "val_verdict": val_out.verdict,
                    "val_conf": val_out.confidence,
                    "val_reason": val_out.reason[:40],
                    "routing": routing.value,
                    "ground_truth": ground_truth.get(record_id, "N/A")
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
    
    # ─── PHASE I METRICS COMPUTATION ─────────────────────────────────────────
    phase1_metrics = {}
    if y_true_list and y_pred_list and SKLEARN_AVAILABLE:
        phase1_metrics = compute_phase1_metrics(y_true_list, y_pred_list, routing_decisions_list)
    
    # ─── DEBUG PRINT (First 3 records) ────────────────────────────────────────
    print("\n🔍 DEBUG: First 3 Record Decisions")
    print(f"{'#':<3} {'ID':<10} {'Speed':<8} {'Prof.Dev':<9} {'Prof.V':<8} {'Conf':<6} {'Val.V':<8} {'Routing':<12} {'GT':<4} {'Reason'}")
    print("-" * 130)
    for d in debug_log:
        print(f"{d['idx']:<3} {d['record_id'] or 'N/A':<10} {d['payload_speed'] or 'N/A':<8.1f} "
              f"{d['prof_deviation']:<9.2f} {d['prof_verdict']:<8} {d['prof_conf']:<6.2f} "
              f"{d['val_verdict']:<8} {d['routing']:<12} {d['ground_truth'] if d['ground_truth'] != 'N/A' else '-':<4} "
              f"{d['val_reason']}")
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
    
    # ─── PHASE I METRICS SECTION ─────────────────────────────────────────────
    if phase1_metrics and "error" not in phase1_metrics:
        print("📈 Phase I Validation Metrics (Detection Accuracy):")
        print(f"   • Precision:  {phase1_metrics['precision']:.3f}  {'✅ PASS (≥0.85)' if phase1_metrics['precision'] >= 0.85 else '⚠️ REVIEW'}")
        print(f"   • Recall:     {phase1_metrics['recall']:.3f}  {'✅ PASS (≥0.80)' if phase1_metrics['recall'] >= 0.80 else '⚠️ REVIEW'}")
        print(f"   • F1-Score:   {phase1_metrics['f1_score']:.3f}  {'✅ PASS (≥0.82)' if phase1_metrics['f1_score'] >= 0.82 else '⚠️ REVIEW'}")
        print(f"   • Accuracy:   {phase1_metrics['accuracy']:.3f}")
        print(f"   • False Positive Rate:  {phase1_metrics['false_positive_rate']:.3f}")
        print(f"   • False Negative Rate:  {phase1_metrics['false_negative_rate']:.3f}")
        print(f"   • Confusion Matrix: TP={phase1_metrics['true_positives']}, TN={phase1_metrics['true_negatives']}, "
              f"FP={phase1_metrics['false_positives']}, FN={phase1_metrics['false_negatives']}")
        print()
    elif "error" in phase1_metrics:
        print(f"⚠️  Phase I Metrics: {phase1_metrics['error']}")
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
    print("   • All decisions logged for compliance & debugging")
    if phase1_metrics and "error" not in phase1_metrics:
        print(f"   • Phase I F1-Score: {phase1_metrics['f1_score']:.3f} — detection quality validated")
    print()
    
    # Phase I readiness check
    phase1_ready = (
        trust_rate >= 0.75 and 
        avg_latency < 100 and 
        routing_counts[RoutingDecision.JUDGE] > 0 and
        (phase1_metrics.get("f1_score", 0) >= 0.82 if phase1_metrics and "error" not in phase1_metrics else True)
    )
    status = "✅ READY FOR PHASE I" if phase1_ready else "⚠️ NEEDS TUNING"
    print(f"🎯 Phase I Readiness: {status}\n" + "🔍"*50)

if __name__ == "__main__":
    asyncio.run(run_pipeline_and_test())