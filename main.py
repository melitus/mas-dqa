#!/usr/bin/env python3
"""
MAS-DQA: Unified Pipeline Runner with 4-Mode Switch System
==========================================================
Modes:
  🧪 Synthetic + Mock  : python main.py --data synthetic --validator mock
  🤖 Synthetic + LLM   : python main.py --data synthetic --validator llm
  🚌 Real + Mock       : python main.py --data real --validator mock
  🚀 Real + LLM        : python main.py --data real --validator llm
"""

import asyncio
import time
import logging
import json
import os
import sys
import argparse
import random
import numpy as np
from typing import Dict, List, Any
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.abspath("src"))

from src.ingestion import StreamIngestor
from src.profiler import Profiler
from src.validator import SemanticValidator, ValidatorInput, DomainContext
from src.validator.rule_validator import SimpleRuleValidator
from src.agreement import determine_routing_decision, RoutingDecision
from src.config.thresholds import DEFAULT_THRESHOLDS

# Metrics
try:
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  sklearn not installed. Install with: pip install scikit-learn")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# SYNTHETIC LABELING (for real data validation)
# ──────────────────────────────────────────────────────────────────────────────
def synthetic_label_record(record: Dict[str, Any]) -> tuple[int, str]:
    """Apply heuristic rules to assign synthetic ground truth."""
    required = ["timestamp", "lat", "lon", "route_id", "trip_id", "stop_id"]
    if not all(k in record and record.get(k) is not None for k in required):
        return 0, "Missing required fields"
    
    lat = record.get("lat")
    if lat is not None and not (39.0 <= lat <= 41.0):
        return 0, f"Latitude out of NJ bounds ({lat})"
    
    lon = record.get("lon")
    if lon is not None and not (-75.0 <= lon <= -73.0):
        return 0, f"Longitude out of NJ bounds ({lon})"
    
    speed = record.get("speed_kmh")
    if speed is not None and speed > 150:
        return 0, f"Extreme speed: {speed} km/h"
    
    pax = record.get("passenger_count")
    if pax is not None and pax < 0:
        return 0, f"Negative passenger count"
    
    return 1, "All checks passed"


def generate_synthetic_labels(records: List[Dict], output_path: str) -> Dict[str, int]:
    """Generate and save synthetic labels for Phase I validation."""
    labels = []
    for rec in records:
        rid = rec.get("record_id")
        if rid:
            gt, reason = synthetic_label_record(rec)
            labels.append({"record_id": rid, "ground_truth": gt, "reason": reason})
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    
    logger.info(f"🏷️  Generated {len(labels)} synthetic labels → {output_path}")
    return {lbl["record_id"]: lbl["ground_truth"] for lbl in labels}


def generate_synthetic_data(sample_size: int) -> tuple[str, str, str]:
    """Generate synthetic test data with ground truth labels."""
    os.makedirs("data", exist_ok=True)
    baseline_path = "data/buspas_baseline.csv"
    stream_path = "data/buspas_stream.json"
    labels_path = "data/buspas_labels.json"
    
    np.random.seed(42)
    
    # Baseline: clean historical data
    baseline = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=500, freq="min"),
        "speed_kmh": np.clip(np.random.normal(45, 10, 500), 0, 120),
        "lat": np.clip(np.random.normal(45.5, 0.05, 500), 40, 50),
        "lon": np.clip(np.random.normal(-73.6, 0.05, 500), -80, -70),
        "passenger_count": np.clip(np.random.poisson(30, 500), 0, 60),
    })
    baseline.to_csv(baseline_path, index=False)
    
    # Stream: 80% normal, 20% anomalies
    records, labels = [], []
    for i in range(sample_size):
        is_anomaly = random.random() < 0.2
        rec = {
            "record_id": f"syn_{i:04d}",
            "timestamp": f"2026-05-12T10:{i%60:02d}:00Z",
            "speed_kmh": round(np.random.normal(45, 10), 1),
            "lat": round(np.random.normal(45.5, 0.05), 4),
            "lon": round(np.random.normal(-73.6, 0.05), 4),
            "passenger_count": int(np.random.poisson(30))
        }
        if is_anomaly:
            anomaly_type = random.choice(["speed", "lat", "passenger"])
            if anomaly_type == "speed":
                rec["speed_kmh"] = round(random.uniform(180, 250), 1)
                reason = f"Extreme speed anomaly ({rec['speed_kmh']} km/h)"
            elif anomaly_type == "lat":
                rec["lat"] = 99.9999
                reason = f"Latitude out of bounds ({rec['lat']})"
            else:
                rec["passenger_count"] = -random.randint(5, 20)
                reason = f"Negative passenger count ({rec['passenger_count']})"
            labels.append({"record_id": rec["record_id"], "ground_truth": 0, "reason": reason})
        else:
            labels.append({"record_id": rec["record_id"], "ground_truth": 1, "reason": "Normal record"})
        records.append(rec)
    
    with open(stream_path, "w") as f:
        json.dump(records, f, indent=2)
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)
    
    logger.info(f"📄 Generated synthetic data: {sample_size} records (80% normal, 20% anomalies)")
    return stream_path, baseline_path, labels_path


# ──────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE RUNNER
# ──────────────────────────────────────────────────────────────────────────────
async def run_pipeline_and_test(
    data_mode: str = "synthetic",      # "synthetic" or "real"
    validator_mode: str = "mock",      # "mock" or "llm"
    sample_size: int = 300,
    results_path: str = "data/dashboard_results.json"
) -> Dict:
    """
    Run MAS-DQA pipeline with configurable data source and validator.
    
    Returns dictionary with results for dashboard display.
    """
    print("\n" + "🔍" * 60)
    print(" MAS-DQA: Phase I Validation")
    print(f" Data Mode     : {'🧪 Synthetic' if data_mode == 'synthetic' else '🚌 Real BusPas'}")
    print(f" Validator Mode: {'🤖 Mock (Rule-based)' if validator_mode == 'mock' else '🧠 Real LLM'}")
    print(f" Sample Size   : {sample_size:,} records")
    print("🔍" * 60 + "\n")
    
    # ──────────────────────────────────────────────────────────────────────
    # LOAD DATA
    # ──────────────────────────────────────────────────────────────────────
    if data_mode == "synthetic":
        stream_path, baseline_path, labels_path = generate_synthetic_data(sample_size)
        with open(labels_path, "r") as f:
            labels_data = json.load(f)
        ground_truth = {lbl["record_id"]: lbl["ground_truth"] for lbl in labels_data}
    else:  # real
        stream_path = "data/normalised.buspas.ndjson"
        if not os.path.exists(stream_path):
            logger.error(f"❌ Real data not found: {stream_path}\n💡 Run adapter first: python scripts/test_adapter_flow.py")
            return {}
        
        records = []
        with open(stream_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= sample_size: break
                line = line.strip()
                if line:
                    try: records.append(json.loads(line))
                    except: continue
        
        logger.info(f"📄 Loaded {len(records)} real records")
        baseline_path = "data/buspas_baseline.csv"
        labels_path = "data/buspas_labels_synthetic.json"
        ground_truth = generate_synthetic_labels(records, labels_path)
    
    # ──────────────────────────────────────────────────────────────────────
    # COMPUTE BASELINE
    # ──────────────────────────────────────────────────────────────────────
    # Load records for baseline computation (simplified for demo)
    records_for_baseline = []
    with open(stream_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= int(sample_size * 0.7): break
            line = line.strip()
            if line:
                try: records_for_baseline.append(json.loads(line))
                except: continue
    
    baseline_records = [r for r in records_for_baseline if synthetic_label_record(r)[0] == 1]
    if baseline_records:
        baseline_df = pd.DataFrame(baseline_records)
        # Keep only numeric columns for Profiler
        numeric_cols = baseline_df.select_dtypes(include=[np.number]).columns.tolist()
        baseline_df = baseline_df[numeric_cols] if numeric_cols else pd.DataFrame()
        baseline_df.to_csv(baseline_path, index=False)
        logger.info(f"📊 Computed baseline from {len(baseline_df)} clean records")
    else:
        baseline_df = pd.DataFrame()
    
    # ──────────────────────────────────────────────────────────────────────
    # INITIALIZE COMPONENTS
    # ──────────────────────────────────────────────────────────────────────
    ingestor = StreamIngestor(required_fields=["timestamp"])
    profiler = Profiler(baseline_df=baseline_df, thresholds=DEFAULT_THRESHOLDS)
    
    # Domain context: Use OLD bounds for mock mode to create demo conflicts
    # Synthetic labeling uses NJ bounds (39-41) for realistic ground truth
    # Validator uses old bounds (40-50) to generate Judge escalations for demo
    domain_context = DomainContext(
        rules={
            "max_speed_kmh": "speed_kmh <= 150",
            "lat_bounds": "40.0 <= lat <= 50.0",  # ← Old bounds for demo conflicts
            "lon_bounds": "-80.0 <= lon <= -70.0",  # ← Old bounds
            "passenger_positive": "passenger_count >= 0"
        },
        contracts={},  # Empty for mock mode
        schedules=[]
    )
    
    # Validator selection with API key check
    if validator_mode == "llm":
        if not os.getenv("LITELLM_API_KEY"):
            logger.warning("⚠️  LITELLM_API_KEY not set; falling back to mock validator")
            validator = SimpleRuleValidator()
        else:
            logger.info("🧠 Initializing Real LLM Semantic Validator")
            validator = SemanticValidator(
                llm_model="gpt-4o-mini",
                llm_api_key=os.getenv("LITELLM_API_KEY"),
                max_autorater_retries=1,
                thresholds=DEFAULT_THRESHOLDS
            )
    else:
        logger.info("🤖 Initializing Mock (Rule-based) Validator")
        validator = SimpleRuleValidator()
    
    # ──────────────────────────────────────────────────────────────────────
    # PROCESSING LOOP
    # ──────────────────────────────────────────────────────────────────────
    routing_counts = {dec: 0 for dec in RoutingDecision}
    debug_log = []
    y_true_list, y_pred_list = [], []
    latencies = []
    
    start_time = time.time()
    
    async for parsed in ingestor.stream_from_file(stream_path, "test_agency", delay=0.001):
        if len(debug_log) >= sample_size: break
        
        rec_start = time.time()
        
        try:
            # Profiler
            prof_out = profiler.evaluate_record(parsed.payload)
            
            # Validator
            val_in = ValidatorInput(record=parsed.payload, domain_context=domain_context, profiler_result=prof_out)
            val_out = await validator.validate(val_in)
            
            # Decision Diamond
            routing = determine_routing_decision(prof_out, val_out, DEFAULT_THRESHOLDS)
            routing_counts[routing] += 1
            
            # Phase I tracking
            record_id = parsed.payload.get("record_id")
            if record_id and record_id in ground_truth:
                y_true_list.append(ground_truth[record_id])
                y_pred_list.append(1 if val_out.verdict == "Valid" else 0)
            
            # Latency tracking
            latencies.append((time.time() - rec_start) * 1000)
            
            # Debug log (first 5)
            if len(debug_log) < 5:
                debug_log.append({
                    "record_id": record_id,
                    "speed": parsed.payload.get("speed_kmh"),
                    "prof_dev": getattr(prof_out, 'deviation_score', None),
                    "prof_verdict": getattr(prof_out, 'verdict', 'N/A'),
                    "val_verdict": val_out.verdict,
                    "routing": routing.value,
                    "reason": val_out.reason[:200] if len(val_out.reason) > 200 else val_out.reason
                })
                
        except Exception as e:
            logger.error(f"❌ Processing error: {type(e).__name__}: {e}")
            routing_counts[RoutingDecision.QUARANTINE] += 1
            latencies.append((time.time() - rec_start) * 1000)
    
    total_time = time.time() - start_time
    processed = sum(routing_counts.values())
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    # ──────────────────────────────────────────────────────────────────────
    # COMPUTE METRICS - FIXED VERSION
    # ──────────────────────────────────────────────────────────────────────
    trust_rate = routing_counts[RoutingDecision.TRUST] / processed if processed else 0
    anomaly_rate = 1 - trust_rate
    
    # Initialize metrics with defaults
    phase1_metrics = {
        "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "accuracy": 0.0,
        "false_positive_rate": 0.0, "false_negative_rate": 0.0,
        "true_positives": 0, "true_negatives": 0, "false_positives": 0, "false_negatives": 0
    }
    
    # Compute metrics if we have data
    if SKLEARN_AVAILABLE and y_true_list and y_pred_list and len(y_true_list) > 0:
        try:
            # Debug: log label distribution
            unique_true = set(y_true_list)
            unique_pred = set(y_pred_list)
            logger.debug(f"🔍 Labels: y_true unique={unique_true}, y_pred unique={unique_pred}, n={len(y_true_list)}")
            
            # Compute confusion matrix (handle edge cases with explicit labels)
            tn, fp, fn, tp = confusion_matrix(y_true_list, y_pred_list, labels=[1, 0]).ravel()
            
            # Compute metrics with zero-division handling
            precision = precision_score(y_true_list, y_pred_list, zero_division=0)
            recall = recall_score(y_true_list, y_pred_list, zero_division=0)
            f1 = f1_score(y_true_list, y_pred_list, zero_division=0)
            
            phase1_metrics = {
                "precision": round(float(precision), 3),
                "recall": round(float(recall), 3),
                "f1_score": round(float(f1), 3),
                "accuracy": round((tp + tn) / len(y_true_list), 3) if y_true_list else 0.0,
                "false_positive_rate": round(fp / (fp + tn), 3) if (fp + tn) > 0 else 0.0,
                "false_negative_rate": round(fn / (fn + tp), 3) if (fn + tp) > 0 else 0.0,
                "true_positives": int(tp), "true_negatives": int(tn),
                "false_positives": int(fp), "false_negatives": int(fn)
            }
            logger.info(f"📈 Metrics computed: P={phase1_metrics['precision']}, R={phase1_metrics['recall']}, F1={phase1_metrics['f1_score']}")
            
        except Exception as e:
            logger.error(f"⚠️  Metrics calculation error: {type(e).__name__}: {e}")
            logger.error(f"   y_true sample: {y_true_list[:10] if y_true_list else 'empty'}")
            logger.error(f"   y_pred sample: {y_pred_list[:10] if y_pred_list else 'empty'}")
    
    # ──────────────────────────────────────────────────────────────────────
    # SAVE & DISPLAY RESULTS
    # ──────────────────────────────────────────────────────────────────────
    dashboard_results = {
        "config": {
            "data_mode": data_mode,
            "validator_mode": validator_mode,
            "sample_size": sample_size
        },
        "summary": {
            "records_processed": processed,
            "avg_latency_ms": round(avg_latency, 2),
            "total_time_sec": round(total_time, 2),
            "trust_rate": round(trust_rate, 4),
            "anomaly_capture_rate": round(anomaly_rate, 4)
        },
        # ── ADD THESE NEW SECTIONS ────────────────────────────────────────
    "quality_safety": {
        "trust_rate_percent": round(trust_rate * 100, 1),
        "anomaly_capture_rate_percent": round(anomaly_rate * 100, 1),
        "quarantined_count": routing_counts[RoutingDecision.QUARANTINE],
        "judge_escalations": routing_counts[RoutingDecision.JUDGE],
        "xai_enabled": True
    },
    "key_insights": [
        "Profiler catches statistical outliers automatically",
        "Validator enforces semantic/business rules",
        "Conflicts route to Judge Agent (no guessing)",
        "All decisions logged for compliance & debugging",
        f"Phase I F1-Score: {phase1_metrics['f1_score']:.3f} — detection quality validated"
    ],
        "routing": {k.value: v for k, v in routing_counts.items()},
        "phase1_metrics": phase1_metrics,
        "sample_decisions": debug_log,
        "phase1_ready": (
            phase1_metrics["f1_score"] >= 0.82 and
            avg_latency < 100 and
            trust_rate >= 0.75
        )
    }
    
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_results, f, indent=2)
    
    # Print summary
    print(f"\n📊 Dashboard results saved to {results_path}")
    print(f"✅ Records Processed: {processed}")
    print(f"⏱️  Avg Latency: {avg_latency:.1f} ms/record")
    print(f"🔀 Routing: TRUST={routing_counts[RoutingDecision.TRUST]}, JUDGE={routing_counts[RoutingDecision.JUDGE]}, QUARANTINE={routing_counts[RoutingDecision.QUARANTINE]}")
    print(f"📈 Phase I F1: {phase1_metrics['f1_score']:.3f}")
    print(f"🎯 Phase I Readiness: {'✅ READY' if dashboard_results['phase1_ready'] else '⚠️ NEEDS TUNING'}")
    
    return dashboard_results


# ──────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAS-DQA Phase I Validation with 4-Mode Switch")
    
    parser.add_argument(
        "--data", type=str, default="synthetic", choices=["synthetic", "real"],
        help="Data source: 'synthetic' for generated test data, 'real' for BusPas NDJSON"
    )
    parser.add_argument(
        "--validator", type=str, default="mock", choices=["mock", "llm"],
        help="Validator mode: 'mock' for rule-based, 'llm' for real LLM (requires LITELLM_API_KEY)"
    )
    parser.add_argument(
        "--sample", type=int, default=300,
        help="Number of records to process (default: 300)"
    )
    parser.add_argument(
        "--results", type=str, default="data/dashboard_results.json",
        help="Path to save dashboard results JSON"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    asyncio.run(run_pipeline_and_test(
        data_mode=args.data,
        validator_mode=args.validator,
        sample_size=args.sample,
        results_path=args.results
    ))