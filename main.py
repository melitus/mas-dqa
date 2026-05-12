# main.py - MAS-DQA MVP Pipeline Runner

"""
MVP pipeline runner for MAS-DQA.
Runs the Profiler + Validator + Agreement logic on a stream slice.
Collects routing metrics and latency for Phase 0/I validation.

Reference: MAS-DQA Knowledge Base §5 (Validation Framework), §9 (Design Principles)
"""

import asyncio
import time
import logging
from typing import Dict, Any

import pandas as pd

from src.profiler import Profiler, ProfilerOutput
from src.validator import SemanticValidator, ValidatorInput, DomainContext, ValidatorOutput
from src.agreement import determine_routing_decision, RoutingDecision
from src.config.thresholds import DEFAULT_THRESHOLDS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-12s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ✅ ADDED: Simple XAI Log callback for demo (per KB §3: "Every decision includes audit trail")
def simple_xai_logger(output: ValidatorOutput, input_ ValidatorInput) -> None:
    """Demo XAI logger: prints verdict + explanation for auditability."""
    logger.debug(f"📋 XAI: verdict={output.verdict}, reason={output.reason[:100]}")


async def run_mvp_pipeline(
    baseline_path: str = "data/buspas_baseline.csv",
    stream_path: str = "data/buspas_stream.csv",
    sample_size: int = 1000,
    llm_model: str = "gpt-4o-mini"
):
    """
    Run the MAS-DQA validation pipeline on a dataset slice.
    
    Args:
        baseline_path: Path to baseline CSV for Profiler
        stream_path: Path to streaming CSV
        sample_size: Number of records to process
        llm_model: Litellm-compatible model for Semantic Validator
    """
    logger.info("🚀 Loading datasets...")
    try:
        baseline = pd.read_csv(baseline_path)
        stream = pd.read_csv(stream_path).head(sample_size)
        logger.info(f"✅ Baseline: {len(baseline)} rows | Stream slice: {len(stream)} rows")
    except FileNotFoundError as e:
        logger.error(f"❌ Dataset not found: {e}")
        logger.info("💡 Place CSVs in /data/ or update paths.")
        return

    # Initialize components
    logger.info("⚙️  Initializing agents...")
    profiler = Profiler(
        baseline_df=baseline,
        thresholds=DEFAULT_THRESHOLDS
    )
    validator = SemanticValidator(
        llm_model=llm_model,
        max_autorater_retries=1,  # MVP: limit retries for speed
        cache_size=500,
        thresholds=DEFAULT_THRESHOLDS
    )
    # ✅ ADDED: Register XAI logger callback
    validator.set_xai_logger(simple_xai_logger)

    # Simple domain context for MVP testing
    domain_context = DomainContext(
        rules={"max_speed_kmh": "speed must be <= 100", "lat_range": "40.0 <= lat <= 50.0"},
        contracts={},
        schedules=[]
    )

    # Metrics tracker
    routing_counts: Dict[RoutingDecision, int] = {dec: 0 for dec in RoutingDecision}
    latencies_ms = []
    errors = 0

    logger.info(f"▶️  Processing {len(stream)} records...")
    start_total = time.time()

    for idx, record in stream.iterrows():
        record_start = time.time()
        
        try:
            # Clean & prepare record
            record_dict = record.dropna().to_dict()
            if not record_dict:
                continue

            # 1. Run Profiler (sync, <2ms)
            prof_output: ProfilerOutput = profiler.evaluate_record(record_dict)

            # 2. Run Validator (async, LLM call)
            val_input = ValidatorInput(
                record=record_dict,
                domain_context=domain_context,
                # ✅ FIXED: Use ProfilerOutput directly (no mapping needed if schemas align)
                profiler_result=prof_output
            )
            val_output: ValidatorOutput = await validator.validate(val_input)

            # 3. ✅ FIXED: Call refactored function name
            routing = determine_routing_decision(prof_output, val_output, DEFAULT_THRESHOLDS)
            routing_counts[routing] += 1

            # ✅ ADDED: Simulate Orchestrator execution (per Slide 8: "Orchestrator routes flows")
            match routing:
                case RoutingDecision.TRUST:
                    logger.debug(f"🟢 ORCHESTRATOR → Trust Agent (SRI update)")
                case RoutingDecision.JUDGE | RoutingDecision.AMBIGUOUS:
                    logger.debug(f"🟠 ORCHESTRATOR → Judge Agent (conflict resolution)")
                case RoutingDecision.QUARANTINE:
                    logger.debug(f"🔴 ORCHESTRATOR → Quarantine (isolate stream)")

        except Exception as e:
            logger.debug(f"⚠️ Record {idx} failed: {str(e)[:100]}")
            routing_counts[RoutingDecision.QUARANTINE] += 1
            errors += 1
        finally:
            latencies_ms.append((time.time() - record_start) * 1000)

            # Progress indicator
            if (idx + 1) % 100 == 0:
                logger.info(f"📊 Processed {idx + 1}/{len(stream)} records...")

    total_time = time.time() - start_total
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    max_latency = max(latencies_ms) if latencies_ms else 0.0

    # Print results
    logger.info("📋 === MVP Pipeline Results ===")
    logger.info(f"✅ TRUST:      {routing_counts[RoutingDecision.TRUST]}")
    logger.info(f"⚖️  JUDGE:      {routing_counts[RoutingDecision.JUDGE]}")
    logger.info(f"🚫 QUARANTINE: {routing_counts[RoutingDecision.QUARANTINE]}")
    logger.info(f"🤔 AMBIGUOUS:  {routing_counts[RoutingDecision.AMBIGUOUS]}")
    logger.info(f"⚠️  ERRORS:     {errors}")
    logger.info(f"⏱️  AVG Latency: {avg_latency:.2f}ms")
    logger.info(f"⏱️  MAX Latency: {max_latency:.2f}ms")
    logger.info(f"🕒 Total Time:  {total_time:.2f}s")
    logger.info(f"📈 Throughput:  {len(stream)/total_time:.1f} records/sec")
    
    # Phase I readiness check
    trust_rate = routing_counts[RoutingDecision.TRUST] / len(stream) if len(stream) > 0 else 0
    logger.info(f"🎯 Trust Rate: {trust_rate:.1%} (Target: >0.80 for Phase I)")


if __name__ == "__main__":
    # Run async pipeline
    asyncio.run(run_mvp_pipeline())