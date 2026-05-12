# main.py - MAS-DQA MVP Pipeline Runner (with Ingestion)

"""
MVP pipeline runner for MAS-DQA.
Integrates StreamIngestor → Profiler → Validator → Agreement → Metrics.
Designed for quick testing with real JSON stream files.

Reference: MAS-DQA Knowledge Base §5 (Validation), §9 (Design Principles)
"""

import asyncio
import time
import logging
from typing import Dict

# Modular imports matching refactored architecture
from src.ingestion import StreamIngestor, ParsedRecord
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


# ──────────────────────────────────────────────────────────────────────────────
# XAI LOG DEMO (Simple callback for audit trail)
# ──────────────────────────────────────────────────────────────────────────────

def simple_xai_logger(output: ValidatorOutput, input_data: ValidatorInput) -> None:
    """Demo XAI logger: prints verdict + explanation for auditability."""
    logger.debug(f"📋 XAI: verdict={output.verdict}, reason={output.reason[:100]}")


# ──────────────────────────────────────────────────────────────────────────────
# MVP PIPELINE RUNNER
# ──────────────────────────────────────────────────────────────────────────────

async def run_mvp_pipeline(
    stream_path: str = "data/buspas_stream.json",
    baseline_path: str = "data/buspas_baseline.csv",
    producer_id: str = "agency_42",
    data_type: str = "GTFS-RT",
    sample_size: int = 1000,
    llm_model: str = "gpt-4o-mini",
    ingest_delay: float = 0.001  # Fast for testing
):
    """
    Run the full MAS-DQA pipeline: Ingestion → Profiler → Validator → Agreement.
    
    Args:
        stream_path: Path to JSON stream file (list of records)
        baseline_path: Path to baseline CSV for Profiler initialization
        producer_id: Identifier for the data source
        data_type: Schema type for validation (e.g., "GTFS-RT")
        sample_size: Max records to process (for quick testing)
        llm_model: Litellm-compatible model for Semantic Validator
        ingest_delay: Seconds between records (0 = max speed)
    """
    logger.info("🚀 Starting MAS-DQA MVP Pipeline")
    
    # ─── 1. Initialize Components ─────────────────────────────────────────────
    logger.info("⚙️  Initializing components...")
    
    # Ingestor: minimal config for MVP
    ingestor = StreamIngestor(required_fields=["timestamp"])
    
    # Profiler: load baseline from CSV
    import pandas as pd
    try:
        baseline_df = pd.read_csv(baseline_path)
        profiler = Profiler(baseline_df=baseline_df, thresholds=DEFAULT_THRESHOLDS)
        logger.info(f"✅ Profiler loaded baseline: {len(baseline_df)} rows")
    except FileNotFoundError:
        logger.warning(f"⚠️  Baseline not found: {baseline_path}. Using empty baseline.")
        baseline_df = pd.DataFrame()
        profiler = Profiler(baseline_df=baseline_df, thresholds=DEFAULT_THRESHOLDS)
    
    # Validator: async LLM-based semantic checks
    validator = SemanticValidator(
        llm_model=llm_model,
        max_autorater_retries=1,  # MVP: limit retries for speed
        cache_size=500,
        thresholds=DEFAULT_THRESHOLDS
    )
    validator.set_xai_logger(simple_xai_logger)  # Enable audit trail
    
    # Domain context for semantic validation (customize per use case)
    domain_context = DomainContext(
        rules={
            "max_speed_kmh": "speed must be <= 100",
            "lat_range": "latitude between 40.0 and 50.0",
            "lon_range": "longitude between -80.0 and -70.0",
        },
        contracts={},
        schedules=[]
    )
    
    # ─── 2. Metrics Trackers ──────────────────────────────────────────────────
    routing_counts: Dict[RoutingDecision, int] = {dec: 0 for dec in RoutingDecision}
    latencies_ms = []
    parsed_count = 0
    error_count = 0
    
    logger.info(f"▶️  Streaming from: {stream_path} | Producer: {producer_id}")
    start_total = time.time()
    
    # ─── 3. Main Processing Loop ──────────────────────────────────────────────
    async for parsed_record in ingestor.stream_from_file(
        file_path=stream_path,
        producer_id=producer_id,
        data_type=data_type,
        delay=ingest_delay
    ):
        record_start = time.time()
        
        try:
            # Skip if we've processed enough records
            if parsed_count >= sample_size:
                break
            
            # Extract payload for validation
            record_dict = parsed_record.payload
            
            # 1️⃣ Run Profiler (sync, <2ms)
            prof_output: ProfilerOutput = profiler.evaluate_record(record_dict)
            
            # 2️⃣ Run Validator (async, LLM call)
            val_input = ValidatorInput(
                record=record_dict,
                domain_context=domain_context,
                profiler_result=prof_output  # Pass directly (schemas aligned)
            )
            val_output: ValidatorOutput = await validator.validate(val_input)
            
            # 3️⃣ Decision Diamond: compute routing directive
            routing = determine_routing_decision(prof_output, val_output, DEFAULT_THRESHOLDS)
            routing_counts[routing] += 1
            
            # 4️⃣ Simulate Orchestrator execution (for demo)
            match routing:
                case RoutingDecision.TRUST:
                    logger.debug(f"🟢 ORCHESTRATOR → Trust Agent (SRI update)")
                case RoutingDecision.JUDGE | RoutingDecision.AMBIGUOUS:
                    logger.debug(f"🟠 ORCHESTRATOR → Judge Agent (conflict resolution)")
                case RoutingDecision.QUARANTINE:
                    logger.debug(f"🔴 ORCHESTRATOR → Quarantine (isolate stream)")
            
            parsed_count += 1
            
        except Exception as e:
            logger.debug(f"⚠️  Record {parsed_count} failed: {str(e)[:100]}")
            routing_counts[RoutingDecision.QUARANTINE] += 1
            error_count += 1
        finally:
            # Track latency per record
            latencies_ms.append((time.time() - record_start) * 1000)
            
            # Progress log every 100 records
            if (parsed_count + 1) % 100 == 0:
                logger.info(f"📊 Processed {parsed_count + 1}/{sample_size} records...")
    
    # ─── 4. Final Metrics & Reporting ─────────────────────────────────────────
    total_time = time.time() - start_total
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    max_latency = max(latencies_ms) if latencies_ms else 0.0
    throughput = parsed_count / total_time if total_time > 0 else 0.0
    
    logger.info("\n" + "="*60)
    logger.info("📋 === MAS-DQA MVP Pipeline Results ===")
    logger.info("="*60)
    logger.info(f"✅ Records Processed: {parsed_count}")
    logger.info(f"⚠️  Errors: {error_count}")
    logger.info(f"\n🔀 Routing Distribution:")
    logger.info(f"   • TRUST:      {routing_counts[RoutingDecision.TRUST]:4d} ({routing_counts[RoutingDecision.TRUST]/parsed_count*100:.1f}%)")
    logger.info(f"   • JUDGE:      {routing_counts[RoutingDecision.JUDGE]:4d} ({routing_counts[RoutingDecision.JUDGE]/parsed_count*100:.1f}%)")
    logger.info(f"   • QUARANTINE: {routing_counts[RoutingDecision.QUARANTINE]:4d} ({routing_counts[RoutingDecision.QUARANTINE]/parsed_count*100:.1f}%)")
    logger.info(f"   • AMBIGUOUS:  {routing_counts[RoutingDecision.AMBIGUOUS]:4d} ({routing_counts[RoutingDecision.AMBIGUOUS]/parsed_count*100:.1f}%)")
    logger.info(f"\n⏱️  Performance:")
    logger.info(f"   • Avg Latency: {avg_latency:.2f} ms/record")
    logger.info(f"   • Max Latency: {max_latency:.2f} ms/record")
    logger.info(f"   • Throughput:  {throughput:.1f} records/sec")
    logger.info(f"   • Total Time:  {total_time:.2f} sec")
    
    # Phase I readiness check (Slide 9 target: Trust Rate > 80%)
    trust_rate = routing_counts[RoutingDecision.TRUST] / parsed_count if parsed_count > 0 else 0
    status = "✅ PASS" if trust_rate >= 0.80 else "⚠️  NEEDS IMPROVEMENT"
    logger.info(f"\n🎯 Phase I Readiness: Trust Rate = {trust_rate:.1%} {status}")
    logger.info("="*60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run the async pipeline
    asyncio.run(run_mvp_pipeline(
        stream_path="data/buspas_stream.json",    # Update path as needed
        baseline_path="data/buspas_baseline.csv",  # Update path as needed
        producer_id="agency_42",
        data_type="GTFS-RT",
        sample_size=1000,
        llm_model="gpt-4o-mini",
        ingest_delay=0.001  # Fast for MVP testing
    ))