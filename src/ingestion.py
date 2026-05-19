# Stream loader & schema parser - MVP Simplified (NDJSON-Compatible)

"""
Minimal stream ingestor for MAS-DQA MVP testing.
Parses records, validates basic structure, computes schema hashes.

Supports both formats:
- NDJSON: One JSON object per line (preferred for streaming)
- JSON Array: [...list of objects...] (for backward compatibility)

TODO for production:
- Add Kafka/Pulsar adapters
- Add schema evolution tracking
- Add backpressure handling
"""

import asyncio
import hashlib
import json
import logging
import os
from typing import Dict, Any, AsyncGenerator, Optional, List
from datetime import datetime, timezone

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# I/O SCHEMAS (Pydantic v2 compliant)
# ──────────────────────────────────────────────────────────────────────────────

class ProducerMetadata(BaseModel):
    """Minimal metadata for routing and drift detection."""
    producer_id: str
    schema_hash: str          # Hash of payload keys for structural drift detection
    timestamp: str            # ISO-8601 ingestion time
    data_type: str            # e.g., "GTFS-RT", "AVL"


class ParsedRecord(BaseModel):
    """Normalized record ready for Event Detection."""
    metadata: ProducerMetadata
    payload: Dict[str, Any]   # Raw data (normalized)
    raw_length_bytes: int     # For basic throughput tracking


# ──────────────────────────────────────────────────────────────────────────────
# SIMPLIFIED STREAM INGESTOR
# ──────────────────────────────────────────────────────────────────────────────

class StreamIngestor:
    """
    MVP ingestor: parses records, validates required fields, emits ParsedRecord.
    
    Keeps it simple:
    - Single schema registry (dict-based)
    - Basic structural validation (required fields only)
    - SHA-256 hash for schema drift detection
    - Async generator for pipeline compatibility
    - NDJSON support (one JSON object per line)
    """
    
    def __init__(self, required_fields: Optional[List[str]] = None):
        """
        Initialize with minimal config.
        
        Args:
            required_fields: List of field names that MUST be present in every record.
                           Default: ["timestamp"] (minimal requirement)
        """
        self.required_fields = required_fields or ["timestamp"]
        logger.info(f"✅ StreamIngestor initialized. Required fields: {self.required_fields}")

    @staticmethod
    def _compute_schema_hash(payload: Dict[str, Any]) -> str:
        """Deterministic SHA-256 hash of payload keys (structure only)."""
        sorted_keys = json.dumps(sorted(payload.keys()), sort_keys=True)
        return hashlib.sha256(sorted_keys.encode()).hexdigest()[:16]

    def _validate_structure(self, payload: Dict[str, Any]) -> bool:
        """Check that all required fields are present."""
        return all(key in payload for key in self.required_fields)

    async def parse_record(
        self, 
        raw_data: Dict[str, Any], 
        producer_id: str, 
        data_type: str = "generic"
    ) -> Optional[ParsedRecord]:
        """
        Parse, validate, and wrap a single record.
        
        Returns None if validation fails (caller should skip/log).
        """
        try:
            if not self._validate_structure(raw_data):
                logger.debug(f"⚠️  Structural validation failed for {producer_id}")
                return None

            metadata = ProducerMetadata(
                producer_id=producer_id,
                schema_hash=self._compute_schema_hash(raw_data),
                timestamp=datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                data_type=data_type
            )
            
            return ParsedRecord(
                metadata=metadata,
                payload=raw_data,
                raw_length_bytes=len(json.dumps(raw_data).encode())
            )
            
        except Exception as e:
            logger.error(f"❌ Parse error for {producer_id}: {e}")
            return None

    async def stream_from_file(
        self, 
        file_path: str, 
        producer_id: str, 
        data_type: str = "generic",
        delay: float = 0.001  # Faster for testing
    ) -> AsyncGenerator[ParsedRecord, None]:
        """
        Stream records from file supporting both NDJSON and JSON Array formats.
        
        NDJSON (preferred): One JSON object per line
        JSON Array: [...list of objects...] (backward compatibility)
        
        For MVP testing only. Replace with Kafka/Pulsar in production.
        """
        if not os.path.exists(file_path):
            logger.error(f"❌ File not found: {file_path}")
            return

        logger.info(f"▶ Starting stream: {file_path} | Producer: {producer_id}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                logger.warning(f"⚠️  Empty file: {file_path}")
                return
            
            # Detect format and parse accordingly
            if content.startswith('['):
                # JSON Array format: parse once, yield each item
                logger.debug("📋 Detected JSON Array format")
                try:
                    records = json.loads(content)
                    if not isinstance(records, list):
                        raise ValueError("JSON array expected list of objects")
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Invalid JSON array in {file_path}: {e}")
                    return
            else:
                # NDJSON format: parse line-by-line
                logger.debug("📋 Detected NDJSON format (one object per line)")
                records = []
                for line_num, line in enumerate(content.split('\n'), 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Invalid JSON on line {line_num} in {file_path}: {e}")
                        logger.error(f"   Problematic line: {line[:100]}...")
                        continue  # Skip bad lines, continue processing
            
            # Stream parsed records
            for idx, record in enumerate(records):
                if not isinstance(record, dict):
                    logger.warning(f"⚠️  Skipping non-dict record at index {idx}")
                    continue
                    
                parsed = await self.parse_record(record, producer_id, data_type)
                if parsed:
                    yield parsed
                
                # Simulate real-time interval (adjust for testing speed)
                if delay > 0:
                    await asyncio.sleep(delay)
                
                # Progress log every 100 records
                if (idx + 1) % 100 == 0:
                    logger.info(f"📊 Streamed {idx + 1} records...")
                    
        except Exception as e:
            logger.error(f"❌ Unexpected error streaming {file_path}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


# ──────────────────────────────────────────────────────────────────────────────
# QUICK USAGE EXAMPLE (for testing)
# ──────────────────────────────────────────────────────────────────────────────

async def demo_ingestion():
    """Simple demo: ingest records from a test file (NDJSON format)."""
    ingestor = StreamIngestor(required_fields=["timestamp"])
    
    # Create a tiny test file if it doesn't exist (NDJSON format)
    test_file = "data/test_stream.ndjson"
    if not os.path.exists(test_file):
        os.makedirs("data", exist_ok=True)
        test_records = [
            {"timestamp": "2026-05-12T10:00:00Z", "value": 42, "sensor": "A"},
            {"timestamp": "2026-05-12T10:00:01Z", "value": 43, "sensor": "A"},
            {"timestamp": "2026-05-12T10:00:02Z", "value": 41, "sensor": "A"},
        ] * 4  # 12 records total
        with open(test_file, 'w', encoding='utf-8') as f:
            for rec in test_records:
                f.write(json.dumps(rec) + '\n')  # NDJSON: one object per line
        logger.info(f"✅ Created test file: {test_file}")
    
    # Stream and print
    count = 0
    async for record in ingestor.stream_from_file(test_file, producer_id="test_sensor_01", delay=0.001):
        count += 1
        if count <= 3:
            print(f"✓ Parsed: {record.metadata.producer_id} | schema_hash={record.metadata.schema_hash} | keys={list(record.payload.keys())}")
    
    logger.info(f"✅ Demo complete: processed {count} records")


if __name__ == "__main__":
    asyncio.run(demo_ingestion())