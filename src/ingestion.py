# Stream loader & schema parser - MVP Simplified

"""
Minimal stream ingestor for MAS-DQA MVP testing.
Parses records, validates basic structure, computes schema hashes.

TODO for production:
- Add Kafka/Pulsar adapters
- Add schema evolution tracking
- Add backpressure handling
"""

import asyncio
import hashlib
import json
import logging
from typing import Dict, Any, AsyncGenerator, Optional, List
from datetime import datetime

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
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
                timestamp=datetime.utcnow().isoformat(),
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
        delay: float = 0.01  # Faster for testing
    ) -> AsyncGenerator[ParsedRecord, None]:
        """
        Simulate streaming from a JSON file (list of records).
        
        For MVP testing only. Replace with Kafka/Pulsar in production.
        """
        try:
            with open(file_path, 'r') as f:
                records = json.load(f)
        except FileNotFoundError:
            logger.error(f"❌ File not found: {file_path}")
            return
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in {file_path}: {e}")
            return

        logger.info(f"▶ Starting stream: {file_path} | Producer: {producer_id}")
        
        for idx, record in enumerate(records):
            parsed = await self.parse_record(record, producer_id, data_type)
            if parsed:
                yield parsed
            
            # Simulate real-time interval (adjust for testing speed)
            if delay > 0:
                await asyncio.sleep(delay)
            
            # Progress log every 100 records
            if (idx + 1) % 100 == 0:
                logger.info(f"📊 Streamed {idx + 1} records...")


# ──────────────────────────────────────────────────────────────────────────────
# QUICK USAGE EXAMPLE (for testing)
# ──────────────────────────────────────────────────────────────────────────────

async def demo_ingestion():
    """Simple demo: ingest 10 records from a test file."""
    ingestor = StreamIngestor(required_fields=["timestamp", "value"])
    
    # Create a tiny test file if it doesn't exist
    import os
    test_file = "data/test_stream.json"
    if not os.path.exists(test_file):
        os.makedirs("data", exist_ok=True)
        test_data = [
            {"timestamp": "2026-05-12T10:00:00Z", "value": 42, "sensor": "A"},
            {"timestamp": "2026-05-12T10:00:01Z", "value": 43, "sensor": "A"},
            {"timestamp": "2026-05-12T10:00:02Z", "value": 41, "sensor": "A"},
        ] * 4  # 12 records total
        with open(test_file, 'w') as f:
            json.dump(test_data, f)
        logger.info(f"✅ Created test file: {test_file}")
    
    # Stream and print
    async for record in ingestor.stream_from_file(test_file, producer_id="test_sensor_01", delay=0.001):
        print(f"✓ Parsed: {record.metadata.producer_id} | schema_hash={record.metadata.schema_hash} | keys={list(record.payload.keys())}")


if __name__ == "__main__":
    asyncio.run(demo_ingestion())