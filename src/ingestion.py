# Stream loader & schema parser
import asyncio
import hashlib
import json
import logging
from typing import Dict, Any, AsyncGenerator, Optional
from datetime import datetime
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# I/O SCHEMAS (Pydantic v2 compliant)
# ──────────────────────────────────────────────────────────────────────────────
class ProducerMetadata(BaseModel):
    producer_id: str
    schema_hash: str          # Deterministic hash of payload keys for drift detection
    timestamp: str            # ISO-8601 ingestion time
    data_type: str            # e.g., "GTFS-RT", "AVL", "Vitals"

class ParsedRecord(BaseModel):
    metadata: ProducerMetadata
    payload: Dict[str, Any]   # Normalized raw data
    raw_length_bytes: int     # For latency/throughput tracking

# ──────────────────────────────────────────────────────────────────────────────
# STREAM INGESTOR
# ──────────────────────────────────────────────────────────────────────────────
class StreamIngestor:
    """
    Ingests heterogeneous streams, validates basic structure, 
    computes schema hashes, and emits parsed records for Event Detection.
    """
    def __init__(self, schema_registry: Dict[str, Dict[str, Any]]):
        """
        schema_registry format:
        {
            "GTFS-RT": {"required": ["vehicle_id", "lat", "lon", "timestamp"]},
            "AVL": {"required": ["bus_id", "speed", "heading", "ts"]},
            "Vitals": {"required": ["patient_id", "hr", "spo2", "ts"]}
        }
        """
        self.schema_registry = schema_registry
        self._supported_types = set(schema_registry.keys())

    def _compute_schema_hash(self, payload: Dict[str, Any]) -> str:
        """Deterministic SHA-256 hash of payload keys (structure only)."""
        sorted_keys = json.dumps(sorted(payload.keys()), sort_keys=True)
        return hashlib.sha256(sorted_keys.encode()).hexdigest()[:16]

    def _validate_structure(self, payload: Dict[str, Any], data_type: str) -> bool:
        """Lightweight structural check before handing off to Profiler/Validator."""
        if data_type not in self._supported_types:
            return False
        required = self.schema_registry[data_type].get("required", [])
        return all(key in payload for key in required)

    async def parse_record(self, raw_data: Dict[str, Any], producer_id: str, data_type: str) -> Optional[ParsedRecord]:
        """Parse, validate, hash, and wrap record for downstream routing."""
        try:
            if not self._validate_structure(raw_data, data_type):
                logger.warning(f"Schema validation failed for {producer_id} ({data_type})")
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
            logger.error(f"Parsing error for {producer_id}: {e}")
            return None

    async def stream_from_file(self, file_path: str, producer_id: str, data_type: str, delay: float = 0.05) -> AsyncGenerator[ParsedRecord, None]:
        """Simulate real-time streaming from JSON file. Replace with Kafka/Pulsar in prod."""
        with open(file_path, 'r') as f:
            records = json.load(f)

        logger.info(f"▶ Starting stream: {file_path} | Producer: {producer_id} | Type: {data_type}")
        for record in records:
            parsed = await self.parse_record(record, producer_id, data_type)
            if parsed:
                yield parsed
            await asyncio.sleep(delay)  # Simulate real-world interval (e.g., 50ms)

# ──────────────────────────────────────────────────────────────────────────────
# PRODUCTION ADAPTERS (Placeholders for real-time deployment)
# ──────────────────────────────────────────────────────────────────────────────
"""
To replace `stream_from_file` in production:

1. KAFKA:
   from aiokafka import AIOKafkaConsumer
   async def stream_from_kafka(topic: str, consumer_group: str):
       consumer = AIOKafkaConsumer(topic, group_id=consumer_group)
       await consumer.start()
       async for msg in consumer:
           yield json.loads(msg.value.decode())

2. AWS PULSAR:
   import pulsar
   async def stream_from_pulsar(topic: str, service_url: str):
       client = pulsar.Client(service_url)
       consumer = client.subscribe(topic, 'mas-dqa-sub')
       while True:
           msg = await consumer.receive_async()
           yield json.loads(msg.data.decode())
           consumer.acknowledge(msg)
"""