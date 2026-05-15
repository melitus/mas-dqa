"""Adapter layer for heterogeneous BusPas data shapes.

Converts arbitrary agency-specific schemas to a normalized format
for Profiler/Validator consumption.
"""
from typing import Dict, Any, Optional
import re
from datetime import datetime

class BusPasAdapter:
    """Normalize heterogeneous BusPas records to pipeline-ready format."""
    
    @staticmethod
    def normalize(record: Dict[str, Any], agency_schema: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Convert arbitrary record to normalized format.
        
        Args:
            record: Raw input (any shape)
            agency_schema: Optional schema hints for specific agency
            
        Returns:
            Normalized record with record_id, timestamp, payload, metadata
        """
        # 1. Generate stable record_id (fallback: hash of raw record)
        record_id = (
            record.get("record_id") or 
            record.get("id") or 
            f"{record.get('agency_id', 'unknown')}_{record.get('timestamp', '')}_{hash(str(record)) % 10000}"
        )
        
        # 2. Parse timestamp (flexible: ISO8601, Unix, custom)
        timestamp = BusPasAdapter._parse_timestamp(record)
        
        # 3. Extract payload: keep original structure intact
        payload = record.copy()
        # Remove adapter-specific fields from payload
        for key in ["record_id", "id", "timestamp", "event_time", "ts", "metadata"]:
            payload.pop(key, None)
        
        # 4. Build metadata from available hints
        metadata = {
            "agency_id": record.get("agency_id") or record.get("source", "unknown"),
            "source_format": BusPasAdapter._detect_format(record, agency_schema),
            "schema_version": record.get("schema_version", "unknown"),
            "original_keys": list(record.keys())  # For debugging
        }
        
        return {
            "record_id": str(record_id),
            "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
            "payload": payload,
            "metadata": metadata
        }
    
    @staticmethod
    def _parse_timestamp(record: Dict) -> Any:
        """Flexible timestamp parsing: ISO8601, Unix, or custom."""
        # Try common timestamp fields
        for key in ["timestamp", "event_time", "ts", "time", "datetime"]:
            if key in record:
                val = record[key]
                # Unix timestamp (int/float)
                if isinstance(val, (int, float)):
                    return datetime.fromtimestamp(val)
                # ISO8601 string
                if isinstance(val, str):
                    try:
                        return datetime.fromisoformat(val.replace("Z", "+00:00"))
                    except ValueError:
                        pass
        # Fallback: current time (for testing)
        return datetime.now()
    
    @staticmethod
    def _detect_format(record: Dict, schema_hints: Optional[Dict] = None) -> str:
        """Detect source format from field names."""
        if "route_id" in record and "trip_id" in record:
            return "GTFS-RT"
        if "lat" in record and "lon" in record:
            return "AVL"
        if "location" in record and isinstance(record["location"], dict):
            return "Nested-GeoJSON"
        return "Custom"