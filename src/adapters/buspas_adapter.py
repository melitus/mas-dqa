"""Dynamic, configuration-driven adapter for MAS-DQA."""
import json
import os
import logging
import glob
from typing import Dict, Any
import yaml

logger = logging.getLogger(__name__)

class DynamicAdapter:
    def __init__(self, config_dir: str = "src/adapters"):
        self.config_dir = config_dir
        self._cache = {}

    def _load_config(self, producer_id: str, schema_version: str = "v1") -> Dict:
        key = f"{producer_id}_{schema_version}"
        if key not in self._cache:
            # 1. Try exact naming convention first
            config_path = os.path.join(self.config_dir, f"{key}.yaml")
            
            # 2. Fallback: search for any YAML starting with producer_id in src/adapters
            if not os.path.exists(config_path):
                pattern = os.path.join(self.config_dir, f"{producer_id}*{schema_version}.yaml")
                matches = glob.glob(pattern)
                if matches:
                    config_path = matches[0]
                else:
                    raise FileNotFoundError(
                        f"No adapter config for {key} in {self.config_dir}. "
                        f"Run schema discovery or create YAML manually."
                    )
                    
            with open(config_path, "r", encoding="utf-8") as f:
                self._cache[key] = yaml.safe_load(f)
                logger.info(f"✅ Loaded adapter config: {os.path.basename(config_path)}")
                
        return self._cache[key]

    def _resolve_path(self, record: Dict, path_expr: str) -> Any:
        """Resolve nested path with fallbacks (e.g., 'stop_location.lat | stop_lat')."""
        for path in path_expr.split("|"):
            keys = path.strip().split(".")
            val = record
            try:
                for k in keys:
                    val = val[k]
                if val is not None:
                    return val
            except (KeyError, TypeError):
                continue
        return None

    def adapt(self, record: Dict[str, Any]) -> Dict[str, Any]:
        producer_id = record.get("source_system", "unknown")
        schema_version = record.get("schema_version", "v1")
        config = self._load_config(producer_id, schema_version)

        unified = {}
        for unified_key, path_expr in config["mappings"].items():
            val = self._resolve_path(record, path_expr)
            t = config.get("transformations", {}).get(unified_key)
            if t == "float" and val is not None: val = float(val)
            elif t == "int" and val is not None: val = int(val)
            unified[unified_key] = val

        unified["record_id"] = f"{unified.get('trip_id')}_{unified.get('stop_id')}_{record.get('service_date')}_{unified.get('stop_sequence')}"
        unified["producer_id"] = producer_id
        unified["record_type"] = config.get("record_type", "unknown")
        unified["metadata"] = {k: self._resolve_path(record, k) for k in config.get("metadata_passthrough", [])}
        return unified