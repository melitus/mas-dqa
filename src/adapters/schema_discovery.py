"""Auto-discovers schema and generates draft adapter config."""
import json
import yaml
from typing import Dict, Any, List

def discover_schema(sample_records: List[Dict], producer_id: str) -> Dict:
    """Generate draft config from 5-10 sample records."""
    mappings = {}
    metadata_passthrough = []
    
    # Heuristic: if field exists in >80% of samples, map it; else metadata
    for key in sample_records[0].keys():
        present_count = sum(1 for r in sample_records if key in r)
        if present_count / len(sample_records) >= 0.8 and key not in ["metadata", "source_indices"]:
            mappings[key] = key  # Direct map for flat fields
        else:
            metadata_passthrough.append(key)
            
    config = {
        "producer_id": producer_id,
        "schema_version": "v1",
        "record_type": "unknown",
        "mappings": mappings,
        "metadata_passthrough": metadata_passthrough,
        "defaults": {"record_type": "auto_discovered"},
        "transformations": {}
    }
    return config

def register_adapter(config: Dict, output_dir: str = "data/adapters"):
    """Save config and make it available to DynamicAdapter."""
    filename = f"{config['producer_id']}_{config['schema_version']}.yaml"
    path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return path