#!/usr/bin/env python3
"""
Export a realistic, filled prompt template using REAL data from your pipeline.
- Loads sample record from normalised.buspas.ndjson
- Loads domain context from adapter config YAML
- Supports any producer via --producer argument

Generates: prompt/example_filled_prompt.txt
"""
import os
import sys
import json
import argparse
import yaml
import random

# 🔑 CRITICAL: Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.validator.prompt import build_validation_prompt, PROMPT_CONFIG
from src.schemas.validator import DomainContext

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_PRODUCER = "njtransit"
DEFAULT_SCHEMA_VERSION = "v1"
NORMALIZED_DATA_PATH = os.path.join(PROJECT_ROOT, "data/normalised.buspas.ndjson")
ADAPTER_CONFIG_DIR = os.path.join(PROJECT_ROOT, "src/adapters")
PROMPT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "prompt")


def load_sample_record(producer_id: str, max_samples: int = 10) -> dict:
    """
    Load a random sample record for the given producer from normalized data.
    
    Args:
        producer_id: Filter records by producer_id field
        max_samples: Stop after finding this many matching records (for efficiency)
    
    Returns:
        A single record dict, or None if not found
    """
    if not os.path.exists(NORMALIZED_DATA_PATH):
        raise FileNotFoundError(f"Normalized data not found: {NORMALIZED_DATA_PATH}")
    
    matching_records = []
    
    with open(NORMALIZED_DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples * 10:  # Safety limit to avoid scanning huge files
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("producer_id") == producer_id:
                    matching_records.append(record)
                    if len(matching_records) >= max_samples:
                        break
            except json.JSONDecodeError:
                continue
    
    if not matching_records:
        raise ValueError(f"No records found for producer '{producer_id}' in {NORMALIZED_DATA_PATH}")
    
    # Return random sample for variety
    return random.choice(matching_records)


def load_domain_context_from_config(producer_id: str, schema_version: str = DEFAULT_SCHEMA_VERSION) -> DomainContext:
    """
    Load domain context (rules, contracts) from adapter config YAML.
    
    Args:
        producer_id: Producer identifier (e.g., 'njtransit')
        schema_version: Schema version (e.g., 'v1')
    
    Returns:
        DomainContext object ready for prompt building
    """
    # Try exact naming first, then glob fallback
    config_filename = f"{producer_id}_{schema_version}.yaml"
    config_path = os.path.join(ADAPTER_CONFIG_DIR, config_filename)
    
    if not os.path.exists(config_path):
        # Fallback: search for any YAML starting with producer_id
        import glob
        pattern = os.path.join(ADAPTER_CONFIG_DIR, f"{producer_id}*{schema_version}.yaml")
        matches = glob.glob(pattern)
        if matches:
            config_path = matches[0]
        else:
            raise FileNotFoundError(
                f"No adapter config found for {producer_id}_{schema_version} in {ADAPTER_CONFIG_DIR}. "
                f"Run adapter discovery or create YAML manually."
            )
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Map config to DomainContext schema
    # Note: contracts must be Dict[str, str], not List[str]
    rules = config.get("validation_rules", {})
    contracts_dict = {}
    raw_contracts = config.get("contracts", [])
    if isinstance(raw_contracts, list):
        for i, c in enumerate(raw_contracts):
            contracts_dict[f"rule_{i+1}"] = str(c)
    elif isinstance(raw_contracts, dict):
        contracts_dict = raw_contracts
    
    # Schedules: if your schema expects List[ScheduleEntry], simplify to empty list for now
    # Or adapt based on your actual DomainContext definition
    schedules_list = []  # Empty if ScheduleEntry not defined yet
    
    return DomainContext(
        rules=rules,
        contracts=contracts_dict,
        schedules=schedules_list
    )


def build_and_export_prompt(producer_id: str, output_filename: str = "example_filled_prompt.txt"):
    """
    Main workflow: load real data + config → build prompt → export to file.
    """
    print(f"🔍 Loading data for producer: {producer_id}")
    
    # 1. Load sample record
    sample_record = load_sample_record(producer_id)
    print(f"✅ Loaded sample record: {sample_record.get('record_id', 'unknown')}")
    
    # 2. Load domain context from config
    domain_context = load_domain_context_from_config(producer_id)
    print(f"✅ Loaded domain context from adapter config")
    
    # 3. Build the prompt
    prompt_text = build_validation_prompt(
        record=sample_record,
        domain_context=domain_context,
        attempt=1,
        config=PROMPT_CONFIG
    )
    
    # 4. Export to prompt/ folder
    os.makedirs(PROMPT_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(PROMPT_OUTPUT_DIR, output_filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    
    # 5. Print summary
    print(f"\n✅ Prompt exported successfully!")
    print(f"📄 Saved to: {output_path}")
    print(f"📏 Length: {len(prompt_text):,} characters")
    print(f"\n👀 Preview (first 400 chars):\n{'─'*40}")
    print(prompt_text[:400] + "...\n")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export dynamic prompt template from real MAS-DQA data")
    parser.add_argument(
        "--producer", 
        type=str, 
        default=DEFAULT_PRODUCER,
        help=f"Producer ID to load data for (default: {DEFAULT_PRODUCER})"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="example_filled_prompt.txt",
        help="Output filename (default: example_filled_prompt.txt)"
    )
    parser.add_argument(
        "--list-producers", 
        action="store_true",
        help="List available producers in normalized data and exit"
    )
    
    args = parser.parse_args()
    
    # Optional: list available producers
    if args.list_producers:
        if not os.path.exists(NORMALIZED_DATA_PATH):
            print(f"❌ Normalized data not found: {NORMALIZED_DATA_PATH}")
            return
        
        producers = set()
        with open(NORMALIZED_DATA_PATH, "r") as f:
            for i, line in enumerate(f):
                if i > 1000:  # Sample first 1000 lines
                    break
                try:
                    record = json.loads(line.strip())
                    if "producer_id" in record:
                        producers.add(record["producer_id"])
                except:
                    continue
        
        print(f"📋 Available producers in {NORMALIZED_DATA_PATH}:")
        for p in sorted(producers):
            print(f"  • {p}")
        return
    
    # Main workflow
    try:
        build_and_export_prompt(args.producer, args.output)
    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
        print(f"💡 Tip: Run adapter first: python scripts/test_adapter_flow.py")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Data error: {e}")
        print(f"💡 Tip: Check producer ID with --list-producers")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()