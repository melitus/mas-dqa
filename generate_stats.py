"""Generate baseline statistics for the MAS‑DQA Profiler.

The Profiler expects a CSV file (``data/stats.csv``) that contains a
``feature,mean,std`` row for every numeric field present in the streaming
input (``data/buspas_stream.json``).  This script reads the full stream,
computes the arithmetic mean and **population** standard deviation for
each numeric column using only the Python standard library, and writes the
resulting table.

Usage
-----
Run the script from the repository root:

    python generate_stats.py

It will create (or overwrite) ``data/stats.csv``.
"""

import json
import csv
import os
import statistics
from typing import List, Dict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_stream(path: str) -> List[Dict]:
    """Load the JSON stream file.

    The file is expected to contain a JSON *array* of records.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_numeric(data: List[Dict]) -> Dict[str, List[float]]:
    """Collect all numeric values for each field.

    Returns a dict mapping ``field_name`` → ``list_of_numbers``.
    """
    numeric: Dict[str, List[float]] = {}
    for rec in data:
        for key, value in rec.items():
            if isinstance(value, (int, float)):
                numeric.setdefault(key, []).append(float(value))
    return numeric


def compute_stats(numeric: Dict[str, List[float]]) -> List[tuple]:
    """Compute mean and standard deviation for each numeric field.

    For a field with a single observation the standard deviation is set to
    ``0.0`` (the ``statistics`` module raises ``StatisticsError`` for that
    case).
    """
    rows = []
    for field, values in numeric.items():
        if len(values) == 0:
            mean = std = 0.0
        elif len(values) == 1:
            mean = values[0]
            std = 0.0
        else:
            mean = statistics.mean(values)
            std = statistics.stdev(values)
        rows.append((field, mean, std))
    return rows


def write_csv(rows: List[tuple], out_path: str) -> None:
    """Write ``feature,mean,std`` CSV.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["feature", "mean", "std"])
        for row in rows:
            writer.writerow([row[0], f"{row[1]:.6f}", f"{row[2]:.6f}"])
    print(f"Baseline statistics written to {out_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    stream_path = os.path.join("data", "buspas_stream.json")
    out_path = os.path.join("data", "stats.csv")

    if not os.path.exists(stream_path):
        raise FileNotFoundError(f"Stream file not found: {stream_path}")

    records = load_stream(stream_path)
    numeric = collect_numeric(records)
    stats = compute_stats(numeric)
    write_csv(stats, out_path)
