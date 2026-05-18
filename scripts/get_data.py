import os
import json
import requests
import urllib3
from requests.auth import HTTPBasicAuth

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ELASTIC_URL = "https://1"
USERNAME = "elastic"
PASSWORD = "xxxxx"
INDEX = "gtfs_daily_stop_times"


def elastic_request(method, path, **kwargs):
    url = f"{ELASTIC_URL}{path}"

    response = requests.request(
        method=method,
        url=url,
        auth=HTTPBasicAuth(USERNAME, PASSWORD),
        verify=False,
        timeout=60,
        **kwargs,
    )

    print(f"{method} {url} -> {response.status_code}")
    response.raise_for_status()
    return response.json()


def export_all_records():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    output_file = os.path.join(
        data_dir,
        "raw.buspas.ndjson"
    )

    query = {
        "size": 5000,
        "query": {
            "match_all": {}
        }
    }

    data = elastic_request(
        "POST",
        f"/{INDEX}/_search?scroll=5m",
        json=query,
    )

    scroll_id = data["_scroll_id"]
    total = 0

    with open(output_file, "a") as f:

        while True:
            hits = data.get("hits", {}).get("hits", [])

            if not hits:
                print("Finished fetching.")
                break

            print(
                f"Fetched {len(hits)} docs | Total: {total}"
            )

            for hit in hits:
                json.dump(hit["_source"], f)
                f.write("\n")
                total += 1

            # force write to disk immediately
            f.flush()
            os.fsync(f.fileno())

            data = elastic_request(
                "POST",
                "/_search/scroll",
                json={
                    "scroll": "5m",
                    "scroll_id": scroll_id
                }
            )

            scroll_id = data.get("_scroll_id")

    # clear scroll
    if scroll_id:
        elastic_request(
            "DELETE",
            "/_search/scroll",
            json={"scroll_id": [scroll_id]}
        )

    print(
        f"\nSaved {total} docs to {output_file}"
    )


if __name__ == "__main__":
    export_all_records()