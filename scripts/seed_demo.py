import os
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
NAMES = ["current-policy.md", "legacy-policy.md", "support-metrics.csv"]


def request(client, method, route, **kwargs):
    response = client.request(method, route, **kwargs)
    if not response.is_success:
        problem = response.json().get("error", {})
        raise RuntimeError(problem.get("message", f"HTTP {response.status_code}"))
    return response.json() if response.status_code != 204 else None


def seed(client):
    collections = request(client, "GET", "/collections?limit=100")["items"]
    collection = next((c for c in collections if c["name"] == "Northstar Support"), None)
    if not collection:
        collection = request(
            client,
            "POST",
            "/collections",
            json={
                "name": "Northstar Support",
                "description": "Original synthetic policies and metrics for a local RAG demo.",
            },
        )
    cid = collection["id"]
    existing = {
        d["name"]: d for d in request(client, "GET", f"/documents?collectionId={cid}&limit=100")["items"]
    }
    missing = [name for name in NAMES if name not in existing]
    if missing:
        uploaded = request(
            client,
            "POST",
            "/documents",
            data={"collectionId": cid, "ocr": "false"},
            files=[("files", (name, (ROOT / "docs" / "demo" / name).read_bytes())) for name in missing],
        )
        existing.update({d["name"]: d for d in uploaded["items"]})
    for name in NAMES:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            doc = request(client, "GET", f"/documents/{existing[name]['id']}")
            if doc["status"] == "ready":
                existing[name] = doc
                break
            if doc["status"] == "failed":
                raise RuntimeError(f"{name}: {doc['error']}")
            time.sleep(0.5)
        else:
            raise TimeoutError(f"{name} did not finish processing.")
    known = {
        c["question"]
        for c in request(client, "GET", f"/evaluations/cases?collectionId={cid}&limit=100")["items"]
    }
    examples = [
        ("What is the warranty for purchases after September 2026?", ["current-policy.md"], True),
        ("How long did the legacy hardware warranty last?", ["legacy-policy.md"], True),
        ("What is the device battery chemistry?", [], False),
    ]
    for question, expected, answerable in examples:
        if question not in known:
            request(
                client,
                "POST",
                "/evaluations/cases",
                json={
                    "collectionId": cid,
                    "question": question,
                    "expectedDocumentIds": [existing[name]["id"] for name in expected],
                    "answerable": answerable,
                },
            )
    return collection


def main():
    base = os.environ.get("RAG_BASE_URL", "http://127.0.0.1:8787/api/v1").rstrip("/")
    with httpx.Client(base_url=base + "/", timeout=180) as client:
        collection = seed(client)
    print(f"Demo ready: {collection['name']} ({collection['id']}), 3 files and 3 evaluation questions.")
    print("Open http://127.0.0.1:5173. No NVIDIA generation request was made.")


if __name__ == "__main__":
    main()
