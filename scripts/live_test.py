import hashlib
import os
import re

import httpx

from .seed_demo import NAMES, ROOT, request


def main():
    if os.environ.get("RAG_LIVE_TEST") != "1":
        raise SystemExit("Set RAG_LIVE_TEST=1 to enable the synthetic live NVIDIA test.")
    base = os.environ.get("RAG_BASE_URL", "").rstrip("/")
    if not base:
        raise SystemExit("Set RAG_BASE_URL to an isolated API URL, then seed its demo collection.")
    with httpx.Client(base_url=base + "/", timeout=180) as client:
        config = request(client, "GET", "/config/status")
        assert config["modelMode"] == "nvidia" and config["nvidiaConfigured"]
        collections = request(client, "GET", "/collections?limit=100")["items"]
        collection = next(c for c in collections if c["name"] == "Northstar Support")
        documents = request(client, "GET", f"/documents?collectionId={collection['id']}&limit=100")["items"]
        assert len(documents) == len(NAMES), "Use only the three synthetic demo files."
        by_name = {doc["name"]: doc for doc in documents}
        for name in NAMES:
            assert by_name[name]["status"] == "ready"
            assert (
                by_name[name]["sha256"]
                == hashlib.sha256((ROOT / "docs" / "demo" / name).read_bytes()).hexdigest()
            )
        cases = [
            (
                "current warranty",
                "What is the warranty period for purchases after September 2026?",
                [],
                "answered",
                r"24.month",
                ["current-policy.md"],
            ),
            (
                "legacy warranty",
                "How long did the legacy hardware warranty last?",
                ["legacy-policy.md"],
                "answered",
                r"12.month",
                ["legacy-policy.md"],
            ),
            (
                "support hours",
                "When is live support available?",
                ["current-policy.md"],
                "answered",
                r"09:00.*18:00",
                ["current-policy.md"],
            ),
            (
                "CSV row",
                "How many tickets did South have in August 2026?",
                ["support-metrics.csv"],
                "answered",
                r"\b110\b",
                ["support-metrics.csv"],
            ),
            (
                "unsupported fact",
                "What is the device battery chemistry?",
                ["current-policy.md"],
                "abstained",
                r"not contain enough evidence|not enough|cannot determine",
                [],
            ),
            (
                "conflicting policies",
                "How long is the Northstar Devices hardware warranty?",
                ["current-policy.md", "legacy-policy.md"],
                "abstained",
                r"24 month.*12 month|12 month.*24 month",
                ["current-policy.md", "legacy-policy.md"],
            ),
        ]
        for name, question, filters, status, pattern, cited_names in cases:
            result = request(
                client,
                "POST",
                "/queries",
                json={
                    "collectionId": collection["id"],
                    "question": question,
                    "documentIds": [by_name[n]["id"] for n in filters],
                    "mode": "hybrid",
                    "generate": True,
                    "topK": 6,
                },
            )
            assert result["answerStatus"] == status, f"{name}: {result}"
            assert re.search(pattern, result["answer"], re.I | re.S), f"{name}: {result['answer']}"
            assert "The source states:" not in result["answer"]
            sources = {s["chunkId"]: s for s in result["sources"]}
            cited = []
            for citation in result["citations"]:
                assert citation["chunkId"] in sources
                assert f"[{citation['marker']}]" in result["answer"]
                source = sources[citation["chunkId"]]
                cited.append(source["documentName"])
                if name == "CSV row":
                    assert source["location"]["row"] == 5
            assert set(cited_names).issubset(cited)
            print(f"PASS {name}: {result['answerStatus']}; citations={','.join(cited) or 'none'}")
    print("Six live-mode cases passed. Live checks use synthetic repository documents only.")


if __name__ == "__main__":
    main()
