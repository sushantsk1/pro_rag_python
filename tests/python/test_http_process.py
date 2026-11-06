import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def test_python_entrypoint_serves_existing_dashboard_contract(tmp_path):
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    env = {
        **os.environ,
        "PORT": str(port),
        "DATA_DIR": str(tmp_path / "http-data"),
        "MODEL_MODE": "stub",
        "EMBEDDING_MODE": "stub",
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "server.main"],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}/api/v1/", timeout=10) as client:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    if client.get("health").status_code == 200:
                        break
                except httpx.ConnectError:
                    pass
                assert process.poll() is None, "Python API exited before becoming healthy"
                time.sleep(0.1)
            else:
                raise AssertionError("Python API did not become healthy")
            collection = client.post("collections", json={"name": "HTTP smoke"}).json()
            uploaded = client.post(
                "documents",
                data={"collectionId": collection["id"]},
                files={"files": ("policy.txt", b"Warranty coverage lasts 24 months.")},
            )
            assert uploaded.status_code == 202
            document_id = uploaded.json()["items"][0]["id"]
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                doc = client.get(f"documents/{document_id}").json()
                if doc["status"] in ("ready", "failed"):
                    break
                time.sleep(0.1)
            assert doc["status"] == "ready", doc
            response = client.post(
                "queries",
                json={
                    "collectionId": collection["id"],
                    "question": "How long is warranty coverage?",
                    "mode": "hybrid",
                    "generate": True,
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["answerStatus"] == "answered"
            assert response.json()["sources"][0]["documentId"] == document_id
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
