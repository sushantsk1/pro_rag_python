import base64
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .errors import RagError


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class LocalStore:
    """Preserves the original SQLite schema, IDs, and JSON record contract."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.files_dir = self.data_dir / "files"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.data_dir / "rag.sqlite", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS records (type TEXT NOT NULL, id TEXT NOT NULL, "
            "collection_id TEXT, created_at TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY(type,id))"
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS records_type_collection ON records(type,collection_id)")
        self.db.commit()

    def put(self, kind: str, value: dict, *, commit: bool = True) -> dict:
        row = {"createdAt": timestamp(), "updatedAt": timestamp(), **value}
        with self.lock:
            self.db.execute(
                "INSERT INTO records VALUES(?,?,?,?,?) ON CONFLICT(type,id) DO UPDATE SET "
                "collection_id=excluded.collection_id,created_at=excluded.created_at,data=excluded.data",
                (kind, row["id"], row.get("collectionId"), row["createdAt"], json.dumps(row)),
            )
            if commit:
                self.db.commit()
        return row

    def get(self, kind: str, record_id: str) -> dict | None:
        with self.lock:
            row = self.db.execute(
                "SELECT data FROM records WHERE type=? AND id=?", (kind, record_id)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def all(self, kind: str, collection_id: str | None = None) -> list[dict]:
        with self.lock:
            if collection_id is None:
                rows = self.db.execute("SELECT data FROM records WHERE type=?", (kind,)).fetchall()
            else:
                rows = self.db.execute(
                    "SELECT data FROM records WHERE type=? AND collection_id=?", (kind, collection_id)
                ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def delete(self, kind: str, record_id: str, *, commit: bool = True) -> bool:
        with self.lock:
            removed = self.db.execute("DELETE FROM records WHERE type=? AND id=?", (kind, record_id)).rowcount
            if commit:
                self.db.commit()
        return removed > 0

    def delete_where(self, kind: str, predicate, collection_id: str | None = None) -> list[dict]:
        with self.lock:
            items = [item for item in self.all(kind, collection_id) if predicate(item)]
            for item in items:
                self.delete(kind, item["id"], commit=False)
            self.db.commit()
        return items

    def replace_chunks(self, document: dict, chunks: list[dict]):
        with self.lock:
            try:
                self.db.execute("BEGIN IMMEDIATE")
                old = [
                    c
                    for c in self.all("chunk", document["collectionId"])
                    if c["documentId"] == document["id"]
                ]
                for chunk in old:
                    self.delete("chunk", chunk["id"], commit=False)
                for chunk in chunks:
                    self.put("chunk", chunk, commit=False)
                self.put(
                    "document",
                    {
                        **document,
                        "status": "ready",
                        "progress": 100,
                        "chunkCount": len(chunks),
                        "error": None,
                        "updatedAt": timestamp(),
                    },
                    commit=False,
                )
                self.db.commit()
            except Exception:
                self.db.rollback()
                raise

    @staticmethod
    def page(items: list[dict], limit=20, cursor=None) -> dict:
        try:
            maximum = int(limit)
            if str(maximum) != str(limit) or not 1 <= maximum <= 100:
                raise ValueError
        except (ValueError, TypeError):
            raise RagError(400, "VALIDATION_ERROR", "limit must be 1 to 100") from None
        ordered = sorted(items, key=lambda item: (item["createdAt"], item["id"]), reverse=True)
        offset = 0
        if cursor:
            try:
                parsed = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
                offset = parsed["offset"]
                if (
                    type(offset) is not int
                    or not 0 <= offset <= len(ordered)
                    or parsed["count"] != len(ordered)
                ):
                    raise ValueError
            except (ValueError, TypeError, KeyError):
                raise RagError(400, "VALIDATION_ERROR", "Invalid or stale cursor.") from None
        next_offset = offset + maximum
        next_cursor = None
        if next_offset < len(ordered):
            next_cursor = (
                base64.urlsafe_b64encode(json.dumps({"offset": next_offset, "count": len(ordered)}).encode())
                .decode()
                .rstrip("=")
            )
        return {"items": ordered[offset:next_offset], "nextCursor": next_cursor}

    def close(self):
        with self.lock:
            self.db.close()
