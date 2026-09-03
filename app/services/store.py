# app/services/store.py
"""父块存储：用 SQLite 持久化 Small-to-Big 检索所需的父块文本。

父块不进向量库（避免污染召回），只在命中子块后按 id 取回，
用于拼接完整上下文。
"""
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class ParentStore:
    """父块键值存储（SQLite 实现，线程安全）"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.PARENT_STORE_PATH
        directory = os.path.dirname(self.db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parent_chunks (
                    id        TEXT PRIMARY KEY,
                    source    TEXT NOT NULL,
                    text      TEXT NOT NULL,
                    page      INTEGER,
                    file_path TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_parent_source ON parent_chunks(source)"
            )

    # ---------------- 写入 ----------------

    def put_many(self, items: List[Dict]) -> None:
        """批量写入父块，同 id 覆盖"""
        if not items:
            return
        rows = [
            (
                item["id"],
                item["source"],
                item["text"],
                item.get("page"),
                item.get("file_path"),
            )
            for item in items
        ]
        with self._lock, self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO parent_chunks (id, source, text, page, file_path) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )

    # ---------------- 读取 ----------------

    def get_many(self, ids: List[str]) -> Dict[str, Dict]:
        """按 id 批量取回父块，返回 {id: {...}}"""
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT id, source, text, page, file_path FROM parent_chunks "
                f"WHERE id IN ({placeholders})",
                list(ids),
            ).fetchall()
        return {row["id"]: dict(row) for row in rows}

    # ---------------- 删除 / 统计 ----------------

    def delete_by_source(self, source: str) -> None:
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM parent_chunks WHERE source = ?", (source,))

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM parent_chunks").fetchone()[0]
