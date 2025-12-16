import json
import sqlite3
import threading
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from .config import DB_PATH, ensure_dirs


class Database:
    def __init__(self, path: Path = DB_PATH) -> None:
        ensure_dirs()
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS photos (
                    photo_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    orig_name TEXT,
                    width INTEGER,
                    height INTEGER,
                    no_face INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    photo_id TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    bbox TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    group_id TEXT,
                    FOREIGN KEY(photo_id) REFERENCES photos(photo_id) ON DELETE CASCADE
                );
                """
            )

    def add_photo(
        self,
        photo_id: str,
        file_path: str,
        orig_name: str,
        width: int,
        height: int,
        no_face: bool,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO photos (photo_id, file_path, orig_name, width, height, no_face)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (photo_id, file_path, orig_name, width, height, int(no_face)),
            )

    def add_faces(
        self,
        photo_id: str,
        faces: Iterable[Tuple[bytes, str, str]],
    ) -> None:
        """
        faces: iterable of (embedding_bytes, bbox_json, model_type)
        """
        with self._lock, self._conn:
            self._conn.executemany(
                """
                INSERT INTO faces (photo_id, embedding, bbox, model_type)
                VALUES (?, ?, ?, ?)
                """,
                ((photo_id, emb, bbox, model) for emb, bbox, model in faces),
            )

    def list_faces(self) -> List[Tuple[int, str, bytes, str, str]]:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "SELECT id, photo_id, embedding, bbox, model_type FROM faces"
            )
            return cur.fetchall()

    def update_group(self, face_id: int, group_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE faces SET group_id = ? WHERE id = ?", (group_id, face_id)
            )

    def clear_groups(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE faces SET group_id = NULL")

    def list_groups(self) -> List[Tuple[str, int, str]]:
        """
        Returns list of (group_id, count, cover_photo_id)
        """
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                SELECT group_id, COUNT(*), MIN(photo_id)
                FROM faces
                WHERE group_id IS NOT NULL
                GROUP BY group_id
                ORDER BY COUNT(*) DESC
                """
            )
            return cur.fetchall()

    def list_group_photos(self, group_id: str) -> List[str]:
        with self._lock, self._conn:
            cur = self._conn.execute(
                """
                SELECT DISTINCT photo_id FROM faces
                WHERE group_id = ?
                """,
                (group_id,),
            )
            return [row[0] for row in cur.fetchall()]

    def get_photo_path(self, photo_id: str) -> Optional[str]:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "SELECT file_path FROM photos WHERE photo_id = ?", (photo_id,)
            )
            row = cur.fetchone()
            return row[0] if row else None

    def list_photos(self) -> List[Tuple[str, str, str]]:
        """
        Returns list of (photo_id, file_path, orig_name)
        """
        with self._lock, self._conn:
            cur = self._conn.execute(
                "SELECT photo_id, file_path, orig_name FROM photos ORDER BY created_at"
            )
            return cur.fetchall()

    def photo_exists(self, photo_id: str) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "SELECT 1 FROM photos WHERE photo_id = ? LIMIT 1", (photo_id,)
            )
            return cur.fetchone() is not None

    def count_stats(self) -> Tuple[int, int]:
        """
        Returns (total_photos, photos_no_face)
        """
        with self._lock, self._conn:
            cur = self._conn.execute("SELECT COUNT(*), SUM(no_face) FROM photos")
            total, no_face = cur.fetchone()
            return total or 0, no_face or 0

    def faces_count(self) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute("SELECT COUNT(*) FROM faces")
            (count,) = cur.fetchone()
            return count or 0

    def remove_missing_files(self) -> None:
        """
        Cleanup database entries whose files were removed manually.
        """
        with self._lock, self._conn:
            cur = self._conn.execute("SELECT photo_id, file_path FROM photos")
            to_delete = []
            for photo_id, path in cur.fetchall():
                if not Path(path).exists():
                    to_delete.append(photo_id)
            for pid in to_delete:
                self._conn.execute("DELETE FROM photos WHERE photo_id = ?", (pid,))

    def close(self) -> None:
        with self._lock:
            self._conn.close()
