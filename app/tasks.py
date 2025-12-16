from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from .database import Database
from .face_engine import FaceEngine


@dataclass
class PhotoJob:
    photo_id: str
    file_path: Path
    orig_name: str


@dataclass
class TaskStatus:
    state: str = "idle"  # idle, running, done, error
    total: int = 0
    processed: int = 0
    current: str = ""
    faces_found: int = 0
    photos_no_face: int = 0
    error: Optional[str] = None


class PhotoProcessor:
    def __init__(self, db: Database, engine: FaceEngine) -> None:
        self.db = db
        self.engine = engine
        self.queue: "queue.SimpleQueue[List[PhotoJob]]" = queue.SimpleQueue()
        self.status = TaskStatus()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def enqueue(self, jobs: List[PhotoJob]) -> None:
        if not jobs:
            return
        self.queue.put(jobs)

    def _reset_status(self, total: int) -> None:
        self.status = TaskStatus(state="running", total=total, processed=0)

    def _worker(self) -> None:
        while True:
            jobs = self.queue.get()
            if not jobs:
                continue
            self._process_jobs(jobs)

    def _process_jobs(self, jobs: List[PhotoJob]) -> None:
        if not self.engine.available:
            self.status = TaskStatus(
                state="error",
                error=self.engine.error_message or "No face engine available",
            )
            return

        self._reset_status(total=len(jobs))
        try:
            for job in jobs:
                self.status.current = job.orig_name
                self._process_single(job)
                self.status.processed += 1
            # recluster all faces
            self._cluster_all()
            self.status.state = "done"
            total_photos, photos_no_face = self.db.count_stats()
            self.status.photos_no_face = photos_no_face
            self.status.faces_found = self.db.faces_count()
        except Exception as exc:  # noqa: BLE001
            self.status = TaskStatus(state="error", error=str(exc))

    def _process_single(self, job: PhotoJob) -> None:
        with Image.open(job.file_path) as im:
            image = im.convert("RGB")
            width, height = image.size
            faces = self.engine.detect_and_embed(image)
            if faces:
                face_rows = []
                for face in faces:
                    bbox_json = json.dumps(face.bbox)
                    face_rows.append(
                        (
                            face.embedding.astype("float32").tobytes(),
                            bbox_json,
                            face.model_type,
                        )
                    )
                self.db.add_photo(
                    job.photo_id,
                    str(job.file_path),
                    job.orig_name,
                    width,
                    height,
                    no_face=False,
                )
                self.db.add_faces(job.photo_id, face_rows)
            else:
                self.db.add_photo(
                    job.photo_id,
                    str(job.file_path),
                    job.orig_name,
                    width,
                    height,
                    no_face=True,
                )
                self.status.photos_no_face += 1

    def _cluster_all(self) -> None:
        faces_rows = self.db.list_faces()
        # faces_rows: (id, photo_id, embedding_bytes, bbox_json, model_type)
        parsed: List[tuple[int, np.ndarray, str]] = []
        for row in faces_rows:
            face_id, _, emb_bytes, _, model_type = row
            emb = np.frombuffer(emb_bytes, dtype="float32")
            parsed.append((face_id, emb, model_type))

        # clear existing groups and reassign
        self.db.clear_groups()
        assignments = self.engine.cluster(parsed)
        for face_id, group_id in assignments:
            self.db.update_group(face_id, group_id)

