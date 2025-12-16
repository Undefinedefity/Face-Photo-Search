from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

from .config import get_thresholds


@dataclass
class DetectedFace:
    embedding: np.ndarray
    bbox: Tuple[int, int, int, int]
    model_type: str


class FaceEngine:
    """
    Wraps face detection/embedding with a simple fallback strategy:
    1) Try insightface
    2) Fallback to face_recognition
    """

    def __init__(self) -> None:
        self.model_type = None
        self._engine = None
        self._init_error: str | None = None
        self._load()

    @property
    def available(self) -> bool:
        return self._engine is not None

    @property
    def error_message(self) -> str | None:
        return self._init_error

    def _load(self) -> None:
        try:
            from insightface.app import FaceAnalysis  # type: ignore

            app = FaceAnalysis()
            app.prepare(ctx_id=0, det_size=(640, 640))
            self._engine = app
            self.model_type = "insightface"
            return
        except Exception as e:  # noqa: BLE001
            first_error = f"insightface not available: {e}"

        try:
            import face_recognition  # type: ignore  # noqa: F401
        except Exception as e:  # noqa: BLE001
            self._init_error = self._help_text(first_error, f"face_recognition not available: {e}")
            self._engine = None
            self.model_type = None
            return

        # face_recognition fallback
        self.model_type = "face_recognition"
        self._engine = "face_recognition"
        self._init_error = None

    def detect_and_embed(self, image: Image.Image) -> List[DetectedFace]:
        if not self.available or not self.model_type:
            return []

        if self.model_type == "insightface":
            return self._detect_insightface(image)
        if self.model_type == "face_recognition":
            return self._detect_face_recognition(image)
        return []

    def _detect_insightface(self, image: Image.Image) -> List[DetectedFace]:
        import numpy as np

        arr = np.asarray(image.convert("RGB"))
        faces = self._engine.get(arr)
        results: List[DetectedFace] = []
        for face in faces:
            embedding = np.array(face["embedding"], dtype="float32")
            # normalize to keep cosine similarity stable
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            bbox = tuple(int(v) for v in face.bbox.astype(int).tolist())
            results.append(DetectedFace(embedding=embedding, bbox=bbox, model_type="insightface"))
        return results

    def _detect_face_recognition(self, image: Image.Image) -> List[DetectedFace]:
        import face_recognition

        arr = np.asarray(image.convert("RGB"))
        boxes = face_recognition.face_locations(arr)
        encodings = face_recognition.face_encodings(arr, boxes)
        results: List[DetectedFace] = []
        for bbox, enc in zip(boxes, encodings):
            top, right, bottom, left = bbox
            results.append(
                DetectedFace(
                    embedding=np.array(enc, dtype="float32"),
                    bbox=(left, top, right, bottom),
                    model_type="face_recognition",
                )
            )
        return results

    def cluster(
        self,
        faces: List[Tuple[int, np.ndarray, str]],
    ) -> List[Tuple[int, str]]:
        """
        faces: list of (face_id, embedding, model_type)
        Returns list of (face_id, group_id)
        """
        assignments: List[Tuple[int, str]] = []
        if not faces:
            return assignments

        # Separate by model_type to avoid mixing spaces
        by_model: dict[str, List[Tuple[int, np.ndarray]]] = {}
        for fid, emb, model in faces:
            by_model.setdefault(model, []).append((fid, emb))

        insightface_threshold, facerec_threshold = get_thresholds()
        for model, items in by_model.items():
            if model == "face_recognition":
                assignments.extend(self._cluster_euclidean(items, facerec_threshold))
            else:
                assignments.extend(self._cluster_cosine(items, insightface_threshold))
        return assignments

    @staticmethod
    def _cluster_cosine(items: List[Tuple[int, np.ndarray]], threshold: float) -> List[Tuple[int, str]]:
        clusters: List[Tuple[str, np.ndarray, int]] = []  # (group_id, centroid, count)
        results: List[Tuple[int, str]] = []
        for face_id, emb in items:
            if emb.ndim == 1:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
            assigned = False
            for idx, (gid, centroid, count) in enumerate(clusters):
                sim = float(np.dot(emb, centroid))
                if sim >= threshold:
                    # update centroid incremental
                    new_centroid = (centroid * count + emb) / (count + 1)
                    clusters[idx] = (gid, new_centroid, count + 1)
                    results.append((face_id, gid))
                    assigned = True
                    break
            if not assigned:
                gid = FaceEngine._new_group_id()
                clusters.append((gid, emb, 1))
                results.append((face_id, gid))
        return results

    @staticmethod
    def _cluster_euclidean(items: List[Tuple[int, np.ndarray]], threshold: float) -> List[Tuple[int, str]]:
        clusters: List[Tuple[str, np.ndarray, int]] = []
        results: List[Tuple[int, str]] = []
        for face_id, emb in items:
            assigned = False
            for idx, (gid, centroid, count) in enumerate(clusters):
                dist = float(np.linalg.norm(emb - centroid))
                if dist <= threshold:
                    new_centroid = (centroid * count + emb) / (count + 1)
                    clusters[idx] = (gid, new_centroid, count + 1)
                    results.append((face_id, gid))
                    assigned = True
                    break
            if not assigned:
                gid = FaceEngine._new_group_id()
                clusters.append((gid, emb, 1))
                results.append((face_id, gid))
        return results

    @staticmethod
    def _new_group_id() -> str:
        import uuid

        return uuid.uuid4().hex

    @staticmethod
    def _help_text(first_error: str, second_error: str) -> str:
        return textwrap.dedent(
            f"""
            No face engine available.
            - Tried insightface: {first_error}
            - Tried face_recognition: {second_error}

            Please install one of them:
            macOS (Homebrew): brew install cmake && pip install insightface
            Ubuntu/Debian: sudo apt-get install -y build-essential cmake libgl1 && pip install insightface
            Windows: install Visual Studio Build Tools (C++), then pip install insightface

            If insightface fails, try face_recognition:
            macOS: brew install cmake dlib && pip install face_recognition
            Ubuntu/Debian: sudo apt-get install -y build-essential cmake libgl1 libopenblas-dev && pip install face_recognition
            Windows: install CMake and Visual Studio Build Tools, then pip install face_recognition
            """
        ).strip()
