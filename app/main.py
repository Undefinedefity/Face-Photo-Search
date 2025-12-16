from __future__ import annotations

import asyncio
import hashlib
import io
import shutil
import sys
import threading
import webbrowser
from pathlib import Path
from typing import List, Optional

import uvicorn
import traceback
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

from .config import (
    DEFAULT_THUMB_WIDTH,
    PHOTOS_DIR,
    TMP_DIR,
    ensure_dirs,
)
from .database import Database
from .face_engine import FaceEngine
from .tasks import PhotoJob, PhotoProcessor


ensure_dirs()
app = FastAPI(title="Face Photo Search", version="0.1.0")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

db = Database()
face_engine = FaceEngine()
processor = PhotoProcessor(db=db, engine=face_engine)


@app.middleware("http")
async def log_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return HTMLResponse(
            content=f"Internal server error: {exc}",
            status_code=500,
        )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    if not face_engine.available:
        error = face_engine.error_message or "No face engine available. See README for install instructions."
    else:
        error = None
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "engine": face_engine.model_type or "unavailable",
            "error": error,
        },
    )


@app.post("/api/upload-folder")
async def upload_folder(files: List[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files received")
    if not face_engine.available:
        raise HTTPException(status_code=500, detail=face_engine.error_message or "Face engine not available")

    jobs: List[PhotoJob] = []
    saved = 0
    for file in files:
        # Only basic image filter by extension
        ext = Path(file.filename or "").suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        tmp_path = TMP_DIR / f"upload_{hashlib.sha1(file.filename.encode('utf-8')).hexdigest()}{ext}"
        hasher = hashlib.sha1()
        with tmp_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
                out.write(chunk)
        photo_id = hasher.hexdigest()
        dest_path = PHOTOS_DIR / f"{photo_id}{ext}"
        if not dest_path.exists():
            shutil.move(str(tmp_path), dest_path)
        else:
            tmp_path.unlink(missing_ok=True)

        if db.photo_exists(photo_id):
            continue
        jobs.append(PhotoJob(photo_id=photo_id, file_path=dest_path, orig_name=file.filename or dest_path.name))
        saved += 1

    if jobs:
        processor.enqueue(jobs)
    return {"accepted": saved, "message": "Processing queued", "queued_jobs": len(jobs)}


@app.get("/api/status")
async def status() -> dict:
    s = processor.status
    total_photos, photos_no_face = db.count_stats()
    return {
        "state": s.state,
        "total": s.total,
        "processed": s.processed,
        "current": s.current,
        "faces_found": s.faces_found,
        "photos_no_face": photos_no_face,
        "error": s.error,
        "total_photos": total_photos,
    }


@app.get("/api/groups")
async def groups() -> dict:
    items = db.list_groups_with_cover()
    return {
        "groups": [
            {"group_id": gid, "count": count, "cover_photo_id": pid, "cover_bbox": bbox}
            for gid, count, pid, bbox in items
        ]
    }


@app.get("/api/groups/{group_id}")
async def group_detail(group_id: str) -> dict:
    photos = db.list_group_photos(group_id)
    if not photos:
        raise HTTPException(status_code=404, detail="Group not found")
    return {"group_id": group_id, "photos": photos}


def _read_image(photo_path: Path, width: Optional[int]) -> bytes:
    with Image.open(photo_path) as im:
        if width and width > 0 and im.width > width:
            ratio = width / float(im.width)
            height = int(im.height * ratio)
            im = im.resize((width, height))
        buf = io.BytesIO()
        im.save(buf, format="JPEG")
        return buf.getvalue()


@app.get("/api/photo/{photo_id}")
async def photo(photo_id: str, w: Optional[int] = None):
    path_str = db.get_photo_path(photo_id)
    if not path_str:
        raise HTTPException(status_code=404, detail="Photo not found")
    photo_path = Path(path_str)
    if not photo_path.exists():
        raise HTTPException(status_code=404, detail="File missing on disk")

    if not w:
        return FileResponse(photo_path)
    data = await asyncio.to_thread(_read_image, photo_path, int(w))
    return StreamingResponse(io.BytesIO(data), media_type="image/jpeg")


def _read_face_crop(photo_path: Path, bbox_json: str, width: Optional[int]) -> bytes:
    bbox = json.loads(bbox_json)
    with Image.open(photo_path) as im:
        x1, y1, x2, y2 = bbox
        # expand a bit to include more context
        dx = int((x2 - x1) * 0.2)
        dy = int((y2 - y1) * 0.2)
        left = max(0, x1 - dx)
        top = max(0, y1 - dy)
        right = min(im.width, x2 + dx)
        bottom = min(im.height, y2 + dy)
        face = im.crop((left, top, right, bottom))
        if width and width > 0:
            ratio = width / float(face.width)
            height = int(face.height * ratio)
            face = face.resize((width, height))
        buf = io.BytesIO()
        face.save(buf, format="JPEG")
        return buf.getvalue()


@app.get("/api/face-cover/{group_id}")
async def face_cover(group_id: str, w: Optional[int] = None):
    # pick first face in group as cover
    items = db.list_groups_with_cover()
    record = next((r for r in items if r[0] == group_id), None)
    if not record:
        raise HTTPException(status_code=404, detail="Group not found")
    _, _, photo_id, bbox = record
    path_str = db.get_photo_path(photo_id)
    if not path_str:
        raise HTTPException(status_code=404, detail="Photo not found")
    data = await asyncio.to_thread(_read_face_crop, Path(path_str), bbox, int(w) if w else None)
    return StreamingResponse(io.BytesIO(data), media_type="image/jpeg")


@app.post("/api/rebuild")
async def rebuild() -> dict:
    if processor.status.state == "running":
        raise HTTPException(status_code=409, detail="Processing already running")
    jobs: List[PhotoJob] = []
    for photo_id, path, orig in db.list_photos():
        if not Path(path).exists():
            continue
        jobs.append(PhotoJob(photo_id=photo_id, file_path=Path(path), orig_name=orig or Path(path).name))
    if not jobs:
        return {"message": "No photos to rebuild"}
    processor.enqueue(jobs)
    return {"message": f"Rebuild started for {len(jobs)} photos"}


def _open_browser() -> None:
    try:
        webbrowser.open("http://localhost:8000", new=2)
    except Exception:
        pass


def main() -> None:
    threading.Timer(1.0, _open_browser).start()
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
