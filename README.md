# Face Photo Search

English · [中文说明](#中文说明)

A local-only web app: pick a photo folder, detect faces on-device, cluster the same person, and browse each person’s photos with face thumbnails and full-size previews. Works on macOS / Windows / Linux.

- Backend: FastAPI + SQLite caching
- Frontend: static HTML + minimal JS served by FastAPI
- Offline: no uploads, no cloud calls

## Quick Start (English)

1) Environment (Python 3.9+, example with conda)
```bash
conda create -n face-photo-search python=3.10 -y
# If you see “Run 'conda init' before 'conda activate'”, run: conda init zsh (or bash/powershell) then open a new terminal
conda activate face-photo-search
```

2) Install deps (numpy pinned <2 to avoid old binary crashes)
```bash
pip install -r requirements.txt
# Choose a face engine (try insightface first, fallback to face_recognition)
# Intel/macOS/Windows/Linux:
pip install insightface onnxruntime
# Apple Silicon:
# pip install insightface onnxruntime-silicon
# If insightface fails:
pip install face_recognition
```

3) Run
- macOS / Linux
  ```bash
  chmod +x run.sh   # first time
  ./run.sh
  ```
- Windows
  ```bat
  run.bat
  ```
- Or
  ```bash
  python -m app.main
  ```
Browser opens at <http://localhost:8000>.

4) Use the web UI
- Click “Choose photo folder” (Chrome/Edge/Safari; Firefox may not support folder input).
- Analysis starts immediately; progress bar shows status. Large folders are auto-batched (~40 files/request).
- “Smart groups” cards: each card = one person; click to view that person’s photos; click a thumbnail to see full image.
- Adjust clustering threshold with the slider → “Save threshold” → “Re-analyze” to re-cluster (settings saved to `data/config.json`).
- “Clear cache” button removes `data/app.db` and copied photos; re-upload to start fresh.

## Data & Cache
- Photos copied to `data/photos/` (content-hash filenames).
- Results stored in `data/app.db` (SQLite); reused on next launch.
- Re-uploading the same folder processes new/changed files incrementally.
- Clear cache via the UI button or delete `data/app.db` and `data/photos/*`.

## Clustering & Thresholds
- InsightFace (cosine): higher threshold = stricter (more splitting).
- face_recognition (euclidean): lower threshold = stricter.
- Adjust in UI (slider + save, then re-analyze) or via env vars:
  - `INSIGHTFACE_THRESHOLD` (default 0.6)
  - `FACEREC_THRESHOLD` (default 0.6)

## API (minimal)
- `POST /api/upload-folder` : multipart multi-file upload (folder picker)
- `GET /api/status` : task state/progress
- `GET /api/groups` : face groups
- `GET /api/groups/{group_id}` : photo_ids in a group
- `GET /api/photo/{photo_id}?w=256` : photo or thumbnail
- `GET /api/face-cover/{group_id}?w=256` : face crop cover
- `POST /api/rebuild` : re-analyze all existing photos
- `POST /api/clear-cache` : clear DB and copied photos
- `GET/POST /api/settings` : get/set thresholds

## Troubleshooting
- Folder picker: use Chrome/Edge/Safari (Firefox lacks `webkitdirectory`).
- Install issues:
  - insightface
    - macOS: `brew install cmake` then `pip install insightface`
    - Ubuntu/Debian: `sudo apt-get install -y build-essential cmake libgl1` then `pip install insightface`
    - Windows: install “Visual Studio Build Tools (C++)”, then `pip install insightface`
  - face_recognition
    - macOS: `brew install cmake dlib` then `pip install face_recognition`
    - Ubuntu/Debian: `sudo apt-get install -y build-essential cmake libgl1 libopenblas-dev` then `pip install face_recognition`
    - Windows: install CMake + VS Build Tools, then `pip install face_recognition`
- One big cluster/poor separation: make threshold stricter (insightface → larger; face_recognition → smaller) then “Re-analyze”.

## License
MIT, see `LICENSE`.

---

## 中文说明

一个完全本地运行的 Web 应用：选择本地照片文件夹 → 自动识别人脸 → 把同一个人归类 → 点击查看该人的所有照片。支持 macOS / Windows / Linux。

- 后端：FastAPI + SQLite 缓存
- 前端：静态 HTML/JS
- 全程离线：不上传、不联网

### 快速开始
```bash
conda create -n face-photo-search python=3.10 -y
# 若提示 “Run 'conda init' before 'conda activate'”，先执行 conda init zsh（或 bash/powershell）后重开终端
conda activate face-photo-search
pip install -r requirements.txt   # 已固定 numpy<2
# Intel/macOS/Windows/Linux:
pip install insightface onnxruntime
# Apple Silicon:
# pip install insightface onnxruntime-silicon
# 如果 insightface 不行，再用：
pip install face_recognition
chmod +x run.sh && ./run.sh   # macOS/Linux
# 或 run.bat / python -m app.main
```
浏览器打开 <http://localhost:8000>。

### 使用
- 点击“选择照片文件夹”上传（推荐 Chrome/Edge/Safari）。
- 自动开始分析，进度条可见；大文件夹自动分批（约 40 张/批）。
- “智能合集”卡片代表一个人；点卡片看该人的所有照片；点缩略图放大。
- 阈值滑条 + “保存阈值” + “重新分析”可调整分组严格度（写入 `data/config.json`）。
- “清空缓存”可删除 `data/app.db` 与已复制照片，重新上传即可。

### 数据与缓存
- 照片复制到 `data/photos/`（内容哈希命名），结果在 `data/app.db`。
- 重复选择同一文件夹可增量处理新增/修改文件。
- 清空缓存：用按钮或手动删 `data/app.db` 与 `data/photos/*`。

### 聚类与阈值
- InsightFace：阈值越大越严格；face_recognition：阈值越小越严格。
- 调节：UI 滑条（保存后写入 `data/config.json`），再点“重新分析”；或启动前用环境变量 `INSIGHTFACE_THRESHOLD` / `FACEREC_THRESHOLD`。

### API
- `POST /api/upload-folder`、`GET /api/status`、`GET /api/groups`、`GET /api/groups/{group_id}`、
  `GET /api/photo/{photo_id}?w=256`、`GET /api/face-cover/{group_id}?w=256`、
  `POST /api/rebuild`、`POST /api/clear-cache`、`GET/POST /api/settings`

### 常见问题
- 浏览器：Chrome/Edge/Safari 支持文件夹选择。
- 安装失败：按上方 insightface/face_recognition 指南安装。
- 分组不准：调严格阈值后点“重新分析”。

### 许可证
MIT（见 `LICENSE`）。
