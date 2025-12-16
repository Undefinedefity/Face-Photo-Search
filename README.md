# Face Photo Search (本地人脸归类相册)

一个完全本地运行的 Web 应用：选择本地照片文件夹 → 自动识别人脸 → 把同一个人归类 → 点击查看该人的所有照片。支持 macOS / Windows / Linux。

- 后端：Python + FastAPI，SQLite 缓存结果
- 前端：纯 HTML + 少量 JS，随 FastAPI 一起提供
- 全程离线：不上传、不联网、不调用云 API

## 快速开始

1) 准备环境（建议 Python 3.9+，示例使用 conda）

```bash
conda create -n face-photo-search python=3.10 -y
# 若提示 “Run 'conda init' before 'conda activate'”，先运行：conda init zsh  （或 bash/powershell）并重开终端
conda activate face-photo-search
```

安装依赖：

```bash
pip install -r requirements.txt   # 已固定 numpy<2 以避免旧编译包崩溃
# 选择一个人脸引擎（推荐先试 insightface，失败就换 face_recognition）
# Intel/macOS/Windows/Linux:
pip install insightface onnxruntime
# Apple Silicon 请用：
# pip install insightface onnxruntime-silicon
# 如果 insightface 不行，再用：
pip install face_recognition
```

2) 启动

- macOS / Linux：
  ```bash
  chmod +x run.sh  # 如首次运行需要权限
  ./run.sh
  ```
- Windows：
  ```bat
  run.bat
  ```
- 或通用方式：
  ```bash
  python -m app.main
  ```

启动后会自动打开浏览器：<http://localhost:8000>。

3) 在网页上操作

- 首页大按钮“选择照片文件夹”，选择你的相册文件夹（推荐 Chrome / Edge / Safari，Firefox 不支持文件夹选择）。
- 选择后自动开始分析，页面显示进度条和当前处理的照片。
- 完成后看到“智能合集”卡片，每张卡代表一个人；点击卡片进入该人的详情页，网格展示所有包含 TA 的照片，点击缩略图可放大预览原图。

## 数据与缓存

- 照片会复制到本地 `data/photos/` 下（按文件内容哈希命名，避免重名）。
- 分析结果写入 `data/app.db`（SQLite），下次启动无需重算。
- 若原始照片改动，再次选择同一文件夹会增量处理新增/修改的文件。

## 聚类与阈值

- 默认使用简单聚类，将同一人的 embedding 合并到同一组。
- 阈值可通过环境变量调整：
  - InsightFace（余弦相似度）：`INSIGHTFACE_THRESHOLD`，默认 `0.5`（越大越严格）
  - face_recognition（欧氏距离）：`FACEREC_THRESHOLD`，默认 `0.6`（越小越严格）

## API（最小可用集）

- `POST /api/upload-folder`：multipart 上传多文件（浏览器文件夹选择自动完成）
- `GET /api/status`：查看任务状态与进度
- `GET /api/groups`：人脸分组列表
- `GET /api/groups/{group_id}`：指定分组的所有 `photo_id`
- `GET /api/photo/{photo_id}?w=256`：原图或缩略图
- `POST /api/rebuild`：重新分析现有所有照片（可选）

## 常见问题

### 浏览器支持
- 文件夹选择需要 Chrome / Edge / Safari。Firefox 暂不支持 `<input type="file" webkitdirectory>`。

### 人脸库安装失败怎么办？

1. 优先试 `insightface`
   - macOS（Homebrew）：`brew install cmake` 然后 `pip install insightface`
   - Ubuntu/Debian：`sudo apt-get install -y build-essential cmake libgl1` 然后 `pip install insightface`
   - Windows：安装 “Visual Studio Build Tools (C++)”，再 `pip install insightface`

2. 如果 insightface 失败，再试 `face_recognition`
   - macOS：`brew install cmake dlib` 然后 `pip install face_recognition`
   - Ubuntu/Debian：`sudo apt-get install -y build-essential cmake libgl1 libopenblas-dev` 然后 `pip install face_recognition`
   - Windows：安装 CMake + Visual Studio Build Tools（C++），再 `pip install face_recognition`

3. 两个都装不了时，应用会在页面显示明确错误，请按提示安装。

### 照片太多很慢？
- 先用少量照片试运行，确认可用后再分批导入。
- 关闭其他高占用程序，确保有可用的 CPU/GPU。
- 调整阈值：严格阈值可能增加分组数量但不会明显提速。

### 重新分析 / 清理
- 点击页面上的“重新分析”按钮（如果你加了自定义 UI）或调用 `POST /api/rebuild`。
- 删除 `data/app.db` 可重置缓存（已有分组会丢失）。

## 许可证

MIT License，详见 `LICENSE`。
