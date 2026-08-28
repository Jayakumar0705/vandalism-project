# 🔍 Vandalism Detection Using Surveillance Camera

> AI-powered surveillance video analysis system that detects real acts of vandalism — spray-painting, fighting, property destruction — using **Google Gemini Vision AI** as the primary classifier, with an automatic **offline fallback** (MobileNetV2 + Isolation Forest) when the API is unavailable.

Built with **Streamlit · OpenCV · TensorFlow · Scikit-learn · Google Gemini**

---

## 📑 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Three-Tier Detection Engine](#-three-tier-detection-engine)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Processing Pipeline](#-processing-pipeline)
- [Module Reference](#-module-reference)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Screenshots & UI](#-screenshots--ui)
- [Troubleshooting](#-troubleshooting)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📹 **Video Upload** | Supports MP4, AVI, MOV, MKV, WMV formats up to 500 MB |
| 🧠 **Gemini Vision AI** | Sends frames to Google Gemini for intelligent vandalism classification |
| 🔄 **Model Rotation** | Cycles through `gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-2.5-flash-lite` to maximise free-tier quota |
| 🏃 **Motion Pre-Filter** | Frame differencing selects only high-activity frames, reducing API calls |
| 🤖 **Offline Fallback** | MobileNetV2 + Isolation Forest anomaly detection when API is unavailable |
| 📋 **AI Incident Reports** | Gemini generates professional security analyst reports |
| 📊 **Detection Timeline** | Markdown table of every flagged event with timestamp, type, and confidence |
| 📈 **Motion Chart** | Interactive area chart showing motion intensity over video duration |
| 🖼️ **Flagged Frame Gallery** | Grid of up to 12 flagged frames with captions and descriptions |
| ⬇️ **Report Download** | Export the full incident report as a Markdown file |
| ✅ **Input Validation** | Extension, file-size, and corruption checks before processing |
| 🎨 **Premium Dark UI** | Glassmorphism theme with Inter font, gradient cards, and pulse animations |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      STREAMLIT FRONTEND                          │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────────┐   │
│  │ Sidebar  │  │  Upload  │  │  Results  │  │   Download    │   │
│  │ Settings │  │  Widget  │  │  Gallery  │  │   Button      │   │
│  └──────────┘  └────┬─────┘  └─────▲─────┘  └───────────────┘   │
└──────────────────────┼─────────────┼─────────────────────────────┘
                       │             │
                       ▼             │
┌──────────────────────┴─────────────┴─────────────────────────────┐
│                     PROCESSING PIPELINE                           │
│                                                                   │
│  ┌────────────┐   ┌────────────┐   ┌─────────────────┐           │
│  │ Validator  │──▶│ VideoLoader│──▶│  FrameChunker   │           │
│  │ extension, │   │ save file, │   │  motion scores  │           │
│  │ size, read │   │ extract    │   │  per frame      │           │
│  └────────────┘   │ frames     │   └────────┬────────┘           │
│                   └────────────┘            │                     │
│                                             ▼                     │
│                              ┌──────────────────────────┐         │
│                              │  VandalismDetector       │         │
│                              │                          │         │
│                              │  Tier 1: Gemini Vision   │         │
│                              │  Tier 2: Local Fallback  │         │
│                              │  Tier 3: Template Report │         │
│                              └────────────┬─────────────┘         │
│                                           │                       │
│                                           ▼                       │
│                              ┌──────────────────────────┐         │
│                              │  Report Generator        │         │
│                              │  (Gemini AI / Template)  │         │
│                              └──────────────────────────┘         │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Three-Tier Detection Engine

The system uses a cascading fallback architecture to ensure detection always works, even without internet or API quota:

### Tier 1 — Gemini Vision AI *(Primary)*

- Sends up to 12 high-motion frames as base64 JPEG images to the Gemini Vision API in a **single API call**
- A detailed prompt instructs the model to classify each frame as `"vandalism"` or `"normal"`
- **Model rotation** cycles through three models (`gemini-2.5-flash` → `gemini-2.0-flash` → `gemini-2.5-flash-lite`), each with its own free-tier quota (~20 req/day each, ~60+ total)
- Only flags **real vandalism**: spray-painting, fighting, breaking/smashing, arson, property damage
- Explicitly ignores normal activity: walking, driving, crowds, weather, construction

### Tier 2 — Local Offline Fallback

Activated automatically when all Gemini models are rate-limited or the API key is missing:

- **MobileNetV2** (ImageNet pre-trained) extracts 1280-dim feature vectors per frame
- **Isolation Forest** anomaly detector identifies statistically unusual frames
- **Score fusion**: `combined = 0.6 × motion_norm + 0.4 × anomaly_score`
- Frames exceeding `(1 - sensitivity)` threshold are flagged as `suspicious_activity`
- Falls back to **OpenCV colour+edge histograms** if TensorFlow is unavailable

### Tier 3 — Template Reports

When no API is available for report generation, a structured Markdown template is used with:
- Executive summary, video metadata, incident details, and recommended actions

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| **Streamlit** | ≥ 1.28.0 | Web UI dashboard |
| **OpenCV** (`opencv-python-headless`) | ≥ 4.8.0 | Video I/O, frame differencing, contour detection |
| **TensorFlow / Keras** | ≥ 2.13.0 | MobileNetV2 feature extraction (offline fallback) |
| **scikit-learn** | ≥ 1.3.0 | Isolation Forest anomaly detection |
| **NumPy** | ≥ 1.24.0 | Numerical operations |
| **Pillow** | ≥ 10.0.0 | Image display in Streamlit |
| **google-genai** | ≥ 1.0.0 | Gemini Vision API + AI report generation |
| **python-dotenv** | ≥ 1.0.0 | `.env` file loading |
| **pytest** | ≥ 7.4.0 | Test framework |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9 – 3.12** (TensorFlow requires ≤ 3.12)
- **pip** package manager
- **Google Gemini API key** — free at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### Installation

```bash
# 1. Clone or navigate to the project
cd "Vandalism Detection Using Surveillance Camera"

# 2. Create and activate a virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Configure your API key
copy .env.example .env
# Open .env in a text editor and paste your Gemini API key

# 5. Run the test suite (optional but recommended)
python test_validation.py

# 6. Launch the application
streamlit run app.py

# 7. Open your browser at http://localhost:8501
```

> **Note:** The system works without an API key — it falls back to local Isolation Forest detection (Tier 2). The Gemini key enables more accurate Vision AI classification and AI-generated reports.

---

## 📁 Project Structure

```
Vandalism Detection Using Surveillance Camera/
│
├── app.py                    # Streamlit frontend — main entry point
├── config.py                 # Centralised configuration & environment loading
├── requirements.txt          # Python dependencies with minimum versions
├── .env.example              # Environment variable template
├── .env                      # Your secrets (git-ignored)
├── test_validation.py        # 7 test classes, 15+ unit tests
├── README.md                 # This file
│
├── utils/                    # Core processing package
│   ├── __init__.py           # Package exports (5 classes)
│   ├── validator.py          # InputValidator — extension, size, corruption checks
│   ├── document_loader.py    # VideoLoader — save uploads, extract frames
│   ├── chunker.py            # FrameChunker — temporal chunking & motion scoring
│   ├── embedder.py           # FeatureExtractor — MobileNetV2 / OpenCV / base64
│   └── retriever.py          # VandalismDetector — 3-tier detection & reports
│
├── temp_uploads/             # Auto-created for uploaded video files
└── detection_output/         # Auto-created for results
```

---

## 🔄 Processing Pipeline

### Step-by-Step Data Flow

```
Upload Video ──▶ Validate ──▶ Save to Disk ──▶ Extract Frames ──▶ Motion Scoring
                                                                       │
                                                                       ▼
                                                              Select High-Motion
                                                              Frames (pre-filter)
                                                                       │
                                               ┌───────────────────────┤
                                               ▼                       ▼
                                        Gemini Vision AI        Local Fallback
                                        (if API available)    (MobileNetV2 + IF)
                                               │                       │
                                               └───────────┬───────────┘
                                                           ▼
                                                    Detection Results
                                                           │
                                              ┌────────────┼────────────┐
                                              ▼            ▼            ▼
                                          Timeline    Frame Gallery   Report
                                           Table      (up to 12)    (AI/Template)
```

1. **Upload** — User uploads a surveillance video via the Streamlit file uploader
2. **Validate** — `InputValidator` checks file extension, size (≤ 500 MB), and readability
3. **Save** — `VideoLoader.save_uploaded_file()` writes bytes to `temp_uploads/`
4. **Extract** — `VideoLoader.load_and_extract()` reads frames at the configured skip rate (every Nth frame)
5. **Motion** — `FrameChunker.compute_motion_scores()` calculates per-frame motion via frame differencing with Gaussian blur, thresholding, and contour area summation
6. **Select** — `VandalismDetector._select_frames_for_analysis()` picks the top high-motion frames (up to 12) to send for classification
7. **Detect** — Gemini Vision classifies each frame; if rate-limited, falls back to local anomaly detection
8. **Report** — Gemini generates a professional incident report; falls back to a structured template
9. **Display** — Results shown as metrics, timeline table, flagged frame gallery, motion chart, and downloadable report

---

## 📦 Module Reference

### `config.py` — Central Configuration

All system-wide constants. Loads `.env` via `python-dotenv`. Key functions:

| Function | Returns | Purpose |
|----------|---------|---------|
| `check_keys()` | `bool` | Validates Gemini API key is present and ≥ 10 chars |
| `setup_logging()` | `None` | Configures root logger with level and format |
| `ensure_directories()` | `None` | Creates `temp_uploads/` and `detection_output/` |

### `utils/validator.py` — InputValidator

Static methods for pre-processing validation:

| Method | Args | Returns | Purpose |
|--------|------|---------|---------|
| `validate_extension()` | `filename` | `(bool, str)` | Checks against `SUPPORTED_VIDEO_EXTENSIONS` |
| `validate_file_size()` | `file_size_bytes` | `(bool, str)` | Enforces `MAX_FILE_SIZE_MB` limit |
| `validate_video_readable()` | `file_path` | `(bool, str)` | Opens with OpenCV, reads one frame |
| `validate_upload()` | `uploaded_file` | `(bool, str)` | Runs extension + size checks |
| `validate_full()` | `uploaded_file, saved_path` | `(bool, str)` | All checks including readability |

### `utils/document_loader.py` — VideoLoader

| Method | Returns | Purpose |
|--------|---------|---------|
| `save_uploaded_file(uploaded_file)` | `Path` | Writes Streamlit upload to disk |
| `load_video(file_path)` | `(VideoCapture, dict)` | Opens video, extracts metadata |
| `extract_frames(cap)` | `(List[ndarray], List[int])` | Reads frames with skip rate |
| `load_and_extract(file_path)` | `(frames, indices, metadata)` | One-shot convenience wrapper |

### `utils/chunker.py` — FrameChunker

| Method | Returns | Purpose |
|--------|---------|---------|
| `chunk_frames(frames, indices)` | `List[Dict]` | Splits into overlapping temporal windows |
| `compute_motion_scores(frames)` | `List[float]` | Per-frame motion via absdiff + contours |
| `compute_chunk_motion(chunks)` | `List[float]` | Average motion score per chunk |

### `utils/embedder.py` — FeatureExtractor

| Method | Returns | Purpose |
|--------|---------|---------|
| `extract_features(frame)` | `ndarray (1-D)` | Single-frame embedding (MobileNetV2 or OpenCV) |
| `extract_batch_features(frames)` | `ndarray (N, D)` | Batch embedding for all frames |
| `frame_to_base64(frame)` | `str` | Encodes frame as base64 JPEG for Vision API |
| `prepare_frames_for_vision(...)` | `List[Dict]` | Selects top-motion frames and encodes them |

### `utils/retriever.py` — VandalismDetector

| Method | Returns | Purpose |
|--------|---------|---------|
| `detect(frames, motion, indices, fps)` | `List[Dict]` | **Main entry** — runs full detection pipeline |
| `create_timeline(detections)` | `str` | Markdown table of all detected events |
| `generate_report(detections, metadata)` | `str` | AI or template incident report |

**Detection dict keys:** `frame_index`, `timestamp_sec`, `timestamp_str`, `motion_score`, `vandalism_type`, `confidence`, `description`, `combined_score`, `list_position`

---

## ⚙️ Configuration

All settings are defined in `config.py` and can be overridden via `.env`:

### API Keys

| Setting | Default | Description |
|---------|---------|-------------|
| `GEMINI_API_KEY` | `""` | Google Gemini API key for Vision + report generation |

### Video Processing

| Setting | Default | Description |
|---------|---------|-------------|
| `FRAME_SKIP` | `5` | Process every Nth frame (adjustable via sidebar) |
| `MAX_FRAMES` | `1000` | Cap on total frames to prevent memory overflow |
| `MAX_FILE_SIZE_MB` | `500` | Maximum upload file size |
| `TARGET_FRAME_SIZE` | `(224, 224)` | MobileNetV2 input resolution |
| `SUPPORTED_VIDEO_EXTENSIONS` | `.mp4, .avi, .mov, .mkv, .wmv` | Allowed file types |

### Chunking

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | `30` | Frames per temporal chunk (adjustable via sidebar) |
| `CHUNK_OVERLAP` | `5` | Overlapping frames between chunks |

### Detection Thresholds

| Setting | Default | Description |
|---------|---------|-------------|
| `VANDALISM_SENSITIVITY` | `0.65` | 0.0 = lenient, 1.0 = strictest (adjustable via sidebar) |
| `MOTION_THRESHOLD` | `25.0` | Pixel-intensity change threshold for motion detection |
| `ANOMALY_CONTAMINATION` | `0.1` | Expected anomaly proportion for Isolation Forest |
| `MIN_CONTOUR_AREA` | `500` | Ignore motion blobs smaller than this (pixels²) |

### Gemini Vision

| Setting | Default | Description |
|---------|---------|-------------|
| `GEMINI_VISION_MODEL` | `gemini-2.5-flash` | Primary Vision model |
| `GEMINI_MODEL_ROTATION` | `[gemini-2.5-flash, gemini-2.0-flash, gemini-2.5-flash-lite]` | Fallback model order |
| `MAX_FRAMES_FOR_VISION` | `12` | Max frames sent to API per analysis |
| `VANDALISM_CONFIDENCE_THRESHOLD` | `0.6` | Minimum confidence to flag as vandalism |

### Vandalism Activities Detected

The system specifically looks for: breaking, smashing, shattering, graffiti, spray-painting, tagging, kicking, hitting, punching, throwing objects, arson, fire-setting, burning, tearing down, ripping, destroying, damaging property, defacing, slashing, cutting, scratching.

---

## 🧪 Testing

### Running Tests

```bash
# Run all tests with unittest
python test_validation.py

# Or use pytest for detailed output
pytest test_validation.py -v
```

### Test Coverage — 7 Test Classes, 15+ Tests

| # | Class | Test Method | What It Validates |
|---|-------|-------------|-------------------|
| 1 | `TestImports` | `test_all_imports` | All 5 project modules import without error |
| 2 | `TestConfig` | `test_check_keys_returns_bool` | `config.check_keys()` returns `bool` |
| 2 | `TestConfig` | `test_constants_exist` | All essential constants defined with correct types |
| 3 | `TestChunking` | `test_chunk_structure` | Chunks have `frames`, `indices`, `chunk_id` keys |
| 3 | `TestChunking` | `test_motion_scores_length` | Motion scores list matches frame count |
| 4 | `TestEmbedding` | `test_single_frame_embedding` | Returns 1-D numpy array |
| 4 | `TestEmbedding` | `test_batch_embedding_shape` | Batch shape is `(N, D)` |
| 4 | `TestEmbedding` | `test_frame_to_base64` | Returns valid JPEG base64 string |
| 4 | `TestEmbedding` | `test_prepare_frames_for_vision` | Selects high-motion frames with correct keys |
| 5 | `TestFrameSelection` | `test_frame_selection` | Detector selects high-motion frames correctly |
| 5 | `TestFrameSelection` | `test_empty_frames` | Handles empty input gracefully |
| 6 | `TestEndToEnd` | `test_full_pipeline` | Full extract→chunk→detect on synthetic video |
| 7 | `TestTimelineAndReport` | `test_timeline_no_detections` | Shows safe message when no vandalism |
| 7 | `TestTimelineAndReport` | `test_timeline_with_detections` | Contains detection info in timeline |
| 7 | `TestTimelineAndReport` | `test_template_report_with_detections` | Report includes incident details |
| 7 | `TestTimelineAndReport` | `test_template_report_no_detections` | Report indicates safety |

---

## 🎨 Screenshots & UI

The application features a **premium dark glassmorphism** design:

- **Dark gradient background** (`#0f0c29` → `#1a1a2e` → `#16213e`)
- **Glass-effect cards** with backdrop blur and hover lift animations
- **Google Inter font** for modern typography
- **Colour-coded badges**: 🔴 red pulsing alert for vandalism, 🟢 green for safe
- **Metric cards** displaying Total Frames, FPS, Resolution, and Analysed Frames
- **Interactive sidebar** with sensitivity, frame skip, and chunk size sliders
- **Motion intensity area chart** in `#ff6b6b` red

---

## 🔧 Troubleshooting

| # | Error | Cause | Fix |
|---|-------|-------|-----|
| 1 | `ModuleNotFoundError: No module named 'cv2'` | OpenCV not installed | `pip install opencv-python-headless` |
| 2 | `TensorFlow not found (fallback active)` | TF not installed or Python > 3.12 | `pip install tensorflow` (requires Python ≤ 3.12) |
| 3 | `GEMINI_API_KEY is not set` | Missing `.env` or empty key | Copy `.env.example` → `.env` and paste your key |
| 4 | `Video file could not be opened` | Corrupted file or missing codec | Try a different video or install ffmpeg |
| 5 | `MemoryError during feature extraction` | Video too large | Increase `FRAME_SKIP` or decrease `MAX_FRAMES` in the sidebar |
| 6 | `429 / RESOURCE_EXHAUSTED` | Gemini free-tier quota exceeded | System auto-rotates models; wait 24h or use a paid key |
| 7 | `JSON parse failed` warning | Gemini returned unstructured text | System auto-falls back to keyword-based parsing |

---

## 🔮 Future Enhancements

- **Real-time RTSP stream** support for live camera feeds
- **Multi-camera dashboard** with simultaneous monitoring
- **Alert notifications** via email, SMS, or webhook
- **Database logging** for historical incident tracking
- **Fine-tuned model** trained on vandalism-specific datasets
- **Edge deployment** with TensorFlow Lite for on-device inference

---

## 📄 Licence 
Academic project
