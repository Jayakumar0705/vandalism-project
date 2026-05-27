"""
app.py — Streamlit UI for Vandalism Detection Using Surveillance Camera.

Premium dark-themed dashboard with sidebar controls, video upload,
real-time processing pipeline with progress indicators, detection
results gallery, incident report, and download button.

Run:
    streamlit run app.py
"""

import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Project imports ──
import config
from utils.validator import InputValidator
from utils.document_loader import VideoLoader
from utils.chunker import FrameChunker
from utils.retriever import VandalismDetector

# ── Logging setup ──
config.setup_logging()
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PAGE CONFIG & CUSTOM CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Vandalism Detection — Surveillance AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject premium dark CSS with glassmorphism and animations
st.markdown(
    """
    <style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, rgba(229,57,53,0.15), rgba(255,152,0,0.10));
        border: 1px solid rgba(229,57,53,0.25);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        text-align: center;
        animation: fadeIn 0.8s ease-out;
    }
    .main-header h1 {
        color: #ff6b6b;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #a0aec0;
        font-size: 1.05rem;
    }

    /* ── Glass cards ── */
    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(8px);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }

    /* ── Metric cards ── */
    .metric-row {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    .metric-card {
        flex: 1;
        min-width: 150px;
        background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.10));
        border: 1px solid rgba(102,126,234,0.3);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-card .label {
        color: #a0aec0;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }

    /* ── Alert badge ── */
    .alert-badge {
        background: linear-gradient(135deg, #e53935, #ff6b6b);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        text-align: center;
        font-size: 1.1rem;
        animation: pulse 2s infinite;
    }
    .safe-badge {
        background: linear-gradient(135deg, #00c853, #69f0ae);
        color: #1a1a2e;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        text-align: center;
        font-size: 1.1rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #667eea;
    }

    /* ── Animations ── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(229,57,53,0.4); }
        50%      { box-shadow: 0 0 0 10px rgba(229,57,53,0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")

    sensitivity = st.slider(
        "Vandalism Sensitivity",
        min_value=0.1,
        max_value=1.0,
        value=config.VANDALISM_SENSITIVITY,
        step=0.05,
        help="Higher = more sensitive (flags more frames).",
    )

    frame_skip = st.slider(
        "Frame Skip Rate",
        min_value=1,
        max_value=30,
        value=config.FRAME_SKIP,
        help="Process every Nth frame. Higher = faster but less accurate.",
    )

    chunk_size = st.slider(
        "Temporal Chunk Size",
        min_value=10,
        max_value=100,
        value=config.CHUNK_SIZE,
        step=5,
        help="Number of frames per analysis window.",
    )

    st.markdown("---")
    st.markdown("## 🔑 API Status")

    if config.check_keys():
        st.success("✅ Gemini API key configured")
    else:
        st.warning("⚠️ No Gemini key — template reports only")

    st.markdown("---")
    st.markdown("## 📖 About")
    st.markdown(
        """
        **Vandalism Detection System** uses computer vision and deep
        learning to automatically detect suspicious activity in
        surveillance footage.

        **Pipeline:**
        1. 📹 Frame extraction
        2. 🏃 Motion analysis
        3. 🧠 Deep-feature embedding
        4. 🔍 Anomaly detection
        5. 📋 Incident reporting
        """
    )
    st.caption("Built with Streamlit • OpenCV • MobileNetV2 • Gemini")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HEADER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown(
    """
    <div class="main-header">
        <h1>🔍 Vandalism Detection System</h1>
        <p>AI-powered surveillance analysis — upload footage, detect threats, generate reports</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FILE UPLOAD SECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("### 📹 Upload Surveillance Footage")

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov", "mkv", "wmv"],
    help="Supported formats: MP4, AVI, MOV, MKV, WMV (max 500 MB)",
    key="video_uploader",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PROCESSING PIPELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if uploaded_file is not None:
    # ── Step 0: Validate input ──
    ok, err_msg = InputValidator.validate_upload(uploaded_file)
    if not ok:
        st.error(f"❌ {err_msg}")
        st.stop()

    # ── Step 1: Save to disk ──
    with st.spinner("💾 Saving uploaded file…"):
        try:
            loader = VideoLoader(frame_skip=frame_skip)
            saved_path = loader.save_uploaded_file(uploaded_file)
        except RuntimeError as exc:
            st.error(f"❌ File save failed: {exc}")
            st.stop()

    # ── Step 1b: Post-save corruption check ──
    ok, err_msg = InputValidator.validate_video_readable(saved_path)
    if not ok:
        st.error(f"❌ {err_msg}")
        st.stop()

    # ── Step 2: Extract frames ──
    with st.spinner("🎬 Extracting frames…"):
        try:
            frames, indices, metadata = loader.load_and_extract(saved_path)
        except RuntimeError as exc:
            st.error(f"❌ Frame extraction failed: {exc}")
            st.stop()

    if not frames:
        st.error("❌ No frames could be extracted from this video.")
        st.stop()

    # ── Video info metrics ──
    duration_sec = metadata["total_frames"] / metadata["fps"]
    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="value">{metadata['total_frames']}</div>
                <div class="label">Total Frames</div>
            </div>
            <div class="metric-card">
                <div class="value">{metadata['fps']:.0f}</div>
                <div class="label">FPS</div>
            </div>
            <div class="metric-card">
                <div class="value">{metadata['width']}×{metadata['height']}</div>
                <div class="label">Resolution</div>
            </div>
            <div class="metric-card">
                <div class="value">{len(frames)}</div>
                <div class="label">Analysed Frames</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Step 3: Compute motion (used as pre-filter only) ──
    with st.spinner("🏃 Analysing motion patterns…"):
        chunker = FrameChunker(chunk_size=chunk_size)
        motion_scores = chunker.compute_motion_scores(frames)

    # ── Step 4: Detect vandalism using Gemini Vision AI ──
    with st.spinner("🧠🔍 Classifying frames with Gemini Vision AI — this may take a moment…"):
        detector = VandalismDetector(sensitivity=sensitivity)
        detections = detector.detect(
            frames, motion_scores, indices, metadata["fps"]
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  RESULTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    st.markdown("---")
    st.markdown("### 📊 Detection Results")

    if detections:
        st.markdown(
            f'<div class="alert-badge">🚨 {len(detections)} '
            f'vandalism event(s) detected!</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="safe-badge">✅ No vandalism detected — footage appears safe</div>',
            unsafe_allow_html=True,
        )

    # ── Timeline ──
    st.markdown("")
    timeline = detector.create_timeline(detections)
    st.markdown(timeline)

    # ── Flagged frames gallery ──
    if detections:
        st.markdown("### 🖼️ Flagged Frames")

        # Show up to 12 flagged frames in a grid
        display_dets = detections[:12]
        cols_per_row = 3

        for row_start in range(0, len(display_dets), cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, det in enumerate(
                display_dets[row_start : row_start + cols_per_row]
            ):
                with cols[col_idx]:
                    # Find the frame — use list_position if available
                    if "list_position" in det and det["list_position"] < len(frames):
                        frame_bgr = frames[det["list_position"]]
                    else:
                        try:
                            list_pos = indices.index(det["frame_index"])
                            frame_bgr = frames[list_pos]
                        except ValueError:
                            list_pos = min(
                                range(len(indices)),
                                key=lambda i: abs(indices[i] - det["frame_index"]),
                            )
                            frame_bgr = frames[list_pos]

                    # Convert BGR → RGB for display
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)

                    st.image(
                        img,
                        caption=(
                            f"⏱ {det['timestamp_str']}  |  "
                            f"Type: {det.get('vandalism_type', 'unknown')}  |  "
                            f"Confidence: {det.get('confidence', det.get('combined_score', 0)):.2f}"
                        ),
                        use_container_width=True,
                    )
                    # Show description below the image
                    desc = det.get('description', '')
                    if desc:
                        st.caption(f"📝 {desc}")

    # ── Incident Report ──
    st.markdown("---")
    st.markdown("### 📋 Incident Report")

    with st.spinner("📝 Generating incident report…"):
        report = detector.generate_report(detections, metadata)

    st.markdown(
        f'<div class="glass-card">{report}</div>',
        unsafe_allow_html=True,
    )

    # Also show in markdown for proper rendering
    with st.expander("📄 View Full Report (Markdown)", expanded=False):
        st.markdown(report)

    # ── Download button ──
    st.download_button(
        label="⬇️ Download Incident Report",
        data=report,
        file_name="vandalism_incident_report.md",
        mime="text/markdown",
        key="download_report",
    )

    # ── Motion score chart ──
    st.markdown("---")
    st.markdown("### 📈 Motion Intensity Over Time")

    import pandas as pd

    chart_data = pd.DataFrame(
        {
            "Frame Index": indices[: len(motion_scores)],
            "Motion Score": motion_scores,
        }
    )
    st.area_chart(chart_data.set_index("Frame Index"), color="#ff6b6b")

    # ── Cleanup status ──
    st.success("✅ Analysis complete!")
    logger.info(
        "Pipeline finished — %d detections from %d frames.",
        len(detections),
        len(frames),
    )

else:
    # ── Empty-state prompt ──
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; padding:3rem;">
            <h3 style="color:#667eea;">👆 Upload a surveillance video to begin</h3>
            <p style="color:#a0aec0;">
                Supported formats: MP4, AVI, MOV, MKV, WMV<br>
                Maximum file size: 500 MB
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FOOTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("---")
st.caption(
    "🔍 Vandalism Detection Using Surveillance Camera  •  "
    "Built with Streamlit, OpenCV, MobileNetV2, Scikit-learn & Google Gemini  •  "
    "Final Year Project © 2026"
)
