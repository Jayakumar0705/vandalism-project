"""
config.py — Central Configuration for Vandalism Detection System.

Loads environment variables from .env and defines all system-wide
constants. Every module imports from here — no hardcoded values anywhere.

Usage:
    from config import GEMINI_API_KEY, FRAME_SKIP, check_keys
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env file so secrets stay out of source control ──
load_dotenv()

# ── Module logger ──
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  API KEYS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  VIDEO PROCESSING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUPPORTED_VIDEO_EXTENSIONS: list = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]
MAX_FILE_SIZE_MB: int = 500
FRAME_SKIP: int = 5          # Process every Nth frame to cut computation
TARGET_FRAME_SIZE: tuple = (224, 224)  # MobileNetV2 expected input
MAX_FRAMES: int = 1000       # Cap to prevent memory overflow

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CHUNKING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHUNK_SIZE: int = 30          # Frames per temporal chunk
CHUNK_OVERLAP: int = 5        # Overlapping frames for continuity

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DETECTION THRESHOLDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MOTION_THRESHOLD: float = 25.0      # Pixel-intensity diff threshold
VANDALISM_SENSITIVITY: float = 0.65 # 0.0 = lenient, 1.0 = strictest
ANOMALY_CONTAMINATION: float = 0.1  # Expected anomaly proportion
MIN_CONTOUR_AREA: int = 500         # Ignore tiny motion blobs

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DEEP-LEARNING MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL_NAME: str = "MobileNetV2"
FEATURE_DIM: int = 1280             # Output size of MobileNetV2 GAP

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  REPORT GENERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REPORT_MAX_LENGTH: int = 2000
GEMINI_MODEL: str = "gemini-2.5-flash"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  GEMINI VISION — VANDALISM CLASSIFICATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GEMINI_VISION_MODEL: str = "gemini-2.5-flash"   # Primary model

# Model rotation — each model has its own 20 req/day free-tier quota.
# System tries them in order; when one is rate-limited, it moves to the next.
GEMINI_MODEL_ROTATION: list = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]

MAX_FRAMES_FOR_VISION: int = 12                  # Max frames to send to Gemini (cost control)
VANDALISM_CONFIDENCE_THRESHOLD: float = 0.6      # Minimum confidence to flag as vandalism

# Specific vandalism activities the system looks for
VANDALISM_ACTIVITIES: list = [
    "breaking", "smashing", "shattering",
    "graffiti", "spray-painting", "tagging",
    "kicking", "hitting", "punching",
    "throwing objects", "hurling",
    "arson", "fire-setting", "burning",
    "tearing down", "ripping", "destroying",
    "damaging property", "defacing",
    "slashing", "cutting", "scratching",
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  FILE PATHS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TEMP_DIR: Path = Path("temp_uploads")
OUTPUT_DIR: Path = Path("detection_output")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LOGGING CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def check_keys() -> bool:
    """
    Verify that all required API keys are present and minimally valid.

    Returns:
        bool: True if every required key is configured, False otherwise.

    Example:
        >>> from config import check_keys
        >>> check_keys()
        True
    """
    # Gemini key must be non-empty and not the placeholder value
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        logger.warning(
            "GEMINI_API_KEY is not set. AI report generation will be disabled."
        )
        return False

    # Basic length sanity check — real keys are 39+ chars
    if len(GEMINI_API_KEY) < 10:
        logger.error("GEMINI_API_KEY appears invalid (too short).")
        return False

    logger.info("All API keys validated successfully.")
    return True


def setup_logging() -> None:
    """
    Configure the root logger using values from this config module.

    Returns:
        None

    Example:
        >>> setup_logging()
    """
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format=LOG_FORMAT,
    )
    logger.info("Logging configured at %s level.", LOG_LEVEL)


def ensure_directories() -> None:
    """
    Create temporary and output directories if they do not exist.

    Returns:
        None

    Example:
        >>> ensure_directories()
    """
    try:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Directories ensured: %s, %s", TEMP_DIR, OUTPUT_DIR)
    except OSError as exc:
        logger.error("Failed to create directories: %s", exc)
        raise
