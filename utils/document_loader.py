"""
document_loader.py — Video Loading & Frame Extraction.

Responsible for reading surveillance video files, extracting individual
frames at a configurable skip-rate, and persisting uploaded files to a
temporary directory so OpenCV can read them from disk.
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config

# ── Module logger ──
logger = logging.getLogger(__name__)


class VideoLoader:
    """Handles video I/O: saving uploads, opening captures, extracting frames."""

    def __init__(self, frame_skip: int = config.FRAME_SKIP) -> None:
        """
        Initialise the loader with a frame-skip rate.

        Args:
            frame_skip: Process every Nth frame (default from config).

        Returns:
            None

        Example:
            >>> loader = VideoLoader(frame_skip=3)
        """
        self.frame_skip = frame_skip
        logger.info("VideoLoader initialised  (frame_skip=%d)", self.frame_skip)

    # ------------------------------------------------------------------ #
    #  Save an uploaded file to disk
    # ------------------------------------------------------------------ #
    def save_uploaded_file(self, uploaded_file) -> Path:
        """
        Write a Streamlit UploadedFile to a temporary path on disk.

        Args:
            uploaded_file: A Streamlit UploadedFile object.

        Returns:
            Path: Absolute path to the saved file.

        Example:
            >>> path = loader.save_uploaded_file(st_file)
        """
        try:
            # Ensure the temp directory exists before writing
            config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
            file_path = config.TEMP_DIR / uploaded_file.name

            # Write bytes to disk so OpenCV can open from a real path
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            logger.info("Saved uploaded file to %s", file_path)
            return file_path

        except (OSError, IOError) as exc:
            logger.error("Failed to save uploaded file: %s", exc)
            raise RuntimeError(f"Could not save file: {exc}") from exc

    # ------------------------------------------------------------------ #
    #  Open a video capture handle
    # ------------------------------------------------------------------ #
    def load_video(self, file_path: Path) -> Tuple[cv2.VideoCapture, dict]:
        """
        Open a video file and return the capture object plus metadata.

        Args:
            file_path: Path to the video file on disk.

        Returns:
            Tuple containing:
                - cv2.VideoCapture: Opened video capture.
                - dict: Metadata with keys fps, width, height, total_frames.

        Example:
            >>> cap, meta = loader.load_video(Path("video.mp4"))
            >>> print(meta["fps"])
        """
        try:
            cap = cv2.VideoCapture(str(file_path))

            # Validate that the capture actually opened
            if not cap.isOpened():
                raise ValueError(f"OpenCV cannot open video: {file_path}")

            # Extract metadata used downstream for timestamps
            metadata = {
                "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            }

            logger.info(
                "Opened video: %s — %d frames @ %.1f FPS",
                file_path.name,
                metadata["total_frames"],
                metadata["fps"],
            )
            return cap, metadata

        except (cv2.error, ValueError) as exc:
            logger.error("Video load failed: %s", exc)
            raise RuntimeError(f"Cannot load video: {exc}") from exc

    # ------------------------------------------------------------------ #
    #  Extract frames from a capture
    # ------------------------------------------------------------------ #
    def extract_frames(
        self,
        cap: cv2.VideoCapture,
        max_frames: int = config.MAX_FRAMES,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Read frames from an open capture, honouring the skip rate and cap.

        Args:
            cap:        An opened cv2.VideoCapture.
            max_frames: Maximum number of frames to return.

        Returns:
            Tuple containing:
                - List[np.ndarray]: BGR frames.
                - List[int]:        Original frame indices (for timestamp calc).

        Example:
            >>> frames, indices = loader.extract_frames(cap)
        """
        frames: List[np.ndarray] = []
        indices: List[int] = []
        frame_idx = 0

        try:
            while cap.isOpened() and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video or read error

                # Only keep every Nth frame to reduce workload
                if frame_idx % self.frame_skip == 0:
                    frames.append(frame)
                    indices.append(frame_idx)

                frame_idx += 1

            logger.info(
                "Extracted %d frames (skipped every %d, max %d)",
                len(frames),
                self.frame_skip,
                max_frames,
            )
            return frames, indices

        except cv2.error as exc:
            logger.error("Frame extraction error at idx %d: %s", frame_idx, exc)
            raise RuntimeError(f"Frame extraction failed: {exc}") from exc

        finally:
            cap.release()  # Always release the capture handle

    # ------------------------------------------------------------------ #
    #  Convenience: load + extract in one call
    # ------------------------------------------------------------------ #
    def load_and_extract(
        self, file_path: Path
    ) -> Tuple[List[np.ndarray], List[int], dict]:
        """
        One-shot helper: open video → extract frames → return everything.

        Args:
            file_path: Path to a video file.

        Returns:
            Tuple of (frames, frame_indices, video_metadata).

        Example:
            >>> frames, idxs, meta = loader.load_and_extract(Path("v.mp4"))
        """
        cap, metadata = self.load_video(file_path)
        frames, indices = self.extract_frames(cap)
        return frames, indices, metadata
