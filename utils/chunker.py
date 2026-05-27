"""
chunker.py — Temporal Frame Chunking & Motion Scoring.

Splits a sequence of frames into overlapping temporal chunks and computes
per-frame motion-intensity scores via frame differencing.  Chunking lets
the detector analyse short segments independently, which mirrors how
human operators review footage in time-windows.
"""

import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np

import config

# ── Module logger ──
logger = logging.getLogger(__name__)


class FrameChunker:
    """Chunks frame sequences and computes motion-intensity metrics."""

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ) -> None:
        """
        Initialise the chunker.

        Args:
            chunk_size:    Number of frames per chunk.
            chunk_overlap: How many frames overlap between consecutive chunks.

        Returns:
            None

        Example:
            >>> chunker = FrameChunker(chunk_size=30, chunk_overlap=5)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(
            "FrameChunker ready  (size=%d, overlap=%d)",
            self.chunk_size,
            self.chunk_overlap,
        )

    # ------------------------------------------------------------------ #
    #  Split frames into overlapping chunks
    # ------------------------------------------------------------------ #
    def chunk_frames(
        self,
        frames: List[np.ndarray],
        indices: List[int],
    ) -> List[Dict]:
        """
        Divide frames into overlapping temporal windows.

        Args:
            frames:  List of BGR numpy arrays.
            indices: Corresponding original frame indices.

        Returns:
            List[Dict]: Each dict has keys
                'frames'  → List[np.ndarray],
                'indices' → List[int],
                'chunk_id'→ int.

        Example:
            >>> chunks = chunker.chunk_frames(frames, indices)
            >>> len(chunks[0]["frames"])
            30
        """
        chunks: List[Dict] = []
        step = max(1, self.chunk_size - self.chunk_overlap)  # stride

        for start in range(0, len(frames), step):
            end = min(start + self.chunk_size, len(frames))
            chunk = {
                "frames": frames[start:end],
                "indices": indices[start:end],
                "chunk_id": len(chunks),
            }
            chunks.append(chunk)

            # Stop if we've reached the last frame
            if end >= len(frames):
                break

        logger.info("Created %d temporal chunks from %d frames", len(chunks), len(frames))
        return chunks

    # ------------------------------------------------------------------ #
    #  Motion-intensity scoring via frame differencing
    # ------------------------------------------------------------------ #
    def compute_motion_scores(
        self,
        frames: List[np.ndarray],
        threshold: float = config.MOTION_THRESHOLD,
    ) -> List[float]:
        """
        Compute a motion-intensity score for every frame by comparing it
        to the previous frame using absolute pixel difference.

        Args:
            frames:    Ordered list of BGR frames.
            threshold: Pixel-intensity change below this is ignored as noise.

        Returns:
            List[float]: One score per frame (first frame gets 0.0).

        Example:
            >>> scores = chunker.compute_motion_scores(frames)
            >>> max(scores)
            4523.7
        """
        if len(frames) < 2:
            logger.warning("Need ≥2 frames for motion scoring; returning zeros.")
            return [0.0] * len(frames)

        scores: List[float] = [0.0]  # First frame has no predecessor

        try:
            # Convert first frame to grey once
            prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

            for i in range(1, len(frames)):
                curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.GaussianBlur(curr_gray, (21, 21), 0)

                # Absolute difference highlights changed pixels
                diff = cv2.absdiff(prev_gray, curr_gray)

                # Threshold to remove camera noise
                _, thresh_img = cv2.threshold(
                    diff, int(threshold), 255, cv2.THRESH_BINARY
                )

                # Find contours of changed regions
                contours, _ = cv2.findContours(
                    thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                # Sum areas of significant contours as the motion score
                motion_area = sum(
                    cv2.contourArea(c)
                    for c in contours
                    if cv2.contourArea(c) > config.MIN_CONTOUR_AREA
                )
                scores.append(float(motion_area))
                prev_gray = curr_gray

            logger.info(
                "Motion scores computed — min=%.1f  max=%.1f  mean=%.1f",
                min(scores),
                max(scores),
                np.mean(scores),
            )
            return scores

        except cv2.error as exc:
            logger.error("Motion scoring failed: %s", exc)
            raise RuntimeError(f"Motion computation error: {exc}") from exc

    # ------------------------------------------------------------------ #
    #  Per-chunk aggregated motion
    # ------------------------------------------------------------------ #
    def compute_chunk_motion(
        self,
        chunks: List[Dict],
        threshold: float = config.MOTION_THRESHOLD,
    ) -> List[float]:
        """
        Return the average motion score for each chunk.

        Args:
            chunks:    Output of chunk_frames().
            threshold: Passed through to compute_motion_scores().

        Returns:
            List[float]: One aggregated score per chunk.

        Example:
            >>> chunk_scores = chunker.compute_chunk_motion(chunks)
        """
        chunk_scores: List[float] = []
        for chunk in chunks:
            frame_scores = self.compute_motion_scores(chunk["frames"], threshold)
            avg = float(np.mean(frame_scores)) if frame_scores else 0.0
            chunk_scores.append(avg)

        logger.info("Chunk-level motion scores: %s", chunk_scores)
        return chunk_scores
