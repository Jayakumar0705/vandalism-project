"""
embedder.py — Deep Feature Extraction & Frame Encoding for Surveillance Frames.

Uses MobileNetV2 (pre-trained on ImageNet) with global-average pooling to
produce a 1280-dim feature vector per frame.  These embeddings capture
high-level visual semantics (object shapes, textures, scene layout) that
the downstream anomaly detector uses to flag unusual activity.

Also provides frame-to-base64 encoding for sending frames to the
Google Gemini Vision API for intelligent vandalism classification.

Falls back to OpenCV-based colour + edge histograms when TensorFlow is
unavailable, so the project still works on low-resource machines.
"""

import base64
import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import config

# ── Module logger ──
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract per-frame feature embeddings using a deep CNN or OpenCV."""

    def __init__(self) -> None:
        """
        Load the feature-extraction model (MobileNetV2 → OpenCV fallback).

        Returns:
            None

        Example:
            >>> extractor = FeatureExtractor()
        """
        self.model = None
        self.use_deep: bool = False
        self._load_model()

    # ------------------------------------------------------------------ #
    #  Model loading with graceful fallback
    # ------------------------------------------------------------------ #
    def _load_model(self) -> None:
        """
        Attempt to load MobileNetV2 via TensorFlow/Keras.
        Falls back to OpenCV histogram features on failure.

        Returns:
            None

        Example:
            >>> extractor._load_model()
        """
        try:
            # Suppress TF info logs during import
            import os
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

            from tensorflow.keras.applications import MobileNetV2  # type: ignore

            # include_top=False + pooling='avg' gives a 1280-dim vector
            self.model = MobileNetV2(
                weights="imagenet", include_top=False, pooling="avg"
            )
            self.use_deep = True
            logger.info("MobileNetV2 loaded — deep feature mode active.")

        except (ImportError, OSError, Exception) as exc:
            logger.warning(
                "TensorFlow unavailable (%s). Using OpenCV histogram fallback.",
                exc,
            )
            self.model = None
            self.use_deep = False

    # ------------------------------------------------------------------ #
    #  Public API: extract features from a single frame
    # ------------------------------------------------------------------ #
    def extract_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Produce a 1-D feature vector for one BGR frame.

        Args:
            frame: A BGR numpy array (H × W × 3).

        Returns:
            np.ndarray: 1-D feature vector.

        Example:
            >>> vec = extractor.extract_features(frame)
            >>> vec.shape
            (1280,)
        """
        if self.use_deep and self.model is not None:
            return self._extract_deep_features(frame)
        return self._extract_opencv_features(frame)

    # ------------------------------------------------------------------ #
    #  Batch extraction
    # ------------------------------------------------------------------ #
    def extract_batch_features(
        self, frames: List[np.ndarray]
    ) -> np.ndarray:
        """
        Extract features for a list of frames.

        Args:
            frames: List of BGR numpy arrays.

        Returns:
            np.ndarray: Shape (N, feature_dim).

        Example:
            >>> matrix = extractor.extract_batch_features(frames)
            >>> matrix.shape
            (50, 1280)
        """
        if not frames:
            logger.warning("Empty frame list — returning empty array.")
            return np.array([])

        try:
            features = [self.extract_features(f) for f in frames]
            result = np.vstack(features)
            logger.info(
                "Batch features extracted — shape %s", result.shape
            )
            return result

        except (cv2.error, ValueError) as exc:
            logger.error("Batch extraction failed: %s", exc)
            raise RuntimeError(f"Feature extraction error: {exc}") from exc

    # ------------------------------------------------------------------ #
    #  MobileNetV2 deep features
    # ------------------------------------------------------------------ #
    def _extract_deep_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Run a single frame through MobileNetV2.

        Args:
            frame: BGR numpy array.

        Returns:
            np.ndarray: 1280-dim vector.

        Example:
            >>> vec = extractor._extract_deep_features(frame)
        """
        from tensorflow.keras.applications.mobilenet_v2 import (  # type: ignore
            preprocess_input,
        )

        # Resize to 224×224 as required by MobileNetV2
        resized = cv2.resize(frame, config.TARGET_FRAME_SIZE)
        # OpenCV is BGR; Keras expects RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Add batch dimension and apply ImageNet pre-processing
        batch = preprocess_input(
            np.expand_dims(rgb.astype(np.float32), axis=0)
        )
        # Forward pass — verbose=0 suppresses per-batch progress bar
        features = self.model.predict(batch, verbose=0)
        return features.flatten()

    # ------------------------------------------------------------------ #
    #  OpenCV fallback features (colour + edge histograms)
    # ------------------------------------------------------------------ #
    def _extract_opencv_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute a lightweight feature vector using colour and edge histograms.

        Args:
            frame: BGR numpy array.

        Returns:
            np.ndarray: Concatenated histogram vector.

        Example:
            >>> vec = extractor._extract_opencv_features(frame)
        """
        # 8-bin colour histogram per channel → 8³ = 512 bins
        colour_hist = cv2.calcHist(
            [frame], [0, 1, 2], None, [8, 8, 8],
            [0, 256, 0, 256, 0, 256],
        )
        colour_hist = cv2.normalize(colour_hist, colour_hist).flatten()

        # Edge histogram adds texture information
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_hist = cv2.calcHist([edges], [0], None, [16], [0, 256])
        edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()

        return np.concatenate([colour_hist, edge_hist])

    # ------------------------------------------------------------------ #
    #  Frame → base64 encoding for Gemini Vision API
    # ------------------------------------------------------------------ #
    @staticmethod
    def frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
        """
        Encode a BGR frame as a base64 JPEG string for the Vision API.

        Args:
            frame:   BGR numpy array (H × W × 3).
            quality: JPEG compression quality (0–100).

        Returns:
            str: Base64-encoded JPEG string.

        Example:
            >>> b64 = FeatureExtractor.frame_to_base64(frame)
        """
        # Resize to reduce payload size while keeping enough detail
        max_dim = 512
        h, w = frame.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(
                frame, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        _, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        return base64.b64encode(buffer).decode("utf-8")

    # ------------------------------------------------------------------ #
    #  Select & encode top-motion frames for Gemini Vision
    # ------------------------------------------------------------------ #
    def prepare_frames_for_vision(
        self,
        frames: List[np.ndarray],
        motion_scores: List[float],
        frame_indices: List[int],
        max_frames: int = 12,
    ) -> List[Dict]:
        """
        Select the frames with highest motion and encode them for the
        Gemini Vision API.

        Args:
            frames:        List of BGR numpy arrays.
            motion_scores: Per-frame motion intensities.
            frame_indices: Original frame indices.
            max_frames:    Maximum frames to prepare.

        Returns:
            List[Dict]: Each dict has keys:
                'frame_index', 'motion_score', 'base64_image'.

        Example:
            >>> prepared = extractor.prepare_frames_for_vision(frames, scores, idx)
        """
        n = min(len(frames), len(motion_scores), len(frame_indices))
        if n == 0:
            return []

        # Rank frames by motion score and pick top N
        scored = [
            (motion_scores[i], i) for i in range(n)
        ]
        scored.sort(reverse=True, key=lambda x: x[0])

        # Only keep frames with non-trivial motion
        selected = [
            (score, idx) for score, idx in scored[:max_frames]
            if score > 0.0
        ]

        prepared: List[Dict] = []
        for score, list_idx in selected:
            b64 = self.frame_to_base64(frames[list_idx])
            prepared.append({
                "frame_index": frame_indices[list_idx],
                "motion_score": score,
                "base64_image": b64,
                "list_position": list_idx,
            })

        logger.info(
            "Prepared %d high-motion frames for Vision API.",
            len(prepared),
        )
        return prepared
