"""
retriever.py — Vandalism Detection & Incident Report Generation.

Three-tier detection system:
  Tier 1: Gemini Vision API with model rotation (tries multiple models,
          each with its own free-tier quota — effectively 60+ req/day).
  Tier 2: Local offline detection using Isolation Forest anomaly detection
          on MobileNetV2 features + motion analysis (unlimited, no API).
  Tier 3: Template reports when no API is available.

Only actual vandalism (spraying, fighting, property damage, graffiti, etc.)
is flagged by Tier 1.  Tier 2 flags anomalous activity as a fallback.
"""

import base64
import json
import logging
import re
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import config

# ── Module logger ──
logger = logging.getLogger(__name__)


class VandalismDetector:
    """Detect vandalism using Gemini Vision AI with local offline fallback."""

    def __init__(
        self,
        sensitivity: float = config.VANDALISM_SENSITIVITY,
    ) -> None:
        """
        Initialise the detector.

        Args:
            sensitivity: Detection sensitivity 0.0–1.0 (higher = more strict).

        Returns:
            None

        Example:
            >>> detector = VandalismDetector(sensitivity=0.7)
        """
        self.sensitivity = sensitivity
        self._gemini_client = None
        self._used_model = None          # Track which model was used
        self._detection_method = None    # "gemini_vision" or "local_fallback"
        logger.info("VandalismDetector ready  (sensitivity=%.2f)", self.sensitivity)

    # ================================================================== #
    #                        GEMINI VISION (Tier 1)                       #
    # ================================================================== #

    def _init_gemini(self) -> bool:
        """
        Lazily initialise the Gemini client.

        Returns:
            bool: True if Gemini is available, False otherwise.

        Example:
            >>> ok = detector._init_gemini()
        """
        if self._gemini_client is not None:
            return True

        key = config.GEMINI_API_KEY
        if not key or key == "your_gemini_api_key_here":
            logger.info("No Gemini API key — will use local fallback.")
            return False

        try:
            from google import genai

            self._gemini_client = genai.Client(api_key=key)
            logger.info("Gemini client initialised.")
            return True

        except Exception as exc:
            logger.error("Failed to initialise Gemini client: %s", exc)
            return False

    # ------------------------------------------------------------------ #
    #  Select best frames for analysis
    # ------------------------------------------------------------------ #
    def _select_frames_for_analysis(
        self,
        frames: List[np.ndarray],
        motion_scores: List[float],
        frame_indices: List[int],
        max_frames: int = 8,
    ) -> List[Dict]:
        """
        Use motion as a PRE-FILTER to pick the most interesting frames.
        This is NOT detection — just frame selection for the API.

        Args:
            frames:        List of BGR numpy arrays.
            motion_scores: Per-frame motion intensities.
            frame_indices: Original frame indices.
            max_frames:    Maximum frames to select.

        Returns:
            List[Dict]: Selected frames with metadata.

        Example:
            >>> selected = detector._select_frames_for_analysis(frames, m, idx)
        """
        from utils.embedder import FeatureExtractor

        n = min(len(frames), len(motion_scores), len(frame_indices))
        if n == 0:
            return []

        # Rank by motion
        scored = [(motion_scores[i], i) for i in range(n)]
        scored.sort(reverse=True, key=lambda x: x[0])

        # Take top motion frames
        motion_frames = [
            (score, idx) for score, idx in scored[:max_frames]
            if score > 0.0
        ]

        # Add evenly spaced samples if too few motion frames
        if len(motion_frames) < 3 and n > 3:
            step = max(1, n // 4)
            for i in range(0, n, step):
                if len(motion_frames) < max_frames:
                    if not any(idx == i for _, idx in motion_frames):
                        motion_frames.append((motion_scores[i], i))

        # Encode selected frames
        selected: List[Dict] = []
        for score, list_idx in motion_frames[:max_frames]:
            b64 = FeatureExtractor.frame_to_base64(frames[list_idx])
            selected.append({
                "frame_index": frame_indices[list_idx],
                "motion_score": score,
                "base64_image": b64,
                "list_position": list_idx,
            })

        logger.info("Selected %d frames for analysis.", len(selected))
        return selected

    # ------------------------------------------------------------------ #
    #  Try all models in rotation
    # ------------------------------------------------------------------ #
    def _classify_with_gemini(
        self,
        selected_frames: List[Dict],
        fps: float,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Try each Gemini model in rotation until one succeeds.
        Each model has its own free-tier quota (20 req/day).

        Args:
            selected_frames: Frames with base64 encoding.
            fps:             Video FPS.

        Returns:
            List[Dict] of detections, or None if all models exhausted.

        Example:
            >>> dets = detector._classify_with_gemini(frames, 30.0)
        """
        from google import genai
        from google.genai import types

        models_to_try = config.GEMINI_MODEL_ROTATION.copy()

        for model_name in models_to_try:
            try:
                logger.info("Trying Gemini model: %s", model_name)
                detections = self._call_gemini(
                    model_name, selected_frames, fps, types
                )
                self._used_model = model_name
                self._detection_method = "gemini_vision"
                logger.info(
                    "Model %s succeeded — %d detections.",
                    model_name, len(detections),
                )
                return detections

            except Exception as exc:
                error_str = str(exc)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    logger.warning(
                        "Model %s rate-limited — trying next model...",
                        model_name,
                    )
                    continue
                else:
                    logger.error(
                        "Model %s failed with error: %s — trying next...",
                        model_name, exc,
                    )
                    continue

        # All models exhausted
        logger.warning("All Gemini models exhausted. Switching to local fallback.")
        return None

    # ------------------------------------------------------------------ #
    #  Single Gemini API call
    # ------------------------------------------------------------------ #
    def _call_gemini(
        self,
        model_name: str,
        selected_frames: List[Dict],
        fps: float,
        types,
    ) -> List[Dict[str, Any]]:
        """
        Make ONE Gemini API call with all frames.

        Args:
            model_name:      Gemini model identifier.
            selected_frames: Frames with base64_image data.
            fps:             Video FPS.
            types:           google.genai.types module.

        Returns:
            List[Dict]: Vandalism detections only.

        Example:
            >>> dets = detector._call_gemini("gemini-2.5-flash", frames, 30.0, types)
        """
        prompt_parts = []

        prompt_text = (
            "You are a surveillance security AI specialising in VANDALISM detection.\n\n"
            "Analyse each surveillance camera frame below and classify it as "
            "'vandalism' or 'normal'.\n\n"
            "VANDALISM — flag ONLY these activities:\n"
            "- Spray-painting / graffiti on walls or property\n"
            "- Breaking / smashing windows, doors, or objects\n"
            "- Fighting / physical violence / assault\n"
            "- Throwing objects at property or people\n"
            "- Kicking, hitting, or damaging vehicles, signs, or equipment\n"
            "- Arson / setting fires\n"
            "- Tearing down fixtures, signs, or fences\n"
            "- Slashing tires, scratching cars\n"
            "- Any deliberate destruction or defacing of property\n\n"
            "NORMAL — do NOT flag these:\n"
            "- People walking, standing, sitting, running\n"
            "- Vehicles driving or parked normally\n"
            "- Crowds gathering peacefully\n"
            "- Empty scenes, weather, animals\n"
            "- Construction work or authorised activity\n\n"
            "Respond ONLY with a JSON array. For each frame:\n"
            "```json\n"
            "[\n"
            '  {"frame_id": 0, "classification": "vandalism" or "normal", '
            '"confidence": 0.0-1.0, "vandalism_type": "spray_painting" or '
            '"fighting" or "breaking" or "arson" or "none", '
            '"description": "what you see"}\n'
            "]\n"
            "```\n\n"
            f"Analysing {len(selected_frames)} frame(s):\n"
        )
        prompt_parts.append(prompt_text)

        for i, frame_data in enumerate(selected_frames):
            ts_sec = frame_data["frame_index"] / fps
            ts_str = str(timedelta(seconds=int(ts_sec)))
            prompt_parts.append(
                f"\n--- Frame {i} (timestamp: {ts_str}) ---\n"
            )
            image_bytes = base64.b64decode(frame_data["base64_image"])
            prompt_parts.append(
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            )

        # SINGLE API call
        response = self._gemini_client.models.generate_content(
            model=model_name,
            contents=prompt_parts,
        )
        response_text = response.text
        logger.info(
            "Gemini (%s) response: %d chars.", model_name, len(response_text)
        )

        # Parse classifications
        classifications = self._parse_vision_response(
            response_text, len(selected_frames)
        )

        # Filter: only vandalism frames
        detections: List[Dict[str, Any]] = []
        confidence_threshold = max(0.3, config.VANDALISM_CONFIDENCE_THRESHOLD - (
            self.sensitivity - 0.5) * 0.3
        )

        for i, cls in enumerate(classifications):
            if i >= len(selected_frames):
                break

            frame_data = selected_frames[i]
            is_vandalism = (
                cls.get("classification", "").lower() == "vandalism"
                and float(cls.get("confidence", 0.0)) >= confidence_threshold
            )

            if is_vandalism:
                fi = frame_data["frame_index"]
                ts_sec = fi / fps
                detections.append({
                    "frame_index": fi,
                    "timestamp_sec": round(ts_sec, 2),
                    "timestamp_str": str(timedelta(seconds=int(ts_sec))),
                    "motion_score": round(float(frame_data["motion_score"]), 4),
                    "vandalism_type": cls.get("vandalism_type", "unknown"),
                    "confidence": round(float(cls.get("confidence", 0.0)), 4),
                    "description": cls.get("description", "Vandalism detected"),
                    "combined_score": round(float(cls.get("confidence", 0.0)), 4),
                    "list_position": frame_data.get("list_position", 0),
                })

        return detections

    # ------------------------------------------------------------------ #
    #  Parse Gemini response
    # ------------------------------------------------------------------ #
    def _parse_vision_response(
        self,
        response_text: str,
        expected_count: int,
    ) -> List[Dict]:
        """
        Extract JSON classification array from Gemini's response.

        Args:
            response_text:  Raw text from Gemini.
            expected_count: Number of frames sent.

        Returns:
            List[Dict]: Parsed classifications.

        Example:
            >>> parsed = detector._parse_vision_response(text, 3)
        """
        try:
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list):
                    logger.info("Parsed %d classifications.", len(parsed))
                    return parsed

        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("JSON parse failed: %s", exc)

        # Fallback: infer from text
        text_lower = response_text.lower()
        vandalism_keywords = [
            "vandalism", "spray", "graffiti", "breaking", "smashing",
            "fighting", "assault", "arson", "fire", "damage", "destroy",
        ]
        has_vandalism = any(kw in text_lower for kw in vandalism_keywords)

        return [{
            "frame_id": i,
            "classification": "vandalism" if has_vandalism else "normal",
            "confidence": 0.6 if has_vandalism else 0.2,
            "vandalism_type": "unknown" if has_vandalism else "none",
            "description": "Parsed from unstructured response",
        } for i in range(expected_count)]

    # ================================================================== #
    #                     LOCAL OFFLINE FALLBACK (Tier 2)                  #
    # ================================================================== #

    def _local_fallback_detect(
        self,
        frames: List[np.ndarray],
        motion_scores: List[float],
        frame_indices: List[int],
        fps: float,
    ) -> List[Dict[str, Any]]:
        """
        Offline detection using MobileNetV2 features + Isolation Forest
        anomaly detection + motion analysis.  No API calls needed.

        This is the fallback when all Gemini models are rate-limited.
        It detects anomalous activity (may include false positives since
        it cannot understand scene content like the Vision API can).

        Args:
            frames:        List of BGR frames.
            motion_scores: Per-frame motion intensities.
            frame_indices: Original frame indices.
            fps:           Video FPS.

        Returns:
            List[Dict]: Detected suspicious frames.

        Example:
            >>> dets = detector._local_fallback_detect(frames, m, idx, 30.0)
        """
        self._detection_method = "local_fallback"
        logger.info("Using LOCAL OFFLINE detection (Isolation Forest + motion).")

        # Step 1: Extract deep features
        from utils.embedder import FeatureExtractor
        extractor = FeatureExtractor()
        features = extractor.extract_batch_features(frames)

        # Step 2: Compute anomaly scores via Isolation Forest
        anomaly_scores = self._compute_anomaly_scores(features)

        # Step 3: Fuse motion + anomaly scores
        detections: List[Dict[str, Any]] = []

        m_arr = np.array(motion_scores, dtype=np.float64)
        m_max = m_arr.max() if m_arr.max() > 0 else 1.0
        m_norm = m_arr / m_max

        n = min(len(m_norm), len(anomaly_scores), len(frame_indices))
        threshold = 1.0 - self.sensitivity

        for i in range(n):
            combined = 0.6 * m_norm[i] + 0.4 * float(anomaly_scores[i])

            if combined >= threshold:
                ts_sec = frame_indices[i] / fps
                detections.append({
                    "frame_index": frame_indices[i],
                    "timestamp_sec": round(ts_sec, 2),
                    "timestamp_str": str(timedelta(seconds=int(ts_sec))),
                    "motion_score": round(float(m_norm[i]), 4),
                    "anomaly_score": round(float(anomaly_scores[i]), 4),
                    "combined_score": round(float(combined), 4),
                    "vandalism_type": "suspicious_activity",
                    "confidence": round(float(combined), 4),
                    "description": (
                        "Suspicious activity detected via local analysis "
                        "(Gemini API unavailable)"
                    ),
                    "list_position": i,
                })

        logger.info(
            "Local fallback: %d / %d frames flagged.", len(detections), n
        )
        return detections

    def _compute_anomaly_scores(
        self, features: np.ndarray
    ) -> np.ndarray:
        """
        Fit Isolation Forest on features and return anomaly scores [0, 1].

        Args:
            features: Shape (N, D) feature matrix.

        Returns:
            np.ndarray: Normalised anomaly scores.

        Example:
            >>> scores = detector._compute_anomaly_scores(features)
        """
        if features.size == 0 or len(features) < 2:
            return np.zeros(max(len(features), 0))

        try:
            from sklearn.ensemble import IsolationForest

            iso = IsolationForest(
                contamination=config.ANOMALY_CONTAMINATION,
                random_state=42,
                n_estimators=100,
            )
            iso.fit(features)

            raw_scores = -iso.decision_function(features)
            mn, mx = raw_scores.min(), raw_scores.max()
            if mx - mn > 0:
                return (raw_scores - mn) / (mx - mn)
            return np.zeros_like(raw_scores)

        except Exception as exc:
            logger.error("Anomaly scoring failed: %s", exc)
            return np.zeros(len(features))

    # ================================================================== #
    #                        MAIN ENTRY POINT                             #
    # ================================================================== #

    def detect(
        self,
        frames: List[np.ndarray],
        motion_scores: List[float],
        frame_indices: List[int],
        fps: float,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run vandalism detection with automatic fallback:

        1. Try Gemini Vision (rotates through multiple models)
        2. If all models rate-limited → use local Isolation Forest fallback

        Args:
            frames:         List of BGR frame arrays.
            motion_scores:  Per-frame motion intensities.
            frame_indices:  Original frame numbers.
            fps:            Video FPS.

        Returns:
            List[Dict]: Detections for vandalism/suspicious frames.

        Example:
            >>> detections = detector.detect(frames, motion, idx, 30.0)
        """
        # Step 1: Select frames for analysis
        selected = self._select_frames_for_analysis(
            frames, motion_scores, frame_indices,
            max_frames=config.MAX_FRAMES_FOR_VISION,
        )

        if not selected:
            logger.info("No frames to analyse — video appears empty.")
            return []

        # Step 2: Try Gemini Vision (model rotation)
        detections = None
        if self._init_gemini():
            detections = self._classify_with_gemini(selected, fps)

        # Step 3: If Gemini failed, use local fallback (unlimited)
        if detections is None:
            logger.info(
                "Gemini unavailable — switching to local offline detection."
            )
            detections = self._local_fallback_detect(
                frames, motion_scores, frame_indices, fps
            )

        # Sort by timestamp
        detections.sort(key=lambda d: d["frame_index"])

        logger.info(
            "Detection complete [%s] — %d event(s) found.",
            self._detection_method or "unknown",
            len(detections),
        )
        return detections

    # ================================================================== #
    #                        TIMELINE & REPORTS                           #
    # ================================================================== #

    def create_timeline(
        self, detections: List[Dict[str, Any]]
    ) -> str:
        """
        Create a human-readable timeline of all detections.

        Args:
            detections: Output of detect().

        Returns:
            str: Markdown-formatted timeline.

        Example:
            >>> print(detector.create_timeline(detections))
        """
        if not detections:
            return "✅ **No vandalism detected** in the analysed footage."

        lines = ["## 🚨 Vandalism Detection Timeline\n"]
        lines.append("| # | Timestamp | Type | Confidence | Description |")
        lines.append("|---|-----------|------|------------|-------------|")

        for idx, d in enumerate(detections, 1):
            vtype = d.get('vandalism_type', 'unknown')
            conf = d.get('confidence', d.get('combined_score', 0.0))
            desc = d.get('description', '')
            if len(desc) > 60:
                desc = desc[:57] + "..."
            lines.append(
                f"| {idx} | {d['timestamp_str']} | "
                f"{vtype} | {conf:.2f} | {desc} |"
            )

        return "\n".join(lines)

    def generate_report(
        self,
        detections: List[Dict[str, Any]],
        video_metadata: dict,
        api_key: Optional[str] = None,
    ) -> str:
        """
        Generate an incident report (Gemini AI or template fallback).

        Args:
            detections:     List of detection dicts.
            video_metadata: Dict with fps, width, height, total_frames.
            api_key:        Gemini API key (falls back to config).

        Returns:
            str: Generated incident report text.

        Example:
            >>> report = detector.generate_report(dets, meta)
        """
        key = api_key or config.GEMINI_API_KEY

        if not key or key == "your_gemini_api_key_here":
            return self._template_report(detections, video_metadata)

        # Try Gemini models for report generation too
        models = config.GEMINI_MODEL_ROTATION.copy()

        for model_name in models:
            try:
                from google import genai

                client = genai.Client(api_key=key)

                det_summary = "\n".join(
                    f"- Frame {d['frame_index']} at {d['timestamp_str']}: "
                    f"type={d.get('vandalism_type', 'unknown')}, "
                    f"confidence={d.get('confidence', 0):.2f}, "
                    f"description={d.get('description', 'N/A')}"
                    for d in detections[:20]
                )

                prompt = (
                    "You are a surveillance security analyst. Write a "
                    "professional incident report based on these CCTV "
                    "vandalism detection results.\n\n"
                    f"**Video:** {video_metadata['total_frames']} frames, "
                    f"{video_metadata['fps']:.1f} FPS, "
                    f"{video_metadata['width']}×{video_metadata['height']}px\n\n"
                    f"**Detections ({len(detections)}):**\n{det_summary}\n\n"
                    "Include: Executive Summary, Incident Details, "
                    "Risk Assessment, Recommended Actions. Under 500 words."
                )

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                logger.info("Report generated via %s.", model_name)
                return response.text

            except Exception as exc:
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    logger.warning("Report model %s rate-limited.", model_name)
                    continue
                logger.error("Report generation failed (%s): %s", model_name, exc)
                continue

        # All models exhausted → template
        return self._template_report(detections, video_metadata)

    def _template_report(
        self,
        detections: List[Dict[str, Any]],
        video_metadata: dict,
    ) -> str:
        """
        Generate a static template report when Gemini is unavailable.

        Args:
            detections:     List of detection dicts.
            video_metadata: Video info dict.

        Returns:
            str: Formatted report string.

        Example:
            >>> print(detector._template_report(dets, meta))
        """
        total_duration = video_metadata["total_frames"] / video_metadata["fps"]

        report_lines = [
            "# 📋 Vandalism Incident Report",
            "",
            "## Executive Summary",
            f"Automated analysis of surveillance footage "
            f"({timedelta(seconds=int(total_duration))} duration) "
            f"detected **{len(detections)} vandalism event(s)**.",
            "",
            "## Video Information",
            f"- **Resolution:** {video_metadata['width']}×{video_metadata['height']}",
            f"- **Frame Rate:** {video_metadata['fps']:.1f} FPS",
            f"- **Total Frames:** {video_metadata['total_frames']}",
            f"- **Duration:** {timedelta(seconds=int(total_duration))}",
            "",
            "## Incident Details",
        ]

        if detections:
            for i, d in enumerate(detections[:10], 1):
                vtype = d.get("vandalism_type", "unknown")
                conf = d.get("confidence", d.get("combined_score", 0.0))
                desc = d.get("description", "Vandalism detected")
                report_lines.append(
                    f"{i}. **{d['timestamp_str']}** — "
                    f"**{vtype}** (confidence: {conf:.0%}) — {desc}"
                )
            if len(detections) > 10:
                report_lines.append(
                    f"\n*… and {len(detections) - 10} more events.*"
                )
        else:
            report_lines.append(
                "No vandalism detected. The footage shows normal activity only."
            )

        report_lines += [
            "",
            "## Recommended Actions",
        ]

        if detections:
            report_lines += [
                "1. Review flagged timestamps in the original footage.",
                "2. Cross-reference with access-control logs.",
                "3. Notify on-site security personnel if confirmed.",
                "4. Preserve evidence for potential law enforcement report.",
            ]
        else:
            report_lines += [
                "1. No immediate action required.",
                "2. Continue routine monitoring.",
            ]

        report_lines += [
            "",
            "---",
            "*Report generated automatically by the Vandalism Detection System.*",
        ]

        return "\n".join(report_lines)
