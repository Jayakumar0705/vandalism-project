"""
test_validation.py — Unit Tests for the Vandalism Detection System.

Tests covering:
    1. All module imports succeed
    2. Config loads and check_keys() returns bool
    3. Frame chunking produces correct structure
    4. Feature embedding returns correct shape
    5. Frame-to-base64 encoding for Vision API
    6. End-to-end pipeline on a synthetic video
    7. Timeline and report generation

Run:
    python test_validation.py
    # or
    pytest test_validation.py -v
"""

import sys
import unittest
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

# ── Ensure project root is on sys.path ──
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.WARNING)


class TestImports(unittest.TestCase):
    """Test 1: Verify every project module can be imported."""

    def test_all_imports(self) -> None:
        """
        Import every module and class the project exposes.

        Args:
            None

        Returns:
            None

        Example:
            >>> TestImports().test_all_imports()
        """
        try:
            import config
            from utils.document_loader import VideoLoader
            from utils.chunker import FrameChunker
            from utils.embedder import FeatureExtractor
            from utils.retriever import VandalismDetector
            from utils.validator import InputValidator
        except ImportError as exc:
            self.fail(f"Import failed: {exc}")


class TestConfig(unittest.TestCase):
    """Test 2: Config module loads correctly and exposes expected items."""

    def test_check_keys_returns_bool(self) -> None:
        """
        check_keys() must return a boolean regardless of key state.

        Args:
            None

        Returns:
            None

        Example:
            >>> TestConfig().test_check_keys_returns_bool()
        """
        import config

        result = config.check_keys()
        self.assertIsInstance(result, bool)

    def test_constants_exist(self) -> None:
        """
        Essential constants must be defined in config.

        Args:
            None

        Returns:
            None

        Example:
            >>> TestConfig().test_constants_exist()
        """
        import config

        self.assertIsInstance(config.FRAME_SKIP, int)
        self.assertIsInstance(config.CHUNK_SIZE, int)
        self.assertIsInstance(config.VANDALISM_SENSITIVITY, float)
        self.assertIsInstance(config.SUPPORTED_VIDEO_EXTENSIONS, list)
        self.assertGreater(config.MAX_FILE_SIZE_MB, 0)
        # Vision-related constants
        self.assertIsInstance(config.GEMINI_VISION_MODEL, str)
        self.assertIsInstance(config.MAX_FRAMES_FOR_VISION, int)
        self.assertIsInstance(config.VANDALISM_CONFIDENCE_THRESHOLD, float)


class TestChunking(unittest.TestCase):
    """Test 3: FrameChunker produces valid chunks and motion scores."""

    def _make_frames(self, n: int = 20) -> List[np.ndarray]:
        """Generate n synthetic BGR frames (240x320x3)."""
        rng = np.random.RandomState(42)
        return [rng.randint(0, 255, (240, 320, 3), dtype=np.uint8) for _ in range(n)]

    def test_chunk_structure(self) -> None:
        """
        chunk_frames() must return dicts with frames, indices, chunk_id.

        Args:
            None

        Returns:
            None

        Example:
            >>> TestChunking().test_chunk_structure()
        """
        from utils.chunker import FrameChunker

        chunker = FrameChunker(chunk_size=10, chunk_overlap=2)
        frames = self._make_frames(20)
        indices = list(range(20))

        chunks = chunker.chunk_frames(frames, indices)

        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIn("frames", chunk)
            self.assertIn("indices", chunk)
            self.assertIn("chunk_id", chunk)
            self.assertLessEqual(len(chunk["frames"]), 10)

    def test_motion_scores_length(self) -> None:
        """
        Motion scores list must be the same length as input frames.

        Args:
            None

        Returns:
            None

        Example:
            >>> TestChunking().test_motion_scores_length()
        """
        from utils.chunker import FrameChunker

        chunker = FrameChunker()
        frames = self._make_frames(10)
        scores = chunker.compute_motion_scores(frames)

        self.assertEqual(len(scores), len(frames))
        self.assertEqual(scores[0], 0.0)


class TestEmbedding(unittest.TestCase):
    """Test 4: FeatureExtractor returns vectors of correct shape."""

    def test_single_frame_embedding(self) -> None:
        """
        extract_features() must return a 1-D numpy array.

        Args:
            None

        Returns:
            None

        Example:
            >>> TestEmbedding().test_single_frame_embedding()
        """
        from utils.embedder import FeatureExtractor

        extractor = FeatureExtractor()
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

        vec = extractor.extract_features(frame)

        self.assertIsInstance(vec, np.ndarray)
        self.assertEqual(vec.ndim, 1)
        self.assertGreater(len(vec), 0)

    def test_batch_embedding_shape(self) -> None:
        """
        extract_batch_features() shape must be (N, D).

        Args:
            None

        Returns:
            None

        Example:
            >>> TestEmbedding().test_batch_embedding_shape()
        """
        from utils.embedder import FeatureExtractor

        extractor = FeatureExtractor()
        frames = [
            np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            for _ in range(3)
        ]

        matrix = extractor.extract_batch_features(frames)

        self.assertEqual(matrix.shape[0], 3)
        self.assertGreater(matrix.shape[1], 0)

    def test_frame_to_base64(self) -> None:
        """
        frame_to_base64() must return a non-empty base64 string.

        Args:
            None

        Returns:
            None

        Example:
            >>> TestEmbedding().test_frame_to_base64()
        """
        import base64
        from utils.embedder import FeatureExtractor

        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        b64 = FeatureExtractor.frame_to_base64(frame)

        self.assertIsInstance(b64, str)
        self.assertGreater(len(b64), 0)

        decoded = base64.b64decode(b64)
        self.assertTrue(decoded[:2] == b'\xff\xd8', "Not a valid JPEG header")

    def test_prepare_frames_for_vision(self) -> None:
        """
        prepare_frames_for_vision() must return dicts with expected keys.

        Args:
            None

        Returns:
            None

        Example:
            >>> TestEmbedding().test_prepare_frames_for_vision()
        """
        from utils.embedder import FeatureExtractor

        extractor = FeatureExtractor()
        frames = [
            np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        motion_scores = [0.0, 100.0, 5000.0, 200.0, 0.0]
        indices = [0, 5, 10, 15, 20]

        prepared = extractor.prepare_frames_for_vision(
            frames, motion_scores, indices, max_frames=3
        )

        self.assertGreater(len(prepared), 0)
        self.assertLessEqual(len(prepared), 3)

        for item in prepared:
            self.assertIn("frame_index", item)
            self.assertIn("motion_score", item)
            self.assertIn("base64_image", item)
            self.assertGreater(item["motion_score"], 0.0)


class TestFrameSelection(unittest.TestCase):
    """Test 5: VandalismDetector selects correct frames for analysis."""

    def test_frame_selection(self) -> None:
        """
        _select_frames_for_analysis() should select high-motion frames.

        Args:
            None

        Returns:
            None

        Example:
            >>> TestFrameSelection().test_frame_selection()
        """
        from utils.retriever import VandalismDetector

        detector = VandalismDetector(sensitivity=0.65)

        frames = [
            np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        motion = [0.0, 0.0, 5000.0, 0.0, 8000.0, 0.0, 3000.0, 0.0, 0.0, 0.0]
        indices = list(range(10))

        selected = detector._select_frames_for_analysis(
            frames, motion, indices, max_frames=4
        )

        self.assertGreater(len(selected), 0)
        self.assertLessEqual(len(selected), 4)

        for item in selected:
            self.assertIn("frame_index", item)
            self.assertIn("base64_image", item)
            self.assertIn("list_position", item)

    def test_empty_frames(self) -> None:
        """
        _select_frames_for_analysis() should return empty list for no frames.

        Args:
            None

        Returns:
            None

        Example:
            >>> TestFrameSelection().test_empty_frames()
        """
        from utils.retriever import VandalismDetector

        detector = VandalismDetector()
        result = detector._select_frames_for_analysis([], [], [])
        self.assertEqual(result, [])


class TestEndToEnd(unittest.TestCase):
    """Test 6: Full pipeline on a programmatically-generated video."""

    def _create_synthetic_video(self) -> Path:
        """
        Write a small .avi video with motion in the middle section
        to simulate activity.

        Returns:
            Path to the generated video file.
        """
        import config
        config.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        path = config.TEMP_DIR / "test_synthetic.avi"

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(path), fourcc, 10.0, (320, 240))

        rng = np.random.RandomState(123)

        for i in range(30):
            frame = np.full((240, 320, 3), 128, dtype=np.uint8)
            if 10 <= i <= 20:
                x, y = rng.randint(50, 200), rng.randint(50, 150)
                cv2.rectangle(frame, (x, y), (x + 80, y + 60), (255, 255, 255), -1)
            writer.write(frame)

        writer.release()
        return path

    def test_full_pipeline(self) -> None:
        """
        Run extract -> chunk -> detect on a synthetic video.
        Pipeline should complete without crashing (Gemini may be unavailable).

        Args:
            None

        Returns:
            None

        Example:
            >>> TestEndToEnd().test_full_pipeline()
        """
        from utils.document_loader import VideoLoader
        from utils.chunker import FrameChunker
        from utils.retriever import VandalismDetector

        video_path = self._create_synthetic_video()
        self.assertTrue(video_path.exists(), "Synthetic video was not created.")

        loader = VideoLoader(frame_skip=1)
        frames, indices, meta = loader.load_and_extract(video_path)
        self.assertGreater(len(frames), 0, "No frames extracted.")
        self.assertIn("fps", meta)

        chunker = FrameChunker()
        motion = chunker.compute_motion_scores(frames)
        self.assertEqual(len(motion), len(frames))

        # Detect — may return empty list if Gemini is rate-limited
        detector = VandalismDetector(sensitivity=0.5)
        detections = detector.detect(frames, motion, indices, meta["fps"])
        self.assertIsInstance(detections, list)

        # Report and timeline should always work
        report = detector.generate_report(detections, meta)
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0, "Report must not be empty.")

        timeline = detector.create_timeline(detections)
        self.assertIsInstance(timeline, str)

        try:
            video_path.unlink()
        except OSError:
            pass


class TestTimelineAndReport(unittest.TestCase):
    """Test 7: Timeline and report generation."""

    def test_timeline_no_detections(self) -> None:
        """
        Timeline should show safe message when no vandalism is found.

        Args:
            None

        Returns:
            None

        Example:
            >>> TestTimelineAndReport().test_timeline_no_detections()
        """
        from utils.retriever import VandalismDetector

        detector = VandalismDetector()
        timeline = detector.create_timeline([])

        self.assertIn("No vandalism detected", timeline)

    def test_timeline_with_detections(self) -> None:
        """
        Timeline should contain detection info when vandalism found.

        Args:
            None

        Returns:
            None

        Example:
            >>> TestTimelineAndReport().test_timeline_with_detections()
        """
        from utils.retriever import VandalismDetector

        detector = VandalismDetector()
        detections = [
            {
                "frame_index": 50,
                "timestamp_str": "0:00:02",
                "combined_score": 0.85,
                "confidence": 0.85,
                "vandalism_type": "spray_painting",
                "description": "Person spray painting wall",
            }
        ]
        timeline = detector.create_timeline(detections)

        self.assertIn("spray_painting", timeline)
        self.assertIn("0:00:02", timeline)

    def test_template_report_with_detections(self) -> None:
        """
        Template report should include incident details when detections exist.

        Args:
            None

        Returns:
            None

        Example:
            >>> TestTimelineAndReport().test_template_report_with_detections()
        """
        from utils.retriever import VandalismDetector

        detector = VandalismDetector()
        detections = [
            {
                "frame_index": 50,
                "timestamp_str": "0:00:02",
                "combined_score": 0.85,
                "confidence": 0.85,
                "vandalism_type": "spray_painting",
                "description": "Graffiti detected",
            }
        ]
        meta = {"fps": 30.0, "width": 320, "height": 240, "total_frames": 300}

        report = detector._template_report(detections, meta)

        self.assertIn("Incident Report", report)
        self.assertIn("0:00:02", report)
        self.assertIn("1 vandalism event", report)

    def test_template_report_no_detections(self) -> None:
        """
        Template report should indicate safety when no detections.

        Args:
            None

        Returns:
            None

        Example:
            >>> TestTimelineAndReport().test_template_report_no_detections()
        """
        from utils.retriever import VandalismDetector

        detector = VandalismDetector()
        meta = {"fps": 30.0, "width": 320, "height": 240, "total_frames": 300}

        report = detector._template_report([], meta)

        self.assertIn("No vandalism detected", report)
        self.assertIn("0 vandalism event", report)


if __name__ == "__main__":
    print("=" * 60)
    print("  Vandalism Detection — Validation Test Suite")
    print("=" * 60)
    unittest.main(verbosity=2)
