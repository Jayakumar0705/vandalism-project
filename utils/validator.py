"""
validator.py — Input Validation for Uploaded Surveillance Files.

Centralises all pre-processing checks (file size, extension, corruption)
so the Streamlit app can give clear error messages instead of crashing
on bad input.
"""

import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

import config

# ── Module logger ──
logger = logging.getLogger(__name__)


class InputValidator:
    """Validate uploaded video files before processing."""

    # ------------------------------------------------------------------ #
    #  Extension check
    # ------------------------------------------------------------------ #
    @staticmethod
    def validate_extension(filename: str) -> Tuple[bool, str]:
        """
        Check whether the file extension is in the supported list.

        Args:
            filename: Original filename with extension.

        Returns:
            Tuple[bool, str]: (is_valid, error_message).

        Example:
            >>> InputValidator.validate_extension("clip.mp4")
            (True, '')
        """
        ext = Path(filename).suffix.lower()

        if ext not in config.SUPPORTED_VIDEO_EXTENSIONS:
            msg = (
                f"Unsupported file type '{ext}'. "
                f"Allowed: {', '.join(config.SUPPORTED_VIDEO_EXTENSIONS)}"
            )
            logger.warning(msg)
            return False, msg

        logger.debug("Extension '%s' is valid.", ext)
        return True, ""

    # ------------------------------------------------------------------ #
    #  File-size check
    # ------------------------------------------------------------------ #
    @staticmethod
    def validate_file_size(file_size_bytes: int) -> Tuple[bool, str]:
        """
        Ensure the file does not exceed the configured maximum size.

        Args:
            file_size_bytes: Size of the uploaded file in bytes.

        Returns:
            Tuple[bool, str]: (is_valid, error_message).

        Example:
            >>> InputValidator.validate_file_size(1024 * 1024)
            (True, '')
        """
        max_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
        size_mb = file_size_bytes / (1024 * 1024)

        if file_size_bytes > max_bytes:
            msg = (
                f"File too large ({size_mb:.1f} MB). "
                f"Maximum allowed: {config.MAX_FILE_SIZE_MB} MB."
            )
            logger.warning(msg)
            return False, msg

        logger.debug("File size %.1f MB is within limit.", size_mb)
        return True, ""

    # ------------------------------------------------------------------ #
    #  Corruption / readability check
    # ------------------------------------------------------------------ #
    @staticmethod
    def validate_video_readable(file_path: Path) -> Tuple[bool, str]:
        """
        Verify OpenCV can open the file and read at least one frame.

        Args:
            file_path: Path to the saved video file.

        Returns:
            Tuple[bool, str]: (is_valid, error_message).

        Example:
            >>> InputValidator.validate_video_readable(Path("vid.mp4"))
            (True, '')
        """
        try:
            cap = cv2.VideoCapture(str(file_path))

            if not cap.isOpened():
                msg = "Video file could not be opened — it may be corrupted."
                logger.warning(msg)
                return False, msg

            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                msg = "Video file is empty or contains no readable frames."
                logger.warning(msg)
                return False, msg

            logger.debug("Video is readable and contains valid frames.")
            return True, ""

        except cv2.error as exc:
            msg = f"OpenCV error while validating video: {exc}"
            logger.error(msg)
            return False, msg

    # ------------------------------------------------------------------ #
    #  Combined validation (runs all checks in order)
    # ------------------------------------------------------------------ #
    @classmethod
    def validate_upload(
        cls, uploaded_file
    ) -> Tuple[bool, str]:
        """
        Run all validations on a Streamlit UploadedFile object.

        Args:
            uploaded_file: Streamlit UploadedFile (must have .name and .size).

        Returns:
            Tuple[bool, str]: (is_valid, error_message).

        Example:
            >>> ok, err = InputValidator.validate_upload(st_file)
        """
        # Guard against None / empty submission
        if uploaded_file is None:
            return False, "No file uploaded. Please select a video file."

        # 1. Extension
        ok, msg = cls.validate_extension(uploaded_file.name)
        if not ok:
            return False, msg

        # 2. Size
        ok, msg = cls.validate_file_size(uploaded_file.size)
        if not ok:
            return False, msg

        logger.info("Upload '%s' passed all pre-save validations.", uploaded_file.name)
        return True, ""

    # ------------------------------------------------------------------ #
    #  Full validation (including post-save readability)
    # ------------------------------------------------------------------ #
    @classmethod
    def validate_full(
        cls, uploaded_file, saved_path: Path
    ) -> Tuple[bool, str]:
        """
        Run pre-save AND post-save (corruption) checks.

        Args:
            uploaded_file: Streamlit UploadedFile.
            saved_path:    Where the file was written on disk.

        Returns:
            Tuple[bool, str]: (is_valid, error_message).

        Example:
            >>> ok, err = InputValidator.validate_full(st_file, path)
        """
        ok, msg = cls.validate_upload(uploaded_file)
        if not ok:
            return False, msg

        ok, msg = cls.validate_video_readable(saved_path)
        if not ok:
            return False, msg

        logger.info("Full validation passed for '%s'.", uploaded_file.name)
        return True, ""
