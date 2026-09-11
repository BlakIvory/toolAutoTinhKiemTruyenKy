"""
AutoQuestTool - Screen Capture
Wrapper cho screenshot operations.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2
from loguru import logger

from .error_codes import ErrorCode, BotError
from .config_loader import AppConfig


class ScreenCapture:
    """
    Quản lý screenshot.
    Ưu tiên: ADB screencap (không phụ thuộc cửa sổ).
    """

    def __init__(self, adb_controller, config: AppConfig):
        self._adb = adb_controller
        self.config = config
        self._last_frame: Optional[np.ndarray] = None
        self._debug_dir = Path(config.mode.save_path)
        if config.mode.debug_screenshots:
            self._debug_dir.mkdir(parents=True, exist_ok=True)

    def capture(self) -> np.ndarray:
        """
        Chụp full screenshot. Trả về numpy BGR array.
        """
        frame = self._adb.screenshot()
        if frame is None or frame.size == 0:
            raise BotError(ErrorCode.VIS_003, "Screenshot is empty")
        self._last_frame = frame
        return frame

    def capture_region(
        self,
        x1: int, y1: int,
        x2: int, y2: int,
        frame: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Crop ROI từ screenshot.
        Nếu frame được cung cấp → dùng frame đó (không chụp lại).
        Nếu không → chụp mới.
        """
        if frame is None:
            frame = self.capture()
        h, w = frame.shape[:2]
        # Clamp bounds
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            raise BotError(
                ErrorCode.VIS_003,
                f"Invalid region: ({x1},{y1}) → ({x2},{y2})",
            )
        return frame[y1:y2, x1:x2]

    def save_debug_image(
        self,
        frame: np.ndarray,
        name: str = "debug",
        annotate: str = "",
    ) -> str:
        """Lưu ảnh debug với timestamp. Trả về path."""
        if not self.config.mode.debug_screenshots:
            return ""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{ts}_{name}.png"
        path = self._debug_dir / filename
        img = frame.copy()
        if annotate:
            cv2.putText(
                img, annotate, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
        cv2.imwrite(str(path), img)
        logger.debug(f"Debug image saved: {path}")
        return str(path)

    def get_last_frame(self) -> Optional[np.ndarray]:
        return self._last_frame

    def wait_and_capture(self, wait_s: float = 1.0) -> np.ndarray:
        """Chờ rồi chụp."""
        time.sleep(wait_s)
        return self.capture()
