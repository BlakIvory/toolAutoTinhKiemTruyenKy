"""
AutoQuestTool - Mock ADB Controller
Dùng cho test workflow mà không cần LDPlayer/game chạy.
"""

from __future__ import annotations

import io
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger

from ..config_loader import AppConfig


class MockADBController:
    """
    MockADB: simulate ADB responses.
    Không cần emulator thật.
    Dùng test assets từ tests/assets/.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._connected = True
        self._mock_state = "LOGIN_SCREEN"  # Có thể thay đổi trong test
        self._screenshot_path: Optional[str] = None  # Dùng ảnh thật từ tests/assets/
        logger.warning("[MOCK] MockADBController active – NO real device")

    def connect(self) -> bool:
        logger.info("[MOCK] ADB connect OK")
        return True

    def disconnect(self) -> None:
        logger.info("[MOCK] ADB disconnect")

    def is_connected(self) -> bool:
        return self._connected

    def get_devices(self):
        return ["emulator-5555"]

    def screenshot(self) -> np.ndarray:
        """Trả về ảnh test từ disk hoặc ảnh đen nếu không có."""
        if self._screenshot_path and Path(self._screenshot_path).exists():
            import cv2
            img = cv2.imread(self._screenshot_path)
            logger.debug(f"[MOCK] screenshot from {self._screenshot_path}")
            return img

        # Ảnh đen 1280x720
        logger.debug(f"[MOCK] screenshot (black {self.config.emulator.width}x{self.config.emulator.height})")
        return np.zeros(
            (self.config.emulator.height, self.config.emulator.width, 3),
            dtype=np.uint8
        )

    def tap(self, x: int, y: int) -> None:
        logger.info(f"[MOCK] tap ({x}, {y})")

    def double_tap(self, x: int, y: int) -> None:
        logger.info(f"[MOCK] double_tap ({x}, {y})")

    def long_press(self, x: int, y: int, duration_ms: int = 1000) -> None:
        logger.info(f"[MOCK] long_press ({x}, {y}) {duration_ms}ms")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        logger.info(f"[MOCK] swipe ({x1},{y1}) → ({x2},{y2}) {duration_ms}ms")

    def drag(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500) -> None:
        logger.info(f"[MOCK] drag ({x1},{y1}) → ({x2},{y2}) {duration_ms}ms")

    def input_text(self, text: str) -> None:
        logger.info(f"[MOCK] input_text [masked]")

    def key_event(self, keycode: int) -> None:
        logger.debug(f"[MOCK] key_event {keycode}")

    def press_back(self) -> None:
        logger.debug("[MOCK] BACK key")

    def press_home(self) -> None:
        logger.debug("[MOCK] HOME key")

    def start_app(self) -> None:
        logger.info(f"[MOCK] start_app {self.config.game.package_name}")

    def stop_app(self) -> None:
        logger.info(f"[MOCK] stop_app {self.config.game.package_name}")

    def restart_app(self) -> None:
        logger.info("[MOCK] restart_app")

    def is_app_running(self) -> bool:
        return True

    def get_current_activity(self) -> str:
        return "com.example.MainActivity"

    def get_resolution(self) -> Tuple[int, int]:
        return (self.config.emulator.width, self.config.emulator.height)

    def shell(self, cmd: str) -> str:
        logger.debug(f"[MOCK] shell: {cmd}")
        return ""

    def set_mock_screenshot(self, path: str) -> None:
        """Set ảnh sẽ trả về khi screenshot() được gọi."""
        self._screenshot_path = path

    def set_mock_state(self, state: str) -> None:
        """Cập nhật mock state để test flow."""
        self._mock_state = state
        logger.debug(f"[MOCK] state set to: {state}")
