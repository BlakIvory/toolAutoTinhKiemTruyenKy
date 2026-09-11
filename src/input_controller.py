"""
AutoQuestTool - Input Controller
Wrapper input actions. Sử dụng ADBController bên dưới.
Không dùng PyAutoGUI để click trực tiếp cửa sổ LDPlayer.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

from loguru import logger

from .adb_controller import ADBController
from .config_loader import AppConfig
from .error_codes import ErrorCode, BotError
from .vision_engine import MatchResult


class InputController:
    """
    Lớp input action cho game.
    API:
        tap(x, y)
        tap_match(result) - tap vào center của MatchResult
        double_tap(x, y)
        long_press(x, y, duration_ms)
        swipe(x1, y1, x2, y2, duration_ms)
        drag(x1, y1, x2, y2, duration_ms)
        input_text(text)
        clear_and_type(text)
    """

    def __init__(self, adb: ADBController, config: AppConfig):
        self._adb = adb
        self.config = config

    # ============================
    # Tap
    # ============================

    def tap(self, x: int, y: int, delay_after: Optional[float] = None) -> None:
        """Tap tại tọa độ."""
        logger.info(f"INPUT: tap ({x}, {y})")
        self._adb.tap(x, y)
        if delay_after is not None:
            time.sleep(delay_after)

    def tap_match(self, result: MatchResult, delay_after: Optional[float] = None) -> None:
        """Tap vào center của MatchResult từ VisionEngine."""
        if not result.found:
            raise BotError(ErrorCode.ADB_004, "Cannot tap: MatchResult not found")
        self.tap(result.center_x, result.center_y, delay_after)

    def double_tap(self, x: int, y: int) -> None:
        logger.info(f"INPUT: double_tap ({x}, {y})")
        self._adb.double_tap(x, y)

    def long_press(self, x: int, y: int, duration_ms: int = 1000) -> None:
        logger.info(f"INPUT: long_press ({x}, {y}) {duration_ms}ms")
        self._adb.long_press(x, y, duration_ms)

    # ============================
    # Swipe / Drag
    # ============================

    def swipe(
        self,
        x1: int, y1: int,
        x2: int, y2: int,
        duration_ms: Optional[int] = None,
    ) -> None:
        dur = duration_ms or self.config.timing.swipe_duration
        logger.info(f"INPUT: swipe ({x1},{y1}) → ({x2},{y2}) {dur}ms")
        self._adb.swipe(x1, y1, x2, y2, dur)

    def drag(
        self,
        x1: int, y1: int,
        x2: int, y2: int,
        duration_ms: int = 500,
    ) -> None:
        logger.info(f"INPUT: drag ({x1},{y1}) → ({x2},{y2}) {duration_ms}ms")
        self._adb.drag(x1, y1, x2, y2, duration_ms)

    def scroll_up(self, center_x: int, center_y: int, distance: int = 300) -> None:
        self.swipe(center_x, center_y + distance // 2, center_x, center_y - distance // 2)

    def scroll_down(self, center_x: int, center_y: int, distance: int = 300) -> None:
        self.swipe(center_x, center_y - distance // 2, center_x, center_y + distance // 2)

    # ============================
    # Text Input
    # ============================

    def input_text(self, text: str) -> None:
        """Nhập text vào field đang focus."""
        logger.info(f"INPUT: input_text [masked]")
        self._adb.input_text(text)

    def clear_and_type(
        self, x: int, y: int, text: str, clear_length: int = 30
    ) -> None:
        """Tap vào field, xóa nội dung cũ rồi nhập text mới."""
        self.tap(x, y)
        time.sleep(0.3)
        # Chọn tất cả rồi xóa
        self._adb.key_event(123)   # KEYCODE_MOVE_END
        for _ in range(clear_length):
            self._adb.key_event(67)  # KEYCODE_DEL
        time.sleep(0.2)
        self.input_text(text)

    # ============================
    # Navigation Keys
    # ============================

    def press_back(self) -> None:
        logger.debug("INPUT: BACK key")
        self._adb.press_back()
        time.sleep(0.5)

    def press_home(self) -> None:
        logger.debug("INPUT: HOME key")
        self._adb.press_home()

    # ============================
    # Coordinate from Config
    # ============================

    def tap_config_point(self, section: str, key: str, coords_config: dict) -> None:
        """Tap vào điểm được khai báo trong ui_coordinates.yaml."""
        try:
            point = coords_config[section][key]
            self.tap(point["x"], point["y"])
        except KeyError:
            raise BotError(
                ErrorCode.CFG_004,
                f"Coordinate not found: {section}.{key}",
            )
