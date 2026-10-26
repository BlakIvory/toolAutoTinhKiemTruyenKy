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
        logger.debug(f"INPUT: tap ({x}, {y})")
        self._adb.tap(x, y)
        if delay_after is not None:
            time.sleep(delay_after)

    def tap_match(self, result: MatchResult, delay_after: Optional[float] = None) -> None:
        """Tap vào center của MatchResult từ VisionEngine."""
        if not result.found:
            raise BotError(ErrorCode.ADB_004, "Cannot tap: MatchResult not found")
        self.tap(result.center_x, result.center_y, delay_after)

    def double_tap(self, x: int, y: int) -> None:
        logger.debug(f"INPUT: double_tap ({x}, {y})")
        self._adb.double_tap(x, y)

    def _send_win32_right_click(self, x: int, y: int, count: int = 2) -> bool:
        """
        Gửi Windows Message WM_RBUTTONDOWN và WM_RBUTTONUP tới cửa sổ LDPlayer.
        Tự động tìm kiếm cửa sổ LDPlayer (RenderWindow / subWin) và chuyển đổi tọa độ tỉ lệ.
        """
        import sys
        if sys.platform != "win32":
            return False

        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            # Đảm bảo gắn với desktop "default" nếu bot chạy trong thread/session cách ly
            h_desk = user32.OpenDesktopW("default", 0, False, 0x01FF)
            if h_desk:
                user32.SetThreadDesktop(h_desk)

            target_hwnds = []

            # 1. Thử tìm nhanh bằng class name LDPlayerMainFrame
            main_hwnd = user32.FindWindowW("LDPlayerMainFrame", None)
            if main_hwnd and user32.IsWindowVisible(main_hwnd):
                target_hwnds.append(main_hwnd)

            # 2. Nếu chưa thấy, EnumWindows tìm theo title/class
            if not target_hwnds:
                def enum_cb(hwnd, extra):
                    if user32.IsWindowVisible(hwnd):
                        length = user32.GetWindowTextLengthW(hwnd)
                        if length > 0:
                            buff = ctypes.create_unicode_buffer(length + 1)
                            user32.GetWindowTextW(hwnd, buff, length + 1)
                            title = buff.value.lower()
                            class_buff = ctypes.create_unicode_buffer(256)
                            user32.GetClassNameW(hwnd, class_buff, 256)
                            c_name = class_buff.value.lower()
                            if any(kw in title or kw in c_name for kw in ["ldplayer", "dnplayer", "tinh kiem"]):
                                target_hwnds.append(hwnd)
                    return True

                WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
                user32.EnumWindows(WNDENUMPROC(enum_cb), 0)

            if not target_hwnds:
                return False

            WM_MOUSEMOVE = 0x0200
            WM_RBUTTONDOWN = 0x0204
            WM_RBUTTONUP = 0x0205
            MK_RBUTTON = 0x0002
            success = False

            for main_hwnd in target_hwnds:
                child_hwnds = []

                def enum_child_cb(child_hwnd, extra):
                    class_buff = ctypes.create_unicode_buffer(256)
                    user32.GetClassNameW(child_hwnd, class_buff, 256)
                    c_name = class_buff.value
                    if "RenderWindow" in c_name or "subWin" in c_name:
                        child_hwnds.append((child_hwnd, c_name))
                    return True

                WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
                user32.EnumChildWindows(main_hwnd, WNDENUMPROC(enum_child_cb), 0)

                # Ưu tiên duy nhất cửa sổ RenderWindow (TheRender) của LDPlayer để nhận sự kiện chuột
                render_hwnds = [ch for ch, c_name in child_hwnds if "renderwindow" in c_name.lower()]
                targets = render_hwnds if render_hwnds else ([ch for ch, _ in child_hwnds] if child_hwnds else [main_hwnd])

                for target_hwnd in targets:
                    rect = wintypes.RECT()
                    user32.GetClientRect(target_hwnd, ctypes.byref(rect))
                    win_w = rect.right - rect.left
                    win_h = rect.bottom - rect.top

                    if win_w > 0 and win_h > 0:
                        scale_x = int(x * win_w / 1600.0)
                        scale_y = int(y * win_h / 900.0)
                    else:
                        scale_x, scale_y = x, y

                    l_param = (scale_y << 16) | (scale_x & 0xFFFF)
                    for i in range(count):
                        user32.PostMessageW(target_hwnd, WM_MOUSEMOVE, 0, l_param)
                        user32.PostMessageW(target_hwnd, WM_RBUTTONDOWN, MK_RBUTTON, l_param)
                        time.sleep(0.04)
                        user32.PostMessageW(target_hwnd, WM_RBUTTONUP, 0, l_param)
                        if i < count - 1:
                            time.sleep(0.08)
                    logger.debug(f"INPUT: PostMessage WM_RBUTTON (x{count}) to hwnd={target_hwnd} at ({scale_x}, {scale_y})")
                    success = True

            return success
        except Exception as e:
            logger.debug(f"INPUT: _send_win32_right_click error: {e}")
            return False

    def right_click(self, x: int, y: int, count: int = 1) -> None:
        """
        Gửi sự kiện chuột phải (Right Click) tại tọa độ (x, y):
        Mặc định click 2 lần (double right-click) theo yêu cầu thực chiến để vừa chọn mục tiêu vừa tấn công.
        1. Gửi qua Windows Message (WM_RBUTTONDOWN / WM_RBUTTONUP) tới cửa sổ LDPlayer.
        2. Chỉ fallback qua ADB khi Windows Message không gửi được.
        """
        logger.info(f"INPUT: right_click x{count} ({x}, {y})")
        sent_by_win32 = self._send_win32_right_click(x, y, count=count)
        sent_by_adb = None
        try:
            # ADBController.right_click thực chất là Android input tap
            # (chuột trái). Không gửi nó sau chuột phải Windows vì cú
            # tap trái kế tiếp có thể ghi đè/hủy lệnh tấn công.
            if not sent_by_win32 and hasattr(self._adb, "right_click"):
                self._adb.right_click(x, y, count=count)
                sent_by_adb = True
            elif not sent_by_win32:
                for i in range(count):
                    self.tap(x, y)
                    if i < count - 1:
                        time.sleep(0.1)
                sent_by_adb = True
        except Exception as exc:
            logger.warning(f"INPUT: ADB right_click failed at ({x}, {y}): {exc}")
        logger.info(
            f"INPUT: right_click delivery win32={'OK' if sent_by_win32 else 'FAIL'}, "
            f"adb={'SKIP' if sent_by_adb is None else ('OK' if sent_by_adb else 'FAIL')}"
        )

    def long_press(self, x: int, y: int, duration_ms: int = 1000) -> None:
        logger.debug(f"INPUT: long_press ({x}, {y}) {duration_ms}ms")
        self._adb.long_press(x, y, duration_ms)

    # ============================
    # Swipe / Drag
    # ============================

    def swipe(
        self,
        x1: int, y1: int,
        x2: int, y2: int,
        duration_ms: Optional[int] = None,
        duration: Optional[int] = None,
    ) -> None:
        dur = duration_ms or duration or self.config.timing.swipe_duration
        logger.debug(f"INPUT: swipe ({x1},{y1}) → ({x2},{y2}) {dur}ms")
        self._adb.swipe(x1, y1, x2, y2, dur)

    def drag(
        self,
        x1: int, y1: int,
        x2: int, y2: int,
        duration_ms: int = 500,
    ) -> None:
        logger.debug(f"INPUT: drag ({x1},{y1}) → ({x2},{y2}) {duration_ms}ms")
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
        self, x: int, y: int, text: str, clear_length: int = 20
    ) -> None:
        """Tap vào field, xóa nội dung cũ và nhập text mới."""
        self.tap(x, y)
        time.sleep(0.8)
        # Gửi backspace để xóa ký tự cũ (nếu có)
        del_keys = " ".join(["67"] * min(clear_length, 15))
        try:
            self._adb._run(["shell", f"input keyevent {del_keys}"])
        except Exception:
            pass
        time.sleep(0.3)
        self.input_text(str(text))
        time.sleep(0.6)

    # ============================
    # Navigation Keys
    # ============================

    def key_event(self, keycode: int) -> None:
        """Gửi keycode Android."""
        logger.debug(f"INPUT: key_event {keycode}")
        self._adb.key_event(keycode)

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
