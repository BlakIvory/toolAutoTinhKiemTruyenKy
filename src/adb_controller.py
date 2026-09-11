"""
AutoQuestTool - ADB Controller
Tất cả ADB command phải đi qua class này. 
Không gọi subprocess ADB rải rác trong project.
"""

from __future__ import annotations

import subprocess
import time
import shutil
import io
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image
from loguru import logger

from .error_codes import ErrorCode, BotError
from .config_loader import AppConfig


class ADBController:
    """
    Wrapper cho ADB commands.
    API chính:
        connect() / disconnect() / is_connected()
        screenshot() → np.ndarray
        tap(x, y)
        double_tap(x, y)
        long_press(x, y, duration_ms)
        swipe(x1, y1, x2, y2, duration_ms)
        input_text(text)
        start_app() / stop_app() / restart_app()
        get_resolution() → (w, h)
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._device_serial: Optional[str] = None
        self._adb_path = self._resolve_adb_path()

    # ============================
    # Setup
    # ============================

    def _resolve_adb_path(self) -> str:
        """Tìm adb.exe: từ config → PATH → thư mục LDPlayer."""
        cfg_path = self.config.adb.adb_path
        if cfg_path and Path(cfg_path).exists():
            logger.debug(f"ADB: using config path: {cfg_path}")
            return cfg_path

        in_path = shutil.which("adb")
        if in_path:
            logger.debug(f"ADB: found in PATH: {in_path}")
            return in_path

        # Tìm trong thư mục LDPlayer
        ldplayer_dir = Path(self.config.emulator.ldplayer_path)
        candidates = [
            ldplayer_dir / "adb.exe",
            ldplayer_dir / "platform-tools" / "adb.exe",
        ]
        for c in candidates:
            if c.exists():
                logger.debug(f"ADB: found near LDPlayer: {c}")
                return str(c)

        raise BotError(
            ErrorCode.ADB_001,
            "adb.exe not found. Set adb_path in config/app.yaml or add to PATH.",
        )

    def _run(
        self,
        args: List[str],
        timeout: Optional[int] = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess:
        """Chạy adb command, trả về CompletedProcess."""
        cmd = [self._adb_path]
        if self._device_serial:
            cmd += ["-s", self._device_serial]
        cmd += args

        t = timeout or self.config.adb.command_timeout
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=t,
            )
            if check and result.returncode != 0:
                err = result.stderr.decode(errors="replace").strip()
                raise BotError(
                    ErrorCode.ADB_006,
                    f"ADB command failed: {' '.join(args)}\n{err}",
                )
            return result
        except subprocess.TimeoutExpired:
            raise BotError(
                ErrorCode.ADB_006,
                f"ADB command timed out ({t}s): {' '.join(args)}",
            )
        except BotError:
            raise
        except Exception as e:
            raise BotError(ErrorCode.ADB_001, f"ADB error: {e}")

    # ============================
    # Connection
    # ============================

    def connect(self) -> bool:
        """Kết nối ADB đến LDPlayer emulator."""
        host = self.config.emulator.adb_host
        port = self.config.emulator.adb_port
        address = f"{host}:{port}"
        logger.info(f"ADB: connecting to {address}...")

        result = self._run(
            ["connect", address],
            timeout=self.config.adb.connect_timeout,
        )
        out = result.stdout.decode(errors="replace").strip()
        logger.debug(f"ADB connect output: {out}")

        if "connected" in out.lower():
            self._device_serial = address
            logger.info(f"ADB: connected to {address}")
            return True
        elif "already" in out.lower():
            self._device_serial = address
            logger.info(f"ADB: already connected to {address}")
            return True
        else:
            logger.warning(f"ADB: connect returned: {out}")
            return False

    def disconnect(self) -> None:
        if self._device_serial:
            self._run(["disconnect", self._device_serial])
            logger.info(f"ADB: disconnected from {self._device_serial}")
            self._device_serial = None

    def is_connected(self) -> bool:
        """Kiểm tra device còn kết nối không."""
        result = self._run(["devices"])
        out = result.stdout.decode(errors="replace")
        serial = self._device_serial or ""
        return serial in out and "device" in out

    def get_devices(self) -> List[str]:
        """Trả về list device serials đang kết nối."""
        result = self._run(["devices"])
        lines = result.stdout.decode(errors="replace").splitlines()
        devices = []
        for line in lines:
            if "\t" in line and "offline" not in line and "List" not in line:
                devices.append(line.split("\t")[0].strip())
        return devices

    # ============================
    # Screenshot
    # ============================

    def screenshot(self) -> np.ndarray:
        """
        Chụp screenshot qua ADB screencap.
        Trả về numpy array BGR (OpenCV format).
        """
        result = self._run(["exec-out", "screencap", "-p"])
        if not result.stdout or len(result.stdout) < 100:
            raise BotError(
                ErrorCode.ADB_003,
                "Screenshot returned empty data",
            )
        try:
            img = Image.open(io.BytesIO(result.stdout))
            img_array = np.array(img)
            # Convert RGBA → BGR hoặc RGB → BGR cho OpenCV
            import cv2
            if img_array.ndim == 3 and img_array.shape[2] == 4:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            elif img_array.ndim == 3 and img_array.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array
            return img_bgr
        except Exception as e:
            raise BotError(ErrorCode.ADB_003, f"Screenshot decode error: {e}")

    # ============================
    # Input
    # ============================

    def tap(self, x: int, y: int) -> None:
        logger.debug(f"ADB: tap ({x}, {y})")
        self._run(["shell", "input", "tap", str(x), str(y)], check=True)
        time.sleep(self.config.timing.tap_delay)

    def double_tap(self, x: int, y: int) -> None:
        logger.debug(f"ADB: double_tap ({x}, {y})")
        self.tap(x, y)
        time.sleep(0.1)
        self.tap(x, y)

    def long_press(self, x: int, y: int, duration_ms: int = 1000) -> None:
        logger.debug(f"ADB: long_press ({x}, {y}) {duration_ms}ms")
        # long press = swipe tại chỗ
        self._run(
            ["shell", "input", "swipe", str(x), str(y), str(x), str(y), str(duration_ms)],
            check=True,
        )
        time.sleep(self.config.timing.tap_delay)

    def swipe(
        self,
        x1: int, y1: int,
        x2: int, y2: int,
        duration_ms: Optional[int] = None,
    ) -> None:
        dur = duration_ms or self.config.timing.swipe_duration
        logger.debug(f"ADB: swipe ({x1},{y1}) → ({x2},{y2}) {dur}ms")
        self._run(
            ["shell", "input", "swipe",
             str(x1), str(y1), str(x2), str(y2), str(dur)],
            check=True,
        )
        time.sleep(self.config.timing.tap_delay)

    def drag(
        self,
        x1: int, y1: int,
        x2: int, y2: int,
        duration_ms: int = 500,
    ) -> None:
        """Drag tương tự swipe nhưng slower."""
        logger.debug(f"ADB: drag ({x1},{y1}) → ({x2},{y2}) {duration_ms}ms")
        self._run(
            ["shell", "input", "draganddrop",
             str(x1), str(y1), str(x2), str(y2), str(duration_ms)],
            check=True,
        )

    def input_text(self, text: str) -> None:
        """Nhập text vào field đang focus."""
        # Escape ký tự đặc biệt cho ADB
        escaped = text.replace(" ", "%s").replace("'", "\\'")
        logger.debug(f"ADB: input_text (masked)")
        self._run(["shell", "input", "text", escaped], check=True)
        time.sleep(0.3)

    def key_event(self, keycode: int) -> None:
        """Gửi keycode Android."""
        self._run(["shell", "input", "keyevent", str(keycode)])

    def press_back(self) -> None:
        self.key_event(4)  # KEYCODE_BACK

    def press_home(self) -> None:
        self.key_event(3)  # KEYCODE_HOME

    # ============================
    # App Management
    # ============================

    def start_app(self) -> None:
        pkg = self.config.game.package_name
        activity = self.config.game.activity_name
        logger.info(f"ADB: starting app {pkg}")
        if activity:
            self._run(["shell", "am", "start", "-n", f"{pkg}/{activity}"])
        else:
            self._run(["shell", "monkey", "-p", pkg, "-c",
                       "android.intent.category.LAUNCHER", "1"])
        time.sleep(self.config.game.startup_wait)

    def stop_app(self) -> None:
        pkg = self.config.game.package_name
        logger.info(f"ADB: stopping app {pkg}")
        self._run(["shell", "am", "force-stop", pkg])
        time.sleep(1)

    def restart_app(self) -> None:
        logger.info("ADB: restarting app")
        self.stop_app()
        time.sleep(2)
        self.start_app()

    def is_app_running(self) -> bool:
        pkg = self.config.game.package_name
        result = self._run(["shell", "pidof", pkg])
        out = result.stdout.decode(errors="replace").strip()
        return bool(out)

    def get_current_activity(self) -> str:
        result = self._run(["shell", "dumpsys", "activity", "activities"])
        out = result.stdout.decode(errors="replace")
        for line in out.splitlines():
            if "mResumedActivity" in line or "mCurrentFocus" in line:
                return line.strip()
        return ""

    # ============================
    # Device Info
    # ============================

    def get_resolution(self) -> Tuple[int, int]:
        """Trả về (width, height)."""
        result = self._run(["shell", "wm", "size"])
        out = result.stdout.decode(errors="replace").strip()
        # Output: "Physical size: 1280x720"
        try:
            part = out.split(":")[-1].strip()
            w, h = part.split("x")
            return (int(w), int(h))
        except Exception:
            # Fallback từ config
            return (self.config.emulator.width, self.config.emulator.height)

    def shell(self, cmd: str) -> str:
        """Chạy lệnh shell tùy ý. Dùng với cẩn thận."""
        result = self._run(["shell"] + cmd.split())
        return result.stdout.decode(errors="replace").strip()
