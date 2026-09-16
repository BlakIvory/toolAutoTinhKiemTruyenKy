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
        # True when the launcher assigned this controller to one LDPlayer tab.
        # A locked controller must never fall back to another online emulator.
        self._device_locked: bool = False
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
        if not self._device_serial:
            devs = self.get_devices()
            if devs:
                emu = [d for d in devs if d.startswith("emulator-")]
                self._device_serial = emu[0] if emu else devs[0]
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

    def set_target_device(self, serial: str) -> None:
        """Pin all ADB commands to one emulator selected by the launcher."""
        self._device_serial = serial
        self._device_locked = True

    def connect(self) -> bool:
        """
        Kết nối ADB đến LDPlayer emulator.
        Hỗ trợ 2 mode:
          - TCP: 127.0.0.1:PORT  (LDPlayer thường)
          - Serial: emulator-5554 (LDPlayer 9 Box / Android emulator mode)
        """
    def _ensure_device_connected(self) -> bool:
        """Đảm bảo thiết bị sẵn sàng; tự động chuyển giữa emulator-XXXX và 127.0.0.1:PORT nếu cần."""
        if not self._device_serial:
            return False

        timeout = max(self.config.adb.connect_timeout, 25)
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                res = self._run(["get-state"], timeout=2)
                out_str = res.stdout.decode(errors="replace").lower()
                if "device" in out_str:
                    return True
            except Exception:
                pass

            # Kiểm tra danh sách thiết bị đang online thực tế
            connected_devices = self.get_devices()
            if connected_devices:
                if self._device_serial in connected_devices:
                    return True
                # Nếu chỉ có duy nhất 1 tab máy ảo đang chạy, tự động chọn tab đó
                if len(connected_devices) == 1 and not self._device_locked:
                    logger.info(f"ADB: Tự động dùng tab máy ảo duy nhất đang mở: {connected_devices[0]}")
                    self._device_serial = connected_devices[0]
                    return True

            # Nếu đang dùng emulator-XXXX -> thử kết nối qua TCP 127.0.0.1:PORT
            if self._device_serial.startswith("emulator-"):
                try:
                    num = int(self._device_serial.split("-")[1])
                    tcp_target = f"127.0.0.1:{num + 1}"
                    subprocess.run([self._adb_path, "connect", tcp_target], capture_output=True, timeout=3)
                    check = subprocess.run([self._adb_path, "-s", tcp_target, "get-state"], capture_output=True, text=True, timeout=3)
                    if "device" in check.stdout:
                        logger.info(f"ADB: Chuyển device serial từ {self._device_serial} -> {tcp_target}")
                        self._device_serial = tcp_target
                        return True
                except Exception as e:
                    logger.debug(f"ADB fallback error: {e}")

            # Nếu đang dùng TCP 127.0.0.1:PORT -> thử emulator-XXXX hoặc connect lại
            elif ":" in self._device_serial:
                try:
                    subprocess.run([self._adb_path, "connect", self._device_serial], capture_output=True, timeout=3)
                    check = subprocess.run([self._adb_path, "-s", self._device_serial, "get-state"], capture_output=True, text=True, timeout=3)
                    if "device" in check.stdout:
                        return True
                except Exception:
                    pass

            time.sleep(1.0)

        logger.warning(f"ADB: Thiết bị {self._device_serial} không sẵn sàng sau {timeout}s")
        return False

    def connect(self) -> bool:
        """
        Kết nối ADB đến LDPlayer emulator.
        Hỗ trợ cả Serial (emulator-XXXX) và TCP (127.0.0.1:PORT).
        Tự động quét các cổng 5555, 5557, 5559 và tự động nhận diện tab đang mở.
        """
        # 1. Thử quét/kết nối các cổng TCP phổ biến của LDPlayer
        for port in [5555, 5557, 5559, 5561]:
            try:
                subprocess.run([self._adb_path, "connect", f"127.0.0.1:{port}"], capture_output=True, timeout=3)
            except Exception:
                pass

        devices = self.get_devices()
        logger.info(f"ADB: active connected devices: {devices}")

        # Nếu có target cụ thể được truyền vào
        if self._device_serial:
            if self._device_serial in devices:
                logger.info(f"ADB: using targeted device serial: {self._device_serial}")
                return True

            # Map giữa serial emulator-555X và TCP 127.0.0.1:555Y
            alt_serial = None
            if self._device_serial.startswith("emulator-"):
                try:
                    num = int(self._device_serial.split("-")[1])
                    alt_serial = f"127.0.0.1:{num + 1}"
                except Exception:
                    pass
            elif "127.0.0.1:" in self._device_serial:
                try:
                    port_num = int(self._device_serial.split(":")[1])
                    alt_serial = f"emulator-{port_num - 1}"
                except Exception:
                    pass

            if alt_serial and alt_serial in devices:
                logger.info(f"ADB: targeted {self._device_serial} mapped to active {alt_serial}")
                self._device_serial = alt_serial
                return True

            # Nếu chỉ có duy nhất 1 tab máy ảo đang mở thực tế, tự động chọn tab đó
            if len(devices) == 1 and not self._device_locked:
                logger.info(f"ADB: Target {self._device_serial} không có, nhưng chỉ có duy nhất 1 tab online ({devices[0]}) -> Dùng {devices[0]}")
                self._device_serial = devices[0]
                return True
            # Nếu không thì thử đảm bảo kết nối
            return self._ensure_device_connected()

        # Nếu không có target chỉ định:
        if devices:
            emu = [d for d in devices if d.startswith("emulator-")]
            self._device_serial = emu[0] if emu else devices[0]
            logger.info(f"ADB: auto-connected to active device: {self._device_serial}")
            return True

        # Fallback TCP
        host = self.config.emulator.adb_host
        port = self.config.emulator.adb_port
        address = f"{host}:{port}"
        logger.info(f"ADB: no device found, trying TCP connect to {address}...")
        try:
            result = subprocess.run([self._adb_path, "connect", address], capture_output=True, text=True, timeout=self.config.adb.connect_timeout)
            if "connected" in result.stdout.lower() or "already" in result.stdout.lower():
                self._device_serial = address
                logger.info(f"ADB: connected to {address}")
                return True
        except Exception:
            pass

        return False

    def disconnect(self) -> None:
        if self._device_serial:
            self._run(["disconnect", self._device_serial])
            logger.info(f"ADB: disconnected from {self._device_serial}")
            self._device_serial = None

    def is_connected(self) -> bool:
        """Kiểm tra device còn kết nối không."""
        try:
            res = subprocess.run([self._adb_path, "devices"], capture_output=True, text=True, timeout=3)
            serial = self._device_serial or ""
            return serial in res.stdout and "device" in res.stdout
        except Exception:
            return False

    def get_devices(self) -> List[str]:
        """Trả về list device serials đang kết nối."""
        try:
            res = subprocess.run([self._adb_path, "devices"], capture_output=True, text=True, timeout=5)
            lines = res.stdout.splitlines()
            devices = []
            for line in lines:
                if "\t" in line and "offline" not in line and "List" not in line:
                    devices.append(line.split("\t")[0].strip())
            return devices
        except Exception:
            return []


    # ============================
    # Screenshot
    # ============================

    def screenshot(self) -> np.ndarray:
        """
        Chụp screenshot qua ADB screencap.
        Trả về numpy array BGR (OpenCV format).
        Hỗ trợ fallback nếu exec-out không khả dụng.
        """
        raw_data = None
        try:
            result = self._run(["exec-out", "screencap", "-p"])
            if result.stdout and len(result.stdout) >= 100:
                raw_data = result.stdout
        except Exception:
            pass

        if not raw_data or len(raw_data) < 100:
            # Fallback 1: screencap qua file tạm
            try:
                self._run(["shell", "screencap", "-p", "/sdcard/autotool_scr.png"])
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = tmp.name
                self._run(["pull", "/sdcard/autotool_scr.png", tmp_path])
                raw_data = Path(tmp_path).read_bytes()
                try:
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception:
                    pass
            except Exception as exc:
                logger.warning(f"ADB screenshot fallback error: {exc}")

        if not raw_data or len(raw_data) < 100:
            raise BotError(
                ErrorCode.ADB_003,
                "Screenshot returned empty data",
            )
        try:
            img = Image.open(io.BytesIO(raw_data))
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
