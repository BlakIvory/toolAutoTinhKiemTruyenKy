"""
AutoQuestTool - LDPlayer Controller
Quản lý emulator instance (start/stop/restart/list).
Tách biệt với ADBController (game bên trong).
"""

from __future__ import annotations

import subprocess
import time
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from loguru import logger

from .error_codes import ErrorCode, BotError
from .config_loader import AppConfig

GAME_PACKAGE = "com.titanstudio.tkcm"


@dataclass
class LDInstance:
    """Thông tin một instance LDPlayer."""
    index: int
    name: str
    is_running: bool = False
    adb_port: int = -1
    width: int = 1600
    height: int = 900
    dpi: int = 240

    @property
    def adb_serial(self) -> str:
        """Serial ADB: emulator-5554, emulator-5556, ..."""
        if self.adb_port > 0:
            return f"emulator-{self.adb_port}"
        # LDPlayer mapping: index 0 → 5554, index 1 → 5556, ...
        return f"emulator-{5554 + self.index * 2}"


class LDPlayerController:
    """
    Quản lý LDPlayer emulator thông qua CLI (ldconsole.exe).
    Hỗ trợ:
    - Liệt kê các instance
    - Khởi động / dừng instance
    - Mở game trong instance
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._ldconsole = self._find_ldconsole()

    # ============================
    # Setup
    # ============================

    def _find_ldconsole(self) -> Optional[str]:
        """Tìm ldconsole.exe theo thứ tự ưu tiên."""
        # 1. Từ config (đường dẫn trực tiếp)
        cfg_path = self.config.emulator.ldconsole_path
        if cfg_path and Path(cfg_path).exists():
            logger.debug(f"LDPlayer: ldconsole at config path: {cfg_path}")
            return str(cfg_path)

        # 2. Trong thư mục LDPlayer từ config
        lddir = Path(self.config.emulator.ldplayer_path)
        candidates = [
            lddir / "ldconsole.exe",
            lddir / "dnconsole.exe",
            lddir / "dnplayer.exe",
        ]
        for c in candidates:
            if c.exists():
                logger.debug(f"LDPlayer: found at {c}")
                return str(c)

        # 3. Các đường dẫn phổ biến
        common = [
            Path("D:/LDPlayer/LDPlayer14/ldconsole.exe"),
            Path("C:/LDPlayer/LDPlayer14/ldconsole.exe"),
            Path("D:/LDPlayer/LDPlayer9/ldconsole.exe"),
            Path("C:/LDPlayer/LDPlayer9/ldconsole.exe"),
            Path("C:/Program Files/LDPlayer/LDPlayer9/ldconsole.exe"),
        ]
        for c in common:
            if c.exists():
                logger.debug(f"LDPlayer: ldconsole found at {c}")
                return str(c)

        # 4. PATH
        in_path = shutil.which("ldconsole")
        if in_path:
            return in_path

        logger.warning(
            "LDPlayerController: ldconsole.exe not found. "
            "Emulator management limited. ADB fallback active."
        )
        return None

    def _run(self, args: list, timeout: int = 30) -> str:
        """Chạy ldconsole với args, trả về stdout."""
        if not self._ldconsole:
            raise BotError(
                ErrorCode.ADB_001,
                "ldconsole.exe not found. Cannot manage emulator.",
            )
        cmd = [self._ldconsole] + [str(a) for a in args]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, encoding="utf-8", errors="ignore"
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            raise BotError(ErrorCode.ADB_006, f"ldconsole timeout: {' '.join(args)}")
        except Exception as e:
            logger.warning(f"ldconsole error: {e}")
            return ""

    @property
    def available(self) -> bool:
        return self._ldconsole is not None

    # ============================
    # Instance Management
    # ============================

    def list_instances(self) -> list[LDInstance]:
        """
        Liệt kê tất cả instance từ 'ldconsole list2'.
        Output format: index,name,top-window,bind-window,is-running,...,width,height,dpi
        """
        raw = self._run(["list2"])
        instances = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 10:
                continue
            try:
                idx = int(parts[0])
                name = parts[1].strip()
                # Bỏ qua instance template/internal của LDPlayer (Index 99999 hoặc tên rỗng)
                if idx >= 99999 or idx < 0 or not name:
                    continue
                is_running = parts[4] == "1"
                if not is_running:
                    # Double check by ADB/process if ldconsole list2 didn't report running
                    is_running = self.is_instance_running(idx)
                width = int(parts[7]) if parts[7].isdigit() else 1600
                height = int(parts[8]) if parts[8].isdigit() else 900
                dpi = int(parts[9]) if parts[9].isdigit() else 240
                instances.append(LDInstance(
                    index=idx,
                    name=name,
                    is_running=is_running,
                    width=width,
                    height=height,
                    dpi=dpi,
                ))
            except (ValueError, IndexError) as e:
                logger.warning(f"Cannot parse instance line: {line!r} – {e}")
        logger.info(f"LDPlayer: {len(instances)} instance(s) found")
        return instances

    def is_instance_running(self, index: int) -> bool:
        """Kiểm tra một instance (tab máy ảo) có đang chạy không (real-time)."""
        # 1. Kiểm tra qua ldconsole isrunning (chính xác nhất từ LDPlayer)
        try:
            out = self._run(["isrunning", "--index", str(index)], timeout=5)
            if "running" in out.lower():
                return True
            if "stop" in out.lower():
                return False
        except Exception:
            pass

        # 2. Kiểm tra qua ldconsole runninglist
        try:
            out = self._run(["runninglist"], timeout=5)
            running_indices = [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]
            return index in running_indices
        except Exception:
            pass

        return False

    def is_app_running(self, index: int, package: str = GAME_PACKAGE) -> bool:
        """Kiểm tra game/app có đang mở trong tab máy ảo này không."""
        if not self.is_instance_running(index):
            return False

        adb_bin = getattr(getattr(self.config, "adb", None), "adb_path", None) or "adb"
        serial = f"emulator-{5554 + index * 2}"

        # 1. Kiểm tra qua adb pidof với serial chính xác
        try:
            flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            res = subprocess.run(
                [adb_bin, "-s", serial, "shell", "pidof", package],
                capture_output=True, text=True, timeout=5,
                creationflags=flags
            )
            pids = res.stdout.strip().split()
            if any(p.isdigit() for p in pids):
                return True
        except Exception:
            pass

        # 2. Fallback qua ldconsole adb
        try:
            out = self._run(["adb", "--index", str(index), "--command", f"shell pidof {package}"], timeout=5)
            pids = out.strip().split()
            if any(p.isdigit() for p in pids):
                return True
        except Exception:
            pass

        return False

    def launch_instance(self, index: int, wait: int = 8) -> None:
        """Khởi động một instance. Chờ `wait` giây."""
        logger.info(f"LDPlayer: launching instance {index}")
        self._run(["launch", "--index", str(index)])
        time.sleep(wait)

    def stop_instance(self, index: int) -> None:
        """Dừng một instance (tắt tab máy ảo và đóng game)."""
        logger.info(f"LDPlayer: stopping instance {index}")
        # 1. Tắt game app trong tab trước
        try:
            self._run(["killapp", "--index", str(index), "--packagename", GAME_PACKAGE])
        except Exception:
            pass

        # 2. Gửi lệnh quit tới ldconsole
        try:
            self._run(["quit", "--index", str(index)])
        except Exception:
            pass

        # 3. Đảm bảo tiến trình dnplayer.exe cho index này dừng hẳn
        try:
            import psutil
            for p in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if p.info['name'] and 'dnplayer' in p.info['name'].lower():
                        cmd = ' '.join(p.info['cmdline'] or [])
                        if f'index={index}' in cmd:
                            p.terminate()
                            try:
                                p.wait(timeout=2)
                            except Exception:
                                p.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            logger.warning(f"Error stopping dnplayer process: {e}")

        time.sleep(1)

    def launch_app(self, index: int, package: str = GAME_PACKAGE, wait: int = 30) -> None:
        """
        Mở ứng dụng trong instance.
        Chờ `wait` giây để game load.
        """
        logger.info(f"LDPlayer: launching {package} on instance {index}")
        self._run(["runapp", "--index", str(index), "--packagename", package])
        logger.info(f"LDPlayer: waiting {wait}s for game to load...")
        time.sleep(wait)

    def ensure_instance_ready(self, index: int, package: str = GAME_PACKAGE) -> None:
        """
        Đảm bảo instance đang chạy và game đã được mở.
        1. Mở instance nếu chưa chạy
        2. Launch game
        """
        if not self.is_instance_running(index):
            logger.info(f"LDPlayer: instance {index} not running, launching...")
            self.launch_instance(index, wait=10)
        else:
            logger.info(f"LDPlayer: instance {index} already running")
        self.launch_app(index, package, wait=30)

    # ============================
    # Legacy (tương thích cũ)
    # ============================

    @property
    def instance_id(self) -> int:
        return self.config.emulator.instance_id

    def start(self) -> None:
        self.launch_instance(self.instance_id)

    def stop(self) -> None:
        self.stop_instance(self.instance_id)

    def restart(self) -> None:
        self.stop()
        time.sleep(3)
        self.start()

    def get_status(self) -> str:
        try:
            return self._run(["isrunning", "--index", str(self.instance_id)])
        except BotError:
            return "unknown"

    def set_resolution(self, width: int, height: int, dpi: int) -> None:
        logger.info(f"LDPlayer: set resolution {width}x{height} dpi={dpi}")
        self._run([
            "modify", "--index", str(self.instance_id),
            "--resolution", f"{width},{height},{dpi}"
        ])

    def input_text(self, index: int, text: str) -> bool:
        """
        Nhập text Unicode có dấu tiếng Việt (ví dụ: 'Chủ thương hội', 'Nạn dân Đinh')
        vào ô input đang focus của instance thông qua ldconsole action call.input.
        """
        if not self._ldconsole:
            return False
        try:
            logger.info(f"LDPlayer: typing Unicode text '{text}' on instance {index}")
            self._run(["action", "--index", str(index), "--key", "call.input", "--value", text])
            return True
        except Exception as e:
            logger.warning(f"LDPlayer input_text error: {e}")
            return False
