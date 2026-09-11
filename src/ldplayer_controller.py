"""
AutoQuestTool - LDPlayer Controller
Quản lý emulator instance (start/stop/restart).
Tách biệt với ADBController (game bên trong).
"""

from __future__ import annotations

import subprocess
import time
import shutil
from pathlib import Path
from typing import Optional

from loguru import logger

from .error_codes import ErrorCode, BotError
from .config_loader import AppConfig


class LDPlayerController:
    """
    Quản lý LDPlayer emulator thông qua CLI.
    Chỉ chịu trách nhiệm quản lý emulator.
    Game bên trong → dùng ADBController.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._ldconsole = self._find_ldconsole()

    def _find_ldconsole(self) -> Optional[str]:
        """Tìm ldconsole.exe hoặc dnplayer.exe."""
        # 1. Từ config
        cfg_path = self.config.emulator.ldconsole_path
        if cfg_path and Path(cfg_path).exists():
            return cfg_path

        # 2. Trong thư mục LDPlayer
        lddir = Path(self.config.emulator.ldplayer_path)
        candidates = [
            lddir / "ldconsole.exe",
            lddir / "LDPlayer" / "ldconsole.exe",
            lddir / "dnplayer.exe",
        ]
        for c in candidates:
            if c.exists():
                logger.debug(f"LDPlayer: ldconsole found at {c}")
                return str(c)

        # 3. PATH
        in_path = shutil.which("ldconsole")
        if in_path:
            return in_path

        logger.warning(
            "LDPlayerController: ldconsole.exe not found. "
            "Emulator management will be limited. Use ADB fallback."
        )
        return None

    def _run_ldconsole(self, args: list, timeout: int = 30) -> str:
        if not self._ldconsole:
            raise BotError(
                ErrorCode.ADB_001,
                "ldconsole.exe not found. Cannot manage emulator.",
            )
        cmd = [self._ldconsole] + args
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            raise BotError(ErrorCode.ADB_006, f"ldconsole timeout: {' '.join(args)}")

    @property
    def instance_id(self) -> int:
        return self.config.emulator.instance_id

    # ============================
    # Emulator Lifecycle
    # ============================

    def start(self) -> None:
        """Khởi động LDPlayer instance."""
        logger.info(f"LDPlayer: starting instance {self.instance_id}")
        self._run_ldconsole(["launch", "--index", str(self.instance_id)])
        time.sleep(5)  # Chờ emulator khởi động

    def stop(self) -> None:
        """Dừng LDPlayer instance."""
        logger.info(f"LDPlayer: stopping instance {self.instance_id}")
        self._run_ldconsole(["quit", "--index", str(self.instance_id)])
        time.sleep(2)

    def restart(self) -> None:
        """Restart LDPlayer instance."""
        logger.info(f"LDPlayer: restarting instance {self.instance_id}")
        self.stop()
        time.sleep(3)
        self.start()

    def is_running(self) -> bool:
        """Kiểm tra emulator có đang chạy không."""
        try:
            out = self._run_ldconsole(["isrunning", "--index", str(self.instance_id)])
            return "running" in out.lower()
        except BotError:
            return False

    def get_status(self) -> str:
        """Lấy trạng thái emulator."""
        try:
            return self._run_ldconsole(["isrunning", "--index", str(self.instance_id)])
        except BotError:
            return "unknown"

    def list_instances(self) -> str:
        """Liệt kê tất cả instance."""
        try:
            return self._run_ldconsole(["list"])
        except BotError:
            return ""

    def set_resolution(self, width: int, height: int, dpi: int) -> None:
        """Thiết lập resolution (cần restart sau)."""
        logger.info(f"LDPlayer: set resolution {width}x{height} dpi={dpi}")
        self._run_ldconsole([
            "modify", "--index", str(self.instance_id),
            "--resolution", f"{width},{height},{dpi}"
        ])

    def available(self) -> bool:
        """Kiểm tra ldconsole có khả dụng không."""
        return self._ldconsole is not None
