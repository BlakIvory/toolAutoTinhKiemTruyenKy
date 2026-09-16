"""
AutoQuestTool - Activity Log
Ghi log đơn giản, dễ đọc: tài khoản nào hoàn thành nhiệm vụ gì.
Mỗi ngày tạo 1 file riêng: logs/YYYY-MM-DD.log
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


class ActivityLog:
    """
    Tự động tạo file log theo ngày: logs/2026-09-11.log, logs/2026-09-12.log,...
    Folder logs/ được tạo tự động nếu chưa có.
    """

    LOG_DIR = Path("logs")

    def __init__(self):
        # Tạo folder logs/ nếu chưa có
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _log_file(self) -> Path:
        """Trả về file log của ngày hôm nay, vd: logs/2026-09-11.log"""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.LOG_DIR / f"{today}.log"

    def _write(self, icon: str, message: str) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        line = f"[{now}] {icon} {message}\n"
        # In ra terminal an toàn với mọi bảng mã console Windows
        try:
            print(line, end="", flush=True)
        except Exception:
            try:
                enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
                safe_line = line.encode(enc, errors="replace").decode(enc, errors="replace")
                print(safe_line, end="", flush=True)
            except Exception:
                pass
        # Ghi vào file log ngày hôm nay (luôn dùng utf-8)
        try:
            with open(self._log_file(), "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

    # =====================
    # Các sự kiện chính
    # =====================

    def quest_done(self, username: str, quest_num: int, quest_type: str = "NPC") -> None:
        """Hoàn thành 1 nhiệm vụ."""
        self._write("✅", f"[{username}] Hoàn thành nhiệm vụ #{quest_num} (Loại: {quest_type})")

    def quest_failed(self, username: str, quest_num: int, reason: str = "") -> None:
        """Thất bại / bỏ qua nhiệm vụ."""
        msg = f"[{username}] Thất bại nhiệm vụ #{quest_num}"
        if reason:
            msg += f" — {reason}"
        self._write("❌", msg)

    def session_start(self, username: str) -> None:
        """Bắt đầu phiên làm việc."""
        self._write("▶️ ", f"[{username}] Bắt đầu chạy tool")

    def session_end(self, username: str, total: int) -> None:
        """Kết thúc phiên làm việc."""
        self._write("⏹️ ", f"[{username}] Kết thúc — Tổng nhiệm vụ hoàn thành: {total}")

    def connected(self, serial: str) -> None:
        """Kết nối giả lập thành công."""
        self._write("📱", f"Kết nối giả lập thành công: {serial}")

    def error(self, username: str, message: str) -> None:
        """Ghi lỗi nghiêm trọng."""
        self._write("⚠️ ", f"[{username}] LỖI: {message}")

    def separator(self) -> None:
        """Dòng phân cách khi bắt đầu run mới."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = (
            f"\n{'='*50}\n"
            f"  Tool khởi động lúc {now}\n"
            f"{'='*50}\n"
        )
        print(line, end="", flush=True)
        with open(self._log_file(), "a", encoding="utf-8") as f:
            f.write(line)


# Singleton dùng chung toàn app
activity = ActivityLog()
