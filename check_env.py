"""
AutoQuestTool - Environment Check Script
Game: Tinh Kiem Truyen Ky
Muc dich: Kiem tra moi truong truoc khi chay tool
"""

import sys
import subprocess
import shutil
import sqlite3
from pathlib import Path

# Fix UTF-8 stdout trên Windows
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ============================
# Màu terminal
# ============================
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}[OK]{RESET}  {msg}")
def fail(msg):  print(f"  {RED}[FAIL]{RESET} {msg}")
def warn(msg):  print(f"  {YELLOW}[WARN]{RESET} {msg}")
def info(msg):  print(f"  {CYAN}[INFO]{RESET} {msg}")


def check_python():
    print(f"\n{BOLD}PYTHON:{RESET}")
    version = sys.version_info
    ver_str = f"{version.major}.{version.minor}.{version.micro}"
    if version >= (3, 11):
        ok(f"Python {ver_str}")
    else:
        fail(f"Python {ver_str} – yêu cầu 3.11+")


def check_adb():
    print(f"\n{BOLD}ADB:{RESET}")
    # Tìm ADB trong các vị trí thường gặp
    common_paths = [
        r"C:\Program Files\ldplayer9box",
        r"C:\LDPlayer\LDPlayer9",
        r"C:\LDPlayer9",
        r"C:\Program Files\LDPlayer",
        r"C:\Android\platform-tools",
        r"C:\Users\nguye\AppData\Local\Android\Sdk\platform-tools",
    ]
    adb_path = shutil.which("adb")
    if adb_path:
        ok(f"ADB found in PATH: {adb_path}")
    else:
        found = None
        for p in common_paths:
            candidate = Path(p) / "adb.exe"
            if candidate.exists():
                found = str(candidate)
                break
        if found:
            warn(f"ADB not in PATH but found at: {found}")
            warn("→ Hãy thêm vào PATH hoặc cấu hình adb_path trong config/app.yaml")
        else:
            fail("ADB not found – cài Android Platform Tools hoặc kiểm tra config adb_path")


def check_ldplayer():
    print(f"\n{BOLD}LDPLAYER:{RESET}")
    from src.config_loader import ConfigLoader
    app_cfg = ConfigLoader().load_app()
    paths = [
        app_cfg.emulator.ldplayer_path,
        r"D:\LDPlayer\LDPlayer9",
        r"D:\LDPlayer",
        r"C:\LDPlayer\LDPlayer9",
        r"C:\Program Files\ldplayer9box",
        r"C:\LDPlayer9",
    ]
    for p in paths:
        if Path(p).exists():
            ok(f"LDPlayer directory: {p}")
            # Tìm ldconsole
            ldc = Path(p) / "ldconsole.exe"
            dnp = Path(p) / "dnplayer.exe"
            if ldc.exists():
                ok(f"ldconsole.exe found")
            elif dnp.exists():
                ok(f"dnplayer.exe found")
            else:
                warn("ldconsole.exe / dnplayer.exe not found – LDPlayerController sẽ dùng ADB")
            return
    fail("LDPlayer không tìm thấy. Kiểm tra ldplayer_path trong config/app.yaml")


def check_adb_device():
    print(f"\n{BOLD}ADB DEVICE:{RESET}")
    adb_cmd = shutil.which("adb")
    if not adb_cmd:
        warn("Skipping – ADB not in PATH")
        return
    try:
        result = subprocess.run(
            [adb_cmd, "devices"],
            capture_output=True, text=True, timeout=5
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        devices = [l for l in lines if "\t" in l and "offline" not in l and "List" not in l]
        if devices:
            for d in devices:
                ok(f"Device: {d}")
        else:
            warn("Không có device nào. Hãy bật LDPlayer và kết nối ADB trước.")
    except Exception as e:
        fail(f"Lỗi khi chạy adb devices: {e}")


def check_resolution():
    print(f"\n{BOLD}RESOLUTION:{RESET}")
    info("Cần ADB device để đọc resolution. Xem config/app.yaml để cấu hình thủ công.")
    info("Mặc định: 1280×720 (có thể thay đổi trong config/app.yaml)")


def check_opencv():
    print(f"\n{BOLD}OPENCV:{RESET}")
    try:
        import cv2
        ok(f"OpenCV {cv2.__version__}")
    except ImportError:
        fail("OpenCV chưa được cài – chạy: pip install opencv-python")


def check_numpy():
    print(f"\n{BOLD}NUMPY:{RESET}")
    try:
        import numpy as np
        ok(f"NumPy {np.__version__}")
    except ImportError:
        fail("NumPy chưa được cài – chạy: pip install numpy")


def check_pillow():
    print(f"\n{BOLD}PILLOW:{RESET}")
    try:
        from PIL import Image
        import PIL
        ok(f"Pillow {PIL.__version__}")
    except ImportError:
        fail("Pillow chưa được cài – chạy: pip install Pillow")


def check_pydantic():
    print(f"\n{BOLD}PYDANTIC:{RESET}")
    try:
        import pydantic
        ok(f"Pydantic {pydantic.VERSION}")
    except ImportError:
        fail("Pydantic chưa được cài – chạy: pip install pydantic")


def check_loguru():
    print(f"\n{BOLD}LOGURU:{RESET}")
    try:
        import loguru
        ok("Loguru available")
    except ImportError:
        fail("Loguru chưa được cài – chạy: pip install loguru")


def check_yaml():
    print(f"\n{BOLD}PYYAML:{RESET}")
    try:
        import yaml
        ok(f"PyYAML available")
    except ImportError:
        fail("PyYAML chưa được cài – chạy: pip install PyYAML")


def check_sqlite():
    print(f"\n{BOLD}SQLITE:{RESET}")
    ok(f"SQLite {sqlite3.sqlite_version} (built-in)")


def check_git():
    print(f"\n{BOLD}GIT:{RESET}")
    git = shutil.which("git")
    if git:
        try:
            result = subprocess.run(
                ["git", "--version"], capture_output=True, text=True, timeout=5
            )
            ok(result.stdout.strip())
        except Exception:
            ok("Git available")
    else:
        fail("Git không tìm thấy – cài Git for Windows")


def check_pyside6():
    print(f"\n{BOLD}PYSIDE6 (GUI):{RESET}")
    try:
        import PySide6
        ok(f"PySide6 {PySide6.__version__}")
    except ImportError:
        warn("PySide6 chưa được cài – OK cho Phase 1 CLI. Cài sau: pip install PySide6")


def main():
    print("=" * 55)
    print(f"  {BOLD}AutoQuestTool – Environment Check{RESET}")
    print(f"  Game: Tình Kiếm Truyền Kỳ")
    print("=" * 55)

    check_python()
    check_adb()
    check_ldplayer()
    check_adb_device()
    check_resolution()
    check_opencv()
    check_numpy()
    check_pillow()
    check_pydantic()
    check_loguru()
    check_yaml()
    check_sqlite()
    check_git()
    check_pyside6()

    print(f"\n{'=' * 55}")
    print(f"  {BOLD}Kiểm tra hoàn tất.{RESET}")
    print(f"  Nếu có [FAIL] → cài package hoặc cấu hình PATH.")
    print(f"  Nếu có [WARN] → xem xét trước khi chạy.")
    print(f"  Xem config/app.yaml để cấu hình adb_path và ldplayer_path thủ công.")
    print("=" * 55)


if __name__ == "__main__":
    main()
