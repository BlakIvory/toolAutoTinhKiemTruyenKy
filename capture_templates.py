"""
Script chụp screenshot từ LDPlayer và tạo template ảnh.
Chạy: python capture_templates.py
"""
import sys
import subprocess
import io
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

ADB = r"C:\Users\nguye\AppData\Local\Microsoft\WinGet\Packages\Google.PlatformTools_Microsoft.Winget.Source_8wekyb3d8bbwe\platform-tools\adb.exe"
SERIAL = "emulator-5554"

def screenshot(save_path: str) -> bool:
    """Chụp screenshot và lưu file PNG."""
    result = subprocess.run(
        [ADB, "-s", SERIAL, "exec-out", "screencap", "-p"],
        capture_output=True, timeout=15
    )
    if not result.stdout or len(result.stdout) < 1000:
        print(f"❌ Screenshot thất bại: {len(result.stdout)} bytes")
        return False
    
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(result.stdout)
    print(f"✅ Saved: {save_path} ({len(result.stdout)/1024:.1f} KB)")
    return True

def main():
    print("=" * 50)
    print("  AutoQuestTool – Screenshot & Template Capture")
    print("=" * 50)
    
    # 1. Chụp màn hình hiện tại
    print("\n[1] Chụp màn hình hiện tại...")
    ok = screenshot("screenshot_current.png")
    if not ok:
        print("Kiểm tra LDPlayer đang bật và ADB kết nối")
        return

    # Hiển thị ảnh
    try:
        from PIL import Image
        img = Image.open("screenshot_current.png")
        print(f"    Resolution: {img.size[0]}x{img.size[1]}")
        img.save("screenshot_current.png")  # Re-save to fix any format issues
    except Exception as e:
        print(f"    PIL error: {e}")

    print("\n[2] Hướng dẫn tạo template:")
    print("    Ảnh hiện tại đã lưu tại: screenshot_current.png")
    print("    Mở ảnh này, crop và lưu vào đúng thư mục template:")
    print()
    templates = [
        ("assets/templates/screens/login/login_bg.png",    "Màn hình login (toàn bộ hoặc logo game)"),
        ("assets/templates/screens/ingame/ingame_bg.png",  "Màn hình ingame (HUD/minimap)"),
        ("assets/templates/screens/loading/loading.png",   "Màn hình loading"),
        ("assets/templates/quest/accept/btn_accept.png",   "Nút chấp nhận nhiệm vụ"),
        ("assets/templates/quest/complete/btn_complete.png","Nút hoàn thành nhiệm vụ"),
        ("assets/templates/popup/btn_ok.png",              "Nút OK popup"),
    ]
    for path, desc in templates:
        status = "✅" if Path(path).exists() else "❌"
        print(f"    {status} {path}")
        print(f"       → {desc}")
    
    print()
    print("[3] Hoặc chạy script này nhiều lần ở các màn hình khác nhau:")
    print("    python capture_templates.py  (sẽ overwrite screenshot_current.png)")

if __name__ == "__main__":
    main()
