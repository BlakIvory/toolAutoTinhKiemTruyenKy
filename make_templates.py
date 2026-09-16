"""
Tạo template ảnh từ screenshot hiện tại.
Chạy sau khi đã có screenshot_current.png
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

from PIL import Image

def crop_and_save(src: str, dest: str, box: tuple, desc: str):
    """Crop vùng box=(x1,y1,x2,y2) từ ảnh src và lưu vào dest."""
    img = Image.open(src)
    cropped = img.crop(box)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    cropped.save(dest)
    print(f"✅ {desc}")
    print(f"   → {dest} ({cropped.size[0]}x{cropped.size[1]})")

def main():
    src = "screenshot_current.png"
    if not Path(src).exists():
        print(f"❌ {src} không tồn tại. Chạy: python capture_templates.py trước")
        return

    img = Image.open(src)
    W, H = img.size
    print(f"Screenshot size: {W}x{H}")
    print()

    # =========================================
    # Login screen: lấy phần bottom tab bar
    # Vùng đặc trưng: thanh tab dưới cùng
    # "Fanpage | Trang chủ | Quên MK | Đăng ký | Thoát"
    # =========================================
    crop_and_save(src,
        "assets/templates/screens/login/login_bg.png",
        (0, int(H*0.8), W, H),   # phần dưới 20% (tab bar)
        "Login screen template (bottom tab bar)")

    # Logo/header vùng trên
    crop_and_save(src,
        "assets/templates/screens/login/login_tab_bar.png",
        (300, int(H*0.82), 1300, H-10),
        "Login tab bar template")

    # Nút XÁC MINH (captcha confirm) - nếu xuất hiện cần dismiss
    crop_and_save(src,
        "assets/templates/popup/btn_xac_minh.png",
        (820, 640, 980, 700),
        "Nút XÁC MINH (captcha)")

    # Toàn bộ màn hình login làm reference
    img.save("assets/templates/screens/login/login_full.png")
    print(f"✅ Full login screen reference saved")

    print()
    print("=" * 50)
    print("Templates đã tạo. Bây giờ cần:")
    print("1. Mở game, login xong để vào màn hình ingame")
    print("2. Chạy: python capture_templates.py")
    print("3. Chạy: python make_templates.py ingame")

if __name__ == "__main__":
    main()
