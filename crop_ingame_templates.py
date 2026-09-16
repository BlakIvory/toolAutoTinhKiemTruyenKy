"""
Script crop template và các thành phần UI từ màn hình ingame (1600x900).
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PIL import Image

def crop_and_save(src_img: Image.Image, dest_path: str, box: tuple, desc: str):
    cropped = src_img.crop(box)
    p = Path(dest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(p)
    print(f"✅ {desc} -> {dest_path} ({cropped.size[0]}x{cropped.size[1]})")

def main():
    src = "screenshot_current.png"
    img = Image.open(src)
    W, H = img.size
    print(f"Crop templates from {src} ({W}x{H})")

    # 1. Ingame screen template: Cụm thanh máu/avatar góc trên trái (hoặc minimap góc trên phải)
    # Avatar + thanh máu: box (80, 30, 320, 130)
    crop_and_save(img, "assets/templates/screens/ingame/ingame_bg.png", (70, 25, 320, 135), "Màn hình Ingame (Thanh máu/mana)")

    # 2. Minimap frame góc trên phải
    crop_and_save(img, "assets/templates/screens/ingame/minimap.png", (1360, 5, 1585, 55), "Minimap header (Tuyền Châu)")

    # 3. Nút "Nhiệm vụ" ở góc dưới phải
    # Thử crop icon Nhiệm vụ: x: 1350-1450, y: 755-855
    crop_and_save(img, "assets/templates/quest/btn_nhiem_vu.png", (1355, 755, 1455, 855), "Nút Nhiệm vụ")

    # 4. Banner nhiệm vụ tân thủ ở giữa màn hình
    # "Hướng dẫn nhiệm vụ tân thủ: Bái kiến Dã Tẩu"
    crop_and_save(img, "assets/templates/quest/banner_tan_thu.png", (555, 720, 990, 785), "Banner hướng dẫn nhiệm vụ")

    # 5. Tên NPC "Dã Tẩu" màu xanh lá
    # Text Dã Tẩu: x: 850-925, y: 340-370
    crop_and_save(img, "assets/templates/npc/npc_da_tau_name.png", (850, 340, 925, 370), "Tên NPC Dã Tẩu")

    # 6. Túi đồ
    crop_and_save(img, "assets/templates/menu/btn_tui.png", (1450, 295, 1520, 375), "Nút Túi đồ")

    print("\nHoàn thành crop templates ingame cơ bản!")

if __name__ == "__main__":
    main()
