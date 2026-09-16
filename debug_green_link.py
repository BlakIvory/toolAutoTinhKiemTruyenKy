"""
Script debug: chụp màn hình, lưu ảnh ROI và mask xanh lá
để kiểm tra xem phát hiện tên NPC màu xanh lá có đúng không.

Chạy khi: game đang mở, bảng nhiệm vụ đang hiện (có tên NPC màu xanh)
    python debug_green_link.py
"""
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from src.config_loader import ConfigLoader
from src.adb_controller import ADBController
from src.screen_capture import ScreenCapture

# Khởi tạo
loader = ConfigLoader()
loader.load_all()
cfg = loader.load_app()
adb = ADBController(cfg)
adb.connect()
screen = ScreenCapture(adb, cfg)

print("Chup man hinh...")
frame = screen.capture()
if frame is None:
    print("LOI: Khong chup duoc man hinh!")
    sys.exit(1)

# Lưu ảnh gốc
cv2.imwrite("debug_full_screen.png", frame)
print("Saved: debug_full_screen.png")

# Cắt vùng bảng nhiệm vụ
panel_x1, panel_y1 = 60,  180
panel_x2, panel_y2 = 600, 520
roi = frame[panel_y1:panel_y2, panel_x1:panel_x2]
cv2.imwrite("debug_panel_roi.png", roi)
print("Saved: debug_panel_roi.png (vung bang nhiem vu)")

# Tạo mask màu xanh lá
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
lower_green = np.array([35, 80, 80])
upper_green = np.array([85, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)
cv2.imwrite("debug_green_mask.png", mask)
print("Saved: debug_green_mask.png (phan mau xanh la phat hien duoc)")

# Tìm contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"\nTim thay {len(contours)} vung mau xanh la")

# Vẽ lên ảnh để xem
vis = roi.copy()
for c in contours:
    area = cv2.contourArea(c)
    if area > 30:
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.drawContours(vis, [c], -1, (0, 0, 255), 2)
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
            print(f"  Contour area={area:.0f}, center roi=({cx},{cy}), screen=({cx+panel_x1},{cy+panel_y1})")

cv2.imwrite("debug_detected.png", vis)
print("\nSaved: debug_detected.png (vung phat hien duoc, khoanh do)")
print("\n=> Mo cac file debug_*.png de xem ket qua!")
