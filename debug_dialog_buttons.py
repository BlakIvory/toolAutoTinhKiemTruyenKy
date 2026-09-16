"""
Debug: phát hiện nút dialog màu vàng/nâu trong game.
Chạy khi dialog đang mở (có nút "Ta chỉ đi ngang thôi" hoặc 3 nút lựa chọn).
    python debug_dialog_buttons.py
"""
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
from src.config_loader import ConfigLoader
from src.adb_controller import ADBController
from src.screen_capture import ScreenCapture

loader = ConfigLoader()
loader.load_all()
cfg = loader.load_app()
adb = ADBController(cfg)
adb.connect()
screen = ScreenCapture(adb, cfg)

print("Chup man hinh (mo dialog truoc)...")
frame = screen.capture()
if frame is None:
    print("LOI: Khong chup duoc!")
    sys.exit(1)

cv2.imwrite("dbg_full.png", frame)
print("Saved: dbg_full.png")

h, w = frame.shape[:2]
print(f"Resolution: {w}x{h}")

# Vùng dialog
roi_x1, roi_y1 = int(w * 0.05), int(h * 0.40)
roi_x2, roi_y2 = int(w * 0.95), int(h * 0.85)
roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
cv2.imwrite("dbg_roi.png", roi)

# Tìm nút màu vàng/nâu
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
lower = np.array([15, 60, 80])
upper = np.array([35, 220, 200])
mask = cv2.inRange(hsv, lower, upper)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 8))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("dbg_mask.png", mask)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
vis = roi.copy()
buttons = []
for c in contours:
    area = cv2.contourArea(c)
    x, y, bw, bh = cv2.boundingRect(c)
    if area > 800 and bw > 80 and bh > 20:
        cx = x + bw // 2 + roi_x1
        cy = y + bh // 2 + roi_y1
        buttons.append((cx, cy))
        cv2.rectangle(vis, (x, y), (x+bw, y+bh), (0, 255, 0), 3)
        cv2.circle(vis, (x + bw//2, y + bh//2), 8, (0, 0, 255), -1)
        print(f"  NUT: area={area:.0f}, size={bw}x{bh}, screen=({cx},{cy})")

cv2.imwrite("dbg_detected.png", vis)
print(f"\nTong so nut tim thay: {len(buttons)}")
print("=> Mo dbg_detected.png de xem nut duoc khoanh xanh!")
