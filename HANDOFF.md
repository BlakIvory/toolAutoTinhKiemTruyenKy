# HANDOFF.md – AutoQuestTool

## Trạng thái hiện tại

**Phase:** 1 – Project Skeleton  
**Last Good State:** Project skeleton hoàn thành  
**Game:** Tình Kiếm Truyền Kỳ (https://tinhkiem.titanstudio.vn/)

## Môi trường xác nhận

| Thành phần | Trạng thái |
|-----------|-----------|
| Python | 3.11.0 ✅ |
| OpenCV | 5.0.0 ✅ |
| SQLite | 3.38.4 ✅ |
| Git | 2.55.0 ✅ |
| ADB | ❌ Chưa trong PATH – cần cấu hình `adb_path` trong config/app.yaml |
| LDPlayer | Tại `C:\Program Files\ldplayer9box` (LDPlayer 9 Box) |
| ldconsole | ❌ Không tìm thấy (LDPlayer 9 Box dùng VirtualBox backend) |

## Việc cần làm ngay (Model tiếp theo)

### 1. Cài dependencies
```bash
pip install -r requirements.txt
```

### 2. Cấu hình ADB path
Tìm `adb.exe` trong LDPlayer9Box hoặc cài Android Platform Tools.
Cập nhật `config/app.yaml`:
```yaml
adb:
  adb_path: "C:/path/to/adb.exe"
```

### 3. Xác nhận package name game
Package name hiện tại: `vn.com.titanstudio.tinhkiem` (cần xác nhận).
Chạy khi LDPlayer đang mở game:
```
adb shell pm list packages | findstr tinhkiem
```

### 4. Chụp template assets thật
Cần chụp screenshot của game và crop các template sau:
- `assets/templates/screens/login/login_bg.png`
- `assets/templates/screens/ingame/ingame_bg.png`
- `assets/templates/quest/accept/btn_accept.png`
- `assets/templates/npc/btn_talk.png`
- `assets/templates/popup/btn_ok.png`

### 5. Chạy Prototype 1 (ADB)
```bash
python check_env.py
python main.py --check-env
```

### 6. Cập nhật ui_coordinates.yaml
Tọa độ hiện tại là placeholder. Cần đo thực tế từ game.

## Các quyết định đã chốt

- Stack: Python + ADB + OpenCV (không thay đổi)
- Resolution: 1280×720 (config/app.yaml)
- Config: YAML + Pydantic
- State machine: Python Enum + dataclass
- DB: SQLite
- Log: Loguru
- Test: pytest

## Không được thay đổi

Theo Master Prompt: không tự ý đổi stack hiện tại sang framework khác.
Nếu cần thêm dependency → ghi lý do vào đây trước.

## TECH_DECISION_PENDING

(Để trống nếu không có quyết định pending)

---
*Cập nhật: Phase 1 Skeleton hoàn thành*
