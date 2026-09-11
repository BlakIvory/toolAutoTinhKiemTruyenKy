# ARCHITECTURE.md – AutoQuestTool

## Tổng quan

AutoQuestTool là game automation tool cho **Tình Kiếm Truyền Kỳ** chạy trên LDPlayer.

## Kiến trúc hệ thống

```
PySide6 GUI
     |
     v
BotController          ← Orchestrator chính
     |
     v
StateMachine           ← Quản lý workflow
     |
     +----------+----------+
     |          |          |
     v          v          v
AccountManager QuestManager  Watchdog
               |
     +---------+---------+
     |         |         |
     v         v         v
NavigationEngine NPCInteractor ShopManager
     |         |         |
     +---------+---------+
               |
               v
          VisionEngine
          /     |     \
    OpenCV    OCR   Templates
          \     |     /
               v
          InputController
               |
               v
          ADBController
               |
               v
            LDPlayer
```

## Persistence

```
StateMachine
     |
     +-----> runtime/state.json   (checkpoint / recovery)
     |
     +-----> runtime/autoquest.db  (lịch sử lâu dài)
     |
     +-----> logs/YYYY-MM-DD.log   (Loguru)
     |
     +-----> logs/debug/           (debug screenshots)
```

## Module Descriptions

| Module | File | Chức năng |
|--------|------|-----------|
| BotController | `src/bot_controller.py` | Orchestrator, đăng ký handlers |
| StateMachine | `src/state_machine.py` | BotState enum + workflow |
| ADBController | `src/adb_controller.py` | Tất cả ADB commands |
| LDPlayerController | `src/ldplayer_controller.py` | Quản lý emulator |
| ScreenCapture | `src/screen_capture.py` | Screenshot + ROI crop |
| VisionEngine | `src/vision_engine.py` | OpenCV template matching |
| TemplateManager | `src/template_manager.py` | Cache template images |
| InputController | `src/input_controller.py` | Tap/swipe/type |
| NavigationEngine | `src/navigation_engine.py` | Di chuyển bản đồ |
| QuestClassifier | `src/quest_classifier.py` | Phân loại quest |
| Checkpoint | `src/checkpoint.py` | state.json read/write |
| Database | `src/database.py` | SQLite layer |
| WatchdogService | `src/watchdog.py` | Monitor crash/freeze |
| ConfigLoader | `src/config_loader.py` | YAML + Pydantic |

## Technology Priority

Khi giải quyết vấn đề UI, luôn theo thứ tự:

1. ADB + tọa độ cố định
2. ADB + OpenCV template matching
3. OpenCV preprocessing + OCR (Phase 2)
4. Appium/UiAutomator2 (chỉ khi element native)
5. YOLO/Object Detection (chỉ khi prototype yêu cầu)

## Config Files

| File | Nội dung |
|------|---------|
| `config/app.yaml` | Emulator, ADB, game package, timing |
| `config/accounts.yaml` | Danh sách accounts |
| `config/ui_coordinates.yaml` | Tọa độ UI 1280×720 |
| `config/templates.yaml` | Template paths + confidence |
| `config/maps.yaml` | Map IDs + navigation graph |
| `config/npcs.yaml` | NPC positions + maps |
| `config/shops.yaml` | Shop items |
