# CHANGELOG.md – AutoQuestTool

## [Phase 1] – Project Skeleton

### Added
- `check_env.py` – environment check script
- `requirements.txt` – Phase 1 dependencies only
- `src/error_codes.py` – ErrorCode enum + BotError exception
- `src/config_loader.py` – ConfigLoader + Pydantic models
- `src/adb_controller.py` – ADBController (all ADB via single class)
- `src/ldplayer_controller.py` – LDPlayerController (emulator lifecycle)
- `src/screen_capture.py` – ScreenCapture (ADB screencap + ROI)
- `src/template_manager.py` – TemplateManager (cache templates on startup)
- `src/vision_engine.py` – VisionEngine (OpenCV template matching)
- `src/input_controller.py` – InputController (tap/swipe/type via ADB)
- `src/state_machine.py` – StateMachine + BotState enum + StateContext
- `src/quest_classifier.py` – QuestClassifier (icon-based)
- `src/navigation_engine.py` – NavigationEngine (deterministic)
- `src/checkpoint.py` – Checkpoint (runtime/state.json)
- `src/database.py` – SQLite layer (accounts, quests, errors, runs)
- `src/watchdog.py` – WatchdogService (background monitor)
- `src/bot_controller.py` – BotController (orchestrator)
- `src/mock/mock_adb.py` – MockADBController (test without emulator)
- `config/app.yaml` – Main config
- `config/accounts.yaml` – Accounts template
- `config/ui_coordinates.yaml` – UI coordinates (1280×720 placeholders)
- `config/templates.yaml` – Template config with per-template confidence
- `config/maps.yaml` – Maps + navigation graph
- `config/npcs.yaml` – NPC positions
- `config/shops.yaml` – Shop items
- `tests/test_config.py` – Config loader tests
- `tests/test_state_machine.py` – State machine tests
- `tests/test_checkpoint.py` – Checkpoint tests
- `ARCHITECTURE.md` – Architecture documentation
- `HANDOFF.md` – Handoff notes

### Status
- Skeleton hoàn chỉnh
- Business logic handlers là placeholder → cần game thật để implement
- Template assets chưa có → cần chụp từ game

### Not yet implemented
- GUI PySide6 (sẽ làm sau khi ADB prototype PASS)
- Login flow thật (chờ template assets)
- Quest flow thật (chờ game observation)
- Gold transfer flow
- OCR (Phase 2)
