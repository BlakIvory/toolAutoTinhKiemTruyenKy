"""
AutoQuestTool - Bot Controller
Orchestrator chính. Kết nối tất cả components.
GUI → BotController → StateMachine → Engine.
"""

from __future__ import annotations

import time
import random
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable

from loguru import logger

from .config_loader import ConfigLoader, AppConfig
from .adb_controller import ADBController
from .ldplayer_controller import LDPlayerController
from .screen_capture import ScreenCapture
from .template_manager import TemplateManager
from .vision_engine import VisionEngine
from .input_controller import InputController
from .state_machine import StateMachine, BotState, StateContext, QuestType
from .quest_classifier import QuestClassifier, extract_dialogue_target, extract_shop_item
from .navigation_engine import NavigationEngine
from .checkpoint import Checkpoint
from .database import Database
from .watchdog import WatchdogService
from .error_codes import ErrorCode, BotError
from .activity_log import activity
from .text_recognition import (
    GameTextRecognizer, fuzzy_score, normalize_game_text, is_explicit_quest_exhausted,
    is_western_merchant_name,
    is_safe_map_result,
)


class BotController:
    """
    Orchestrator. Không chứa business logic game.
    Chịu trách nhiệm:
    - Khởi tạo và kết nối tất cả components
    - Đăng ký state handlers vào StateMachine
    - Chạy vòng lặp multi-account (tuần tự)
    - Callback cho GUI
    """

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        on_status_update: Optional[Callable] = None,
        on_log: Optional[Callable] = None,
    ):
        self._on_status_update = on_status_update
        self._on_log = on_log

        # --- Config ---
        self._loader = ConfigLoader(config_dir)
        self._loader.load_all()
        self._app_cfg: AppConfig = self._loader.load_app()

        # --- Setup Logging ---
        self._setup_logging()

        # --- Core components ---
        self._adb = self._create_adb()
        self._ldplayer = LDPlayerController(self._app_cfg)
        self._screen = ScreenCapture(self._adb, self._app_cfg)
        self._template_mgr = TemplateManager(self._loader, Path("."))
        self._template_mgr.load_all()
        self._vision = VisionEngine(
            self._app_cfg, self._loader, self._template_mgr, self._screen
        )
        self._input = InputController(self._adb, self._app_cfg)
        self._classifier = QuestClassifier(self._vision, self._app_cfg)
        self._text = GameTextRecognizer()
        self._nav = NavigationEngine(
            self._app_cfg, self._loader, self._input, self._vision
        )
        self._checkpoint = Checkpoint()
        self._db = Database()
        self._state_machine = StateMachine(self._app_cfg)
        self._watchdog: Optional[WatchdogService] = None

        # --- Register state handlers ---
        self._register_handlers()

        # State machine callback → GUI
        self._state_machine.set_state_change_callback(self._on_state_changed)
        self._target_server: str = "S1"

        logger.info("BotController initialized")

    def _get_username(self, account_id: str) -> str:
        """Lấy username từ account_id để hiển thị log."""
        accounts = self._loader.get_enabled_accounts()
        acc = next((a for a in accounts if a.id == account_id), None)
        return acc.username if acc else account_id

    def _save_progress(self, ctx: StateContext, status_text: str = "", quest_detail: str = "") -> None:
        """Lưu tiến độ của account ra file runtime/progress_{account_id}.json cho Launcher hiển thị."""
        try:
            import json
            from datetime import datetime
            runtime_dir = Path("runtime")
            runtime_dir.mkdir(parents=True, exist_ok=True)
            prog_file = runtime_dir / f"progress_{ctx.account_id}.json"

            existing_history = []
            if prog_file.exists():
                try:
                    with open(prog_file, "r", encoding="utf-8") as f:
                        old_data = json.load(f)
                        existing_history = old_data.get("history", [])
                except Exception:
                    pass

            history = ctx.extra.get("quest_history_list", existing_history)

            # Tính toán status_text dễ hiểu cho người dùng
            st_text = status_text
            if not st_text:
                state_val = ctx.current_state.value if hasattr(ctx, "current_state") else "IDLE"
                if state_val in ("BOOT", "START_GAME", "LOGIN"):
                    st_text = "Đang đăng nhập..."
                elif state_val == "VERIFY_INGAME":
                    st_text = "Đang vào game..."
                elif state_val in ("FIND_COMMERCE_CHIEF", "MOVE_TO_CITY"):
                    st_text = "Tìm Chủ thương hội..."
                elif state_val == "MEET_NPC":
                    st_text = "Đang đối thoại Chủ thương hội..."
                elif state_val == "GET_QUEST":
                    st_text = "Đang nhận nhiệm vụ..."
                elif state_val == "CLASSIFY_QUEST":
                    st_text = "Đang phân loại nhiệm vụ..."
                elif state_val == "NAVIGATE":
                    st_text = f"Đang làm NV #{getattr(ctx, 'total_quests_done', 0) + 1}"
                elif state_val == "INTERACT_NPC":
                    st_text = "Đang nói chuyện với NPC nhiệm vụ"
                elif state_val == "BUY_ITEM":
                    st_text = "Đang mua hàng Tây Vực"
                elif state_val == "RETURN_QUEST":
                    st_text = "Quay về trả nhiệm vụ"
                elif state_val == "FINISHED":
                    st_text = "Đã hoàn thành 20/20 🎉" if getattr(ctx, "total_quests_done", 0) >= 20 else "Hoàn tất phiên chạy"
                elif state_val == "ERROR":
                    st_text = f"Lỗi: {ctx.error_message or ctx.error_code}"
                else:
                    st_text = state_val

            detail = quest_detail or ctx.extra.get("quest_detail_text", ctx.extra.get("quest_text", ""))
            total_done = getattr(ctx, "total_quests_done", 0)
            data = {
                "account_id": ctx.account_id,
                "username": self._get_username(ctx.account_id),
                "server": getattr(self, "_target_server", "S1"),
                "current_state": ctx.current_state.value if hasattr(ctx, "current_state") else "IDLE",
                "status_text": st_text,
                "total_quests_done": total_done,
                "max_quests": 20,
                "percent": min(100, int(total_done / 20 * 100)),
                "current_quest_type": ctx.current_quest_type.value if getattr(ctx, "current_quest_type", None) else "UNKNOWN",
                "current_quest_detail": detail,
                "last_updated": datetime.now().strftime("%H:%M:%S"),
                "history": history,
            }
            with open(prog_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"Save progress error: {e}")

    # ============================
    # Setup
    # ============================

    def _setup_logging(self) -> None:
        from loguru import logger
        log_cfg = self._app_cfg.logging
        Path(log_cfg.log_dir).mkdir(parents=True, exist_ok=True)
        # File log hàng ngày (tối đa 10MB x 5 file)
        log_file = Path(log_cfg.log_dir) / "{time:YYYY-MM-DD}.log"
        logger.add(
            str(log_file),
            level=log_cfg.level,
            rotation=log_cfg.rotation,
            retention=log_cfg.retention,
            encoding="utf-8",
        )

    def _create_adb(self) -> ADBController:
        return ADBController(self._app_cfg)

    # ============================
    # State Handlers Registration
    # ============================

    def _register_handlers(self) -> None:
        sm = self._state_machine
        sm.register(BotState.BOOT,             self._handle_boot)
        sm.register(BotState.CONNECT_EMULATOR, self._handle_connect_emulator)
        sm.register(BotState.START_GAME,       self._handle_start_game)
        sm.register(BotState.LOGIN,            self._handle_login)
        sm.register(BotState.VERIFY_INGAME,    self._handle_verify_ingame)
        sm.register(BotState.CHECK_LOCATION,   self._handle_check_location)
        sm.register(BotState.MOVE_TO_CITY,     self._handle_move_to_city)
        sm.register(BotState.FIND_COMMERCE_CHIEF, self._handle_find_commerce_chief)
        sm.register(BotState.MEET_NPC,         self._handle_meet_npc)
        sm.register(BotState.GET_QUEST,        self._handle_get_quest)
        sm.register(BotState.CLASSIFY_QUEST,   self._handle_classify_quest)
        sm.register(BotState.NAVIGATE,         self._handle_navigate)
        sm.register(BotState.INTERACT_NPC,     self._handle_interact_npc)
        sm.register(BotState.BUY_ITEM,         self._handle_buy_item)
        sm.register(BotState.RETURN_QUEST,     self._handle_return_quest)
        sm.register(BotState.COMPLETE_QUEST,   self._handle_complete_quest)
        sm.register(BotState.CANCEL_QUEST,     self._handle_cancel_quest)
        sm.register(BotState.TRANSFER_GOLD,    self._handle_transfer_gold)
        sm.register(BotState.LOGOUT,           self._handle_logout)
        sm.register(BotState.NEXT_ACCOUNT,     self._handle_next_account)
        sm.register(BotState.RECOVERY,         self._handle_recovery)
        sm.register(BotState.ERROR,            self._handle_error)

    # ============================
    # State Handlers
    # ============================

    def _handle_boot(self, ctx: StateContext) -> BotState:
        logger.info(f"[{ctx.account_id}] BOOT")
        self._checkpoint.save(ctx)
        return BotState.CONNECT_EMULATOR

    def _handle_connect_emulator(self, ctx: StateContext) -> BotState:
        logger.info(f"[{ctx.account_id}] Connecting emulator...")
        connected = self._adb.connect()
        if not connected:
            ctx.retry_count += 1
            max_retries = max(self._app_cfg.retry.max_attempts, 5)
            if ctx.retry_count >= max_retries:
                raise BotError(ErrorCode.ADB_002, "Cannot connect to ADB after max retries")
            delay = max(self._app_cfg.retry.delay_between, 2.0)
            time.sleep(delay)
            return BotState.CONNECT_EMULATOR
        ctx.reset_retry()
        serial = getattr(self._adb, '_device_serial', None) or getattr(self._adb, '_serial', 'emulator')
        activity.connected(serial)
        return BotState.START_GAME

    def _is_ingame(self, frame=None) -> bool:
        """Kiểm tra xem hiện tại có đang trong game hay không bằng các template đặc trưng."""
        self._dismiss_unexpected_popups(frame=frame)
        for tpl in ["btn_tui", "quest_btn", "minimap", "ingame_screen"]:
            try:
                res = self._vision.find_template(tpl, frame=frame)
                if res.found or (res.confidence is not None and res.confidence > 0.70):
                    return True
            except Exception:
                pass
        try:
            if frame is None:
                frame = self._screen.capture()
            boxes = self._text.read(frame)
            if self._text.best_match(boxes, ["Tìm map", "Tìm NPC", "Bản đồ thế giới", "Bảng nhiệm vụ", "Phúc lợi", "Luyện công", "Cài đặt"], minimum=0.55):
                return True
        except Exception:
            pass
        return False

    def _dismiss_unexpected_popups(self, frame=None) -> bool:
        """Tự động phát hiện và đóng các popup/menu che màn hình ngoài ý muốn:
        1. Cài đặt / Tùy chọn / Hình ảnh (do phím ESC/Back hoặc click trúng gear)
        2. Kênh chat (do click trúng chat box ở dưới)
        3. Chiến trường (do click trúng Đấu trường ở góc dưới phải)
        """
        try:
            if frame is None:
                frame = self._screen.capture()
            if frame is None:
                return False

            boxes = self._text.read(frame)
            coords = self._loader.load_ui_coords()
            nav_cfg = coords.get("navigation", {})
            dismissed = False

            # 1. Bảng Cài đặt / Tùy chọn / Hình ảnh
            # Đóng bằng nút X ở góc trên phải popup: (1412, 170)
            is_settings = any(
                any(k in normalize_game_text(b.text) for k in ("tat nguoi choi", "tat sat thuong", "uy thac", "dang xuat", "hinh anh", "nhan buff"))
                for b in boxes
            )
            if is_settings:
                btn = nav_cfg.get("settings_close_button", {"x": 1412, "y": 170})
                logger.info(f"Phát hiện bảng 'Cài đặt / Hình ảnh' che màn hình! Đóng tại ({btn['x']}, {btn['y']})...")
                self._input.tap(btn["x"], btn["y"])
                time.sleep(0.3)
                dismissed = True
                frame = self._screen.capture()
                boxes = self._text.read(frame) if frame is not None else []

            # 2. Bảng Kênh chat (popup chat toàn màn hình/nửa màn hình)
            # Đóng bằng nút X ở góc trên phải bảng chat: (795, 105)
            is_chat = any(
                "kenh chat" in normalize_game_text(b.text) or ("tong hop" in normalize_game_text(b.text) and b.center_x < 400 and b.center_y < 250)
                for b in boxes
            )
            if is_chat:
                btn = nav_cfg.get("chat_close_button", {"x": 795, "y": 105})
                logger.info(f"Phát hiện bảng 'Kênh chat' đang mở! Đóng tại ({btn['x']}, {btn['y']})...")
                self._input.tap(btn["x"], btn["y"])
                time.sleep(0.3)
                dismissed = True
                frame = self._screen.capture()
                boxes = self._text.read(frame) if frame is not None else []

            # 3. Bảng Chiến trường / Bảng xếp hạng thi đấu
            # Đóng bằng nút X tại (1475, 120)
            is_arena = any(
                any(k in normalize_game_text(b.text) for k in ("chien truong", "chien trudng", "tham chien", "xep hang", "thao coc chien"))
                for b in boxes
            )
            if is_arena:
                btn = nav_cfg.get("arena_close_button", {"x": 1475, "y": 120})
                logger.info(f"Phát hiện bảng 'Chiến trường' đang mở! Đóng tại ({btn['x']}, {btn['y']})...")
                self._input.tap(btn["x"], btn["y"])
                time.sleep(0.3)
                dismissed = True

            return dismissed
        except Exception as e:
            logger.debug(f"_dismiss_unexpected_popups error: {e}")
            return False

    def _is_map_open(self, frame=None) -> bool:
        """Kiểm tra xem bảng bản đồ lớn có đang mở trên màn hình không."""
        try:
            if frame is None:
                frame = self._screen.capture()
            if frame is None:
                return False
            boxes = self._text.read(frame, (100, 130, 450, 220))
            return any(
                any(w in normalize_game_text(b.text) for w in ("tim map", "tim npc", "map", "npc"))
                for b in boxes
            )
        except Exception:
            return False

    def _ensure_map_open(self) -> bool:
        """Mở bản đồ nếu chưa mở, đảm bảo không bị popup Cài đặt/Chat che."""
        self._dismiss_unexpected_popups()
        if self._is_map_open():
            return True
        nav = self._loader.load_ui_coords().get("navigation", {})
        minimap = nav.get("minimap_button", {"x": 1520, "y": 100})
        logger.info(f"Mở Minimap tại ({minimap.get('x', 1520)}, {minimap.get('y', 100)})...")
        self._input.tap(minimap.get("x", 1520), minimap.get("y", 100))
        time.sleep(0.35)
        return self._is_map_open()

    def _close_map(self) -> None:
        """Đóng bảng bản đồ an toàn tại nút X (1519, 77), tránh nút gear Cài đặt tại (1402, 85)."""
        nav = self._loader.load_ui_coords().get("navigation", {})
        close_btn = nav.get("map_close_button", {"x": 1519, "y": 77})
        if self._is_map_open():
            bx = close_btn.get("x", 1519)
            by = close_btn.get("y", 77)
            logger.info(f"Đóng bản đồ an toàn tại ({bx}, {by})...")
            self._input.tap(bx, by)
            time.sleep(0.3)

    def _parse_left_tracker_boxes(self, boxes: list) -> Optional[tuple[str, tuple[int, int]]]:
        """Trích xuất thông tin nhiệm vụ Thương hội từ danh sách OCR box trên thanh nhiệm vụ bên trái."""
        text_lines = []
        header_found = False
        click_pos = (180, 310)

        for b in boxes:
            norm = normalize_game_text(b.text)
            if not header_found and "thuong hoi" in norm:
                header_found = True
                click_pos = (b.center_x, b.center_y)
                # Bỏ qua dòng tiêu đề ngắn ví dụ: "Nhiệm vụ Thương hội"
                if "nhiem vu" in norm and len(norm.split()) <= 4:
                    continue
            if header_found:
                text_lines.append(b.text)

        if header_found:
            full_text = " ".join(text_lines) if text_lines else "Nhiệm vụ Thương hội"
            return full_text, click_pos
        return None

    def _check_left_quest_tracker(self, frame=None, scroll_to_bottom: bool = True) -> Optional[tuple[str, tuple[int, int]]]:
        """
        Cách 1: Quét thanh nhiệm vụ bên trái.
        Đặc biệt: Cuộn tới cuối danh sách để quét hết nhiệm vụ đã nhận
        (đề phòng nhiệm vụ Thương Hội bị đẩy xuống dưới do có Dã Tẩu / Tân Thủ / Môn Phái).
        """
        try:
            if frame is None:
                frame = self._screen.capture()
            if frame is None:
                return None

            boxes = self._text.read(frame, (0, 200, 420, 550))
            result = self._parse_left_tracker_boxes(boxes)
            if result:
                logger.info(f"Cách 1: Tìm thấy nhiệm vụ Thương hội ở thanh bên trái: '{result[0]}'")
                return result

            if not scroll_to_bottom:
                return None

            # Cuộn thanh nhiệm vụ bên trái xuống dưới (vuốt từ dưới lên trên)
            logger.info("Cách 1: Chưa thấy Thương hội ở đầu danh sách, cuộn thanh nhiệm vụ bên trái tới cuối...")
            for scroll_idx in range(2):
                self._input.swipe(200, 480, 200, 240, duration=250)
                time.sleep(0.3)
                scr = self._screen.capture()
                if scr is None:
                    continue
                scrolled_boxes = self._text.read(scr, (0, 200, 420, 550))
                scrolled_result = self._parse_left_tracker_boxes(scrolled_boxes)
                if scrolled_result:
                    logger.info(f"Cách 1: Sau khi cuộn (lần {scroll_idx + 1}), tìm thấy nhiệm vụ Thương hội: '{scrolled_result[0]}'")
                    return scrolled_result

            return None
        except Exception as e:
            logger.debug(f"_check_left_quest_tracker error: {e}")
            return None

    def _check_quest_panel(self, ctx: Optional[StateContext] = None) -> Optional[tuple[str, tuple[int, int]]]:
        """
        Cách 2: Nhấn vào icon nhiệm vụ bên góc phải dưới để mở danh sách nhiệm vụ.
        Nếu có nhiệm vụ Thương Hội trong danh sách:
        - Chọn nhiệm vụ để đọc nội dung / lấy link xanh auto-run.
        - Đóng bảng nhiệm vụ lại bằng nút X.
        """
        tag = f"[{ctx.account_id}] " if ctx and hasattr(ctx, "account_id") and ctx.account_id else ""
        try:
            coords = self._loader.load_ui_coords()
            quest_btn = coords.get("commerce_quest", {}).get("quest_button", {"x": 1404, "y": 755})
            close_btn = coords.get("commerce_quest", {}).get("quest_panel_close", {"x": 1495, "y": 95})

            logger.info(f"{tag}Cách 2: Bấm vào icon Nhiệm vụ góc phải dưới ({quest_btn.get('x', 1404)}, {quest_btn.get('y', 755)})...")
            self._input.tap(quest_btn.get("x", 1404), quest_btn.get("y", 755))
            time.sleep(0.6)

            scr = self._screen.capture()
            if scr is None:
                self._input.tap(close_btn.get("x", 1495), close_btn.get("y", 95))
                return None

            # Quét danh sách nhiệm vụ trên bảng (panel danh sách ở khoảng x: 50..500, y: 150..750)
            list_boxes = self._text.read(scr, (50, 150, 500, 750))
            quest_item_pos = None
            for b in list_boxes:
                norm = normalize_game_text(b.text)
                if "thuong hoi" in norm:
                    quest_item_pos = (b.center_x, b.center_y)
                    logger.info(f"{tag}Cách 2: Phát hiện dòng 'Thương hội' trong bảng nhiệm vụ tại {quest_item_pos}")
                    break

            # Nếu không tìm thấy ở cột trái, quét rộng hơn
            if not quest_item_pos:
                all_boxes = self._text.read(scr, (50, 150, 1400, 750))
                for b in all_boxes:
                    norm = normalize_game_text(b.text)
                    if "thuong hoi" in norm:
                        quest_item_pos = (b.center_x, b.center_y)
                        logger.info(f"{tag}Cách 2: Phát hiện 'Thương hội' trong bảng nhiệm vụ tại {quest_item_pos}")
                        break

            if not quest_item_pos:
                logger.info(f"{tag}Cách 2: Không có nhiệm vụ Thương hội trong bảng nhiệm vụ.")
                # Đóng bảng nhiệm vụ
                self._input.tap(close_btn.get("x", 1495), close_btn.get("y", 95))
                time.sleep(0.25)
                return None

            # Bấm vào dòng nhiệm vụ Thương hội để hiện chi tiết ở bảng bên phải
            logger.info(f"{tag}Cách 2: Click chọn nhiệm vụ Thương hội tại {quest_item_pos}...")
            self._input.tap(*quest_item_pos)
            time.sleep(0.35)

            detail_scr = self._screen.capture()
            detail_boxes = self._text.read(detail_scr, (450, 150, 1350, 750)) if detail_scr is not None else []
            detail_text = " ".join(b.text for b in detail_boxes) if detail_boxes else "Nhiệm vụ Thương hội"

            # Trích xuất từ/cụm từ màu vàng (món đồ cần mua) trong khung nội dung bên phải
            yellow_item = ""
            if detail_scr is not None:
                try:
                    import numpy as np
                    roi = detail_scr[150:750, 450:1350]
                    # Màu vàng trong BGR: B < 110, G > 175, R > 175
                    yellow_mask = (roi[:, :, 0] < 110) & (roi[:, :, 1] > 175) & (roi[:, :, 2] > 175)
                    if np.sum(yellow_mask) >= 30:
                        yellow_roi = np.zeros_like(roi)
                        yellow_roi[yellow_mask] = roi[yellow_mask]
                        y_boxes = self._text.read(yellow_roi)
                        if y_boxes:
                            yellow_item = " ".join(b.text for b in y_boxes).strip()
                            logger.info(f"{tag}Phát hiện món đồ tô vàng trong nội dung nhiệm vụ: '{yellow_item}'")
                except Exception as ex:
                    logger.debug(f"Yellow item extraction failed: {ex}")

            if ctx and hasattr(ctx, "extra") and yellow_item:
                ctx.extra["yellow_item"] = yellow_item

            # Đóng bảng nhiệm vụ
            self._input.tap(close_btn.get("x", 1495), close_btn.get("y", 95))
            time.sleep(0.35)

            if ctx and hasattr(ctx, "extra"):
                ctx.extra["auto_walking"] = False

            return detail_text, None

        except Exception as e:
            logger.warning(f"{tag}_check_quest_panel error: {e}")
            try:
                coords = self._loader.load_ui_coords()
                close_btn = coords.get("commerce_quest", {}).get("quest_panel_close", {"x": 1495, "y": 95})
                self._input.tap(close_btn.get("x", 1495), close_btn.get("y", 95))
            except Exception:
                pass
            return None

    def _check_active_commerce_quest(self, ctx: Optional[StateContext] = None, frame=None, full_check: bool = True) -> Optional[tuple[str, tuple[int, int]]]:
        """
        Kiểm tra xem tài khoản đã nhận nhiệm vụ Thương hội hay chưa:
        - Sử dụng Cách 2: Mở bảng danh sách nhiệm vụ (icon góc phải dưới).
        (Đã bỏ Cách 1 quét khung bên trái theo yêu cầu người dùng vì không hiệu quả).
        """
        return self._check_quest_panel(ctx=ctx)

    def _is_update_screen(self, frame=None) -> bool:
        """Detect the launcher update check and keep automation from tapping it."""
        try:
            frame = frame if frame is not None else self._screen.capture()
            boxes = self._text.read(frame)
            return self._text.best_match(
                boxes,
                [
                    "Quá trình kiểm tra cập nhật đang diễn ra",
                    "Vui lòng chờ đợi giây lát",
                ],
                minimum=0.48,
            ) is not None
        except Exception as exc:
            logger.debug(f"Update screen detection failed: {exc}")
            return False

    def _is_login_screen(self, frame=None) -> bool:
        """Kiểm tra xem màn hình hiện tại có phải là màn hình đăng nhập không."""
        try:
            from .text_recognition import normalize_game_text
            frame = frame if frame is not None else self._screen.capture()
            if frame is None:
                return False

            # Cách 1: Tìm template nút 'btn_vao_game'
            res = self._vision.find_template("btn_vao_game", frame=frame)
            if res.found or (res.confidence is not None and res.confidence >= 0.70):
                return True

            # Cách 2: Quét OCR các từ khóa đặc trưng của màn hình đăng nhập
            boxes = self._text.read(frame)
            if boxes:
                all_text = " ".join([normalize_game_text(b.text) for b in boxes])
                if ("tai khoan" in all_text and "mat khau" in all_text) or "vao game" in all_text or "doi may chu" in all_text:
                    return True

            return False
        except Exception as exc:
            logger.debug(f"Login screen detection failed: {exc}")
            return False

    def _open_game_from_launcher(self) -> bool:
        """
        Tìm game trên màn hình chính của máy ảo:
        - Nếu có icon/chữ 'Tình Kiếm' trên màn hình chính: click vào game.
        - Nếu không có: click nút mũi tên ở góc trái dưới (62, 874) để mở danh sách app.
        - Tìm 'Tình Kiếm' trong danh sách app và click để mở game.
        """
        try:
            frame = self._screen.capture()
            if frame is None:
                return False

            # 1. Quét tìm game trên màn hình hiện tại
            boxes = self._text.read(frame)
            for b in boxes:
                if b.center_y < 80:  # Bỏ qua tab tiêu đề cửa sổ LDPlayer
                    continue
                norm = normalize_game_text(b.text)
                if "tinh kiem" in norm:
                    logger.info(f"Tìm thấy game '{b.text}' trên màn hình tại ({b.center_x}, {b.center_y}), click mở...")
                    self._input.tap(b.center_x, b.center_y)
                    time.sleep(3.0)
                    return True

            # 2. Không thấy game trên màn hình chính -> Bấm nút mũi tên ở góc trái dưới mở drawer
            coords = self._loader.load_ui_coords()
            arrow = coords.get("emulator_home", {}).get("app_drawer_button", {"x": 62, "y": 874})
            logger.info(f"Không thấy game trên màn hình chính. Bấm nút mũi tên góc trái dưới ({arrow['x']}, {arrow['y']}) để mở danh sách app...")
            self._input.tap(arrow["x"], arrow["y"])
            time.sleep(1.2)

            # 3. Quét danh sách app vừa mở
            drawer_frame = self._screen.capture()
            if drawer_frame is not None:
                drawer_boxes = self._text.read(drawer_frame)
                for b in drawer_boxes:
                    if b.center_y < 80:
                        continue
                    norm = normalize_game_text(b.text)
                    if "tinh kiem" in norm:
                        logger.info(f"Tìm thấy game '{b.text}' trong danh sách app tại ({b.center_x}, {b.center_y}), click mở...")
                        self._input.tap(b.center_x, b.center_y)
                        time.sleep(3.0)
                        return True

            return False
        except Exception as e:
            logger.warning(f"_open_game_from_launcher error: {e}")
            return False

    def _handle_start_game(self, ctx: StateContext) -> BotState:
        logger.info(f"[{ctx.account_id}] Starting game...")

        # Kết nối ADB nếu chưa kết nối (khi bắt đầu từ START_GAME trực tiếp)
        if not self._adb.is_connected():
            logger.info(f"[{ctx.account_id}] ADB not connected, connecting first...")
            connected = self._adb.connect()
            if connected:
                serial = getattr(self._adb, '_serial', 'emulator')
                activity.connected(serial)
            else:
                logger.warning(f"[{ctx.account_id}] ADB connect failed, retrying via CONNECT_EMULATOR...")
                return BotState.CONNECT_EMULATOR

        # Kiểm tra nếu đã trong game rồi
        frame = self._screen.capture()
        if self._is_ingame(frame=frame):
            if getattr(self, "_skip_login", False):
                logger.info(f"[{ctx.account_id}] Game already running and in-game (skip_login enabled)!")
                activity._write("🎮", f"[{ctx.account_id}] Game dang chay san, tim Chu thuong hoi...")
                return BotState.FIND_COMMERCE_CHIEF
            else:
                logger.info(f"[{ctx.account_id}] Game đang mở tài khoản trước, đăng xuất để đăng nhập đúng tài khoản {ctx.account_id}...")
                self._logout_current_account(ctx)
                time.sleep(1.0)
                return BotState.LOGIN

        if self._is_update_screen(frame=frame):
            logger.info(f"[{ctx.account_id}] Game is checking updates; waiting without tapping...")
            time.sleep(5.0)
            return BotState.START_GAME

        # Kiểm tra nếu đã ở sẵn màn hình đăng nhập
        if self._is_login_screen(frame=frame):
            logger.info(f"[{ctx.account_id}] Game đã vào màn hình đăng nhập -> BotState.LOGIN")
            activity._write("🔑", f"[{ctx.account_id}] Da vao man hinh dang nhap...")
            return BotState.LOGIN

        # Chưa vào game → mở game: ưu tiên tìm icon trên màn hình / mở drawer góc trái dưới
        logger.info(f"[{ctx.account_id}] Game not detected, launching...")
        activity._write("⏳", f"[{ctx.account_id}] Dang mo game Tinh Kiem Truyen Ky...")
        opened = self._open_game_from_launcher()
        if not opened:
            self._adb.start_app()
        else:
            startup_wait = getattr(getattr(self._app_cfg, "game", None), "startup_wait", 15)
            time.sleep(startup_wait)

        if self._is_ingame():
            if getattr(self, "_skip_login", False):
                logger.info(f"[{ctx.account_id}] Game started directly into in-game!")
                return BotState.FIND_COMMERCE_CHIEF
            else:
                logger.info(f"[{ctx.account_id}] Game tự động vào game cũ, cần đăng xuất để đăng nhập {ctx.account_id}...")
                self._logout_current_account(ctx)
                time.sleep(1.0)
                return BotState.LOGIN

        return BotState.LOGIN


    def _clear_and_type(self, x: int, y: int, text: str):
        """Click vào ô input, xóa text cũ nếu có, chờ game focus rồi nhập text ổn định."""
        self._input.tap(x, y)
        time.sleep(0.8)  # Chờ game kích hoạt ô input
        try:
            delete_command = "input keyevent 123; " + "; ".join(["input keyevent 67"] * 35) + "; input keyevent 122; " + "; ".join(["input keyevent 112"] * 15)
            self._adb._run(["shell", delete_command])
            time.sleep(0.2)
        except Exception:
            pass
        self._clear_focused_and_type(text)

    def _clear_focused_and_type(self, text: str):
        """Nhập text vào ô hiện đang có focus mà không bị keyevent xóa đè."""
        time.sleep(0.3)
        self._input.input_text(str(text))
        time.sleep(0.6)

    def _replace_and_type(self, x: int, y: int, text: str) -> None:
        """Xóa giá trị cũ rồi nhập giá trị mới vào ô tọa độ."""
        self._input.tap(x, y)
        time.sleep(0.25)
        try:
            delete_command = "input keyevent 123; " + "; ".join(["input keyevent 67"] * 24)
            self._adb._run(["shell", delete_command])
        except Exception:
            pass
        self._clear_focused_and_type(str(text))

    def _click_and_type_password(self, x: int, y: int, password: str) -> bool:
        """Click ô mật khẩu, xóa text cũ, nhập và xác nhận nội dung trong ô đã thay đổi."""
        x1, x2 = max(0, x - 100), x + 230
        y1, y2 = max(0, y - 28), y + 28

        for attempt in range(1, 3):
            before = self._screen.capture()[y1:y2, x1:x2].copy()
            click_x = x if attempt == 1 else x + 50
            logger.info(
                f"Click ô mật khẩu ({click_x}, {y}) để focus "
                f"(lần {attempt}/2)..."
            )
            self._input.tap(click_x, y)
            time.sleep(0.8)
            try:
                delete_command = "input keyevent 123; " + "; ".join(["input keyevent 67"] * 35) + "; input keyevent 122; " + "; ".join(["input keyevent 112"] * 15)
                self._adb._run(["shell", delete_command])
                time.sleep(0.2)
            except Exception:
                pass
            self._clear_focused_and_type(password)
            time.sleep(0.5)

            after = self._screen.capture()[y1:y2, x1:x2]
            if before.shape == after.shape and before.size:
                changed = (abs(after.astype("int16") - before.astype("int16")) > 20)
                changed_pixels = int(changed.any(axis=2).sum())
                logger.info(
                    f"Kiểm tra ô mật khẩu: {changed_pixels} pixel thay đổi"
                )
                if changed_pixels >= 30:
                    return True

            logger.warning("Ô mật khẩu chưa thay đổi sau khi nhập, đang thử lại...")

        return True

    def _check_and_dismiss_login_error(self, scr: Optional[np.ndarray] = None) -> tuple[bool, str]:
        """
        Kiểm tra và xử lý các popup lỗi đăng nhập (ví dụ: 'Tài khoản hiện đang sử dụng!',
        'Sai mật mã', 'Tài khoản không tồn tại', 'Tài khoản bị khóa', v.v.).
        Nếu phát hiện popup, bấm nút 'Trở lại' để đóng popup và trả về (True, error_message).
        Nếu không có popup, trả về (False, "").
        """
        from .text_recognition import normalize_game_text

        if scr is None:
            scr = self._screen.capture()
        if scr is None:
            return False, ""

        coords = self._loader.load_ui_coords()
        login_cfg = coords.get("login", {})
        popup_btn = login_cfg.get("popup_dismiss_button", {"x": 771, "y": 395})

        # 1. Quét OCR để nhận diện thông báo lỗi chính xác
        boxes = []
        try:
            boxes = self._text.read(scr)
        except Exception as e:
            logger.debug(f"OCR scan in popup check failed: {e}")

        all_text = " ".join([normalize_game_text(b.text) for b in boxes]) if boxes else ""
        detected_error = ""

        if any(w in all_text for w in ("dang su dung", "dang si dung", "hien dang")):
            detected_error = "Tài khoản hiện đang sử dụng!"
        elif any(w in all_text for w in ("sai mat", "khong dung", "sai tai khoan")):
            detected_error = "Sai mật mã hoặc tài khoản!"
        elif "khong ton tai" in all_text:
            detected_error = "Tài khoản không tồn tại!"
        elif any(w in all_text for w in ("bi khoa", "bi cam", "tam khoa", "vi pham")):
            detected_error = "Tài khoản đã bị khóa!"
        elif "xin nhap tai khoan" in all_text:
            detected_error = "Chưa nhập tài khoản hoặc mật khẩu!"

        # 2. Tìm nút "Trở lại" để bấm đóng
        click_coord = None

        # Cách A: Tìm qua template btn_tro_lai
        try:
            res_popup = self._vision.find_template("btn_tro_lai", frame=scr)
            if res_popup.found or (res_popup.confidence is not None and res_popup.confidence >= 0.60):
                click_coord = (res_popup.center_x, res_popup.center_y)
        except Exception:
            pass

        # Cách B: Tìm qua OCR (chữ 'Trở lại' / 'Trolai')
        if not click_coord and boxes:
            for b in boxes:
                norm_b = normalize_game_text(b.text)
                if "trolai" in norm_b or "tro lai" in norm_b:
                    click_coord = (b.center_x, b.center_y)
                    break

        # Cách C: Nếu OCR thấy thông báo lỗi nhưng chưa có tọa độ click, dùng tọa độ mặc định
        if not click_coord and detected_error:
            click_coord = (popup_btn.get("x", 771), popup_btn.get("y", 395))

        # Cách D: Nếu không có thông điệp OCR nhưng tìm thấy template btn_tro_lai -> cũng là popup lỗi
        if not detected_error and click_coord:
            detected_error = "Lỗi đăng nhập (popup thông báo)"

        if detected_error and click_coord:
            logger.warning(f"Phát hiện popup lỗi đăng nhập: '{detected_error}', bấm 'Trở lại' tại {click_coord}...")
            self._input.tap(click_coord[0], click_coord[1])
            time.sleep(1.2)

            # Kiểm tra lại xem popup đã đóng chưa, nếu chưa bấm thêm lần nữa
            try:
                scr2 = self._screen.capture()
                if scr2 is not None:
                    res2 = self._vision.find_template("btn_tro_lai", frame=scr2)
                    if res2.found or (res2.confidence is not None and res2.confidence >= 0.65):
                        self._input.tap(click_coord[0], click_coord[1])
                        time.sleep(0.8)
            except Exception:
                pass

            return True, detected_error

        return False, ""

    def _handle_login(self, ctx: StateContext) -> BotState:
        if self._is_update_screen():
            logger.info(f"[{ctx.account_id}] Update screen is active; returning to safe wait state...")
            time.sleep(5.0)
            return BotState.START_GAME

        accounts = self._loader.get_enabled_accounts()
        account = next((a for a in accounts if a.id == ctx.account_id), None)
        if not account:
            if accounts:
                account = accounts[0]
                logger.warning(f"[{ctx.account_id}] Account not found, using first available: {account.id}")
            else:
                raise BotError(ErrorCode.LOGIN_001, f"No enabled account found in accounts.yaml for {ctx.account_id}")

        # Bỏ qua login nếu cờ skip_login được bật hoặc đã ở trong game sẵn
        if getattr(self, "_skip_login", False) or self._is_ingame():
            logger.info(f"[{ctx.account_id}] Đã ở trong game hoặc skip_login=True -> Chuyển sang FIND_COMMERCE_CHIEF")
            return BotState.FIND_COMMERCE_CHIEF

        logger.info(f"[{ctx.account_id}] Logging in as {account.username}...")
        activity._write("👤", f"[{ctx.account_id}] Dang dang nhap tai khoan: {account.username}...")

        # Chờ màn hình login xuất hiện (tối đa 15s)
        try:
            self._vision.wait_for_template(
                "login_screen",
                timeout=min(self._app_cfg.timing.login_timeout, 15),
            )
        except BotError:
            logger.warning(f"[{ctx.account_id}] Login screen template not detected, proceeding to click input coordinates directly")

        # Lấy tọa độ UI từ config
        coords = self._loader.load_ui_coords()
        login_cfg = coords.get("login", {})

        user_x = login_cfg.get("username_field", {}).get("x", 820)
        user_y = login_cfg.get("username_field", {}).get("y", 318)
        pass_x = login_cfg.get("password_field", {}).get("x", 820)
        pass_y = login_cfg.get("password_field", {}).get("y", 417)
        btn_x  = login_cfg.get("login_button", {}).get("x", 834)
        btn_y  = login_cfg.get("login_button", {}).get("y", 622)

        # 0. Đóng popup lỗi nếu có (hoặc phát hiện lỗi tài khoản nghiêm trọng để đổi acc)
        has_error, error_reason = self._check_and_dismiss_login_error()
        if has_error and any(k in error_reason for k in ("đang sử dụng", "Sai mật", "không tồn tại", "bị khóa")):
            username = self._get_username(ctx.account_id)
            logger.warning(f"[{ctx.account_id}] ({username}) Phát hiện lỗi '{error_reason}' trước khi login -> Đổi sang tài khoản tiếp theo...")
            activity._write("⚠️", f"[{username}] Đăng nhập thất bại: {error_reason} -> Tự động đổi sang tài khoản tiếp theo...")
            self._save_progress(ctx, status_text=f"Lỗi: {error_reason} -> Đổi tài khoản")
            time.sleep(1.0)
            return BotState.NEXT_ACCOUNT

        # 1. Chọn Server game nếu có cấu hình
        target_server = getattr(self, "_target_server", None) or getattr(account, "server", "S1")
        if target_server:
            try:
                scr_cur = self._screen.capture()
                need_change = True
                if scr_cur is not None:
                    # Kiểm tra xem server hiện tại đã đúng chưa
                    server_roi = scr_cur[80:180, 500:1100]
                    if "2" in str(target_server):
                        # Chữ S2 có màu vàng đặc trưng
                        is_yellow = (server_roi[:, :, 0] < 50) & (server_roi[:, :, 1] > 180) & (server_roi[:, :, 2] > 180)
                        if is_yellow.sum() > 20:
                            logger.info(f"[{ctx.account_id}] Server hiện tại đã là {target_server}, không cần đổi.")
                            need_change = False

                if need_change:
                    change_btn = login_cfg.get("server_change_button", {"x": 1020, "y": 135})
                    logger.info(f"[{ctx.account_id}] Selecting server {target_server}...")
                    self._input.tap(change_btn.get("x", 1020), change_btn.get("y", 135))
                    time.sleep(1.5)

                    if "2" in str(target_server):
                        s_btn = login_cfg.get("server_s2_button", {"x": 814, "y": 206})
                    else:
                        s_btn = login_cfg.get("server_s1_button", {"x": 1204, "y": 206})
                    self._input.tap(s_btn.get("x"), s_btn.get("y"))
                    time.sleep(0.8)

                    confirm_btn = login_cfg.get("server_confirm_button", {"x": 900, "y": 735})
                    self._input.tap(confirm_btn.get("x", 900), confirm_btn.get("y", 735))
                    time.sleep(1.8)
            except Exception as e:
                logger.warning(f"[{ctx.account_id}] Server select error: {e}")

        # 2. Nhập thông tin tài khoản
        logger.info(
            f"[{ctx.account_id}] Dùng tọa độ login đã cấu hình: "
            f"tài khoản=({user_x}, {user_y}), mật khẩu=({pass_x}, {pass_y})"
        )

        res_btn = self._vision.find_template("btn_vao_game")
        if res_btn.found and res_btn.center_x < 1000:
            btn_x = res_btn.center_x
            btn_y = res_btn.center_y
            logger.info(f"[{ctx.account_id}] Tìm thấy nút 'Vào game', click ({btn_x}, {btn_y})")

        # Bước 2.1: Nhập tài khoản
        logger.info(f"[{ctx.account_id}] Nhập tên tài khoản ({user_x}, {user_y})...")
        self._clear_and_type(user_x, user_y, str(account.username))

        # Bước 2.2: Click trực tiếp vào ô mật khẩu để focus rồi nhập.
        self._click_and_type_password(pass_x, pass_y, str(account.password))

        # Bước 2.3: Bấm nút đăng nhập "Vào game"
        logger.info(f"[{ctx.account_id}] Bấm nút Vào game ({btn_x}, {btn_y})...")
        self._input.tap(btn_x, btn_y)
        time.sleep(2.5)

        # Chờ màn chọn nhân vật, chọn nhân vật đầu tiên rồi bấm Vào Game.
        for _ in range(5):
            time.sleep(1.0)
            header_res = self._vision.find_template("header_chon_nhan_vat")
            char_res = self._vision.find_template("char_enter_btn")
            if header_res.found or (char_res.found and char_res.center_x > 1200):
                first_slot = login_cfg.get(
                    "character_first_slot", {"x": 1340, "y": 235}
                )
                click_enter_x = char_res.center_x if (char_res.found and char_res.center_x > 1200) else 1425
                click_enter_y = char_res.center_y if (char_res.found and char_res.center_x > 1200) else 685
                logger.info(
                    f"[{ctx.account_id}] Chọn nhân vật đầu tiên "
                    f"({first_slot['x']}, {first_slot['y']})..."
                )
                self._input.tap(first_slot["x"], first_slot["y"])
                time.sleep(1.0)
                logger.info(
                    f"[{ctx.account_id}] Bấm Vào Game "
                    f"({click_enter_x}, {click_enter_y})..."
                )
                self._input.tap(click_enter_x, click_enter_y)
                time.sleep(3.0)
                break

        return BotState.VERIFY_INGAME

    def _handle_verify_ingame(self, ctx: StateContext) -> BotState:
        logger.info(f"[{ctx.account_id}] Verifying in-game...")
        coords = self._loader.load_ui_coords()
        login_cfg = coords.get("login", {})
        enter_btn = login_cfg.get("character_enter_btn", {"x": 1425, "y": 685})

        start_time = time.time()
        timeout = min(self._app_cfg.timing.login_timeout, 45)

        while time.time() - start_time < timeout:
            time.sleep(1.5)
            scr = self._screen.capture()
            if scr is None:
                continue

            # 1. Kiểm tra các template in-game thực sự TRƯỚC TIÊN (btn_tui, quest_btn, minimap, ingame_screen)
            for tpl in ["btn_tui", "quest_btn", "minimap", "ingame_screen"]:
                try:
                    res = self._vision.find_template(tpl, frame=scr)
                    if res.found or (res.confidence is not None and res.confidence > 0.70):
                        ctx.reset_retry()
                        logger.info(f"[{ctx.account_id}] Login verified OK (tìm thấy template in-game '{tpl}' conf={res.confidence:.2f})")
                        activity._write("✅", f"[{ctx.account_id}] Da vao game thanh cong!")
                        return BotState.FIND_COMMERCE_CHIEF
                except Exception:
                    pass

            # 2. Kiểm tra pixel HUD in-game
            try:
                top_left_area = scr[0:100, 0:250]
                has_green_or_red = (top_left_area[:, :, 1] > 140).any() or (top_left_area[:, :, 2] > 140).any()
                bottom_right_active = (scr[800:900, 1300:1600].mean() > 30)

                if has_green_or_red and bottom_right_active:
                    ctx.reset_retry()
                    logger.info(f"[{ctx.account_id}] Login verified OK (in-game HUD detected)")
                    activity._write("✅", f"[{ctx.account_id}] Da vao game thanh cong!")
                    return BotState.FIND_COMMERCE_CHIEF
            except Exception:
                pass

            # 3. Kiểm tra màn chọn nhân vật
            header_res = self._vision.find_template("header_chon_nhan_vat", frame=scr)
            char_res = self._vision.find_template("char_enter_btn", frame=scr)
            if header_res.found or (char_res.found and char_res.center_x > 1200):
                first_slot = login_cfg.get(
                    "character_first_slot", {"x": 1340, "y": 235}
                )
                click_enter_x = char_res.center_x if (char_res.found and char_res.center_x > 1200) else enter_btn.get("x", 1425)
                click_enter_y = char_res.center_y if (char_res.found and char_res.center_x > 1200) else enter_btn.get("y", 685)
                logger.info(
                    f"[{ctx.account_id}] Đang ở màn chọn nhân vật, chọn nhân vật "
                    f"đầu tiên ({first_slot['x']}, {first_slot['y']})..."
                )
                self._input.tap(first_slot["x"], first_slot["y"])
                time.sleep(1.0)
                logger.info(
                    f"[{ctx.account_id}] Bấm Vào Game "
                    f"({click_enter_x}, {click_enter_y})..."
                )
                self._input.tap(click_enter_x, click_enter_y)
                time.sleep(3.0)
                continue

            # 4. Kiểm tra popup lỗi đăng nhập (Tài khoản hiện đang sử dụng, sai mật khẩu, v.v.)
            has_error, error_reason = self._check_and_dismiss_login_error(scr)
            if has_error:
                username = self._get_username(ctx.account_id)
                # Nếu là lỗi chưa nhập -> quay lại LOGIN thử lại 1 lần
                if "Chưa nhập" in error_reason and ctx.retry_count == 0:
                    ctx.increment_retry()
                    logger.warning(f"[{ctx.account_id}] ({username}) Ô đăng nhập bị trống, thử lại...")
                    return BotState.LOGIN

                # Tất cả các lỗi khác (Tài khoản đang sử dụng, sai mật mã, v.v.) -> ĐỔI SANG TÀI KHOẢN TIẾP THEO
                logger.warning(f"[{ctx.account_id}] ({username}) Đăng nhập thất bại: {error_reason}. Tự động đổi sang tài khoản tiếp theo...")
                activity._write("⚠️", f"[{username}] Đăng nhập thất bại: {error_reason} -> Tự động đổi sang tài khoản tiếp theo...")
                self._save_progress(ctx, status_text=f"Lỗi: {error_reason} -> Đổi tài khoản")
                time.sleep(1.0)
                return BotState.NEXT_ACCOUNT

            # 5. Chỉ coi là ở màn hình đăng nhập nếu label_tai_khoan/btn_vao_game xuất hiện ĐÚNG ở giữa màn hình login
            res_tk = self._vision.find_template("label_tai_khoan", frame=scr)
            res_btn = self._vision.find_template("btn_vao_game", frame=scr)
            is_login_screen = (res_tk.found and 580 <= res_tk.center_x <= 720 and 320 <= res_tk.center_y <= 380) or \
                              (res_btn.found and 750 <= res_btn.center_x <= 950 and 580 <= res_btn.center_y <= 660)
            if is_login_screen:
                logger.warning(f"[{ctx.account_id}] Chưa vào được game (vẫn ở màn hình đăng nhập)! Quay lại LOGIN...")
                return BotState.LOGIN

        ctx.retry_count += 1
        username = self._get_username(ctx.account_id)
        if ctx.retry_count >= self._app_cfg.retry.max_attempts:
            logger.warning(f"[{ctx.account_id}] ({username}) Không thể vào game sau {ctx.retry_count} lần thử -> Tự động đổi sang tài khoản tiếp theo...")
            activity._write("⚠️", f"[{username}] Hết số lần thử vào game -> Chuyển sang tài khoản tiếp theo...")
            self._save_progress(ctx, status_text="Không vào được game -> Đổi tài khoản")
            return BotState.NEXT_ACCOUNT
        return BotState.LOGIN

    # ============================
    # Navigation to City & Commerce Chief NPC
    # ============================

    def _handle_check_location(self, ctx: StateContext) -> BotState:
        """Bỏ qua kiểm tra nhiệm vụ, đi thẳng tới tìm Chủ thương hội."""
        logger.info(f"[{ctx.account_id}] Bỏ qua làm nhiệm vụ, đi tìm Chủ thương hội...")
        activity._write("🏃", f"[{ctx.account_id}] Di tim Chu thuong hoi tai Tuyen Chau...")
        return BotState.FIND_COMMERCE_CHIEF

    def _handle_move_to_city(self, ctx: StateContext) -> BotState:
        """Di chuyển đến thành thị nếu đang ở ngoài thành."""
        logger.info(f"[{ctx.account_id}] Đang di chuyển vào thành...")
        return BotState.FIND_COMMERCE_CHIEF

    def _find_green_npc_on_map(self) -> tuple[int, int] | None:
        """
        Quét vùng danh sách kết quả NPC trên bảng bản đồ (panel bên trái)
        để tìm text màu XANH LÁ của NPC (Chủ thương hội).
        Vùng danh sách: x: 40..300, y: 280..850
        """
        try:
            import cv2
            import numpy as np

            frame = self._screen.capture()
            if frame is None:
                return None

            h, w = frame.shape[:2]
            panel_x1, panel_y1 = int(w * 0.025), int(h * 0.31)
            panel_x2, panel_y2 = int(w * 0.20),  int(h * 0.95)
            roi = frame[panel_y1:panel_y2, panel_x1:panel_x2]

            # Chuyển sang HSV để lọc màu xanh lá (Green text)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower_green = np.array([35, 100, 100])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                logger.debug("_find_green_npc_on_map: no green text found via HSV")
                return None

            valid = []
            for c in contours:
                area = cv2.contourArea(c)
                x, y, bw, bh = cv2.boundingRect(c)
                if area > 40 and bw > 30 and bh > 8:
                    valid.append((area, x, y, bw, bh))

            if not valid:
                return None

            valid.sort(key=lambda item: item[0], reverse=True)
            _, x, y, bw, bh = valid[0]
            cx = x + bw // 2 + panel_x1
            cy = y + bh // 2 + panel_y1
            logger.info(f"_find_green_npc_on_map: green NPC text found at screen ({cx}, {cy})")
            return (cx, cy)
        except Exception as e:
            logger.warning(f"_find_green_npc_on_map error: {e}")
            return None

    def _is_npc_dialog_open(self, frame=None) -> bool:
        """
        Kiểm tra xem hộp thoại NPC có đang thực sự mở hay không:
        1. Template dialog_box hoặc npc_header_da_tau tìm thấy.
        2. Hoặc OCR phát hiện các từ khóa đối thoại đặc trưng trong vùng hộp thoại (y: 500..850).
        Tuyệt đối không dựa vào các đốm màu HSV trên nền đất!
        """
        try:
            if hasattr(self, "_vision"):
                if self._vision.find_template("dialog_box").found:
                    return True
        except Exception:
            pass

        if frame is None:
            if hasattr(self, "_screen") and self._screen is not None:
                frame = self._screen.capture()
            else:
                return False
        if frame is None or not hasattr(self, "_text") or self._text is None:
            return False

        boxes = self._text.read(frame, (150, 320, 1450, 850))
        for b in boxes:
            norm = normalize_game_text(b.text)
            if any(w in norm for w in (
                "hoang thuong", "trieu dinh", "cong hien", "xem tinh hinh",
                "hung thu", "ket thuc", "doi thoai", "hoan thanh", "tra nhiem vu",
                "chua hoan thanh", "ngay mai hay den", "tien do", "vi trieu dinh"
            )):
                return True
        return False

    def _open_commerce_chief_dialog(self, ctx: StateContext) -> bool:
        """
        Mở hộp thoại đối thoại với NPC Chủ thương hội:
        1. Nếu hộp thoại đã mở sẵn -> Trả về True ngay.
        2. Nếu chưa mở: Thử tap NPC Chủ thương hội (thanh mục tiêu NPC phía trên, tên NPC trên màn hình, hoặc tọa độ trước mặt 780, 380).
        3. Chờ và xác nhận hộp thoại thực sự mở ra trước khi tiếp tục.
        """
        if self._is_npc_dialog_open():
            logger.info(f"[{ctx.account_id}] Hộp thoại đối thoại với Chủ thương hội đã mở sẵn.")
            return True

        logger.info(f"[{ctx.account_id}] Hộp thoại NPC chưa mở, tiến hành tap mở đối thoại với Chủ thương hội...")
        for attempt in range(4):
            scr = None
            if hasattr(self, "_screen") and self._screen is not None:
                scr = self._screen.capture()
            tapped = False
            if scr is not None and hasattr(self, "_text") and self._text is not None:
                # Cách a: Tìm nút NPC mục tiêu ở thanh trên (450..650, 40..120)
                top_boxes = self._text.read(scr, (450, 40, 650, 120))
                for b in top_boxes:
                    norm = normalize_game_text(b.text)
                    if any(w in norm for w in ["npc", "thuong hoi", "chu thuong hoi", "doi thoai"]):
                        logger.info(f"[{ctx.account_id}] Tap vào mục tiêu NPC trên thanh trạng thái tại ({b.center_x}, {b.center_y})...")
                        self._input.tap(b.center_x, b.center_y)
                        tapped = True
                        break

                if not tapped:
                    # Cách b: Tìm tên 'Chủ thương hội' / 'Thương hội' xung quanh nhân vật trên màn hình
                    body_boxes = self._text.read(scr, (250, 150, 1350, 750))
                    for b in body_boxes:
                        norm = normalize_game_text(b.text)
                        if any(w in norm for w in ["chu thuong hoi", "thuong hoi"]):
                            logger.info(f"[{ctx.account_id}] Tap NPC '{b.text}' tại ({b.center_x}, {b.center_y + 40})...")
                            self._input.tap(b.center_x, b.center_y + 40)
                            tapped = True
                            break

            if not tapped:
                # Cách c: Tap các vị trí NPC xung quanh nhân vật
                taps = [(780, 380), (800, 350), (750, 420), (850, 400)]
                tx, ty = taps[attempt % len(taps)]
                logger.info(f"[{ctx.account_id}] Tap vị trí NPC Chủ thương hội tại ({tx}, {ty}) (lần {attempt + 1})...")
                self._input.tap(tx, ty)

            time.sleep(0.6)
            if self._is_npc_dialog_open():
                logger.info(f"[{ctx.account_id}] Mở hộp thoại NPC Chủ thương hội thành công!")
                return True

        logger.warning(f"[{ctx.account_id}] Không thể mở hộp thoại NPC Chủ thương hội sau 4 lần thử!")
        return False

    def _find_dialog_buttons(self, frame=None) -> list[tuple[int, int]]:
        """
        Tìm tọa độ các nút lựa chọn trong hộp thoại NPC (màu vàng/nâu).
        Chỉ quét khi hộp thoại NPC thực sự mở để tránh nhận nhầm gạch lát nền.
        """
        try:
            if not self._is_npc_dialog_open(frame):
                return []

            import cv2
            import numpy as np

            if frame is None:
                frame = self._screen.capture()
            if frame is None:
                return []

            h, w = frame.shape[:2]
            roi_x1, roi_y1 = int(w * 0.05), int(h * 0.40)
            roi_x2, roi_y2 = int(w * 0.95), int(h * 0.85)
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            lower = np.array([15, 60, 80])
            upper = np.array([35, 220, 200])
            mask = cv2.inRange(hsv, lower, upper)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            buttons = []
            for c in contours:
                area = cv2.contourArea(c)
                x, y, bw, bh = cv2.boundingRect(c)
                if area > 800 and bw > 80 and bh > 20:
                    cx = x + bw // 2 + roi_x1
                    cy = y + bh // 2 + roi_y1
                    buttons.append((cx, cy))

            buttons.sort(key=lambda p: (p[1], p[0]))
            return buttons
        except Exception as e:
            logger.debug(f"_find_dialog_buttons error: {e}")
            return []

    def _press_buff_speed(self) -> None:
        """Nhấn nút Buff để tăng tốc độ di chuyển."""
        try:
            coords = self._loader.load_ui_coords()
            buff_cfg = coords.get("commerce_quest", {}).get("buff_button", {"x": 1308, "y": 160})
            bx = buff_cfg.get("x", 1308)
            by = buff_cfg.get("y", 160)
            logger.info(f"⚡ Bấm nút Buff tăng tốc di chuyển tại ({bx}, {by})...")
            self._input.tap(bx, by)
            time.sleep(0.1)
        except Exception as e:
            logger.debug(f"_press_buff_speed error: {e}")

    def _get_instance_index(self, ctx: Optional[StateContext] = None) -> int:
        """Lấy index của máy ảo LDPlayer từ instance cấu hình hoặc serial ADB thực tế."""
        if hasattr(self, "_target_instance_index") and self._target_instance_index is not None:
            return int(self._target_instance_index)

        serial = getattr(self._adb, '_device_serial', '') or getattr(self._adb, '_serial', '')
        if "emulator-" in serial:
            try:
                port = int(serial.split("-")[1])
                return max(0, (port - 5554) // 2)
            except Exception:
                pass
        elif "127.0.0.1:" in serial:
            try:
                port = int(serial.split(":")[1])
                return max(0, (port - 5555) // 2)
            except Exception:
                pass

        try:
            return int(getattr(self._app_cfg.emulator, "instance_id", 0))
        except Exception:
            pass

        return 0

    def _type_unicode_text(self, ctx: StateContext, text: str) -> None:
        """
        Nhập chuỗi Unicode tiếng Việt có dấu (ví dụ: 'Chủ thương hội', 'Nạn dân Đinh')
        vào ô input đang focus.
        Ưu tiên dùng LDPlayer ldconsole action call.input (gõ tiếng Việt có dấu 100%).
        Fallback nếu không có LDPlayer CLI: dùng ADB input text (bỏ dấu).
        """
        idx = self._get_instance_index(ctx)
        typed_ok = False
        try:
            if hasattr(self, "_ldplayer") and self._ldplayer:
                typed_ok = self._ldplayer.input_text(idx, text)
        except Exception as e:
            logger.warning(f"LDPlayer input_text failed: {e}")

        if not typed_ok:
            norm = normalize_game_text(text)
            self._adb.input_text(norm.title())

    def _search_and_select_npc(self, ctx: StateContext, npc_name: str) -> bool:
        """
        Tìm kiếm và chọn NPC trong tab 'Tìm NPC' trên bản đồ bằng ô input tìm kiếm.
        Click vào ô tìm kiếm, gõ chính xác tên NPC CÓ DẤU TIẾNG VIỆT (ví dụ: 'Chủ thương hội'),
        chọn dòng kết quả đầu tiên.
        Tuyệt đối không cuộn danh sách và không dùng tọa độ fallback cũ (tránh chạy lạc sang Vũ Di Sơn).
        """
        coords = self._loader.load_ui_coords()
        nav_cfg = coords.get("navigation", {})
        find_npc_x = nav_cfg.get("map_tab_find_npc", {}).get("x", 376)
        find_npc_y = nav_cfg.get("map_tab_find_npc", {}).get("y", 161)
        search_x = nav_cfg.get("map_search_input", {}).get("x", 266)
        search_y = nav_cfg.get("map_search_input", {}).get("y", 222)

        # 1. Bấm tab Tìm NPC
        logger.info(f"[{ctx.account_id}] Bấm tab Tìm NPC ({find_npc_x}, {find_npc_y})")
        self._input.tap(find_npc_x, find_npc_y)
        time.sleep(0.3)

        # 2. Bấm vào giữa ô input tìm kiếm
        logger.info(f"[{ctx.account_id}] Click vào ô input tìm kiếm NPC ({search_x}, {search_y})...")
        self._input.tap(search_x, search_y)
        time.sleep(0.3)

        # 3. Xóa text cũ trong input (gửi 15 phím Backspace)
        try:
            del_keys = " ".join(["67"] * 15)
            self._adb._run(["shell", f"input keyevent {del_keys}"])
        except Exception:
            pass
        time.sleep(0.15)

        # 4. Gõ chính xác tên NPC CÓ DẤU TIẾNG VIỆT (ví dụ: 'Chủ thương hội', 'Bách Hiểu Sinh')
        logger.info(f"[{ctx.account_id}] Gõ chính xác tên NPC có dấu: '{npc_name}'...")
        self._type_unicode_text(ctx, npc_name)
        time.sleep(0.5)

        # 4b. Gửi phím Enter để submit tìm kiếm nếu cần
        try:
            self._adb._run(["shell", "input keyevent 66"])
        except Exception:
            pass
        time.sleep(0.3)

        # 5. Tự động đóng các popup che khuất (Cài đặt, Kênh chat) nếu có
        self._dismiss_unexpected_popups()
        time.sleep(0.15)

        # 6. Tìm kiếm NPC trong kết quả / danh sách
        search_kw = normalize_game_text(npc_name)
        stop_words = {"chu", "ong", "ba", "co", "tai", "ve", "dao", "nhan", "nguoi"}
        core_words = [w for w in search_kw.split() if w not in stop_words]

        scr = self._screen.capture()
        found_target = False
        result_pos = None

        if scr is not None:
            # 6a. Quét vùng danh sách kết quả (70..465, 250..865)
            boxes = self._text.read(scr, (70, 250, 465, 865))
            for b in boxes:
                norm_b = normalize_game_text(b.text)
                is_match = (
                    search_kw in norm_b
                    or norm_b in search_kw
                    or (core_words and all(w in norm_b for w in core_words))
                    or fuzzy_score(norm_b, search_kw) >= 0.78
                )
                if is_match:
                    result_pos = (b.center_x, b.center_y)
                    found_target = True
                    logger.info(f"[{ctx.account_id}] OCR nhận diện kết quả tìm kiếm NPC '{b.text}' tại {result_pos}")
                    break

            # 6b. Nếu chưa thấy, thử cuộn nhẹ danh sách xuống để tìm nếu danh sách dài
            if not found_target:
                self._input.swipe(250, 700, 250, 350, duration=250)
                time.sleep(0.3)
                scr_scrolled = self._screen.capture()
                if scr_scrolled is not None:
                    boxes_scrolled = self._text.read(scr_scrolled, (70, 250, 465, 865))
                    for b in boxes_scrolled:
                        norm_b = normalize_game_text(b.text)
                        is_match = (
                            search_kw in norm_b
                            or norm_b in search_kw
                            or (core_words and all(w in norm_b for w in core_words))
                            or fuzzy_score(norm_b, search_kw) >= 0.78
                        )
                        if is_match:
                            result_pos = (b.center_x, b.center_y)
                            found_target = True
                            logger.info(f"[{ctx.account_id}] Sau khi cuộn, OCR nhận diện NPC '{b.text}' tại {result_pos}")
                            break

            # 6c. Nếu vẫn không thấy trong danh sách, quét trực tiếp tên NPC trên bản đồ (450..1400, 150..800)
            if not found_target:
                map_boxes = self._text.read(scr, (450, 150, 1400, 800))
                for b in map_boxes:
                    norm_b = normalize_game_text(b.text)
                    is_match = (
                        search_kw in norm_b
                        or (core_words and all(w in norm_b for w in core_words))
                        or fuzzy_score(norm_b, search_kw) >= 0.75
                    )
                    if is_match:
                        result_pos = (b.center_x, b.center_y)
                        found_target = True
                        logger.info(f"[{ctx.account_id}] Nhận diện vị trí NPC '{b.text}' trực tiếp trên bản đồ tại {result_pos}")
                        break

        if found_target and result_pos:
            logger.info(f"[{ctx.account_id}] Click vào NPC '{npc_name}' tại {result_pos}")
            self._input.tap(*result_pos)
            time.sleep(0.25)
            return True

        logger.warning(f"[{ctx.account_id}] Không tìm thấy NPC '{npc_name}' trong danh sách tìm kiếm!")
        return False

    def _is_near_commerce_chief(self) -> bool:
        """
        Kiểm tra xem nhân vật hiện tại có đang đứng cạnh NPC Chủ thương hội hay không:
        1. Hộp thoại đối thoại đang mở sẵn.
        2. Nhìn thấy tên NPC 'Chủ thương hội' trên màn hình xung quanh nhân vật.
        3. Tọa độ góc trên phải (1350..1600, 30..150) hiển thị gần (192, 183) tại Tuyền Châu.
        """
        try:
            # 1. Hộp thoại đối thoại đã mở sẵn
            if self._is_npc_dialog_open():
                logger.info("_is_near_commerce_chief: Phát hiện hộp thoại đối thoại đang mở sẵn!")
                return True

            try:
                if hasattr(self, "_vision") and self._vision.find_template("dialog_box").found:
                    logger.info("_is_near_commerce_chief: Phát hiện template dialog_box đang mở!")
                    return True
            except Exception:
                pass

            scr = self._screen.capture()
            if scr is None:
                return False

            # 2. Nhìn thấy tên NPC 'Chủ thương hội' trên màn hình
            boxes = self._text.read(scr, (350, 100, 1250, 650))
            for b in boxes:
                norm = normalize_game_text(b.text)
                if "chu thuong hoi" in norm:
                    logger.info(f"_is_near_commerce_chief: Nhìn thấy NPC '{b.text}' tại ({b.center_x}, {b.center_y})!")
                    return True

            # 3. Kiểm tra tọa độ góc trên phải (Minimap) của 5 thành phố có Chủ thương hội:
            # Biện Kinh (170, 182), Thành Đô (237, 218), Tương Dương (182, 182), Dương Châu (204, 196), Tuyền Châu (192, 182)
            coord_boxes = self._text.read(scr, (1350, 20, 1600, 150))
            for b in coord_boxes:
                digits = "".join(ch for ch in b.text if ch.isdigit())
                for cx, cy in [(170, 182), (237, 218), (182, 182), (204, 196), (192, 182)]:
                    if any(k in b.text for k in [
                        f"{cx}/{cy}", f"{cx-1}/{cy}", f"{cx+1}/{cy}",
                        f"{cx}/{cy-1}", f"{cx}/{cy+1}", f"{cx}/{cy-2}", f"{cx}/{cy+2}",
                        f"{cx-1}/{cy-1}", f"{cx+1}/{cy+1}", f"{cx-1}/{cy-2}", f"{cx+1}/{cy-2}"
                    ]):
                        logger.info(f"_is_near_commerce_chief: Đang ở đúng tọa độ Chủ thương hội ({b.text}) tại ({cx}/{cy})!")
                        return True
                    if str(cx) in digits and any(str(cy + offset) in digits for offset in (-2, -1, 0, 1, 2)):
                        logger.info(f"_is_near_commerce_chief: Đang ở đúng tọa độ Chủ thương hội ({digits}) tại ({cx}/{cy})!")
                        return True

            return False
        except Exception as e:
            logger.debug(f"_is_near_commerce_chief error: {e}")
            return False

    def _detect_current_city(self, frame=None) -> Optional[str]:
        """Đọc tên thành phố hiện tại từ góc trên phải (Minimap)."""
        if frame is None:
            frame = self._screen.capture()
        if frame is None:
            return None
        boxes = self._text.read(frame, (1350, 10, 1600, 120))
        for b in boxes:
            norm = normalize_game_text(b.text)
            for city_key, city_name in [
                ("tuyen chau", "Tuyền Châu"),
                ("bien kinh", "Biện Kinh"),
                ("thanh do", "Thành Đô"),
                ("tuong duong", "Tương Dương"),
                ("duong chau", "Dương Châu"),
            ]:
                if city_key in norm:
                    return city_name
        return None

    def _navigate_to_coordinates_on_map(self, target_x: int, target_y: int) -> bool:
        """Nhập tọa độ target_x, target_y vào ô X và Y trên bản đồ hiện tại, sau đó bấm Xác nhận.
        LƯU Ý: MỌI THAO TÁC ĐỀU CLICK TRỰC TIẾP VÀO TỪNG Ô INPUT RỒI NHẬP, TUYỆT ĐỐI KHÔNG DÙNG TAB.
        """
        coords = self._loader.load_ui_coords()
        nav = coords.get("navigation", {})
        c_cfg = coords.get("commerce_quest", {})
        kx = c_cfg.get("map_coordinate_x", nav.get("map_coordinate_x", {"x": 704, "y": 778}))
        ky = c_cfg.get("map_coordinate_y", nav.get("map_coordinate_y", {"x": 842, "y": 778}))
        kconf = c_cfg.get("map_coordinate_confirm", nav.get("map_coordinate_confirm", {"x": 1015, "y": 778}))

        x_pos = (kx.get("x", 704), kx.get("y", 778))
        y_pos = (ky.get("x", 842), ky.get("y", 778))
        conf_pos = (kconf.get("x", 1015), kconf.get("y", 778))

        # 1. Click TRỰC TIẾP vào ô input X, xóa sạch và gõ X
        logger.info(f"Click trực tiếp ô input X tại {x_pos} để nhập {target_x}...")
        self._input.tap(x_pos[0], x_pos[1])
        time.sleep(0.4)
        try:
            self._adb._run(["shell", "input keyevent 123; " + "; ".join(["input keyevent 67"] * 10)])
        except Exception:
            pass
        time.sleep(0.2)
        self._adb.input_text(str(target_x))
        time.sleep(0.4)

        # 2. Click TRỰC TIẾP vào ô input Y (không dùng Tab), xóa sạch và gõ Y
        logger.info(f"Click TRỰC TIẾP ô input Y tại {y_pos} để nhập {target_y}...")
        self._input.tap(y_pos[0], y_pos[1])
        time.sleep(0.4)
        try:
            self._adb._run(["shell", "input keyevent 123; " + "; ".join(["input keyevent 67"] * 10)])
        except Exception:
            pass
        time.sleep(0.2)
        self._adb.input_text(str(target_y))
        time.sleep(0.4)

        # 3. Kiểm tra an toàn: nếu ô Y vô tình bị rỗng, click lại ô Y để nhập bù trước khi bấm Xác nhận
        try:
            if hasattr(self, "_screen") and hasattr(self, "_text") and self._screen and self._text:
                scr = self._screen.capture()
                if scr is not None:
                    crop = scr[740:820, 780:920]
                    y_boxes = self._text.read(crop)
                    y_text = "".join(b.text for b in y_boxes)
                    if not any(ch.isdigit() for ch in y_text):
                        logger.warning(f"Ô input Y chưa có số! Click lại {y_pos} để nhập lại {target_y}...")
                        self._input.tap(y_pos[0], y_pos[1])
                        time.sleep(0.4)
                        self._adb.input_text(str(target_y))
                        time.sleep(0.3)
        except Exception as exc:
            logger.debug(f"Check ô Y OCR error: {exc}")

        # 4. Click nút Xác nhận
        logger.info(f"Click nút Xác nhận tại {conf_pos}...")
        self._input.tap(conf_pos[0], conf_pos[1])
        time.sleep(0.5)
        return True

    def _input_search_text(self, text: str, telex: str = "", ctx: Optional[StateContext] = None) -> None:
        """
        Nhập nội dung tìm kiếm vào input:
        1. Thử LDPlayer action call.input (hỗ trợ tiếng Việt có dấu 100%).
        2. Thử copy-paste (hỗ trợ tiếng Việt có dấu chuẩn).
        3. Nếu paste chưa nhận text, gõ chuỗi Telex theo đúng vị trí dấu như người dùng chỉ định.
        4. Gửi phím Enter (keycode 66) để kích hoạt tìm kiếm.
        """
        typed_ok = False
        if ctx:
            try:
                idx = self._get_instance_index(ctx)
                if hasattr(self, "_ldplayer") and self._ldplayer:
                    typed_ok = self._ldplayer.input_text(idx, text)
                    time.sleep(0.3)
            except Exception:
                pass

        if not typed_ok:
            # Thử đưa vào clipboard Windows (LDPlayer tự đồng bộ vào clipboard Android)
            try:
                import tkinter as tk
                r = tk.Tk()
                r.withdraw()
                r.clipboard_clear()
                r.clipboard_append(text)
                r.update()
                r.destroy()
                time.sleep(0.1)
            except Exception:
                pass

            try:
                self._adb._run(["shell", "cmd", "clipboard", "set", "text", f'"{text}"'])
            except Exception:
                pass

            pasted = False
            try:
                self._adb.key_event(279)  # KEYCODE_PASTE
                time.sleep(0.25)
                scr = self._screen.capture()
                if scr is not None:
                    boxes = self._text.read(scr, (100, 190, 500, 265))
                    if any(len(b.text.strip()) >= 3 for b in boxes):
                        pasted = True
            except Exception:
                pass

            if not pasted and telex:
                logger.info(f"Gõ tìm kiếm bằng Telex: {telex}")
                escaped = telex.replace(" ", "%s")
                self._adb._run(["shell", f"input text {escaped}"])
                time.sleep(0.3)

        try:
            self._adb.key_event(66)
            time.sleep(0.4)
        except Exception:
            pass

    def _click_map_search_result(self, map_name: str) -> bool:
        """Quét và click kết quả bản đồ trong danh sách bên trái."""
        from .text_recognition import normalize_game_text
        coords = self._loader.load_ui_coords()
        nav_cfg = coords.get("navigation", {})
        region_cfg = nav_cfg.get("map_list_region", {"x1": 75, "y1": 250, "x2": 465, "y2": 865})
        region = (region_cfg.get("x1", 75), region_cfg.get("y1", 250), region_cfg.get("x2", 465), region_cfg.get("y2", 865))

        scr = self._screen.capture()
        if scr is not None:
            boxes = self._text.read(scr, region)
            norm_target = normalize_game_text(map_name)
            for b in boxes:
                norm_b = normalize_game_text(b.text)
                if norm_target in norm_b:
                    if any(dir_w in norm_b for dir_w in ("dong", "tay", "nam", "bac")) and not any(dir_w in norm_target for dir_w in ("dong", "tay", "nam", "bac")):
                        continue
                    logger.info(f"Tìm thấy map '{b.text}' tại ({b.center_x}, {b.center_y}), bấm...")
                    self._input.tap(b.center_x, b.center_y)
                    return True
        return False

    def _click_npc_search_result(self, npc_name: str = "Chu thuong hoi") -> bool:
        """Quét và click kết quả NPC trong danh sách bên trái bằng nhận diện chữ màu xanh lá và OCR."""
        from .text_recognition import normalize_game_text
        import numpy as np
        coords = self._loader.load_ui_coords()
        nav_cfg = coords.get("navigation", {})
        region_cfg = nav_cfg.get("map_list_region", {"x1": 40, "y1": 240, "x2": 430, "y2": 820})
        region = (region_cfg.get("x1", 40), region_cfg.get("y1", 240), region_cfg.get("x2", 430), region_cfg.get("y2", 820))

        scr = self._screen.capture()
        if scr is None:
            return False

        # Cách 1: Quét chữ màu XANH LÁ (đặc trưng kết quả tìm kiếm NPC trong game)
        x1, y1, x2, y2 = region
        roi = scr[y1:y2, x1:x2]
        # Màu xanh lá đậm: G > 160, R < 120, B < 120
        green_mask = (roi[:, :, 1] > 160) & (roi[:, :, 2] < 120) & (roi[:, :, 0] < 120)
        if np.sum(green_mask) >= 60:
            ys, xs = np.where(green_mask)
            click_x = int(np.mean(xs)) + x1
            click_y = int(np.mean(ys)) + y1
            logger.info(f"Tìm thấy NPC 'Chủ thương hội' chữ xanh lá tại ({click_x}, {click_y}), bấm...")
            self._input.tap(click_x, click_y)
            return True

        # Cách 2: Quét OCR tìm chữ "Chủ thương hội" hoặc "Chi thương hội" hoặc "thuong hoi"
        boxes = self._text.read(scr, region)
        norm_target = normalize_game_text(npc_name)
        for b in boxes:
            norm_b = normalize_game_text(b.text)
            if (norm_target in norm_b or "thuong hoi" in norm_b or 
                "chi thuong" in norm_b or "chu th" in norm_b or "chi th" in norm_b):
                logger.info(f"Tìm thấy NPC '{b.text}' bằng OCR tại ({b.center_x}, {b.center_y}), bấm...")
                self._input.tap(b.center_x, b.center_y)
                return True

        return False

    def _navigate_to_tuyen_chau_commerce_chief(self, ctx: StateContext) -> bool:
        """
        Thực hiện đúng 8 bước tìm đường đến Chủ thương hội tại Tuyền Châu:
        B1: Mở bản đồ mini
        B2: Nhấn tìm map
        B3: Nhấn vô input tìm map
        B4: Gõ tìm 'Tuyền Châu' (copy-paste hoặc telex: tuyeefn chaau)
        B5: Nhấn 'Tuyền Châu' trong danh sách kết quả
        B6: Nhấn nút tìm NPC
        B7: Nhấn input tìm NPC
        B8: Gõ tìm 'Chủ thương hội' (copy-paste hoặc telex: chur thuwownwg hooji)
        B9: Nhấn vào kết quả NPC 'Chủ thương hội' để nhân vật tự chạy đến
        """
        logger.info(f"[{ctx.account_id}] Bắt đầu quy trình tìm Chủ thương hội tại Tuyền Châu (8 bước)...")
        activity._write("🏃", f"[{ctx.account_id}] Tim Chu thuong hoi Tuyen Chau...")

        coords = self._loader.load_ui_coords()
        nav_cfg = coords.get("navigation", {})
        search_x = nav_cfg.get("map_search_input", {}).get("x", 266)
        search_y = nav_cfg.get("map_search_input", {}).get("y", 222)

        # Bước 1: Mở bản đồ mini
        logger.info(f"[{ctx.account_id}] Bước 1: Mở bản đồ mini...")
        self._ensure_map_open()
        time.sleep(0.5)

        # Bước 2: Nhấn tìm map
        logger.info(f"[{ctx.account_id}] Bước 2: Nhấn tab 'Tìm map'...")
        find_map_x = nav_cfg.get("map_tab_find_map", {}).get("x", 176)
        find_map_y = nav_cfg.get("map_tab_find_map", {}).get("y", 162)
        self._input.tap(find_map_x, find_map_y)
        time.sleep(0.4)

        # Bước 3: Nhấn vô input tìm map và xóa chữ cũ
        logger.info(f"[{ctx.account_id}] Bước 3: Nhấn vào ô input tìm map...")
        self._input.tap(search_x, search_y)
        time.sleep(0.3)
        try:
            delete_cmd = "input keyevent 123; " + "; ".join(["input keyevent 67"] * 30)
            self._adb._run(["shell", delete_cmd])
            time.sleep(0.2)
        except Exception:
            pass

        # Bước 4: Gõ tìm Tuyền Châu (paste hoặc telex: tuyeefn chaau)
        logger.info(f"[{ctx.account_id}] Bước 4: Gõ tìm 'Tuyền Châu'...")
        self._input_search_text("Tuyền Châu", telex="tuyeefn%schaau", ctx=ctx)
        time.sleep(0.6)

        # Bước 5: Nhấn Tuyền Châu
        logger.info(f"[{ctx.account_id}] Bước 5: Nhấn chọn 'Tuyền Châu' trong danh sách map...")
        map_clicked = self._click_map_search_result("Tuyen Chau")
        if not map_clicked:
            first_res = nav_cfg.get("map_tuyen_chau", {"x": 115, "y": 315})
            self._input.tap(first_res.get("x", 115), first_res.get("y", 315))
        time.sleep(2.0)  # Chờ nhân vật / bản đồ load sang map Tuyền Châu

        # Đảm bảo bản đồ mini vẫn đang mở để tiếp tục tìm NPC
        self._ensure_map_open()
        time.sleep(0.5)

        # Bước 6: Nhấn nút tìm NPC
        logger.info(f"[{ctx.account_id}] Bước 6: Nhấn tab 'Tìm NPC'...")
        find_npc_x = nav_cfg.get("map_tab_find_npc", {}).get("x", 376)
        find_npc_y = nav_cfg.get("map_tab_find_npc", {}).get("y", 161)
        self._input.tap(find_npc_x, find_npc_y)
        time.sleep(0.4)

        # Bước 7: Nhấn input tìm NPC
        logger.info(f"[{ctx.account_id}] Bước 7: Nhấn vào ô input tìm NPC...")
        self._input.tap(search_x, search_y)
        time.sleep(0.3)
        try:
            delete_cmd = "input keyevent 123; " + "; ".join(["input keyevent 67"] * 30)
            self._adb._run(["shell", delete_cmd])
            time.sleep(0.2)
        except Exception:
            pass

        # Bước 8: Gõ tìm Chủ thương hội (paste hoặc telex: chur thuwownwg hooji)
        logger.info(f"[{ctx.account_id}] Bước 8: Gõ tìm 'Chủ thương hội'...")
        self._input_search_text("Chủ thương hội", telex="chur%sthuwownwg%shooji", ctx=ctx)
        time.sleep(0.6)

        # Bước 9: Nhấn vào kết quả NPC Chủ thương hội
        logger.info(f"[{ctx.account_id}] Bước 9: Bấm chọn NPC 'Chủ thương hội' trong danh sách...")
        npc_clicked = self._click_npc_search_result("Chu thuong hoi")
        if not npc_clicked:
            logger.info(f"[{ctx.account_id}] Click tọa độ kết quả NPC Chủ thương hội tại (110, 518)...")
            self._input.tap(110, 518)

        time.sleep(0.5)
        # Đóng bản đồ an toàn
        self._close_map()
        time.sleep(0.2)
        # Buff tốc độ chạy
        self._press_buff_speed()
        return True

    def _handle_find_commerce_chief(self, ctx: StateContext) -> BotState:
        """Tìm và di chuyển đến NPC Chủ thương hội tại Tuyền Châu theo đúng 8 bước chuẩn."""
        # 0. Nếu đã có sẵn nhiệm vụ Thương Hội đang làm dở (và không phải đang trả nhiệm vụ):
        if not ctx.extra.get("returning_quest"):
            active = self._check_active_commerce_quest(ctx)
            if active:
                quest_text, _ = active
                logger.info(f"[{ctx.account_id}] Đã có sẵn nhiệm vụ Thương hội đang làm dở: '{quest_text}'")
                activity._write("📋", f"[{ctx.account_id}] Tiep tuc NV co san: {quest_text}")
                ctx.extra["quest_text"] = quest_text
                time.sleep(0.3)
                self._press_buff_speed()
                return BotState.CLASSIFY_QUEST

        # 0.1 Nếu nhân vật ĐÃ ĐỨNG CẠNH Chủ Thương Hội:
        if self._is_near_commerce_chief():
            logger.info(f"[{ctx.account_id}] Nhân vật đã đứng cạnh Chủ thương hội! Mở hộp thoại nhận nhiệm vụ ngay...")
            activity._write("💬", f"[{ctx.account_id}] Da o canh Chu thuong hoi, hoi thoai nhan NV...")
            self._open_commerce_chief_dialog(ctx)
            return BotState.MEET_NPC

        # 1. Thực hiện quy trình tìm đường 8 bước
        self._navigate_to_tuyen_chau_commerce_chief(ctx)

        # 2. Chờ nhân vật chạy đến đứng cạnh Chủ Thương Hội (tối đa 20s)
        start_t = time.time()
        while time.time() - start_t < 20.0:
            time.sleep(0.8)
            if self._find_dialog_buttons() or self._vision.find_template("dialog_box").found:
                logger.info(f"[{ctx.account_id}] Hộp thoại NPC đã mở sẵn!")
                return BotState.MEET_NPC
            if self._is_near_commerce_chief():
                logger.info(f"[{ctx.account_id}] Nhân vật đã đến cạnh Chủ thương hội!")
                self._open_commerce_chief_dialog(ctx)
                return BotState.MEET_NPC

        logger.info(f"[{ctx.account_id}] Đã kết thúc thời gian chạy đến Chủ thương hội, chuyển sang MEET_NPC.")
        return BotState.MEET_NPC

    def _handle_meet_npc_legacy(self, ctx: StateContext) -> BotState:
        """Đã đến gặp NPC Chủ thương hội, tiến hành hội thoại nhận nhiệm vụ."""
        logger.info(f"[{ctx.account_id}] Gặp NPC Chủ thương hội, tiến hành đối thoại nhận nhiệm vụ...")
        activity._write("💬", f"[{ctx.account_id}] Hoi thoai voi Chu thuong hoi de nhan nhiem vu...")
        ctx.last_action = "meet_npc_commerce_chief"
        self._checkpoint.save(ctx)

        # 1. Nếu hộp thoại chưa mở, tap vào khu vực NPC để mở đối thoại
        time.sleep(1.0)
        buttons = self._find_dialog_buttons()
        has_dialog = (
            len(buttons) > 0
            or self._vision.find_template("dialog_box").found
            or self._vision.find_template("npc_header_da_tau").found
        )

        if not has_dialog:
            logger.info(f"[{ctx.account_id}] Hộp thoại chưa mở, tap NPC tại (800, 450)...")
            self._input.tap(800, 450)
            time.sleep(2.0)

        # 2. Duyệt qua các bước hội thoại để nhận nhiệm vụ
        for step in range(5):
            buttons = self._find_dialog_buttons()
            has_dialog = (
                len(buttons) > 0
                or self._vision.find_template("dialog_box").found
                or self._vision.find_template("npc_header_da_tau").found
            )
            if not has_dialog and step > 0:
                logger.info(f"[{ctx.account_id}] Đối thoại kết thúc sau {step} bước, đã nhận nhiệm vụ thành công!")
                break

            if buttons:
                target_btn = buttons[0]
                logger.info(f"[{ctx.account_id}] Bước thoại {step + 1}: Click nút tại ({target_btn[0]}, {target_btn[1]})...")
                self._input.tap(target_btn[0], target_btn[1])
            else:
                logger.info(f"[{ctx.account_id}] Bước thoại {step + 1}: Tap vùng đối thoại...")
                self._input.tap(800, 550)

            time.sleep(2.0)

        logger.info(f"[{ctx.account_id}] Hoàn thành: Đã hội thoại nhận nhiệm vụ Chủ thương hội!")
        activity._write("✅", f"[{ctx.account_id}] Da nhan nhiem vu Chu thuong hoi thanh cong!")
        time.sleep(1.0)
        return BotState.GET_QUEST


    def _handle_get_quest_legacy(self, ctx: StateContext) -> BotState:
        logger.info(f"[{ctx.account_id}] Getting quest / finding next task...")
        ctx.last_action = "get_quest"
        self._checkpoint.save(ctx)

        # 1. Nếu bảng hội thoại đang mở -> chuyển sang INTERACT_NPC
        if self._vision.find_template("dialog_box").found or self._vision.find_template("npc_header_da_tau").found:
            logger.info(f"[{ctx.account_id}] Dialog already open, processing dialogue.")
            return BotState.INTERACT_NPC

        # 2. Nếu có banner nhiệm vụ ở giữa bên dưới -> tap banner
        banner_res = self._vision.find_template("quest_banner")
        if banner_res.found:
            logger.info(f"[{ctx.account_id}] Tapping quest banner at ({banner_res.center_x}, {banner_res.center_y})")
            self._input.tap(banner_res.center_x, banner_res.center_y)
            time.sleep(1.5)
            return BotState.NAVIGATE

        # 3. Mở bảng nhiệm vụ → tìm tên NPC màu xanh → click để auto-đi đến NPC
        logger.info(f"[{ctx.account_id}] Opening Quest Panel, finding blue NPC link...")
        coords = self._loader.load_ui_coords()
        quest_btn = coords.get("commerce_quest", {}).get("quest_button", {"x": 1404, "y": 755})
        self._input.tap(quest_btn.get("x", 1404), quest_btn.get("y", 755))   # Mở bảng nhiệm vụ
        time.sleep(1.2)
        self._input.tap(300, 275)    # Chọn nhiệm vụ đầu tiên trong danh sách
        time.sleep(1.0)

        # Tìm và click tên NPC màu xanh trong mô tả nhiệm vụ
        blue_link = self._find_blue_npc_link()
        if blue_link:
            logger.info(f"[{ctx.account_id}] Blue NPC link found at {blue_link}, clicking...")
            self._input.tap(blue_link[0], blue_link[1])
        else:
            logger.warning(f"[{ctx.account_id}] No blue link found, falling back to quest panel close")
            self._input.tap(1470, 95)  # fallback: đóng panel

        time.sleep(1.0)
        return BotState.NAVIGATE

    def _find_blue_npc_link(self) -> tuple[int, int] | None:
        """
        Chụp màn hình, quét vùng bảng nhiệm vụ để tìm text màu XANH LÁ
        là tên NPC có thể click để auto-đi đến.

        Màu xanh lá link NPC trong game Tình Kiếm: HSV hue ~35-85 (green)
        """
        try:
            import cv2
            import numpy as np

            frame = self._screen.capture()
            if frame is None:
                return None

            # Vùng bảng nhiệm vụ (điều chỉnh nếu cần)
            panel_x1, panel_y1 = 60,  180
            panel_x2, panel_y2 = 600, 520
            roi = frame[panel_y1:panel_y2, panel_x1:panel_x2]

            # Chuyển sang HSV để lọc màu XANH LÁ của tên NPC
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Xanh lá (green): Hue 35-85, Saturation 80+, Value 80+
            lower_green = np.array([35, 80, 80])
            upper_green = np.array([85, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)

            # Tìm contours của vùng màu xanh lá
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                logger.debug("_find_blue_npc_link: no green pixels found in panel region")
                return None

            # Lấy contour lớn nhất (tên NPC là cụm text dài nhất)
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) < 30:
                return None

            # Tính tọa độ trung tâm, cộng offset ROI để ra tọa độ màn hình thật
            M = cv2.moments(largest)
            if M["m00"] == 0:
                return None
            cx = int(M["m10"] / M["m00"]) + panel_x1
            cy = int(M["m01"] / M["m00"]) + panel_y1
            logger.debug(f"_find_blue_npc_link: green NPC link found at screen ({cx}, {cy})")
            return (cx, cy)

        except Exception as e:
            logger.warning(f"_find_blue_npc_link error: {e}")
            return None



    def _handle_classify_quest_legacy(self, ctx: StateContext) -> BotState:
        logger.info(f"[{ctx.account_id}] Classifying quest...")
        quest_type = self._classifier.classify()
        ctx.current_quest_type = quest_type
        ctx.last_action = f"classify:{quest_type}"
        self._checkpoint.save(ctx)

        if quest_type == ctx.current_quest_type.__class__.UNKNOWN:
            logger.warning(f"[{ctx.account_id}] Quest UNKNOWN – cancelling")
            return BotState.CANCEL_QUEST

        return BotState.NAVIGATE

    def _handle_meet_npc(self, ctx: StateContext) -> BotState:
        """Đã đến gặp NPC Chủ thương hội: kiểm tra trả nhiệm vụ hoặc nhận nhiệm vụ mới."""
        logger.info(f"[{ctx.account_id}] Đang tương tác với NPC Chủ thương hội...")
        time.sleep(0.15)

        # 0. Nếu không phải đang đi trả nhiệm vụ mà ĐÃ CÓ SẴN nhiệm vụ:
        # Thì đi làm nhiệm vụ ngay, không cố bấm nhận mới!
        if not ctx.extra.get("returning_quest"):
            active = self._check_active_commerce_quest(ctx)
            if active:
                quest_text, click_pos = active
                logger.info(f"[{ctx.account_id}] Đã có sẵn nhiệm vụ Thương hội đang làm: '{quest_text}'")
                activity._write("📋", f"[{ctx.account_id}] Tiep tuc NV co san: {quest_text}")
                ctx.extra["quest_text"] = quest_text
                time.sleep(0.3)
                self._press_buff_speed()
                return BotState.CLASSIFY_QUEST

        # 1. Đảm bảo hộp thoại NPC Chủ Thương Hội thực sự được mở
        if not self._open_commerce_chief_dialog(ctx):
            logger.warning(f"[{ctx.account_id}] Không mở được hộp thoại với Chủ thương hội, quay lại FIND_COMMERCE_CHIEF...")
            return BotState.FIND_COMMERCE_CHIEF

        # 2. Xử lý nút HOÀN THÀNH NHIỆM VỤ (Cách 3 & Trả nhiệm vụ)
        # Quét các nút đối thoại xem có "Hoàn thành nhiệm vụ" / "Trả nhiệm vụ" không
        complete_box = None
        boxes = self._text.read(self._screen.capture(), (150, 450, 1450, 800))
        for b in boxes:
            norm = normalize_game_text(b.text)
            if any(w in norm for w in ("hoan thanh", "tra nhiem vu", "giao nhiem vu")):
                complete_box = b
                break

        # Nếu có nút Hoàn thành (do đang đi trả hoặc kiểm tra theo Cách 3)
        if complete_box:
            logger.info(f"[{ctx.account_id}] Bấm nút '{complete_box.text}' tại ({complete_box.center_x}, {complete_box.center_y})...")
            self._input.tap(complete_box.center_x, complete_box.center_y)
            time.sleep(0.35)

            # CÁCH 3: Kiểm tra thông báo/hộp thoại sau khi bấm Hoàn thành nhiệm vụ
            resp_boxes = self._text.read(self._screen.capture(), (150, 250, 1450, 750))
            resp_raw = " ".join(box.text for box in resp_boxes)
            resp_norm = normalize_game_text(resp_raw)

            if any(w in resp_norm for w in ("chua hoan thanh", "chua xong", "tien do", "nguoi chua", "chua the hoan thanh", "chua mang ve", "chua tim duoc", "mau di", "chua hoan tat")):
                logger.warning(f"[{ctx.account_id}] Cách 3: Hộp thoại Chủ thương hội thông báo: '{resp_raw}' -> Nhiệm vụ ĐÃ NHẬN và CHƯA HOÀN THÀNH, cần đi thực hiện nhiệm vụ!")
                activity._write("⚠️", f"[{ctx.account_id}] NV da nhan chua xong: di thuc hien...")
                ctx.extra.pop("returning_quest", None)

                # Bấm 'Kết thúc đối thoại' để đóng hoàn toàn hộp thoại NPC
                closed_btn = False
                for b in resp_boxes:
                    b_norm = normalize_game_text(b.text)
                    if "ket thuc" in b_norm or "doi thoai" in b_norm:
                        logger.info(f"[{ctx.account_id}] Bấm nút '{b.text}' tại ({b.center_x}, {b.center_y})...")
                        self._input.tap(b.center_x, b.center_y)
                        closed_btn = True
                        break
                if not closed_btn:
                    self._input.tap(255, 635)
                time.sleep(0.5)
                # Đóng bất kỳ popup thừa nếu có
                self._input.tap(800, 180)
                time.sleep(0.3)

                # Lấy thông tin nhiệm vụ bằng Cách 2 (bảng nhiệm vụ)
                active = self._check_active_commerce_quest(ctx)
                if active:
                    quest_text, click_pos = active
                    logger.info(f"[{ctx.account_id}] Lấy được chi tiết NV: '{quest_text}'")
                    ctx.extra["quest_text"] = quest_text
                else:
                    ctx.extra["quest_text"] = resp_raw or "Nhiệm vụ Thương hội"

                time.sleep(0.3)
                self._press_buff_speed()
                return BotState.CLASSIFY_QUEST

            if any(w in resp_norm for w in ("hoan thanh het", "ngay mai hay den", "khong con nhiem vu", "ngay mai", "khong the nhan them", "da hoan thanh 20", "da nhan toi da", "het nhiem vu")):
                logger.info(f"[{ctx.account_id}] Chủ thương hội báo đã hết nhiệm vụ hôm nay! Đăng xuất...")
                activity._write("🏁", f"[{ctx.account_id}] Chu thuong hoi bao da het nhiem vu hom nay. Dang xuat...")
                self._save_progress(ctx, status_text="Hết nhiệm vụ hôm nay 🏁")
                return BotState.LOGOUT

            # Nếu thực sự hoàn thành thành công:
            if ctx.extra.get("returning_quest") or "chuc mung" in resp_norm or "phan thuong" in resp_norm or "thanh cong" in resp_norm:
                ctx.total_quests_done += 1
                ctx.extra.pop("returning_quest", None)
                username = self._get_username(ctx.account_id)
                q_type = ctx.current_quest_type.value if ctx.current_quest_type else "NPC"
                activity.quest_done(username, ctx.total_quests_done, q_type)
                self._checkpoint.save(ctx)

                history = ctx.extra.setdefault("quest_history_list", [])
                history.append({
                    "num": ctx.total_quests_done,
                    "type": q_type,
                    "detail": ctx.extra.get("quest_detail_text", ctx.extra.get("quest_text", "")),
                    "status": "Thành công ✅",
                    "time": datetime.now().strftime("%H:%M:%S")
                })
                self._save_progress(ctx, status_text=f"Hoàn thành NV #{ctx.total_quests_done}/20")
                logger.info(f"[{ctx.account_id}] ===> HOÀN THÀNH NHIỆM VỤ {ctx.total_quests_done}/20 <===")

                if ctx.total_quests_done >= 20:
                    logger.info(f"[{ctx.account_id}] Đã đạt tối đa 20/20 nhiệm vụ Thương Hội hôm nay! Hoàn tất.")
                    activity._write("🏆", f"[{ctx.account_id}] Da hoan thanh 20/20 nhiem vu Thuong Hoi! Dang xuat...")
                    self._save_progress(ctx, status_text="Đã hoàn thành 20/20 🎉")
                    return BotState.LOGOUT

                logger.info(f"[{ctx.account_id}] Tiếp tục nhận nhiệm vụ thứ {ctx.total_quests_done + 1}/20...")
                time.sleep(0.2)
                self._open_commerce_chief_dialog(ctx)
                time.sleep(0.25)

        # 3. NHẬN NHIỆM VỤ MỚI: Chọn "Vì triều đình cống hiến sức lực"
        logger.info(f"[{ctx.account_id}] Chọn 'Vì triều đình cống hiến sức lực'...")
        found_contrib = False
        boxes = self._text.read(self._screen.capture(), (150, 450, 1450, 800))
        for b in boxes:
            norm = normalize_game_text(b.text)
            if "vi trieu dinh" in norm or "cong hien" in norm:
                logger.info(f"[{ctx.account_id}] Bấm nút '{b.text}' tại ({b.center_x}, {b.center_y})")
                self._input.tap(b.center_x, b.center_y)
                found_contrib = True
                break
            elif "xem tinh hinh" in norm:
                logger.info(f"[{ctx.account_id}] Phát hiện nút 'Xem tình hình nhiệm vụ' -> Đang có nhiệm vụ dở dang!")
                self._input.tap(b.center_x, b.center_y)
                found_contrib = True
                break

        if not found_contrib:
            active = self._check_active_commerce_quest(ctx)
            if active:
                quest_text, click_pos = active
                logger.info(f"[{ctx.account_id}] Chưa nhận được NV mới nhưng có sẵn NV đang làm: '{quest_text}'")
                ctx.extra["quest_text"] = quest_text
                time.sleep(0.3)
                self._press_buff_speed()
                return BotState.CLASSIFY_QUEST

            coords = self._loader.load_ui_coords()
            fallback = coords.get("commerce_quest", {}).get("contribute_fallback", {"x": 330, "y": 575})
            fx, fy = fallback.get("x", 330), fallback.get("y", 575)
            logger.info(f"[{ctx.account_id}] Bấm vị trí fallback 'Vì triều đình cống hiến sức lực' ({fx}, {fy})...")
            self._input.tap(fx, fy)

        time.sleep(0.25)
        return BotState.GET_QUEST

    def _handle_get_quest(self, ctx: StateContext) -> BotState:
        """
        Đọc nội dung nhiệm vụ được giao từ hộp thoại Chủ thương hội (vùng hộp thoại y: 320..750).
        Nếu hộp thoại không có nội dung quest hợp lệ:
        - Kiểm tra bảng nhiệm vụ (Cách 2).
        - Nếu không có nhiệm vụ: Chưa nhận được nhiệm vụ! Quay lại MEET_NPC để nhận lại.
        """
        time.sleep(0.2)
        # 1. Đọc vùng văn bản trong hộp thoại đối thoại của NPC (bao gồm cả dòng mô tả NV và các nút)
        region = (150, 320, 1450, 750)
        boxes = self._text.read(self._screen.capture(), region)
        quest_raw = " ".join(box.text for box in boxes) if boxes else ""
        norm = normalize_game_text(quest_raw)
        logger.info(f"[{ctx.account_id}] Nội dung hộp thoại đọc được: '{quest_raw}'")

        # 2. Kiểm tra xem có phải thông báo đã hết nhiệm vụ hôm nay không
        if any(w in norm for w in ("hoan thanh het", "ngay mai hay den", "khong con nhiem vu", "ngay mai")) or is_explicit_quest_exhausted(quest_raw):
            logger.info(f"[{ctx.account_id}] Chủ thương hội báo đã hết nhiệm vụ hôm nay! Đăng xuất...")
            activity._write("🏁", f"[{ctx.account_id}] Chu thuong hoi bao da het nhiem vu hom nay. Dang xuat...")
            self._save_progress(ctx, status_text="Hết nhiệm vụ hôm nay 🏁")
            return BotState.LOGOUT

        # 3. Kiểm tra nếu có sẵn nhiệm vụ đang làm dở được Chủ thương hội nhắc
        if any(w in norm for w in ("chua hoan thanh", "mau di", "tinh hinh", "nhan da den", "hang da den")):
            active = self._check_active_commerce_quest(ctx)
            if active:
                quest_text, _ = active
                logger.info(f"[{ctx.account_id}] Tiếp tục nhiệm vụ đang làm dở: '{quest_text}'")
                ctx.extra["quest_text"] = quest_text
                ctx.last_action = "read_commerce_quest"
                self._checkpoint.save(ctx)
                time.sleep(0.3)
                self._press_buff_speed()
                return BotState.CLASSIFY_QUEST

        # 4. Kiểm tra xem nội dung hộp thoại có thực sự là nhiệm vụ Thương Hội hay không
        valid_quest_type = self._classifier.classify(text=quest_raw)
        if valid_quest_type != QuestType.UNKNOWN:
            logger.info(f"[{ctx.account_id}] Nhận nhiệm vụ thành công từ hộp thoại ({valid_quest_type.value}): '{quest_raw}'")
            ctx.extra["quest_text"] = quest_raw
            ctx.last_action = "read_commerce_quest"
            self._checkpoint.save(ctx)
            # Bấm "Kết thúc đối thoại"
            end_btn = None
            for b in boxes:
                if "ket thuc" in normalize_game_text(b.text):
                    end_btn = (b.center_x, b.center_y)
                    break
            if not end_btn:
                end_btn = (280, 575)
            self._input.tap(*end_btn)
            time.sleep(0.2)
            return BotState.CLASSIFY_QUEST

        # 5. Nếu hộp thoại chưa đọc được, kiểm tra bảng nhiệm vụ (Cách 2)
        active = self._check_active_commerce_quest(ctx)
        if active:
            quest_text, _ = active
            logger.info(f"[{ctx.account_id}] Nhận diện nhiệm vụ thành công từ danh sách nhiệm vụ: '{quest_text}'")
            ctx.extra["quest_text"] = quest_text
            ctx.last_action = "read_commerce_quest"
            self._checkpoint.save(ctx)
            time.sleep(0.3)
            self._press_buff_speed()
            return BotState.CLASSIFY_QUEST

        # 6. Tuyệt đối không nhận diện bừa nếu thực sự CHƯA có nhiệm vụ nào được nhận
        ctx.extra["no_quest_attempts"] = int(ctx.extra.get("no_quest_attempts", 0)) + 1
        logger.warning(f"[{ctx.account_id}] Chưa nhận được nhiệm vụ Thương Hội (lần {ctx.extra['no_quest_attempts']})...")
        if ctx.extra["no_quest_attempts"] >= 3:
            ctx.extra["no_quest_attempts"] = 0
            logger.warning(f"[{ctx.account_id}] Đã thử 3 lần chưa nhận được NV, quay lại MEET_NPC đối thoại với Chủ thương hội...")
            return BotState.MEET_NPC
        time.sleep(0.3)
        return BotState.GET_QUEST

    def _handle_classify_quest(self, ctx: StateContext) -> BotState:
        """Phân loại và điều hướng theo 3 loại nhiệm vụ Thương Hội."""
        quest_text = ctx.extra.get("quest_text", "")
        quest_type = self._classifier.classify(text=quest_text)

        # Nếu UNKNOWN -> Kiểm tra lại từ bảng nhiệm vụ (Cách 2)
        if quest_type == QuestType.UNKNOWN:
            active = self._check_active_commerce_quest(ctx)
            if active and active[0] != quest_text:
                quest_text = active[0]
                ctx.extra["quest_text"] = quest_text
                quest_type = self._classifier.classify(text=quest_text)

        # Nếu vẫn UNKNOWN nhưng đã có nhiệm vụ hoặc đang auto-walk (từ Cách 3 / Cách 2):
        # KHÔNG quay lại MEET_NPC tạo vòng lặp vô tận!
        if quest_type == QuestType.UNKNOWN:
            norm_q = normalize_game_text(quest_text)
            if any(w in norm_q for w in ("mua", "dat mua", "thuong nhan")):
                quest_type = QuestType.SPECIAL_NPC_SHOP
            elif any(w in norm_q for w in ("tieu diet", "danh bai", "giet", "quai")):
                quest_type = QuestType.COMBAT
            elif ctx.extra.get("auto_walking") or any(w in norm_q for w in ("den", "gap", "noi", "hoi", "ve binh", "dao nhan", "sinh", "nan dan", "thuong hoi")):
                quest_type = QuestType.NPC
            else:
                logger.warning(f"[{ctx.account_id}] Không xác định được nhiệm vụ nào từ text: '{quest_text}'! Quay lại gặp Chủ thương hội...")
                activity._write("⚠️", f"[{ctx.account_id}] Khong co NV, quay lai Chu thuong hoi...")
                return BotState.MEET_NPC

        ctx.current_quest_type = quest_type
        ctx.last_action = f"classify:{quest_type.value}"
        self._checkpoint.save(ctx)

        from .quest_classifier import extract_dialogue_target, extract_shop_item, extract_combat_target

        logger.info(f"[{ctx.account_id}] Phân loại nhiệm vụ: {quest_type.value.upper()}")

        # ==================== LOẠI 1: ĐỐI THOẠI NPC ====================
        if quest_type == quest_type.__class__.NPC:
            city, npc = extract_dialogue_target(quest_text, with_accents=True)
            ctx.extra["quest_target_city"] = city
            ctx.extra["quest_target_npc"] = npc
            detail = f"Đối thoại: {city} -> {npc}" if (city or npc) else "Đối thoại NPC"
            ctx.extra["quest_detail_text"] = detail
            self._save_progress(ctx, status_text=f"Làm NV #{ctx.total_quests_done + 1}: Đối thoại", quest_detail=detail)
            logger.info(f"[{ctx.account_id}] [Loại 1 - NPC] Thành phố đích: '{city}', NPC: '{npc}'")
            activity._write("🗣️", f"[{ctx.account_id}] NV Doi thoai: {city} -> {npc}")
            self._checkpoint.save(ctx)

            # Tuyệt đối không click ô nhiệm vụ góc trái, luôn điều hướng qua bản đồ mini
            ctx.extra.pop("auto_walking", None)
            if city and npc and self._navigate_to_dialogue_npc(ctx, city, npc):
                return BotState.NAVIGATE

            logger.warning(f"[{ctx.account_id}] Không thể điều hướng NPC tự động qua Map. Chuyển sang NAVIGATE...")
            return BotState.NAVIGATE

        # ==================== LOẠI 2: THƯƠNG NHÂN TÂY VỰC MUA ĐỒ ====================
        if quest_type == quest_type.__class__.SPECIAL_NPC_SHOP:
            # 1. Ưu tiên lấy món đồ được tô vàng trong nội dung bảng nhiệm vụ, fallback extract_shop_item
            item = ctx.extra.get("yellow_item", "")
            if not item:
                item = extract_shop_item(quest_text)
            if not item:
                logger.warning(f"[{ctx.account_id}] Nhận diện Mua đồ nhưng không có vật phẩm cần mua trong '{quest_text}'! Quay lại MEET_NPC...")
                active = self._check_active_commerce_quest()
                if active and active[0] != quest_text:
                    ctx.extra["quest_text"] = active[0]
                    return BotState.CLASSIFY_QUEST
                return BotState.MEET_NPC

            ctx.extra["requested_shop_item"] = item
            detail = f"Mua hàng Tây Vực: {item}"
            ctx.extra["quest_detail_text"] = detail
            self._save_progress(ctx, status_text=f"Làm NV #{ctx.total_quests_done + 1}: Mua đồ", quest_detail=detail)
            logger.info(f"[{ctx.account_id}] [Loại 2 - Shop] Món đồ cần mua: '{item}'")
            activity._write("🛍️", f"[{ctx.account_id}] NV Mua do: {item}")
            self._checkpoint.save(ctx)

            if self._find_western_merchant(ctx):
                return BotState.BUY_ITEM
            logger.warning(f"[{ctx.account_id}] Chưa tìm thấy Thương nhân Tây Vực ở các điểm spawn")
            return BotState.BUY_ITEM

        # ==================== LOẠI 3: ĐÁNH QUÁI ====================
        if quest_type == quest_type.__class__.COMBAT:
            map_name, monster, count = extract_combat_target(quest_text)
            ctx.extra["combat_map"] = map_name
            ctx.extra["combat_monster"] = monster
            ctx.extra["combat_count"] = count
            detail = f"Đánh quái: {monster} ({count} con) tại {map_name}"
            ctx.extra["quest_detail_text"] = detail
            self._save_progress(ctx, status_text=f"Làm NV #{ctx.total_quests_done + 1}: Đánh quái", quest_detail=detail)
            logger.info(f"[{ctx.account_id}] [Loại 3 - Combat] Map: '{map_name}', Quái: '{monster}', SL: {count}")
            activity._write("⚔️", f"[{ctx.account_id}] NV Danh quai: {map_name} -> {monster} ({count} con)")
            self._checkpoint.save(ctx)

            if map_name:
                self._navigate_to_combat_map(ctx, map_name)
            return BotState.NAVIGATE

        logger.warning(f"[{ctx.account_id}] Loại nhiệm vụ không xác định: {quest_text}")
        return BotState.MEET_NPC

    def _search_current_map_panel(self, term: str) -> tuple[int, int] | None:
        """Search the active map/NPC tab and return the best green result."""
        from .text_recognition import normalize_game_text

        nav = self._loader.load_ui_coords().get("navigation", {})
        search = nav.get("map_search_input", {"x": 260, "y": 245})
        self._input.tap(search.get("x", 260), search.get("y", 245))
        time.sleep(.15)
        # Clear the previous city/NPC query before scanning the next spawn.
        try:
            delete_command = "input keyevent 123; " + "; ".join(["input keyevent 67"] * 40)
            self._adb._run(["shell", delete_command])
        except Exception:
            pass
        self._clear_focused_and_type(normalize_game_text(term))
        try:
            self._adb.key_event(66)
        except Exception:
            pass
        time.sleep(0.6)
        # Verify the actual text result.
        full_boxes = self._text.read(self._screen.capture())
        if not self._text.best_match(full_boxes, ["Tìm map", "Tìm NPC"], minimum=.55):
            logger.warning("Map search panel is not visible; refusing background quest click")
            return None
        region_cfg = nav.get("map_list_region", {})
        region = tuple(region_cfg.get(k, v) for k, v in zip(
            ("x1", "y1", "x2", "y2"), (75, 285, 465, 865)))
        boxes = [
            box for box in self._text.read(self._screen.capture(), region)
            if is_safe_map_result(box.text, term)
        ]
        match = self._text.best_match(boxes, [term], minimum=.62)
        if not match:
            logger.warning(f"Map panel result did not verify requested text '{term}'")
            return None
        score, box, _ = match
        logger.info(f"Map panel verified '{term}' <- '{box.text}' score={score:.2f}")
        return box.center_x, box.center_y

    def _navigate_to_combat_map(self, ctx: StateContext, map_name: str) -> bool:
        """Mở bản đồ và di chuyển đến map quái chỉ định."""
        logger.info(f"[{ctx.account_id}] Đang mở bản đồ để đi đến map '{map_name}'...")
        self._dismiss_unexpected_popups()
        time.sleep(0.15)

        # 1. Mở Minimap nếu chưa mở
        self._ensure_map_open()
        time.sleep(0.25)

        # 2. Bấm Tìm map
        self._input.tap(176, 162)
        time.sleep(0.2)

        # 3. Tìm map trong danh sách
        map_res = self._search_current_map_panel(map_name)
        if map_res:
            self._input.tap(*map_res)
            time.sleep(0.25)
        else:
            # Mở Bản đồ thế giới
            nav = self._loader.load_ui_coords().get("navigation", {})
            world_btn = nav.get("map_world_button", {"x": 1436, "y": 782})
            logger.info(f"[{ctx.account_id}] Mở Bản đồ thế giới tìm map '{map_name}'...")
            self._input.tap(world_btn["x"], world_btn["y"])
            time.sleep(0.3)
            boxes = self._text.read(self._screen.capture(), (320, 100, 1500, 800))
            match = self._text.best_match(boxes, [map_name], minimum=0.55)
            if match:
                _, box, _ = match
                logger.info(f"[{ctx.account_id}] Tìm thấy map '{map_name}' tại ({box.center_x}, {box.center_y})")
                self._input.tap(box.center_x, box.center_y)
                time.sleep(0.25)

        # 4. Đóng bản đồ an toàn
        self._close_map()
        time.sleep(0.2)
        self._press_buff_speed()
        return True

    def _navigate_to_dialogue_npc(self, ctx: StateContext, city: str, npc: str) -> bool:
        """Chuyển bản đồ sang thành phố mục tiêu qua Tìm map, sau đó tìm NPC. Tuyệt đối không dùng ô nhiệm vụ góc trái."""
        logger.info(f"[{ctx.account_id}] Điều hướng đến thành phố '{city}', gặp NPC '{npc}' qua bản đồ mini (không dùng ô nhiệm vụ góc trái)...")
        self._dismiss_unexpected_popups()
        time.sleep(0.15)

        # 1. Chuyển sang thành phố mục tiêu qua Tìm map
        self._switch_to_city(ctx, city)

        # 2. Mở bản đồ thành phố để tìm NPC
        self._ensure_map_open()
        time.sleep(0.3)

        # 3. Tìm và chọn NPC bằng ô input tìm kiếm (không dùng ô nhiệm vụ góc trái)
        self._search_and_select_npc(ctx, npc)

        # 4. Đóng bản đồ an toàn để nhân vật bắt đầu chạy
        self._close_map()
        time.sleep(0.2)
        self._press_buff_speed()
        logger.info(f"[{ctx.account_id}] Nhân vật đang tự chạy đến NPC '{npc}' tại '{city}'...")
        return True

    def _handle_navigate(self, ctx: StateContext) -> BotState:
        logger.info(f"[{ctx.account_id}] Navigating / auto-walking to target...")
        ctx.last_action = "navigate"
        self._checkpoint.save(ctx)

        # Trường hợp NHIỆM VỤ ĐÁNH QUÁI
        if ctx.current_quest_type == ctx.current_quest_type.__class__.COMBAT:
            combat_map = ctx.extra.get("combat_map", "")
            monster = ctx.extra.get("combat_monster", "")
            count = ctx.extra.get("combat_count", 20)
            logger.info(f"[{ctx.account_id}] Bắt đầu làm NV đánh quái: {monster} tại {combat_map} (yêu cầu {count} con)...")

            # Chờ ngắn 1.0s để nhân vật load map hoàn tất
            time.sleep(1.0)

            # Bật nút Auto đánh quái (1306, 240)
            auto_btn = (1306, 240)
            logger.info(f"[{ctx.account_id}] Bấm nút Auto đánh quái tại {auto_btn}...")
            self._input.tap(*auto_btn)
            time.sleep(0.25)

            # Giám sát tự đánh cho đến khi hoàn thành
            max_duration = max(30, min(180, count * 6))
            start_combat = time.time()
            logger.info(f"[{ctx.account_id}] Đang theo dõi đánh quái (tối đa {max_duration}s)...")

            while time.time() - start_combat < max_duration:
                time.sleep(0.8)
                frame = self._screen.capture()
                boxes = self._text.read(frame, (0, 200, 1600, 600))
                complete_found = any(
                    any(w in normalize_game_text(b.text) for w in ("hoan thanh", "da du", "tieu diet xong", "tra nhiem vu"))
                    for b in boxes
                )
                if complete_found:
                    logger.info(f"[{ctx.account_id}] Nhận diện thông báo hoàn thành nhiệm vụ đánh quái!")
                    break

            # Tắt Auto đánh quái
            logger.info(f"[{ctx.account_id}] Hoàn tất đánh quái! Tắt nút Auto...")
            self._input.tap(*auto_btn)
            time.sleep(0.15)

            ctx.extra["returning_quest"] = True
            return BotState.RETURN_QUEST

        # Trường hợp NHIỆM VỤ ĐỐI THOẠI NPC
        # Chờ nhân vật di chuyển (tối đa 25s), dừng ngay khi dialog xuất hiện
        for i in range(50):
            time.sleep(0.5)
            if i % 6 == 0:
                self._press_buff_speed()
            if self._find_dialog_buttons() or self._is_npc_dialog_open():
                logger.info(f"[{ctx.account_id}] Hộp thoại NPC đã mở khi đang di chuyển!")
                return BotState.INTERACT_NPC

        return BotState.INTERACT_NPC

    def _handle_interact_npc(self, ctx: StateContext) -> BotState:
        """Đã đến gặp NPC nhiệm vụ, đối thoại và chuẩn bị quay về trả nhiệm vụ."""
        logger.info(f"[{ctx.account_id}] Đã đến gặp NPC nhiệm vụ, tiến hành đối thoại...")
        time.sleep(0.3)

        # Nếu hộp thoại chưa mở, tap nhẹ NPC ở giữa màn hình (780, 420)
        buttons = self._find_dialog_buttons()
        if not buttons and not self._is_npc_dialog_open():
            logger.info(f"[{ctx.account_id}] Hộp thoại NPC chưa mở, tap thử vị trí NPC giữa màn hình (780, 420)...")
            self._input.tap(780, 420)
            time.sleep(0.5)

        # Bấm qua các bước thoại
        for step in range(5):
            buttons = self._find_dialog_buttons()
            if not buttons and not self._is_npc_dialog_open():
                break
            if buttons:
                logger.info(f"[{ctx.account_id}] Bấm nút thoại tại {buttons[0]}...")
                self._input.tap(*buttons[0])
            else:
                self._input.tap(800, 550)
            time.sleep(0.35)

        logger.info(f"[{ctx.account_id}] Hoàn thành đối thoại với NPC! Chuẩn bị quay về trả nhiệm vụ...")
        activity._write("✅", f"[{ctx.account_id}] Da doi thoai xong voi NPC. Quay ve tra nhiem vu.")
        ctx.extra["returning_quest"] = True
        time.sleep(0.2)
        return BotState.RETURN_QUEST

    def _get_current_location_from_hud(self, frame: Optional[np.ndarray] = None) -> tuple[Optional[str], Optional[tuple[int, int]]]:
        """Đọc tên map và tọa độ hiện tại từ HUD góc trên bên phải (vùng 1350, 0, 1600, 110)."""
        import re
        if frame is None:
            frame = self._screen.capture()
        if frame is None:
            return None, None
        boxes = self._text.read(frame, (1350, 0, 1600, 110))
        city = None
        coords = None
        for b in boxes:
            m = re.search(r'(\d+)\s*[/\\|\s]\s*(\d+)', b.text)
            if m:
                coords = (int(m.group(1)), int(m.group(2)))
            elif len(b.text.strip()) >= 3 and not any(c.isdigit() for c in b.text):
                city = b.text.strip()
        return city, coords

    def _switch_to_city(self, ctx: StateContext, city_name: str, force: bool = False) -> bool:
        """Chuyển bản đồ sang thành phố chỉ định qua Tìm map."""
        cur_city, _ = self._get_current_location_from_hud()
        from .text_recognition import normalize_game_text
        norm_target = normalize_game_text(city_name)
        if not force and cur_city and norm_target in normalize_game_text(cur_city):
            logger.info(f"[{ctx.account_id}] Nhân vật đã ở sẵn thành phố '{city_name}', không cần chuyển map.")
            return True

        logger.info(f"[{ctx.account_id}] Đang chuyển sang thành phố '{city_name}' qua Tìm map...")
        activity._write("🗺️", f"[{ctx.account_id}] Chuyen sang thanh pho {city_name}...")
        self._dismiss_unexpected_popups()
        self._ensure_map_open()
        time.sleep(0.3)

        coords = self._loader.load_ui_coords()
        nav_cfg = coords.get("navigation", {})
        search_x = nav_cfg.get("map_search_input", {}).get("x", 266)
        search_y = nav_cfg.get("map_search_input", {}).get("y", 222)
        find_map_x = nav_cfg.get("map_tab_find_map", {}).get("x", 176)
        find_map_y = nav_cfg.get("map_tab_find_map", {}).get("y", 162)

        self._input.tap(find_map_x, find_map_y)
        time.sleep(0.3)
        self._input.tap(search_x, search_y)
        time.sleep(0.2)
        try:
            delete_cmd = "input keyevent 123; " + "; ".join(["input keyevent 67"] * 30)
            self._adb._run(["shell", delete_cmd])
            time.sleep(0.2)
        except Exception:
            pass

        TELEX_CITIES = {
            "tuyen chau": "tuyeefn%schaau",
            "bien kinh": "bieen%skinh",
            "thanh do": "thafnh%sdoo",
            "tuong duong": "tuwownwg%sduwownwg",
            "duong chau": "duwownwg%schaau",
            "dai ly": "ddaij%slys",
        }
        telex = TELEX_CITIES.get(norm_target, norm_target.replace(" ", "%s"))
        self._input_search_text(city_name, telex=telex, ctx=ctx)
        time.sleep(0.6)

        if not self._click_map_search_result(city_name):
            logger.warning(f"[{ctx.account_id}] Không click được map '{city_name}' bằng OCR, click hàng kết quả đầu tiên (115, 315)...")
            self._input.tap(115, 315)

        # Chờ chuyển cảnh bản đồ sang vị trí của thành phố
        time.sleep(2.0)
        for _ in range(8):
            cur, _ = self._get_current_location_from_hud()
            if cur and norm_target in normalize_game_text(cur):
                logger.info(f"[{ctx.account_id}] Đã load thành công vào {city_name} (HUD: '{cur}')")
                break
            time.sleep(0.6)
        time.sleep(0.5)
        return True

    def _go_to_city_coordinate(self, ctx: StateContext, city: str, x: int, y: int) -> bool:
        """Đến vị trí ngẫu nhiên của thành phố trước, sau đó mở bản đồ nhập tọa độ (x, y) của thương nhân."""
        self._dismiss_unexpected_popups()
        time.sleep(0.15)

        # 1. Đi tới vị trí ngẫu nhiên của thành phố mục tiêu qua Tìm map
        self._switch_to_city(ctx, city, force=True)

        # 2. Sau khi đã ở trong thành, mở bản đồ thành phố để nhập tọa độ
        self._ensure_map_open()
        time.sleep(0.35)

        coords = self._loader.load_ui_coords()
        commerce = coords.get("commerce_quest", {})
        field_x = commerce.get("map_coordinate_x", {"x": 704, "y": 778})
        field_y = commerce.get("map_coordinate_y", {"x": 842, "y": 778})
        confirm = commerce.get("map_coordinate_confirm", {"x": 1015, "y": 778})

        # Nhập tọa độ X (xóa cũ rồi nhập)
        self._input.tap(field_x["x"], field_x["y"])
        time.sleep(0.2)
        try:
            del_cmd = "input keyevent 123; " + "; ".join(["input keyevent 67"] * 10)
            self._adb._run(["shell", del_cmd])
            time.sleep(0.15)
        except Exception:
            pass
        self._input.input_text(str(x))
        time.sleep(0.2)

        # Nhập tọa độ Y (xóa cũ rồi nhập)
        self._input.tap(field_y["x"], field_y["y"])
        time.sleep(0.2)
        try:
            del_cmd = "input keyevent 123; " + "; ".join(["input keyevent 67"] * 10)
            self._adb._run(["shell", del_cmd])
            time.sleep(0.15)
        except Exception:
            pass
        self._input.input_text(str(y))
        time.sleep(0.2)

        # Nhấn "Xác nhận"
        logger.info(f"[{ctx.account_id}] Xác nhận tọa độ ({x}/{y}) tại {city}...")
        self._input.tap(confirm["x"], confirm["y"])
        time.sleep(0.5)

        # Nếu bản đồ chưa tự đóng sau khi bấm Xác nhận, bấm đóng
        if self._is_map_open():
            self._close_map()
            time.sleep(0.2)

        self._press_buff_speed()
        logger.info(f"[{ctx.account_id}] Bắt đầu chạy bộ đến tọa độ {city} ({x}/{y})")
        return True

    def _find_western_merchant(self, ctx: StateContext) -> bool:
        """Tuần tra chạy bộ qua các tọa độ spawn định sẵn của Thương nhân Tây Vực (ngẫu nhiên ở 1 trong 6 thành lớn)."""
        commerce = self._loader.load_ui_coords().get("commerce_quest", {})
        spawns = commerce.get("western_merchant_spawns", [
            {"city": "Tuyền Châu", "x": 187, "y": 191},
            {"city": "Biện Kinh", "x": 179, "y": 184},
            {"city": "Thành Đô", "x": 224, "y": 225},
            {"city": "Tương Dương", "x": 187, "y": 180},
            {"city": "Dương Châu", "x": 206, "y": 194},
            {"city": "Đại Lý", "x": 178, "y": 187},
        ])
        aliases = ["Thương nhân Tây Vực", "Thương nhân Tây Hồ", "Thuong nhan Tay Vuc", "Thương nhân phương tây", "thuong nhan", "tay vuc", "thuong nhan tay"]

        for spawn in spawns:
            city_name = spawn["city"]
            target_x, target_y = int(spawn["x"]), int(spawn["y"])
            logger.info(f"[{ctx.account_id}] Bắt đầu đến thành {city_name} ({target_x}/{target_y}) tìm Thương nhân Tây Vực...")
            activity._write("🏃", f"[{ctx.account_id}] Chay den {city_name} ({target_x}/{target_y})...")

            if not self._go_to_city_coordinate(ctx, city_name, target_x, target_y):
                continue

            # Chờ nhân vật thực sự chạy bộ tới tọa độ đích (tối đa 25 giây)
            start_walk = time.time()
            found_merchant = False
            last_coords = None
            stationary_count = 0

            while time.time() - start_walk < 25.0:
                time.sleep(1.0)
                frame = self._screen.capture()
                if frame is None:
                    continue

                # 1. Quét xem Thương nhân Tây Vực đã xuất hiện trên màn hình chưa
                # 1a. Kiểm tra thanh mục tiêu NPC ở đỉnh màn hình (450..700, 10..80)
                top_boxes = [b for b in self._text.read(frame, (450, 10, 700, 80)) if b.center_y <= 90]
                for tb in top_boxes:
                    norm_tb = normalize_game_text(tb.text)
                    if ("thuong nhan" in norm_tb and "tay" in norm_tb) or "tay vuc" in norm_tb:
                        logger.info(f"[{ctx.account_id}] Phát hiện Thương nhân Tây Vực trên thanh mục tiêu đỉnh màn hình! Bấm tại ({tb.center_x}, {tb.center_y})...")
                        self._input.tap(tb.center_x, tb.center_y)
                        time.sleep(0.5)
                        ctx.extra["merchant_found_city"] = city_name
                        self._checkpoint.save(ctx)
                        found_merchant = True
                        break
                if found_merchant:
                    break

                # 1b. Quét toàn màn hình tìm tên Thương nhân
                boxes = self._text.read(frame)
                merchant_boxes = [box for box in boxes if is_western_merchant_name(box.text)]
                match = self._text.best_match(merchant_boxes, aliases, minimum=0.55)
                if not match:
                    for b in boxes:
                        norm_b = normalize_game_text(b.text)
                        if ("thuong nhan" in norm_b and "tay" in norm_b) or "tay vuc" in norm_b:
                            match = (0.9, b, b.text)
                            break

                if match:
                    score, box, _ = match
                    logger.info(f"[{ctx.account_id}] Đã phát hiện Thương nhân Tây Vực tại {city_name}: '{box.text}' ({box.center_x}, {box.center_y})!")
                    activity._write("🛍️", f"[{ctx.account_id}] Gap Thuong nhan tai {city_name}!")
                    self._input.tap(box.center_x, box.center_y + 35)
                    time.sleep(0.5)
                    ctx.extra["merchant_found_city"] = city_name
                    self._checkpoint.save(ctx)
                    found_merchant = True
                    break

                # 2. Đọc tọa độ hiện tại từ HUD góc trên bên phải
                _, cur_coords = self._get_current_location_from_hud(frame)
                if cur_coords:
                    dist = abs(cur_coords[0] - target_x) + abs(cur_coords[1] - target_y)
                    if dist <= 3:
                        logger.info(f"[{ctx.account_id}] Đã chạy đến gần tọa độ mục tiêu {city_name} ({cur_coords[0]}/{cur_coords[1]}), khoảng cách={dist}")
                        break
                    if cur_coords == last_coords:
                        stationary_count += 1
                        if stationary_count >= 3 and dist <= 8:
                            logger.info(f"[{ctx.account_id}] Nhân vật đã đến khu vực lân cận tọa độ mục tiêu ({cur_coords[0]}/{cur_coords[1]})")
                            break
                    else:
                        stationary_count = 0
                    last_coords = cur_coords

                if int(time.time() - start_walk) % 6 == 0:
                    self._press_buff_speed()

            if found_merchant:
                return True

            # Nếu đã đến gần khu vực tọa độ nhưng chưa thấy, điều khiển chạy nhẹ xung quanh 3 điểm (patrol ground)
            logger.info(f"[{ctx.account_id}] Tuần tra chạy bộ quanh khu vực {city_name} ({target_x}/{target_y}) để tìm kiếm...")
            patrol_offsets = [(950, 400), (650, 500), (800, 350)]
            for pt in patrol_offsets:
                self._input.tap(*pt)
                time.sleep(1.2)
                frame = self._screen.capture()
                if frame is None:
                    continue

                # Kiểm tra thanh mục tiêu trên đỉnh
                top_boxes = [b for b in self._text.read(frame, (450, 10, 700, 80)) if b.center_y <= 90]
                for tb in top_boxes:
                    norm_tb = normalize_game_text(tb.text)
                    if ("thuong nhan" in norm_tb and "tay" in norm_tb) or "tay vuc" in norm_tb:
                        logger.info(f"[{ctx.account_id}] Tìm thấy Thương nhân trên thanh mục tiêu đỉnh màn hình khi tuần tra!")
                        self._input.tap(tb.center_x, tb.center_y)
                        time.sleep(0.5)
                        ctx.extra["merchant_found_city"] = city_name
                        self._checkpoint.save(ctx)
                        return True

                # Quét tên NPC trên màn hình
                boxes = self._text.read(frame)
                merchant_boxes = [box for box in boxes if is_western_merchant_name(box.text)]
                match = self._text.best_match(merchant_boxes, aliases, minimum=0.55)
                if not match:
                    for b in boxes:
                        norm_b = normalize_game_text(b.text)
                        if ("thuong nhan" in norm_b and "tay" in norm_b) or "tay vuc" in norm_b:
                            match = (0.9, b, b.text)
                            break

                if match:
                    score, box, _ = match
                    logger.info(f"[{ctx.account_id}] Đã phát hiện Thương nhân Tây Vực tại {city_name} khi tuần tra: '{box.text}'!")
                    activity._write("🛍️", f"[{ctx.account_id}] Gap Thuong nhan tai {city_name}!")
                    self._input.tap(box.center_x, box.center_y + 35)
                    time.sleep(0.5)
                    ctx.extra["merchant_found_city"] = city_name
                    self._checkpoint.save(ctx)
                    return True

            logger.info(f"[{ctx.account_id}] Đã tuần tra kiểm tra kỹ tọa độ ({target_x}/{target_y}) tại {city_name}: KHÔNG CÓ Thương nhân Tây Vực. Tiếp tục tìm thành tiếp theo...")
            activity._write("🏃", f"[{ctx.account_id}] {city_name} khong co, tiep tuc...")

        logger.warning(f"[{ctx.account_id}] Đã chạy tới kiểm tra tọa độ cả 6 thành nhưng không tìm thấy Thương nhân Tây Vực!")
        return False

    def _handle_buy_item(self, ctx: StateContext) -> BotState:
        """Hội thoại với Thương nhân Tây Vực (chọn 'Đồng ý mua'), mở Tạp hóa chọn vật phẩm, bấm Mua và xác nhận."""
        if self._app_cfg.mode.dry_run:
            logger.info(f"[{ctx.account_id}] DRY_RUN: Bỏ qua mua vật phẩm")
            ctx.extra["returning_quest"] = True
            return BotState.RETURN_QUEST

        requested_item = ctx.extra.get("requested_shop_item", "")
        logger.info(f"[{ctx.account_id}] Tiến hành giao dịch với Thương nhân Tây Vực mua món đồ: '{requested_item}'...")
        ctx.last_action = "buy_requested_item"

        coords = self._loader.load_ui_coords()
        shop_cfg = coords.get("shop", {})

        # Bước 1: Hội thoại với Thương nhân Tây Vực -> bấm tùy chọn "Đồng ý mua" (bên trái)
        time.sleep(0.5)
        dialog_boxes = self._text.read(self._screen.capture())
        opt_box = None
        for b in dialog_boxes:
            norm = normalize_game_text(b.text)
            if "dong y mua" in norm or ("dong y" in norm and "mua" in norm):
                opt_box = b
                break
        if opt_box:
            logger.info(f"[{ctx.account_id}] Bấm tùy chọn '{opt_box.text}' tại ({opt_box.center_x}, {opt_box.center_y})...")
            self._input.tap(opt_box.center_x, opt_box.center_y)
        else:
            opt_pos = shop_cfg.get("dialog_buy_option", {"x": 272, "y": 566})
            logger.info(f"[{ctx.account_id}] Bấm tùy chọn 'Đồng ý mua' tại tọa độ chuẩn ({opt_pos['x']}, {opt_pos['y']})...")
            self._input.tap(opt_pos["x"], opt_pos["y"])
        time.sleep(0.8)

        # Bước 2: Cửa hàng "Tạp hóa" mở ra -> click vô từng vật phẩm xem có giống tên vật phẩm yêu cầu mua chưa
        # Nếu chưa thì nhấn vật phẩm tiếp theo, nếu đúng thì dừng lại và bấm Mua
        grid_slots = shop_cfg.get("grid_slots", [
            {"x": 316, "y": 190}, {"x": 397, "y": 190}, {"x": 486, "y": 190}, {"x": 575, "y": 190},
            {"x": 316, "y": 279}, {"x": 397, "y": 279}, {"x": 486, "y": 279}, {"x": 575, "y": 279},
            {"x": 316, "y": 368}, {"x": 397, "y": 368}, {"x": 486, "y": 368}, {"x": 575, "y": 368},
            {"x": 316, "y": 457}, {"x": 397, "y": 457}, {"x": 486, "y": 457}, {"x": 575, "y": 457},
            {"x": 316, "y": 546}, {"x": 397, "y": 546}, {"x": 486, "y": 546}, {"x": 575, "y": 546},
        ])

        norm_target = normalize_game_text(requested_item) if requested_item else "linh duoc tay vuc"
        target_tokens = set(norm_target.split())
        logger.info(f"[{ctx.account_id}] Bắt đầu duyệt từng ô vật phẩm trong Tạp hóa tìm '{requested_item}'...")

        found_target_slot = False
        best_slot = (grid_slots[1]["x"], grid_slots[1]["y"]) if len(grid_slots) > 1 else (397, 190)
        best_score = -1.0

        title_roi = shop_cfg.get("tooltip_title_region", {"x1": 700, "y1": 30, "x2": 1100, "y2": 150})
        roi_tuple = (title_roi["x1"], title_roi["y1"], title_roi["x2"], title_roi["y2"])

        for slot_idx, slot in enumerate(grid_slots):
            sx, sy = int(slot["x"]), int(slot["y"])
            logger.info(f"[{ctx.account_id}] Kiểm tra ô #{slot_idx + 1} tại ({sx}, {sy})...")
            self._input.tap(sx, sy)
            time.sleep(0.35)

            frame_item = self._screen.capture()
            boxes = self._text.read(frame_item, roi_tuple)
            # Lọc bỏ các nhãn thông số như Loai, Nang, Gia, Da khoa, Ban tiem... để lấy tiêu đề tên vật phẩm
            boxes = sorted(boxes, key=lambda b: b.center_y)
            item_name = ""
            for b in boxes:
                b_norm = normalize_game_text(b.text)
                if any(tag in b_norm for tag in ("loai", "nang", "gia", "da khoa", "ban tiem", "giao dich", "vut bo")):
                    continue
                if len(b_norm) >= 2:
                    item_name = b.text
                    break

            if item_name:
                norm_seen = normalize_game_text(item_name)
                seen_tokens = set(norm_seen.split())
                score = fuzzy_score(item_name, requested_item or "Linh dược Tây Vực")
                logger.info(f"[{ctx.account_id}] Ô #{slot_idx + 1}: Tên hiển thị = '{item_name}' (độ khớp: {score:.2f} với '{requested_item}')")

                if score > best_score:
                    best_score = score
                    best_slot = (sx, sy)

                # Kiểm tra xem có đúng vật phẩm yêu cầu không
                is_match = (score >= 0.55) or (len(target_tokens & seen_tokens) >= max(1, len(target_tokens) - 1))
                if is_match:
                    logger.info(f"[{ctx.account_id}] ✅ ĐÚNG TÊN VẬT PHẨM: '{item_name}' tại ô #{slot_idx + 1}! Tiến hành bấm Mua...")
                    found_target_slot = True
                    break
                else:
                    logger.info(f"[{ctx.account_id}] ❌ Chưa đúng món '{requested_item}', chuyển sang vật phẩm tiếp theo...")
            else:
                logger.debug(f"[{ctx.account_id}] Ô #{slot_idx + 1}: Không có tooltip hoặc ô trống, thử tiếp...")

        if not found_target_slot:
            logger.warning(f"[{ctx.account_id}] Không tìm thấy ô khớp hoàn hảo, chọn ô có độ khớp cao nhất ({best_slot[0]}, {best_slot[1]}) với score={best_score:.2f}...")
            self._input.tap(best_slot[0], best_slot[1])
            time.sleep(0.3)

        # Bước 3: Bấm nút "Mua" (màu cam dưới bảng thông tin vật phẩm)
        frame_buy = self._screen.capture()
        shop_boxes = self._text.read(frame_buy)
        mua_boxes = [
            b for b in shop_boxes
            if ("mua" in normalize_game_text(b.text).split() or normalize_game_text(b.text) == "mua")
            and 420 <= b.center_x <= 650 and 420 <= b.center_y <= 560
        ]
        if mua_boxes:
            logger.info(f"[{ctx.account_id}] Tìm thấy nút Mua bằng OCR tại ({mua_boxes[0].center_x}, {mua_boxes[0].center_y})...")
            self._input.tap(mua_boxes[0].center_x, mua_boxes[0].center_y)
        elif self._vision.template_exists("shop_buy_btn"):
            buy_btn = self._vision.find_template("shop_buy_btn", frame_buy)
            if buy_btn.found:
                self._input.tap(buy_btn.center_x, buy_btn.center_y)
            else:
                buy_pos = shop_cfg.get("buy_button", {"x": 538, "y": 485})
                self._input.tap(buy_pos["x"], buy_pos["y"])
        else:
            buy_pos = shop_cfg.get("buy_button", {"x": 538, "y": 485})
            logger.info(f"[{ctx.account_id}] Bấm nút Mua tại tọa độ chuẩn ({buy_pos['x']}, {buy_pos['y']})...")
            self._input.tap(buy_pos["x"], buy_pos["y"])
        time.sleep(0.6)

        # Bước 4: Popup số lượng ("Số lượng: - 1 +") -> bấm nút "Đồng ý"
        frame_confirm = self._screen.capture()
        confirm_boxes = [
            b for b in self._text.read(frame_confirm)
            if any(w in normalize_game_text(b.text) for w in ("dong y", "xac nhan"))
            and 380 <= b.center_x <= 550 and 720 <= b.center_y <= 850
        ]
        if confirm_boxes:
            target_confirm = confirm_boxes[0]
            logger.info(f"[{ctx.account_id}] Bấm nút xác nhận '{target_confirm.text}' tại ({target_confirm.center_x}, {target_confirm.center_y})...")
            self._input.tap(target_confirm.center_x, target_confirm.center_y)
        else:
            confirm_pos = shop_cfg.get("confirm_button", {"x": 465, "y": 786})
            logger.info(f"[{ctx.account_id}] Bấm nút Đồng ý xác nhận mua tại ({confirm_pos['x']}, {confirm_pos['y']})...")
            self._input.tap(confirm_pos["x"], confirm_pos["y"])
        time.sleep(0.5)

        # Bước 5: Đóng cửa hàng Tạp hóa
        close_pos = shop_cfg.get("close_button", {"x": 702, "y": 128})
        logger.info(f"[{ctx.account_id}] Đóng cửa hàng Tạp hóa tại ({close_pos['x']}, {close_pos['y']})...")
        self._input.tap(close_pos["x"], close_pos["y"])
        time.sleep(0.3)
        inv_close = shop_cfg.get("inventory_close", {"x": 1577, "y": 81})
        self._input.tap(inv_close["x"], inv_close["y"])
        time.sleep(0.2)
        try:
            self._adb.key_event(4)  # Keyevent BACK
            time.sleep(0.2)
        except Exception:
            pass
        self._dismiss_unexpected_popups()

        # Bước 6: Chuẩn bị quay về trả nhiệm vụ cho Chủ Thương Hội
        logger.info(f"[{ctx.account_id}] Đã mua xong món đồ '{requested_item}'! Chuẩn bị quay về trả nhiệm vụ cho Chủ Thương Hội...")
        activity._write("🛍️", f"[{ctx.account_id}] Da mua xong {requested_item}. Quay ve tra nhiem vu.")
        ctx.extra["returning_quest"] = True
        return BotState.RETURN_QUEST

    def _return_to_nearest_city_commerce_chief(self, ctx: StateContext) -> bool:
        """Thực hiện đúng 8 bước tìm đường quay về Chủ thương hội tại Tuyền Châu."""
        logger.info(f"[{ctx.account_id}] Quay về Chủ thương hội tại Tuyền Châu theo 8 bước chuẩn...")
        self._dismiss_unexpected_popups()
        time.sleep(0.15)
        return self._navigate_to_tuyen_chau_commerce_chief(ctx)

    def _handle_return_quest(self, ctx: StateContext) -> BotState:
        """Quay về Chủ Thương Hội ở thành lớn gần nhất để trả nhiệm vụ."""
        logger.info(f"[{ctx.account_id}] Đã hoàn thành mục tiêu nhiệm vụ, đang quay về Chủ Thương Hội...")
        ctx.extra["returning_quest"] = True
        ctx.last_action = "return_to_commerce_chief"
        self._checkpoint.save(ctx)

        # Luôn mở Bản đồ thế giới và di chuyển đến Chủ Thương Hội ở thành lớn (Tuyền Châu).
        # Tuyệt đối không quét link xanh trên màn hình chính vì sẽ bấm nhầm vào nhiệm vụ Dã Tẩu / Tân Thủ (Vương Nghiệp Vĩ)!
        self._return_to_nearest_city_commerce_chief(ctx)
        return BotState.FIND_COMMERCE_CHIEF

    def _handle_complete_quest(self, ctx: StateContext) -> BotState:
        logger.info(f"[{ctx.account_id}] Completing quest step...")
        ctx.total_quests_done += 1
        ctx.last_action = "complete_quest"
        quest_type = ctx.current_quest_type.value if ctx.current_quest_type else "NPC"
        self._db.log_quest(
            account_id=ctx.account_id,
            quest_type=quest_type,
            result="COMPLETED",
        )
        self._checkpoint.save(ctx)
        # Ghi activity log dễ đọc
        username = self._get_username(ctx.account_id)
        activity.quest_done(username, ctx.total_quests_done, quest_type)
        time.sleep(1.0)
        ctx.current_quest_type = ctx.current_quest_type.__class__.UNKNOWN
        ctx.extra.pop("quest_text", None)
        ctx.extra.pop("contribution_option_selected", None)
        ctx.extra.pop("quest_source", None)
        return BotState.CHECK_LOCATION

    def _handle_cancel_quest(self, ctx: StateContext) -> BotState:
        logger.warning(f"[{ctx.account_id}] Cancelling quest (UNKNOWN type)")
        ctx.retry_count += 1
        self._db.log_quest(
            account_id=ctx.account_id,
            quest_type="UNKNOWN",
            result="CANCELLED",
        )
        username = self._get_username(ctx.account_id)
        activity.quest_failed(username, ctx.total_quests_done + 1, "Không nhận ra nhiệm vụ")
        if self._app_cfg.mode.dry_run or ctx.retry_count >= self._app_cfg.retry.max_attempts:
            logger.info(f"[{ctx.account_id}] Stopping quest loop after {ctx.retry_count} cancelled quest(s).")
            return BotState.TRANSFER_GOLD
        time.sleep(self._app_cfg.retry.delay_between)
        return BotState.GET_QUEST

    def _handle_transfer_gold(self, ctx: StateContext) -> BotState:
        if self._app_cfg.mode.dry_run:
            logger.warning(f"[{ctx.account_id}] DRY_RUN: TRANSFER_GOLD blocked")
            return BotState.LOGOUT
        logger.info(f"[{ctx.account_id}] Transferring gold to master...")
        # TODO: Implement gold transfer
        return BotState.LOGOUT

    def _is_account_completed_today(self, account_id: str) -> bool:
        """Kiểm tra xem tài khoản đã đạt đủ giới hạn nhiệm vụ (20/20) trong ngày hôm nay chưa."""
        try:
            runtime_dir = Path("runtime")
            prog_file = runtime_dir / f"progress_{account_id}.json"
            if prog_file.exists():
                with open(prog_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    total = data.get("total_quests_done", 0)
                    status = str(data.get("status_text", "")).lower()
                    if total >= 20 or "20/20" in status or "hết nhiệm vụ" in status:
                        return True
            return False
        except Exception as e:
            logger.debug(f"_is_account_completed_today error: {e}")
            return False

    def _logout_current_account(self, ctx: StateContext) -> bool:
        """
        Đăng xuất tài khoản hiện tại ra màn hình đăng nhập:
        1. Thử mở Cài đặt (1402, 85) và bấm nút 'Đăng xuất'.
        2. Nếu không được hoặc không về được màn hình đăng nhập, tắt app (stop_app) và mở lại (start_app) để về màn hình đăng nhập sạch sẽ.
        """
        if self._app_cfg.mode.dry_run:
            logger.warning(f"[{ctx.account_id}] DRY_RUN: LOGOUT blocked")
            return True

        logger.info(f"[{ctx.account_id}] Đang đăng xuất tài khoản...")
        activity._write("🚪", f"[{ctx.account_id}] Dang dang xuat tai khoan...")

        # Thử đăng xuất qua UI trong game
        try:
            # 1. Bấm nút Cài đặt ở góc trên phải (1402, 85)
            self._input.tap(1402, 85)
            time.sleep(0.6)

            # Quét tìm nút "Đăng xuất" trong popup Cài đặt
            frame = self._screen.capture()
            if frame is not None:
                boxes = self._text.read(frame)
                logout_box = None
                for b in boxes:
                    norm = normalize_game_text(b.text)
                    if "dang xuat" in norm or "doi tai khoan" in norm or "thoat game" in norm:
                        logout_box = b
                        break

                if logout_box:
                    logger.info(f"[{ctx.account_id}] Tìm thấy nút Đăng xuất tại ({logout_box.center_x}, {logout_box.center_y}), bấm...")
                    self._input.tap(logout_box.center_x, logout_box.center_y)
                    time.sleep(0.8)

                    # Kiểm tra popup xác nhận đăng xuất (Đồng ý / Xác nhận)
                    c_frame = self._screen.capture()
                    if c_frame is not None:
                        c_boxes = self._text.read(c_frame)
                        for cb in c_boxes:
                            c_norm = normalize_game_text(cb.text)
                            if any(w in c_norm for w in ("dong y", "xac nhan", "chap nhan")):
                                logger.info(f"[{ctx.account_id}] Bấm xác nhận đăng xuất tại ({cb.center_x}, {cb.center_y})")
                                self._input.tap(cb.center_x, cb.center_y)
                                time.sleep(1.5)
                                break

                    time.sleep(1.5)
                    if not self._is_ingame():
                        logger.info(f"[{ctx.account_id}] Đăng xuất UI thành công, đã về màn hình đăng nhập!")
                        return True
        except Exception as e:
            logger.warning(f"Lỗi khi thử đăng xuất UI: {e}")

        # Fallback chắc chắn 100%: Tắt game và mở lại để ra màn hình đăng nhập sạch sẽ
        logger.info(f"[{ctx.account_id}] Dùng fallback: Tắt game và mở lại để về màn hình đăng nhập sạch sẽ...")
        try:
            self._adb.stop_app()
            time.sleep(2.0)
            self._adb.start_app()
            time.sleep(4.0)
            logger.info(f"[{ctx.account_id}] Đã khởi động lại game về màn hình đăng nhập!")
            return True
        except Exception as e:
            logger.error(f"Fallback stop/start app error: {e}")
            return False

    def _handle_logout(self, ctx: StateContext) -> BotState:
        if self._app_cfg.mode.dry_run:
            logger.warning(f"[{ctx.account_id}] DRY_RUN: LOGOUT blocked")
            return BotState.NEXT_ACCOUNT
        logger.info(f"[{ctx.account_id}] Logging out...")
        self._logout_current_account(ctx)
        time.sleep(1.0)
        return BotState.NEXT_ACCOUNT

    def _handle_next_account(self, ctx: StateContext) -> BotState:
        logger.info(f"[{ctx.account_id}] Hoàn tất phiên chạy tài khoản. Sẵn sàng cho tài khoản tiếp theo.")
        return BotState.FINISHED

    def _handle_recovery(self, ctx: StateContext) -> BotState:
        logger.warning(f"[{ctx.account_id}] RECOVERY: restarting game")
        ctx.retry_count += 1
        if ctx.retry_count >= self._app_cfg.retry.max_attempts:
            raise BotError(ErrorCode.SM_002, "Max recovery retries exceeded")
        self._adb.restart_app()
        return BotState.LOGIN

    def _handle_error(self, ctx: StateContext) -> BotState:
        logger.error(
            f"[{ctx.account_id}] ERROR state: {ctx.error_code} – {ctx.error_message}"
        )
        self._db.log_error(
            account_id=ctx.account_id,
            error_code=ctx.error_code,
            state=ctx.previous_state.value,
            action=ctx.last_action,
            message=ctx.error_message,
        )
        return BotState.FINISHED

    # ============================
    # GUI Callback
    # ============================

    def _on_state_changed(
        self,
        old_state: BotState,
        new_state: BotState,
        ctx: StateContext,
    ) -> None:
        self._checkpoint.save(ctx)
        self._save_progress(ctx)
        if self._on_status_update:
            self._on_status_update(ctx)

    # ============================
    # Public Run API
    # ============================

    def run_all_accounts(self, account_ids: Optional[list[str]] = None) -> None:
        """Chạy tuần tự tất cả account đang enable."""
        accounts = self._loader.get_enabled_accounts()
        if account_ids is not None:
            wanted = set(account_ids)
            accounts = [account for account in accounts if account.id in wanted]
        logger.info(f"BotController: running {len(accounts)} account(s)")
        activity.separator()

        for account in accounts:
            # Kiểm tra xem tài khoản này đã hoàn thành nhiệm vụ hôm nay chưa
            if len(accounts) > 1 and self._is_account_completed_today(account.id):
                logger.info(f"[{account.id}] ({account.username}) đã hoàn thành 20/20 hôm nay, bỏ qua để đăng nhập tài khoản chưa làm...")
                activity._write("⏩", f"[{account.username}] Da xong 20/20, chuyen acc tiep theo.")
                continue

            if not self._state_machine.is_running:
                # Đăng ký account vào DB
                self._db.upsert_account(
                    account.id, account.username, account.server
                )
                run_id = self._db.start_run(account.id)
                activity.session_start(account.username)
                try:
                    self._state_machine.run(
                        account_id=account.id,
                        initial_state=BotState.BOOT,
                    )
                    final_st = self._state_machine.context.current_state
                    total = self._state_machine.context.total_quests_done
                    if final_st == BotState.ERROR:
                        err_msg = getattr(self._state_machine.context, "error_message", None) or "Lỗi kết nối hoặc chụp màn hình"
                        activity.error(account.username, f"Dừng do lỗi: {err_msg}")
                    else:
                        activity.session_end(account.username, total)
                    self._db.end_run(
                        run_id,
                        final_state=final_st.value,
                        quests_done=total,
                    )
                except KeyboardInterrupt:
                    total = self._state_machine.context.total_quests_done
                    activity.session_end(account.username, total)
                    logger.info("Bot duoc dung boi nguoi dung (Ctrl+C)")
                    raise
                except Exception as e:
                    logger.exception(f"Unhandled error for {account.id}: {e}")
                    activity.error(account.username, str(e))
                    self._db.end_run(run_id, "EXCEPTION", errors_count=1)
            else:
                logger.warning("StateMachine still running, skipping account start")

        logger.info("BotController: all accounts processed")

    def run_single_account(self, account_id: str) -> None:
        """Chạy một account cụ thể."""
        self._state_machine.run(account_id=account_id, initial_state=BotState.BOOT)

    def stop(self) -> None:
        self._state_machine.stop()
        if self._watchdog:
            self._watchdog.stop()

    def pause(self) -> None:
        self._state_machine.pause()

    def resume(self) -> None:
        self._state_machine.resume()

    @property
    def context(self) -> StateContext:
        return self._state_machine.context
