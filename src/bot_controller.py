"""
AutoQuestTool - Bot Controller
Orchestrator chính. Kết nối tất cả components.
GUI → BotController → StateMachine → Engine.
"""

from __future__ import annotations

import time
import math
import random
import json
import re
import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
from difflib import SequenceMatcher

from loguru import logger

from .config_loader import ConfigLoader, AppConfig
from .adb_controller import ADBController
from .ldplayer_controller import LDPlayerController
from .screen_capture import ScreenCapture
from .template_manager import TemplateManager
from .vision_engine import VisionEngine
from .input_controller import InputController
from .state_machine import StateMachine, BotState, StateContext, QuestType
from .quest_classifier import (
    QuestClassifier,
    extract_dialogue_target,
    extract_shop_item,
    clean_shop_item_name,
    match_western_merchant_item,
)
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
    match_known_city,
    match_known_town,
    is_same_city,
    KNOWN_CITIES,
    KNOWN_TOWNS,
    AMBIGUOUS_CITY_WORDS,
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

    FOUR_REFUGEE_NPCS = (
        "Nạn dân Giáp",
        "Nạn dân Ất",
        "Nạn dân Bính",
        "Nạn dân Đinh",
    )

    @classmethod
    def _is_four_refugee_quest(cls, quest_text: str) -> bool:
        """Nhận diện nhiệm vụ phải ghé đủ bốn Nạn dân tại Thành Đô."""
        norm = normalize_game_text(quest_text)
        return (
            "thanh do" in norm
            and "nan dan" in norm
            and ("4 cua" in norm or "bon cua" in norm)
        )

    def _continue_four_refugee_quest(self, ctx: StateContext) -> Optional[BotState]:
        """Chuyển sang Nạn dân kế tiếp; trả None khi đã gặp đủ bốn người."""
        route = ctx.extra.get("multi_npc_route")
        if not isinstance(route, list) or not route:
            return None

        current_index = int(ctx.extra.get("multi_npc_index", 0))
        next_index = current_index + 1
        if next_index >= len(route):
            ctx.extra["multi_npc_completed"] = True
            self._checkpoint.save(ctx)
            return None

        next_npc = route[next_index]
        ctx.extra["multi_npc_index"] = next_index
        ctx.extra["quest_target_city"] = "Thành Đô"
        ctx.extra["quest_target_npc"] = next_npc
        ctx.extra["multi_npc_needs_navigation"] = True
        ctx.extra.pop("npc_finish_state_failures", None)
        ctx.extra.pop("returning_quest", None)
        ctx.extra["quest_detail_text"] = (
            f"Đối thoại Nạn dân {next_index + 1}/{len(route)}: Thành Đô -> {next_npc}"
        )
        self._checkpoint.save(ctx)
        logger.info(
            f"[{ctx.account_id}] Đã gặp {current_index + 1}/{len(route)} Nạn dân; "
            f"tiếp tục tới '{next_npc}', chưa quay về trả nhiệm vụ."
        )
        activity._write(
            "🗣️",
            f"[{ctx.account_id}] Da gap {current_index + 1}/{len(route)} Nan dan; "
            f"tiep tuc den {next_npc}.",
        )
        self._navigate_to_dialogue_npc(ctx, "Thành Đô", next_npc)
        return BotState.NAVIGATE

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
        self._disable_buff_on_move: bool = True

        logger.info("BotController initialized")

    def _get_username(self, account_id: str) -> str:
        """Lấy username từ account_id để hiển thị log."""
        if not hasattr(self, "_loader") or self._loader is None:
            return account_id
        try:
            accounts = self._loader.get_enabled_accounts()
            acc = next((a for a in accounts if a.id == account_id), None)
            return acc.username if acc else account_id
        except Exception:
            return account_id

    def _save_progress(self, ctx: StateContext, status_text: str = "", quest_detail: str = "") -> None:
        """Lưu tiến độ của account ra file runtime/progress_{account_id}.json cho Launcher hiển thị."""
        try:
            import json
            from datetime import datetime
            today_str = datetime.now().strftime("%Y-%m-%d")
            runtime_dir = Path("runtime")
            runtime_dir.mkdir(parents=True, exist_ok=True)
            prog_file = runtime_dir / f"progress_{ctx.account_id}.json"

            existing_history = []
            if prog_file.exists():
                try:
                    with open(prog_file, "r", encoding="utf-8") as f:
                        old_data = json.load(f)
                        old_date = old_data.get("date")
                        if not old_date:
                            try:
                                old_date = datetime.fromtimestamp(prog_file.stat().st_mtime).strftime("%Y-%m-%d")
                            except Exception:
                                old_date = today_str
                        if old_date == today_str:
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
                    is_done = getattr(ctx, "total_quests_done", 0) >= 20 or bool(ctx.extra.get("quest_exhausted_today"))
                    st_text = "Đã hoàn thành 20/20 🎉" if is_done else "Hoàn tất phiên chạy"
                elif state_val == "ERROR":
                    st_text = f"Lỗi: {ctx.error_message or ctx.error_code}"
                else:
                    st_text = state_val

            detail = quest_detail or ctx.extra.get("quest_detail_text", ctx.extra.get("quest_text", ""))
            total_done = 20 if (ctx.extra.get("quest_exhausted_today") or getattr(ctx, "total_quests_done", 0) >= 20) else getattr(ctx, "total_quests_done", 0)
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
                "date": today_str,
                "last_updated": datetime.now().strftime("%H:%M:%S"),
                "history": history,
            }
            with open(prog_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"Save progress error: {e}")

    def _sync_quest_progress_from_text(self, ctx: Optional[StateContext], text: str, source: str = "") -> bool:
        """
        Đồng bộ số lượng nhiệm vụ Thương hội đã hoàn thành (tiến độ X/20) từ văn bản đọc được
        (hộp thoại NPC, nút 'Xem tình hình nhiệm vụ', Bảng nhiệm vụ, tracker...).
        Cập nhật ngay vào ctx.total_quests_done, checkpoint và file progress của Launcher.
        """
        if not ctx or not text:
            return False
        import re
        from .text_recognition import normalize_game_text
        norm = normalize_game_text(text)

        in_game_done = None

        # Mẫu 1: "Hôm nay đã hoàn thành X lần nhiệm vụ"
        m1 = re.search(r"hoan\s*thanh\s*(\d+)\s*(?:lan|1an|[l1]an)\b", norm)
        if m1:
            in_game_done = int(m1.group(1))

        # Mẫu 2: "Còn có thể nhận Y lần!" (tổng 20 lần -> đã làm 20 - Y)
        if in_game_done is None:
            m2 = re.search(r"con\s*co\s*the\s*nhan\s*(\d+)\s*(?:lan|1an|[l1]an)\b", norm)
            if not m2:
                # Tránh OCR dính số ví dụ 17lan -> 171an
                m2_alt = re.search(r"con\s*co\s*the\s*nhan\s*(\d+?)(?:1an|lan)\b", norm)
                if m2_alt:
                    m2 = m2_alt
            if m2:
                remain = int(m2.group(1))
                if 0 <= remain <= 20:
                    in_game_done = 20 - remain

        # Mẫu 3: "Thương hội (X/20)", "tiến độ: X/20", "hoàn thành X/20", "(X/20)"
        if in_game_done is None:
            m3 = re.search(r"(?:thuong\s*hoi|tien\s*do|hoan\s*thanh|nhiem\s*vu)?[^0-9\n\r]{0,25}\b(\d+)\s*[/]\s*20\b", text, re.IGNORECASE)
            if not m3:
                m3 = re.search(r"\(\s*(\d+)\s*[/]\s*20\s*\)", text)
            if not m3:
                m3 = re.search(r"(?:thuong\s*hoi|tien\s*do|hoan\s*thanh|nhiem\s*vu)\s+(\d+)\s+20\b", norm)
            if m3:
                val = int(m3.group(1))
                if 0 <= val <= 20:
                    in_game_done = val

        # Mẫu 4: "nhiệm vụ thứ X", "làm NV #X", "nhiệm vụ lần thứ X" -> đã xong X - 1
        if in_game_done is None:
            m4 = re.search(r"nhiem\s*vu\s*(?:lan\s*)?thu\s*(\d+)", norm)
            if m4:
                idx = int(m4.group(1))
                if 1 <= idx <= 21:
                    in_game_done = idx - 1

        # Mẫu 5: Bất kỳ cụm "(X/20)", "X/20", hoặc OCR nhận nhầm số 0 thành O/o
        if in_game_done is None:
            m5 = re.search(r"\b(\d+)\s*[/]\s*2[0oO]\b", text)
            if not m5:
                m5 = re.search(r"\b(\d+)\s*[/]\s*20\b", text)
            if m5:
                val = int(m5.group(1))
                if 0 <= val <= 20:
                    in_game_done = val

        db_done = 0
        if hasattr(self, "_db") and self._db:
            try:
                res = self._db.get_today_completed_quests(ctx.account_id)
                if isinstance(res, int) and not isinstance(res, bool):
                    db_done = res
            except Exception:
                db_done = 0

        target_done = in_game_done if in_game_done is not None else getattr(ctx, "total_quests_done", 0)
        target_done = max(target_done, db_done)

        old_done = getattr(ctx, "total_quests_done", 0)
        should_update = False
        if in_game_done is not None and 0 <= in_game_done <= 20:
            if in_game_done != old_done:
                target_done = in_game_done
                should_update = True
        elif target_done > old_done and 0 <= target_done <= 20:
            should_update = True

        if should_update:
            logger.info(
                f"[{ctx.account_id}] 🎯 [Cập nhật tiến độ] Đồng bộ số nhiệm vụ hoàn thành từ {source or 'game/db'}: "
                f"{target_done}/20 (cũ: {old_done}/20)"
            )
            ctx.total_quests_done = target_done
            current_idx = min(20, target_done + 1)
            self._save_progress(ctx, status_text=f"Đang làm NV #{current_idx}/20")
            if hasattr(self, "_checkpoint") and self._checkpoint:
                self._checkpoint.save(ctx)
            return True
        return False

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
        today_str = datetime.now().strftime("%Y-%m-%d")
        # Reset context quest counter nếu phát hiện sang ngày mới
        if ctx.extra.get("date") and ctx.extra.get("date") != today_str:
            logger.info(f"[{ctx.account_id}] Context sang ngày mới ({ctx.extra.get('date')} -> {today_str}), reset nhiệm vụ")
            ctx.total_quests_done = 0
            ctx.extra["quest_history_list"] = []
            ctx.extra.pop("quest_exhausted_today", None)
        ctx.extra["date"] = today_str

        # Khôi phục số lượng nhiệm vụ đã hoàn thành hôm nay từ database và file progress
        db_today = 0
        if hasattr(self, "_db") and self._db:
            try:
                res = self._db.get_today_completed_quests(ctx.account_id)
                if isinstance(res, int) and not isinstance(res, bool):
                    db_today = res
            except Exception:
                db_today = 0
        prog_today = 0
        try:
            p_file = Path("runtime") / f"progress_{ctx.account_id}.json"
            if p_file.exists():
                with open(p_file, "r", encoding="utf-8") as f:
                    p_data = json.load(f)
                    file_date = p_data.get("date")
                    if not file_date:
                        try:
                            file_date = datetime.fromtimestamp(p_file.stat().st_mtime).strftime("%Y-%m-%d")
                        except Exception:
                            file_date = today_str
                    if file_date == today_str:
                        prog_today = int(p_data.get("total_quests_done", 0))
                        if p_data.get("history") and not ctx.extra.get("quest_history_list"):
                            ctx.extra["quest_history_list"] = p_data.get("history")
                    else:
                        logger.info(
                            f"[{ctx.account_id}] File tiến độ cũ ({file_date} != {today_str}), "
                            f"reset nhiệm vụ Thương hội cho ngày mới (0/20)"
                        )
        except Exception:
            pass
        ctx.total_quests_done = max(ctx.total_quests_done, db_today, prog_today)
        self._save_progress(ctx)
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

    def _collapse_chat_bar(self, frame=None) -> bool:
        """Không thao tác vùng chat đáy giữa màn hình.

        Nút thu gọn chat nằm ngay trong vùng cấm; OCR có thể nhận sai
        trạng thái và biến cú bấm "thu gọn" thành mở/nhập chat.
        Giữ hàm no-op để các luồng cũ gọi an toàn.
        """
        return False

    def _prepare_ingame_hud_after_login(self, ctx: StateContext) -> None:
        """Thu gọn các khung HUD dễ che mục tiêu ngay sau khi vào game."""
        if ctx.extra.get("login_hud_collapsed"):
            return

        coords = self._loader.load_ui_coords() if hasattr(self, "_loader") else {}
        nav_cfg = coords.get("navigation", {})
        quest_arrow = nav_cfg.get("quest_tracker_collapse_button", {"x": 62, "y": 165})
        bottom_arrow = nav_cfg.get("chat_collapse_button", {"x": 490, "y": 843})

        # Chat bar nằm sát đáy màn hình, thường dưới vùng an toàn của thao tác
        # ADB/UI và nút thu gọn dễ bị lệch giữa các kích thước capture. Không
        # click mù vào đó; chỉ thu gọn quest tracker tại tọa độ đã cấu hình.
        logger.info(
            f"[{ctx.account_id}] Dọn HUD sau đăng nhập: thu gọn khung nhiệm vụ "
            f"tại ({quest_arrow['x']}, {quest_arrow['y']})..."
        )
        self._input.tap(quest_arrow["x"], quest_arrow["y"])
        time.sleep(0.2)
        ctx.extra["login_hud_collapsed"] = True
        self._checkpoint.save(ctx)

    def _dismiss_unexpected_popups(self, frame=None) -> bool:
        """Tự động phát hiện và đóng các popup/menu che màn hình ngoài ý muốn:
        1. Cài đặt / Tùy chọn / Hình ảnh (do phím ESC/Back hoặc click trúng gear)
        2. Kênh chat (do click trúng chat box ở dưới)
        3. Chiến trường (do click trúng Đấu trường ở góc dưới phải)
        """
        try:
            if frame is None and hasattr(self, "_screen") and self._screen is not None:
                frame = self._screen.capture()
            if frame is None or frame.shape[0] < 500 or frame.shape[1] < 800:
                return False

            # Nếu bảng bản đồ lớn đang mở, không xử lý popup để tránh nhận diện nhầm NPC (như 'Chủ Tạp hóa') và đóng nhầm bản đồ
            if self._is_map_open(frame):
                return False

            raw_boxes = self._text.read(frame)
            # Loại bỏ hoàn toàn thanh chat phía dưới màn hình (y >= 720) và hàng icon tiện ích/HUD cố định góc trên phải (x >= 800, y <= 165)
            boxes = [b for b in raw_boxes if b.center_y < 720 and not (b.center_x >= 800 and b.center_y <= 165)]
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
                raw_boxes = self._text.read(frame) if frame is not None else []
                boxes = [b for b in raw_boxes if b.center_y < 720 and not (b.center_x >= 800 and b.center_y <= 165)]

            # 2. Bảng Kênh chat (popup chat toàn màn hình/nửa màn hình)
            # Đóng bằng nút X ở góc trên phải bảng chat: (800, 135)
            is_chat = any(
                "kenh chat" in normalize_game_text(b.text) or ("tong hop" in normalize_game_text(b.text) and b.center_x < 400 and b.center_y < 250)
                for b in boxes
            )
            if is_chat:
                btn = nav_cfg.get("chat_close_button", {"x": 800, "y": 135})
                logger.info(f"Phát hiện bảng 'Kênh chat' đang mở! Đóng tại ({btn['x']}, {btn['y']})...")
                self._input.tap(btn["x"], btn["y"])
                time.sleep(0.3)
                dismissed = True
                frame = self._screen.capture()
                # Sau khi đóng bảng chat lớn, thu gọn luôn thanh chat dưới đáy màn hình nếu còn mở rộng
                self._collapse_chat_bar(frame)
                raw_boxes = self._text.read(frame) if frame is not None else []
                boxes = [b for b in raw_boxes if b.center_y < 720 and not (b.center_x >= 800 and b.center_y <= 165)]

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
            # 4. Bảng Hành trang (Túi đồ) đang mở ngoài ý muốn
            is_inv = any(
                b.center_x >= 1350 and b.center_y >= 150 and any(
                    k in normalize_game_text(b.text)
                    for k in ("sap xep", "thong tin", "tach", "bay ban", "shop kim phieu", "shop ngan phieu", "doi ma", "hanh trang")
                )
                for b in boxes
            )
            if is_inv:
                shop_cfg = coords.get("shop", {})
                btn_inv = shop_cfg.get("inventory_close", {"x": 1565, "y": 95})
                logger.info(f"Phát hiện bảng 'Hành trang / Túi đồ' đang mở ngoài ý muốn! Đóng tại ({btn_inv['x']}, {btn_inv['y']})...")
                self._input.tap(btn_inv["x"], btn_inv["y"])
                time.sleep(0.2)
                dismissed = True

            # 5. Bảng Cửa hiệu / Shop NPC mua bán đang mở ngoài ý muốn
            is_shop = any(
                (b.center_x <= 750 and b.center_y <= 250 and "cua hieu" in normalize_game_text(b.text))
                or any(k in normalize_game_text(b.text) for k in ("mua lai", "giam dinh", "sua chua", "trai cay ri"))
                for b in boxes
                if not (b.center_x <= 400 and b.center_y >= 200) # loại trừ danh sách NPC
            )
            if is_shop:
                shop_cfg = coords.get("shop", {})
                btn_close = shop_cfg.get("close_button", {"x": 679, "y": 110})
                logger.info(f"Phát hiện bảng 'Cửa hiệu NPC' đang mở ngoài ý muốn! Đóng tại ({btn_close['x']}, {btn_close['y']})...")
                self._input.tap(btn_close["x"], btn_close["y"])
                time.sleep(0.2)
                dismissed = True

            # 5. Bảng xác nhận tự động sắp xếp Hành trang: "Bạn đồng ý dùng chế độ tự động sắp xếp không?"
            is_sort_confirm = any(
                any(k in normalize_game_text(b.text) for k in ("tu dong sap xep", "dung che do tu dong", "sap xep khong"))
                for b in boxes
            )
            if is_sort_confirm:
                logger.info("Phát hiện popup 'Xác nhận tự động sắp xếp'! Bấm 'Hủy bỏ' để đóng...")
                cancel_boxes = [b for b in boxes if any(w in normalize_game_text(b.text) for w in ("huy bo", "huy")) and b.center_y > 450]
                if cancel_boxes:
                    self._input.tap(cancel_boxes[0].center_x, cancel_boxes[0].center_y)
                else:
                    shop_cfg = coords.get("shop", {})
                    btn_sort_cancel = shop_cfg.get("sort_cancel_button", {"x": 1059, "y": 546})
                    self._input.tap(btn_sort_cancel["x"], btn_sort_cancel["y"])
                time.sleep(0.3)
                dismissed = True

            # 6. Bảng thông báo / Cảnh báo hệ thống (chống lừa đảo, giả mạo NPH, đổi mật khẩu...)
            # Chỉ nhận diện khi có từ khóa cảnh báo trong vùng trung tâm VÀ có nút xác nhận đóng popup rõ ràng
            # Tuyệt đối không bấm Back vu vơ khi chỉ thấy dòng chữ chạy trên thanh chat / thông báo hệ thống làm dừng nhân vật
            is_warning = any(
                any(k in normalize_game_text(b.text) for k in ("canh bao", "fanpage gia mao", "tai san cua nguoi choi", "doi mat khau", "thay doi mat khau", "chu y:thuong xuyen thay doi mat khau"))
                for b in boxes if 200 <= b.center_y <= 650 and 300 <= b.center_x <= 1300
            )
            if is_warning:
                # Nút xác nhận phải nằm ở vùng giữa màn hình (x: 300..1300, y > 400), tránh icon "Động tác" (52, 646)
                confirm_btn = next((
                    b for b in boxes
                    if any(
                        normalize_game_text(b.text).strip() == k
                        or f" {k} " in f" {normalize_game_text(b.text)} "
                        for k in ("dong y", "xac nhan", "bo qua", "huy", "dong")
                    )
                    and "dong tac" not in normalize_game_text(b.text)
                    and 300 <= b.center_x <= 1300
                    and b.center_y > 400
                ), None)
                if confirm_btn:
                    logger.info(f"Phát hiện nút đóng popup cảnh báo '{confirm_btn.text}' tại ({confirm_btn.center_x}, {confirm_btn.center_y}), bấm để đóng...")
                    self._input.tap(confirm_btn.center_x, confirm_btn.center_y)
                    time.sleep(0.3)
                    dismissed = True
                else:
                    now = time.time()
                    last_back = getattr(self, "_last_warning_back_time", 0)
                    if now - last_back > 15.0:
                        self._last_warning_back_time = now
                        logger.info("Phát hiện popup cảnh báo không có nút bấm rõ ràng, dùng press_back để đóng an toàn...")
                        self._input.press_back()
                        time.sleep(0.3)
                        dismissed = True

            # 7. Bảng Ngự Các / Kỳ Trân Các
            # Chỉ nhận diện khi popup Ngự Các / Kỳ Trân Các thực sự mở ở trung tâm màn hình (y >= 200)
            # kèm theo các từ khóa đặc trưng của cửa hàng/nạp thẻ (tránh nhầm với icon Ngự Các trên HUD)
            is_ngu_cac = any(
                any(k in normalize_game_text(b.text) for k in ("ngu cac", "ky tran cac"))
                for b in boxes if 200 <= b.center_y <= 650 and 300 <= b.center_x <= 1300
            ) and any(
                any(k in normalize_game_text(b.text) for k in ("nap the", "kim nguyen bao", "gia ban", "so luong", "mua vat pham", "cua hang"))
                for b in boxes if 200 <= b.center_y <= 700
            )
            if is_ngu_cac:
                btn_close = nav_cfg.get("settings_close_button", {"x": 1412, "y": 170})
                logger.info(f"Phát hiện bảng 'Ngự Các' đang mở ngoài ý muốn! Bấm nút 'X' tại ({btn_close['x']}, {btn_close['y']}) để đóng...")
                self._input.tap(btn_close["x"], btn_close["y"])
                time.sleep(0.3)
                dismissed = True

            # 8. Bảng đối thoại NPC không mong muốn (Hướng dẫn tân thủ, Võ sư... có tùy chọn 'Rời khỏi')
            is_stray_dialog = any(
                any(k in normalize_game_text(b.text) for k in (
                    "vi bang huu", "chuyen gi", "tim hieu", "giang ho chi nam", "huong dan su dung", "roi khoi"
                ))
                for b in boxes if 250 <= b.center_y <= 850
            )
            if is_stray_dialog:
                exit_btn = next((b for b in boxes if "roi khoi" in normalize_game_text(b.text)), None)
                if exit_btn:
                    logger.info(f"Phát hiện hộp thoại NPC không mong muốn, bấm nút '{exit_btn.text}' tại ({exit_btn.center_x}, {exit_btn.center_y}) để thoát...")
                    self._input.tap(exit_btn.center_x, exit_btn.center_y)
                else:
                    logger.info("Phát hiện hộp thoại NPC lạ, tap an toàn tại (450, 770) để thoát...")
                    self._input.tap(450, 770)
                time.sleep(0.3)
                dismissed = True

            # 9. Thông báo trừ điểm danh vọng do vừa hủy nhiệm vụ: "Bạn bị trừ 5 điểm danh vọng vì hủy nhiệm vụ!"
            is_cancel_notice = any(
                any(k in normalize_game_text(b.text) for k in ("bi tru 5 diem", "tru 5 diem", "danh vong vi huy", "huy nhiem vu!"))
                for b in boxes if 250 <= b.center_y <= 650
            )
            if is_cancel_notice:
                logger.info("Phát hiện thông báo 'Bạn bị trừ 5 điểm danh vọng vì hủy nhiệm vụ!' che màn hình -> Click đúp để tắt thông báo...")
                self._input.tap(303, 571)
                time.sleep(0.4)
                self._input.tap(303, 571)
                time.sleep(0.6)
                dismissed = True

            return dismissed
        except Exception as e:
            logger.debug(f"_dismiss_unexpected_popups error: {e}")
            return False

    def _dismiss_dialogue_completion_banner(self, frame: Optional[np.ndarray] = None) -> bool:
        """Đã vô hiệu hóa: tuyệt đối không bấm vào vùng đáy màn hình (x ~ 735, y ~ 808).
        Bấm vào đây sẽ click trúng thanh chat dưới đáy, làm mở bảng Kênh chat che toàn màn hình.
        Giữ hàm trả về False để các luồng cũ gọi an toàn.
        """
        return False

    def _is_map_open(self, frame=None) -> bool:
        """Kiểm tra xem bảng bản đồ lớn có đang mở trên màn hình không."""
        try:
            if frame is None and hasattr(self, "_screen") and self._screen is not None:
                frame = self._screen.capture()
            if frame is None:
                return False
            h_f, w_f = frame.shape[:2]
            scale_x = w_f / 1600.0
            scale_y = h_f / 900.0
            crop = (int(100 * scale_x), int(130 * scale_y), int(450 * scale_x), int(230 * scale_y))
            boxes = self._text.read(frame, crop)
            return any(
                any(w in normalize_game_text(b.text) for w in ("tim map", "tim npc", "map", "npc"))
                for b in boxes
            )
        except Exception:
            return False

    def _ensure_map_open(self) -> bool:
        """Mở bản đồ nếu chưa mở, đảm bảo không bị popup Cài đặt/Chat che."""
        if self._is_map_open():
            return True
        nav = self._loader.load_ui_coords().get("navigation", {})
        minimap = nav.get("minimap_button", {"x": 1480, "y": 60})
        for attempt in range(1, 4):
            logger.info(f"Mở Minimap tại ({minimap.get('x', 1480)}, {minimap.get('y', 60)}) (lần {attempt})...")
            self._input.tap(minimap.get("x", 1480), minimap.get("y", 60))
            time.sleep(0.35)
            if self._is_map_open():
                return True
            self._dismiss_unexpected_popups()
            time.sleep(0.15)
        return False

    def _close_map(self) -> None:
        """Đóng bảng bản đồ an toàn tại nút X (1512, 68), tránh nút gear Cài đặt."""
        if not self._is_map_open():
            return
        nav = self._loader.load_ui_coords().get("navigation", {})
        close_btn = nav.get("map_close_button", {"x": 1512, "y": 68})
        bx = close_btn.get("x", 1512)
        by = close_btn.get("y", 68)
        logger.info(f"Đóng bản đồ an toàn tại ({bx}, {by})...")
        self._input.tap(bx, by)
        time.sleep(0.3)
        if self._is_map_open():
            self._input.tap(bx, by)
            time.sleep(0.2)

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

    def _check_left_quest_tracker(self, frame=None, scroll_to_bottom: bool = True, ctx: Optional[StateContext] = None) -> Optional[tuple[str, tuple[int, int]]]:
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
                if ctx:
                    self._sync_quest_progress_from_text(ctx, result[0], source="thanh nhiệm vụ HUD trái")
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
                    if ctx:
                        self._sync_quest_progress_from_text(ctx, scrolled_result[0], source="thanh nhiệm vụ HUD trái")
                    return scrolled_result

            return None
        except Exception as e:
            logger.debug(f"_check_left_quest_tracker error: {e}")
            return None

    def _is_quest_panel_open(self, frame: Optional[np.ndarray] = None) -> bool:
        """Kiểm tra xem Bảng nhiệm vụ có đang mở hay không."""
        try:
            if frame is None:
                frame = self._screen.capture() if hasattr(self, "_screen") and self._screen is not None else None
            if frame is None or not (hasattr(frame, "shape") and len(getattr(frame, "shape", ())) == 3):
                return False
            if frame.shape[1] < 500 or frame.shape[0] < 300:
                return False
            if not hasattr(self, "_text") or self._text is None:
                return False
            from .text_recognition import normalize_game_text
            header_boxes = self._text.read(frame, (260, 40, 1350, 190))
            # Tiêu đề "Bảng nhiệm vụ" hoặc đồng thời có "Danh sách nhiệm vụ" và "Nội dung nhiệm vụ"
            texts = [normalize_game_text(b.text) for b in header_boxes]
            texts_ns = [t.replace(" ", "") for t in texts]
            has_title = any(
                "bang nhiem vu" in t or "bangnhiemvu" in ns or "bangnhiem" in ns
                for t, ns in zip(texts, texts_ns)
            )
            has_columns = (
                any("danh sach" in t or "sach" in ns for t, ns in zip(texts, texts_ns))
                and any("noi dung" in t or "noidung" in ns for t, ns in zip(texts, texts_ns))
            )
            return has_title or has_columns
        except Exception:
            return False

    def _ensure_quest_panel_closed(
        self,
        ctx: Optional[StateContext] = None,
        frame: Optional[np.ndarray] = None,
        max_attempts: int = 3,
        assume_open: bool = False,
    ) -> bool:
        """Đảm bảo đóng hoàn toàn Bảng nhiệm vụ bằng nút X hoặc phím BACK."""
        tag = f"[{ctx.account_id}] " if ctx and hasattr(ctx, "account_id") and ctx.account_id else ""
        try:
            current_frame = frame
            if current_frame is None:
                current_frame = self._screen.capture() if hasattr(self, "_screen") and self._screen is not None else None

            # Không bao giờ bấm tọa độ nút X nếu chưa xác nhận panel đang mở.
            # Khi panel đã đóng, vị trí này nằm trên vùng minimap và cú bấm sẽ
            # mở bản đồ ngay trước lúc bắt đầu tuần tra combat.
            if not assume_open:
                if current_frame is None or not self._is_quest_panel_open(current_frame):
                    return True

            # The quest panel close button is the orange diamond at roughly
            # 91.6% width / 13.9% height. Old YAML coordinates pointed at the
            # top-right HUD instead of this button and left the panel open.
            if current_frame is not None and hasattr(current_frame, "shape"):
                h_frame, w_frame = current_frame.shape[:2]
                cx = int(round(w_frame * 0.916))
                cy = int(round(h_frame * 0.139))
            else:
                coords = self._loader.load_ui_coords() if hasattr(self, "_loader") else {}
                close_btn = coords.get("commerce_quest", {}).get("quest_panel_close", {"x": 1466, "y": 125})
                cx = close_btn.get("x", 1466)
                cy = close_btn.get("y", 125)

            for attempt in range(max_attempts):
                logger.info(f"{tag}Đóng Bảng nhiệm vụ (lần {attempt + 1}) tại ({cx}, {cy})...")
                self._input.tap(cx, cy)
                time.sleep(0.3)
                if not self._is_quest_panel_open():
                    return True

                # Nếu vẫn chưa đóng: thử gửi phím BACK
                logger.info(f"{tag}Bảng nhiệm vụ vẫn còn sau nút X, gửi phím BACK/ESC để đóng dứt điểm...")
                if hasattr(self._input, "press_back"):
                    self._input.press_back()
                elif hasattr(self._input, "key_event"):
                    self._input.key_event(4)
                elif hasattr(self._input, "press_key"):
                    self._input.press_key("escape")
                time.sleep(0.3)
                if not self._is_quest_panel_open():
                    return True

            return not self._is_quest_panel_open()
        except Exception as e:
            logger.debug(f"{tag}_ensure_quest_panel_closed error: {e}")
            return True

    def _check_quest_panel(self, ctx: Optional[StateContext] = None) -> Optional[tuple[str, tuple[int, int]]]:
        """
        : Nhấn vào icon nhiệm vụ bên góc phải dưới để mở danh sách nhiệm vụ.
        Nếu có nhiệm vụ Thương Hội trong danh sách:
        - Chọn nhiệm vụ để đọc nội dung / lấy link xanh auto-run.
        - Đóng bảng nhiệm vụ lại bằng nút X.
        """
        tag = f"[{ctx.account_id}] " if ctx and hasattr(ctx, "account_id") and ctx.account_id else ""
        from .text_recognition import normalize_game_text
        try:
            coords = self._loader.load_ui_coords() if hasattr(self, "_loader") and self._loader else {}
            quest_btn = coords.get("commerce_quest", {}).get("quest_button", {"x": 1403, "y": 755})
            close_btn = coords.get("commerce_quest", {}).get("quest_panel_close", {"x": 1466, "y": 125})

            # Đảm bảo đóng minimap nếu đang mở (trong môi trường chạy thật)
            if hasattr(self, "_vision") and hasattr(self, "_is_map_open") and self._is_map_open():
                logger.info(f"{tag}Bản đồ đang mở, đóng trước khi mở bảng nhiệm vụ...")
                self._close_map()
                time.sleep(0.4)

            # Đảm bảo không bị hộp thoại NPC che khuất nút Nhiệm vụ
            if hasattr(self, "_vision") and hasattr(self, "_is_npc_dialog_open") and self._is_npc_dialog_open():
                logger.info(f"{tag}Hộp thoại NPC đang mở, đóng trước khi mở bảng nhiệm vụ...")
                self._ensure_npc_dialog_closed()
                time.sleep(0.4)

            # Kiểm tra xem icon Nhiệm vụ có đang hiển thị không (hay đang bị ẩn bởi bộ phím kỹ năng)
            qx = int(quest_btn.get("x", 1403))
            qy = int(quest_btn.get("y", 755))
            if qx > 1435:
                qx, qy = 1403, 755
            scr_chk = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
            if scr_chk is not None and hasattr(self, "_text") and self._text:
                rb_boxes = self._text.read(scr_chk, (1050, 680, 1600, 900))
                nv_box = next((b for b in rb_boxes if any(k in normalize_game_text(b.text) for k in ("nhiem vu", "nhiem", "nhiemvu")) and 1360 <= b.center_x <= 1500 and 720 <= b.center_y <= 800), None)
                if nv_box:
                    # Nếu OCR gộp với icon Tổ đội bên cạnh làm tâm bị kéo sang phải (> 1435), đưa về tâm nút Nhiệm vụ là 1403
                    if nv_box.center_x > 1435:
                        qx, qy = 1403, nv_box.center_y
                    else:
                        qx, qy = nv_box.center_x, nv_box.center_y
                else:
                    # Nếu không thấy icon Nhiệm vụ, menu chức năng có thể đang đóng (chế độ phím kỹ năng)
                    # Bấm vào icon 'Tùy chọn' (1080, 876) để mở menu ra
                    opt_boxes = self._text.read(scr_chk, (1000, 820, 1150, 900))
                    if any("tuy chon" in normalize_game_text(b.text) for b in opt_boxes):
                        logger.info(f"{tag}Không thấy icon Nhiệm vụ, bấm nút 'Tùy chọn' (1080, 876) để mở menu chức năng...")
                        self._input.tap(1080, 876)
                        time.sleep(0.4)

            # The panel can already be open from a previous state/retry. Tapping
            # the quest icon again may toggle it or leave us scanning the game
            # HUD as if it were the quest list.
            panel_open = self._is_quest_panel_open(scr_chk) if scr_chk is not None else False
            scr = scr_chk
            if panel_open:
                logger.info(f"{tag}Bảng nhiệm vụ đã mở sẵn; đọc danh sách hiện tại, không bấm lại icon.")
            else:
                logger.info(f"{tag}: Bấm vào icon Nhiệm vụ góc phải dưới ({qx}, {qy})...")
                self._input.tap(qx, qy)
                # Wait for the panel header to appear instead of assuming a
                # fixed one-second animation; capture each retry for OCR below.
                for panel_attempt in range(5):
                    time.sleep(0.35)
                    scr = self._screen.capture()
                    if scr is not None and self._is_quest_panel_open(scr):
                        panel_open = True
                        break
                if not panel_open:
                    # Thử kiểm tra lại trên frame mới nhất xem bảng đã bung lên chưa
                    scr = self._screen.capture()
                    if scr is not None and self._is_quest_panel_open(scr):
                        panel_open = True
                    else:
                        logger.warning(
                            f"{tag}Chưa mở được Bảng nhiệm vụ sau khi bấm icon Nhiệm vụ ({qx}, {qy})! "
                            "Tuyệt đối không quét chữ ngoài màn hình để tránh báo nhầm không có nhiệm vụ."
                        )
                        return None

            if scr is None:
                scr = self._screen.capture()
            if scr is None:
                self._ensure_quest_panel_closed(ctx=ctx)
                return None

            # ====================================================================
            # BƯỚC 1: Tìm và click trực tiếp vào dòng "Nhiệm vụ Thương hội"
            # (Bỏ qua nhánh "Tuần hoàn", bấm thẳng vào Thương hội)
            # ====================================================================
            list_boxes = self._text.read(scr, (320, 130, 660, 750))
            if ctx and list_boxes:
                list_raw = " ".join(b.text for b in list_boxes)
                self._sync_quest_progress_from_text(ctx, list_raw, source="danh sách Bảng nhiệm vụ")
            quest_item_pos = None
            for b in list_boxes:
                norm = normalize_game_text(b.text)
                if "thuong hoi" in norm or "thuonghoi" in norm.replace(" ", ""):
                    quest_item_pos = (b.center_x, b.center_y)
                    logger.info(f"{tag}Phát hiện dòng 'Thương hội' qua OCR tại {quest_item_pos}")
                    break

            if not quest_item_pos:
                fallback_boxes = self._text.read(scr, (200, 120, 700, 780))
                for b in fallback_boxes:
                    norm = normalize_game_text(b.text)
                    if "thuong hoi" in norm or "thuonghoi" in norm.replace(" ", ""):
                        quest_item_pos = (b.center_x, b.center_y)
                        logger.info(f"{tag}Phát hiện dòng 'Thương hội' qua OCR vùng rộng tại {quest_item_pos}")
                        break

            if not quest_item_pos:
                if panel_open or (scr is not None and self._is_quest_panel_open(scr)):
                    # Bảng nhiệm vụ đang mở: ưu tiên dùng tọa độ cấu hình chuẩn dòng 'Nhiệm vụ Thương hội'
                    row_cfg = coords.get("commerce_quest", {}).get(
                        "quest_panel_merchant_row", {"x": 474, "y": 428}
                    )
                    quest_item_pos = (int(row_cfg["x"]), int(row_cfg["y"]))
                    logger.info(f"{tag}Bảng nhiệm vụ đang mở: dùng tọa độ cấu hình dòng 'Nhiệm vụ Thương hội' tại {quest_item_pos}")
                else:
                    logger.warning(f"{tag}Bảng nhiệm vụ chưa mở được (panel_open=False) và không thấy dòng Thương hội! Bỏ qua thay vì bấm ra ngoài màn hình.")
                    self._ensure_quest_panel_closed(ctx=ctx)
                    return None

            if not quest_item_pos:
                logger.info(f"{tag}Không có nhiệm vụ Thương hội trong bảng nhiệm vụ. (OCR boxes: {[(b.text, b.center_x, b.center_y) for b in list_boxes]})")
                # Đóng bảng nhiệm vụ
                self._ensure_quest_panel_closed(ctx=ctx)
                time.sleep(0.25)
                return None

            # Bấm vào dòng nhiệm vụ Thương hội để hiện chi tiết ở bảng bên phải
            logger.info(f"{tag}Click chọn nhiệm vụ Thương hội tại {quest_item_pos}...")
            self._input.tap(*quest_item_pos)
            time.sleep(0.5)

            # Chờ nội dung chi tiết hiển thị ở cột bên phải (x: 650..1440, y: 130..620)
            detail_boxes = []
            detail_scr = None
            for attempt in range(5):
                detail_scr = self._screen.capture()
                if detail_scr is not None:
                    boxes_r = self._text.read(detail_scr, (650, 130, 1440, 620))
                    # Lọc những box có nội dung thực tế (không chỉ là tiêu đề 'Nội dung nhiệm vụ')
                    meaningful = [
                        b for b in boxes_r
                        if not any(k in normalize_game_text(b.text) for k in ("noi dung nhiem vu", "nhiem vu"))
                    ]
                    if meaningful or not hasattr(detail_scr, "shape"):
                        detail_boxes = boxes_r
                        break
                if attempt == 2 and quest_item_pos:
                    # Nếu sau 2 lần vẫn chưa hiện nội dung bên phải, thử tap lại vào dòng Thương hội
                    logger.info(f"{tag}Khung bên phải chưa hiện nội dung, click lại vào Thương hội tại {quest_item_pos}...")
                    self._input.tap(*quest_item_pos)
                time.sleep(0.4)

            if not detail_boxes and detail_scr is not None:
                detail_boxes = self._text.read(detail_scr, (650, 130, 1440, 620))
            detail_text = " ".join(b.text for b in detail_boxes) if detail_boxes else "Nhiệm vụ Thương hội"
            logger.info(f"{tag}Nội dung chi tiết nhiệm vụ OCR: '{detail_text}'")
            if ctx and detail_text:
                self._sync_quest_progress_from_text(ctx, detail_text, source="chi tiết Bảng nhiệm vụ")

            detail_norm = normalize_game_text(detail_text)
            if any(w in detail_norm for w in ("vi trieu dinh", "cong hien suc luc", "hoang thuong vua ban chieu", "ban chieu")):
                logger.info(f"{tag}Phát hiện đây là hộp thoại nhận nhiệm vụ từ Chủ thương hội (chưa có NV), không phải Bảng nhiệm vụ.")
                return None

            # The panel must remain open after selecting a row. If it closed,
            # the coordinate was a miss: never treat unrelated HUD OCR as the
            # quest description or extract entities from that screen.
            panel_after_select = bool(
                detail_scr is not None
                and (self._is_quest_panel_open(detail_scr) or not hasattr(detail_scr, "shape"))
            )
            has_quest_detail = any(
                key in detail_norm
                for key in (
                    "thuong hoi", "thanh do", "bien kinh", "tuyen chau", "tuong duong",
                    "duong chau", "dai ly", "phuong tuong", "mua", "tieu diet", "danh",
                    "thu thap", "gap", "noi", "hoi", "binh dinh quoc", "linh duoc",
                )
            )
            if not panel_after_select or not has_quest_detail:
                logger.warning(
                    f"{tag}Chưa xác nhận được Bảng nhiệm vụ đang mở sau khi chọn Thương hội "
                    f"(panel_open={panel_after_select}, detail='{detail_text[:180]}'); bỏ qua OCR ngoài bảng."
                )
                if panel_after_select:
                    self._ensure_quest_panel_closed(ctx=ctx, frame=detail_scr, assume_open=True)
                if ctx and hasattr(ctx, "extra"):
                    ctx.extra["auto_walking"] = False
                return None

            has_active_npc_turnin = (
                "cung voi" in detail_norm and any(
                    k in detail_norm for k in ("doi thoai", "hoan thanh", "hoan thanhnhiem vu")
                )
            ) or (
                "doi thoai" in detail_norm and "hoan thanh nhiem vu" in detail_norm
            )

            # Trích xuất các thực thể chữ màu vàng (vật phẩm, quái, số lượng, NPC, vị trí) trong khung nội dung bên phải
            yellow_tokens = []
            entities = {}
            if detail_scr is not None and hasattr(detail_scr, "shape") and len(getattr(detail_scr, "shape", ())) == 3:
                try:
                    from .quest_classifier import extract_yellow_text_boxes, parse_yellow_entities
                    y_boxes = extract_yellow_text_boxes(detail_scr, (650, 130, 1440, 620))
                    for b in y_boxes:
                        norm_b = normalize_game_text(b.text)
                        if any(w in norm_b for w in ("thuong hoi", "tuan hoan", "nhiem vu", "danh sach", "noi dung")):
                            continue
                        yellow_tokens.append(b.text)

                    if yellow_tokens:
                        logger.info(f"{tag}Phát hiện các từ khóa màu vàng trong nhiệm vụ: {yellow_tokens}")
                        entities = parse_yellow_entities(yellow_tokens)
                        logger.info(f"{tag}Thực thể trích xuất được từ chữ vàng: {entities}")
                except Exception as ex:
                    logger.debug(f"Yellow entities extraction failed: {ex}")

            # Chỉ lưu thực thể vào ctx.extra khi Bảng nhiệm vụ THỰC SỰ ĐANG MỞ HỢP LỆ
            if yellow_tokens and ctx and hasattr(ctx, "extra"):
                ctx.extra["yellow_tokens"] = yellow_tokens
                has_yellow_combat = bool(
                    (entities.get("monster") and int(entities.get("count") or 0) > 0)
                    or (entities.get("map_name") and entities.get("monster"))
                )
                if not has_active_npc_turnin or has_yellow_combat:
                    if entities.get("item") and not has_yellow_combat:
                        ctx.extra["yellow_item"] = entities["item"]
                        ctx.extra["requested_shop_item"] = entities["item"]
                    if entities.get("map_name"):
                        ctx.extra["combat_map"] = entities["map_name"]
                    if entities.get("sect"):
                        ctx.extra["combat_sect"] = entities["sect"]
                    if entities.get("monster"):
                        ctx.extra["combat_monster"] = entities["monster"]
                    if entities.get("count"):
                        ctx.extra["combat_count"] = entities["count"]
                if entities.get("city") and not has_yellow_combat:
                    ctx.extra["quest_target_city"] = entities["city"]
                if entities.get("npc") and not has_yellow_combat:
                    ctx.extra["quest_target_npc"] = entities["npc"]

            # Luôn đọc cả câu mô tả để bổ sung trường mà OCR chữ vàng bỏ sót.
            # Trường hợp thực tế: chữ vàng chỉ có ['Nt Kiem khach', '10'], còn
            # địa điểm 'Nga My Phái' nằm trong phần chữ trắng.
            if ctx and hasattr(ctx, "extra"):
                from .quest_classifier import extract_combat_target, extract_dialogue_target
                has_yellow_combat = bool(
                    (ctx.extra.get("combat_monster") and int(ctx.extra.get("combat_count") or 0) > 0)
                    or (ctx.extra.get("combat_map") and ctx.extra.get("combat_monster"))
                )
                if not has_active_npc_turnin or has_yellow_combat:
                    c_map, c_monster, c_count = extract_combat_target(detail_text, yellow_tokens)
                    if c_monster and not ctx.extra.get("combat_monster"):
                        ctx.extra["combat_monster"] = c_monster
                    if c_map and not ctx.extra.get("combat_map"):
                        ctx.extra["combat_map"] = c_map
                    if c_monster or c_map:
                        ctx.extra.pop("quest_target_city", None)
                        ctx.extra.pop("quest_target_npc", None)
                else:
                    for stale in ("combat_monster", "combat_map", "combat_count", "combat_sect", "abandon_combat_quest"):
                        ctx.extra.pop(stale, None)

                is_combat_detail = bool(ctx.extra.get("combat_monster") or ctx.extra.get("combat_map"))
                if not is_combat_detail and (not ctx.extra.get("quest_target_npc") or has_active_npc_turnin):
                    d_city, d_npc = extract_dialogue_target(detail_text, with_accents=True, yellow_tokens=yellow_tokens)
                    if d_city and not ctx.extra.get("quest_target_city"):
                        ctx.extra["quest_target_city"] = d_city
                    if d_npc:
                        ctx.extra["quest_target_npc"] = d_npc

                from .quest_classifier import is_auto_abandon_dialogue_quest
                if is_auto_abandon_dialogue_quest(npc=ctx.extra.get("quest_target_npc", ""), text=detail_text):
                    logger.warning(
                        f"[{ctx.account_id}] ⚠️ Phát hiện nhiệm vụ gặp NPC '{ctx.extra.get('quest_target_npc', 'Ấu đồng')}' "
                        "(thuộc danh sách tự động hủy). Đánh dấu cờ hủy nhiệm vụ..."
                    )
                    ctx.extra["abandon_dialogue_quest"] = True
                    ctx.extra["abandon_combat_quest"] = True
                    ctx.extra["abandon_combat_quest_reason"] = f"Nhiệm vụ gặp NPC {ctx.extra.get('quest_target_npc', 'Ấu đồng')} (tự động hủy)"

            # Đóng bảng nhiệm vụ
            self._ensure_quest_panel_closed(ctx=ctx, frame=detail_scr, assume_open=True)

            if ctx and hasattr(ctx, "extra"):
                ctx.extra["auto_walking"] = False

            return detail_text, None

        except Exception as e:
            logger.warning(f"{tag}_check_quest_panel error: {e}")
            try:
                self._ensure_quest_panel_closed(ctx=ctx)
            except Exception:
                pass
            return None

    def _check_active_commerce_quest(self, ctx: Optional[StateContext] = None, frame=None, full_check: bool = True) -> Optional[tuple[str, tuple[int, int]]]:
        """
        Chỉ mở icon Nhiệm vụ để đọc lại nội dung của nhiệm vụ đã được Chủ
        thương hội xác nhận là đang tồn tại. Kết quả hàm này không được dùng để
        quyết định tài khoản có nhiệm vụ hay chưa.
        """
        return self._check_quest_panel(ctx=ctx)

    def _is_update_screen(self, frame=None) -> bool:
        """Detect the launcher update check and keep automation from tapping it."""
        try:
            from .text_recognition import normalize_game_text
            frame = frame if frame is not None else self._screen.capture()
            if frame is None:
                return False

            # Nếu đã có HUD in-game thì chắc chắn không phải màn hình update
            if self._is_ingame(frame=frame):
                return False

            boxes = self._text.read(frame)
            for b in boxes:
                norm = normalize_game_text(b.text)
                if (
                    "kiem tra cap nhat" in norm
                    or "kiem tra tai nguyen" in norm
                    or ("cap nhat" in norm and "dang dien ra" in norm)
                    or ("cho doi" in norm and "giay lat" in norm)
                    or "tai xuong" in norm
                    or "du lieu dang" in norm
                ):
                    return True
            return False
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

            # : Quét OCR các từ khóa đặc trưng của màn hình đăng nhập
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
                    time.sleep(1.0)
                    return True

            # 2. Không thấy game trên màn hình chính -> Bấm nút mũi tên ở góc trái dưới mở drawer
            coords = self._loader.load_ui_coords()
            arrow = coords.get("emulator_home", {}).get("app_drawer_button", {"x": 62, "y": 874})
            logger.info(f"Không thấy game trên màn hình chính. Bấm nút mũi tên góc trái dưới ({arrow['x']}, {arrow['y']}) để mở danh sách app...")
            self._input.tap(arrow["x"], arrow["y"])
            time.sleep(0.8)

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
                        time.sleep(1.0)
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

        # 1. Kiểm tra nếu đã trong game rồi TRƯỚC TIÊN (để không bị nhầm màn hình update)
        frame = self._screen.capture()
        if self._is_ingame(frame=frame):
            if getattr(self, "_skip_login", False):
                logger.info(f"[{ctx.account_id}] Game already running and in-game (skip_login enabled)!")
                activity._write("🎮", f"[{ctx.account_id}] Game dang chay san, tim Chu thuong hoi...")
                return BotState.FIND_COMMERCE_CHIEF
            else:
                logger.info(f"[{ctx.account_id}] Game đang mở tài khoản trước, đăng xuất để đăng nhập đúng tài khoản {ctx.account_id}...")
                self._logout_current_account(ctx)
                time.sleep(0.8)
                return BotState.LOGIN

        # 2. Kiểm tra màn hình đăng nhập trước
        if self._is_login_screen(frame=frame):
            logger.info(f"[{ctx.account_id}] Game đã vào sẵn màn hình đăng nhập -> BotState.LOGIN")
            activity._write("🔑", f"[{ctx.account_id}] Da vao man hinh dang nhap...")
            return BotState.LOGIN

        # 3. Kiểm tra màn hình update
        if self._is_update_screen(frame=frame):
            logger.info(f"[{ctx.account_id}] Game is checking updates; waiting without tapping...")
            time.sleep(2.0)
            return BotState.START_GAME

        # Chưa vào game → mở game: ưu tiên tìm icon trên màn hình / mở drawer góc trái dưới
        logger.info(f"[{ctx.account_id}] Game not detected, launching...")
        activity._write("⏳", f"[{ctx.account_id}] Dang mo game Tinh Kiem Truyen Ky...")
        opened = self._open_game_from_launcher()
        if not opened:
            pkg = self._app_cfg.game.package_name
            logger.info(f"ADB: starting app {pkg}")
            self._adb._run(["shell", "monkey", "-p", pkg, "-c", "android.intent.category.LAUNCHER", "1"])

        # Nhận diện tức thì: kiểm tra liên tục mỗi 0.3s, màn hình đăng nhập hiện ra là vào nhập ngay
        logger.info(f"[{ctx.account_id}] Đang chờ game mở lên màn hình đăng nhập (nhận diện ngay)...")
        for _ in range(40):
            time.sleep(0.3)
            frame = self._screen.capture()
            if self._is_login_screen(frame=frame):
                logger.info(f"[{ctx.account_id}] Màn hình đăng nhập đã sẵn sàng! Chuyển sang LOGIN.")
                activity._write("🔑", f"[{ctx.account_id}] Da vao man hinh dang nhap...")
                return BotState.LOGIN
            if self._is_ingame(frame=frame):
                if getattr(self, "_skip_login", False):
                    logger.info(f"[{ctx.account_id}] Game started directly into in-game!")
                    return BotState.FIND_COMMERCE_CHIEF
                else:
                    self._logout_current_account(ctx)
                    time.sleep(0.8)
                    return BotState.LOGIN
            if self._is_update_screen(frame=frame):
                logger.info(f"[{ctx.account_id}] Game đang tải dữ liệu cập nhật... Chờ tải xong.")
                activity._write("⏳", f"[{ctx.account_id}] Dang tai cap nhat...")
                time.sleep(3.0)
                return BotState.START_GAME

        # Nếu sau thời gian chờ mà chưa thấy màn hình login (có thể game load chậm hoặc đang tải)
        frame = self._screen.capture()
        if self._is_update_screen(frame=frame):
            logger.info(f"[{ctx.account_id}] Game vẫn đang tải cập nhật, tiếp tục chờ...")
            time.sleep(3.0)
            return BotState.START_GAME

        if not self._is_login_screen(frame=frame):
            logger.warning(f"[{ctx.account_id}] Chưa thấy màn hình đăng nhập, tiếp tục chờ game khởi động...")
            time.sleep(2.0)
            return BotState.START_GAME

        return BotState.LOGIN


    def _clear_and_type(self, x: int, y: int, text: str):
        """Click vào ô input, xóa text cũ nếu có, chờ game focus rồi nhập text ổn định."""
        self._input.tap(x, y)
        time.sleep(0.15)  # Chờ game kích hoạt ô input
        try:
            del_67 = " ".join(["67"] * 30)
            del_112 = " ".join(["112"] * 15)
            self._adb._run(["shell", f"input keyevent 123 {del_67} 122 {del_112}"])
            time.sleep(0.05)
        except Exception:
            pass
        self._clear_focused_and_type(text)

    def _clear_focused_and_type(self, text: str):
        """Nhập text vào ô hiện đang có focus mà không bị keyevent xóa đè."""
        time.sleep(0.05)
        self._input.input_text(str(text))
        time.sleep(0.1)

    def _replace_and_type(self, x: int, y: int, text: str) -> None:
        """Xóa giá trị cũ rồi nhập giá trị mới vào ô tọa độ."""
        self._input.tap(x, y)
        time.sleep(0.15)
        try:
            del_67 = " ".join(["67"] * 24)
            self._adb._run(["shell", f"input keyevent 123 {del_67}"])
        except Exception:
            pass
        self._clear_focused_and_type(str(text))

    def _click_and_type_password(self, x: int, y: int, password: str) -> bool:
        """Click ô mật khẩu, xóa text cũ, nhập và xác nhận nội dung trong ô đã thay đổi."""
        x1, x2 = max(0, x - 100), x + 230
        y1, y2 = max(0, y - 28), y + 28

        for attempt in range(1, 3):
            before = self._screen.capture()[y1:y2, x1:x2].copy()
            click_x = x if attempt == 1 else x + 30
            logger.info(
                f"Click ô mật khẩu ({click_x}, {y}) để focus "
                f"(lần {attempt}/2)..."
            )
            self._input.tap(click_x, y)
            time.sleep(0.15)
            try:
                del_67 = " ".join(["67"] * 30)
                del_112 = " ".join(["112"] * 15)
                self._adb._run(["shell", f"input keyevent 123 {del_67} 122 {del_112}"])
                time.sleep(0.05)
            except Exception:
                pass
            self._clear_focused_and_type(password)
            time.sleep(0.15)

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

        click_coord = None
        has_popup = False
        try:
            res_popup = self._vision.find_template("btn_tro_lai", frame=scr)
            if res_popup.found or (res_popup.confidence is not None and res_popup.confidence >= 0.60):
                click_coord = (res_popup.center_x, res_popup.center_y)
                has_popup = True
        except Exception:
            pass

        # Quét OCR để nhận diện thông báo lỗi chính xác
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
        elif has_popup:
            detected_error = "Lỗi đăng nhập (popup thông báo)"

        # Tìm tọa độ nút "Trở lại" qua OCR nếu chưa tìm thấy qua template
        if not click_coord and boxes:
            for b in boxes:
                norm_b = normalize_game_text(b.text)
                if "trolai" in norm_b or "tro lai" in norm_b:
                    click_coord = (b.center_x, b.center_y)
                    break

        if not detected_error and not click_coord:
            return False, ""

        if not click_coord:
            click_coord = (popup_btn.get("x", 771), popup_btn.get("y", 395))

        logger.warning(f"Phát hiện popup lỗi đăng nhập: '{detected_error}', bấm 'Trở lại' tại {click_coord}...")
        self._input.tap(click_coord[0], click_coord[1])
        time.sleep(0.8)

        # Kiểm tra lại xem popup đã đóng chưa
        try:
            scr2 = self._screen.capture()
            if scr2 is not None:
                res2 = self._vision.find_template("btn_tro_lai", frame=scr2)
                if res2.found or (res2.confidence is not None and res2.confidence >= 0.65):
                    self._input.tap(click_coord[0], click_coord[1])
                    time.sleep(0.5)
        except Exception:
            pass

        return True, detected_error

    def _handle_login(self, ctx: StateContext) -> BotState:
        if self._is_ingame():
            logger.info(f"[{ctx.account_id}] Đã ở trong game -> Chuyển sang FIND_COMMERCE_CHIEF")
            activity._write("🎮", f"[{ctx.account_id}] Da o trong game, tim Chu thuong hoi...")
            ctx.extra["check_previous_quest_on_login"] = True
            self._prepare_ingame_hud_after_login(ctx)
            return BotState.FIND_COMMERCE_CHIEF

        if self._is_update_screen():
            logger.info(f"[{ctx.account_id}] Update screen is active; returning to safe wait state...")
            time.sleep(3.0)
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
        is_login = self._is_login_screen()
        if not is_login:
            if getattr(self, "_skip_login", False) or self._is_ingame():
                logger.info(f"[{ctx.account_id}] Đã ở trong game hoặc skip_login=True -> Chuyển sang FIND_COMMERCE_CHIEF")
                ctx.extra["check_previous_quest_on_login"] = True
                self._prepare_ingame_hud_after_login(ctx)
                return BotState.FIND_COMMERCE_CHIEF
            # Kiểm tra nếu đang ở màn chọn nhân vật sẵn -> chuyển sang VERIFY_INGAME để bấm Vào Game
            scr_chk = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
            if scr_chk is not None:
                header_res = self._vision.find_template("header_chon_nhan_vat", frame=scr_chk)
                char_res = self._vision.find_template("char_enter_btn", frame=scr_chk)
                if header_res.found or (char_res.found and char_res.center_x > 1200):
                    logger.info(f"[{ctx.account_id}] Đang ở màn chọn nhân vật sẵn -> Chuyển sang VERIFY_INGAME để bấm Vào Game")
                    return BotState.VERIFY_INGAME
                if hasattr(self, "_text") and self._text:
                    c_boxes = self._text.read(scr_chk, (800, 50, 1600, 850))
                    if any(any(k in normalize_game_text(b.text) for k in ("chon nhan vat", "vao game", "tao nhan vat")) for b in c_boxes):
                        logger.info(f"[{ctx.account_id}] Phát hiện màn chọn nhân vật qua OCR -> Chuyển sang VERIFY_INGAME")
                        return BotState.VERIFY_INGAME
        elif getattr(self, "_skip_login", False):
            logger.info(f"[{ctx.account_id}] skip_login=True -> Chuyển sang FIND_COMMERCE_CHIEF")
            ctx.extra["check_previous_quest_on_login"] = True
            return BotState.FIND_COMMERCE_CHIEF

        logger.info(f"[{ctx.account_id}] Logging in as {account.username}...")
        activity._write("👤", f"[{ctx.account_id}] Dang dang nhap tai khoan: {account.username}...")

        # Chờ màn hình login xuất hiện nếu chưa thấy
        if not is_login:
            try:
                self._vision.wait_for_template(
                    "login_screen",
                    timeout=min(self._app_cfg.timing.login_timeout, 8),
                )
            except BotError:
                logger.warning(f"[{ctx.account_id}] Login screen template not detected, proceeding to click input coordinates directly")

        # Lấy tọa độ UI từ config
        coords = self._loader.load_ui_coords()
        login_cfg = coords.get("login", {})

        user_x = login_cfg.get("username_field", {}).get("x", 860)
        user_y = login_cfg.get("username_field", {}).get("y", 318)
        pass_x = login_cfg.get("password_field", {}).get("x", 860)
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

        # Chờ màn chọn nhân vật một cách chủ động (polling mỗi 0.25s thay vì sleep cứng)
        first_slot = login_cfg.get(
            "character_first_slot", {"x": 1340, "y": 235}
        )
        enter_btn = login_cfg.get("character_enter_btn", {"x": 1425, "y": 685})

        start_wait_char = time.time()
        while time.time() - start_wait_char < 8.0:
            time.sleep(0.25)
            frame = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
            if frame is None:
                continue

            # Nếu đã trực tiếp vào game (bỏ qua chọn nhân vật)
            if self._is_ingame(frame=frame):
                self._prepare_ingame_hud_after_login(ctx)
                return BotState.FIND_COMMERCE_CHIEF

            # Kiểm tra popup lỗi đăng nhập
            has_err, _ = self._check_and_dismiss_login_error(frame)
            if has_err:
                break

            header_res = self._vision.find_template("header_chon_nhan_vat", frame=frame)
            char_res = self._vision.find_template("char_enter_btn", frame=frame)
            if header_res.found or (char_res.found and char_res.center_x > 1200):
                click_enter_x = char_res.center_x if (char_res.found and char_res.center_x > 1200) else enter_btn.get("x", 1425)
                click_enter_y = char_res.center_y if (char_res.found and char_res.center_x > 1200) else enter_btn.get("y", 685)
                logger.info(
                    f"[{ctx.account_id}] Màn hình chọn nhân vật đã sẵn sàng! Chọn nhân vật đầu tiên "
                    f"({first_slot['x']}, {first_slot['y']})..."
                )
                self._input.tap(first_slot["x"], first_slot["y"])
                time.sleep(0.2)
                logger.info(
                    f"[{ctx.account_id}] Bấm Vào Game "
                    f"({click_enter_x}, {click_enter_y})..."
                )
                self._input.tap(click_enter_x, click_enter_y)
                time.sleep(0.5)
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
            time.sleep(0.25)
            scr = self._screen.capture()
            if scr is None:
                continue

            # 1. Kiểm tra màn chọn nhân vật TRƯỚC TIÊN (bằng template và OCR)
            # Tránh nhận diện nhầm màn chọn nhân vật là đã vào game
            header_res = self._vision.find_template("header_chon_nhan_vat", frame=scr)
            char_res = self._vision.find_template("char_enter_btn", frame=scr)
            is_char_screen = header_res.found or (char_res.found and char_res.center_x > 1200)

            if not is_char_screen and hasattr(self, "_text") and self._text:
                char_boxes = self._text.read(scr, (800, 50, 1600, 850))
                is_char_screen = any(
                    any(k in normalize_game_text(b.text) for k in ("chon nhan vat", "vao game", "tao nhan vat"))
                    for b in char_boxes
                )

            if is_char_screen:
                first_slot = login_cfg.get(
                    "character_first_slot", {"x": 1340, "y": 235}
                )
                click_enter_x = char_res.center_x if (char_res.found and char_res.center_x > 1200) else enter_btn.get("x", 1425)
                click_enter_y = char_res.center_y if (char_res.found and char_res.center_x > 1200) else enter_btn.get("y", 685)
                logger.info(
                    f"[{ctx.account_id}] Đang ở màn chọn nhân vật! Chọn nhân vật "
                    f"đầu tiên ({first_slot['x']}, {first_slot['y']})..."
                )
                self._input.tap(first_slot["x"], first_slot["y"])
                time.sleep(0.3)
                logger.info(
                    f"[{ctx.account_id}] Bấm Vào Game "
                    f"({click_enter_x}, {click_enter_y})..."
                )
                self._input.tap(click_enter_x, click_enter_y)
                time.sleep(2.0)
                continue

            # 2. Kiểm tra các template in-game thực sự (btn_tui, quest_btn, minimap, ingame_screen)
            for tpl in ["btn_tui", "quest_btn", "minimap", "ingame_screen"]:
                try:
                    res = self._vision.find_template(tpl, frame=scr)
                    if res.found or (res.confidence is not None and res.confidence > 0.70):
                        ctx.reset_retry()
                        logger.info(f"[{ctx.account_id}] Login verified OK (tìm thấy template in-game '{tpl}' conf={res.confidence:.2f})")
                        activity._write("✅", f"[{ctx.account_id}] Đã vào game thành công!")
                        ctx.extra["check_previous_quest_on_login"] = True
                        self._prepare_ingame_hud_after_login(ctx)
                        return BotState.FIND_COMMERCE_CHIEF
                except Exception:
                    pass

            # 3. Kiểm tra bằng nhận diện text in-game (Phúc lợi, Luyện công, Võ công, v.v.)
            if self._is_ingame(frame=scr):
                ctx.reset_retry()
                logger.info(f"[{ctx.account_id}] Login verified OK (tìm thấy các thành phần HUD in-game)")
                activity._write("✅", f"[{ctx.account_id}] Đã vào game thành công!")
                ctx.extra["check_previous_quest_on_login"] = True
                self._prepare_ingame_hud_after_login(ctx)
                return BotState.FIND_COMMERCE_CHIEF

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
        if frame is None:
            if hasattr(self, "_screen") and self._screen is not None:
                frame = self._screen.capture()
            else:
                return False

        if frame is None or not hasattr(self, "_text") or self._text is None:
            return False
        shape = getattr(frame, "shape", None)
        if shape is not None and isinstance(shape, tuple) and len(shape) != 3:
            return False

        try:
            # Template and OCR must inspect the exact same captured frame. The
            # previous call omitted frame here, causing a fresh hidden capture
            # whose result could disagree with the screenshot passed in.
            if getattr(self, "_vision", None) is not None and hasattr(self._vision, "find_template") and self._vision.find_template("dialog_box", frame).found:
                return True

            boxes = self._text.read(frame, (150, 300, 1450, 850))
            for b in boxes:
                norm = normalize_game_text(b.text)
                norm_ns = norm.replace(" ", "")

                # "Kết thúc đối thoại" là nút trong hộp thoại NPC, tuyệt đối không bị bỏ qua dù nằm ở x < 320!
                is_finish_dialog_btn = (
                    "ket thuc" in norm
                    or "ket thic" in norm
                    or "ketthuc" in norm_ns
                    or "ketthic" in norm_ns
                )

                # Bỏ qua khu vực Quest Tracker bên trái (x < 320, y < 600) để không bị nhầm lẫn
                # với tiến độ nhiệm vụ "Tiến độ: 0/10", "Chưa hoàn thành", v.v.
                if not is_finish_dialog_btn and b.center_x < 320 and b.center_y < 600:
                    continue
                # Bỏ qua khung chat và tab hệ thống phía dưới màn hình (y >= 750) để không bắt nhầm thông báo nhận vật phẩm
                if not is_finish_dialog_btn and b.center_y >= 750:
                    continue
                # Bỏ qua icon bong bóng chat nhanh (x: 1100..1250, y: 300..450)
                if 1100 <= b.center_x <= 1250 and 300 <= b.center_y <= 450:
                    continue
                # Bỏ qua các banner thông báo hệ thống xuất hiện giữa màn hình khi đánh quái
                if any(k in norm_ns for k in ("truysat", "tieudiet")):
                    continue
                # Bỏ qua banner thông báo hoàn tất nhiệm vụ thương hội giữa màn hình
                if "nhiem vu thuong hoi" in norm and any(k in norm for k in ("hoan thanh", "da hoan thanh")):
                    continue
                # Đây là banner thông báo hoàn tất mục tiêu đối thoại, không
                # phải hộp thoại NPC và cũng không có nút để bấm.
                if (
                    "cung voi" in norm
                    and "doi thoai" in norm
                    and "hoan thanh nhiem vu" in norm
                ):
                    continue
                # Bỏ qua thông báo trừ điểm danh vọng do hủy nhiệm vụ (không phải hộp thoại NPC để đối thoại)
                if any(k in norm for k in ("bi tru 5 diem", "tru 5 diem", "danh vong vi huy", "huy nhiem vu!")):
                    continue
                if is_finish_dialog_btn or norm in ("duoc", "duoc roi") or any(w in norm for w in (
                    "hoang thuong", "trieu dinh", "cong hien", "xem tinh hinh",
                    "hung thu", "ket thuc", "tra nhiem vu",
                    "chua hoan thanh", "ngay mai hay den", "tien do", "vi trieu dinh", "muon di dau",
                    "lan truoc da hoan thanh", "hoan thanh chua", "20 nhiem vu nhe",
                    "nhiem vu da hoan thanh", "da hoan thanh", "muon huy bo",
                    # Từ khóa Chủ thương hội hỏi đưa đi / giao nhiệm vụ (chỉ các cụm từ trong câu hỏi/nút, không dùng tên NPC)
                    "phien dua ta", "di mot chuyen", "khong can dau", "co can ta dua",
                    "noi lam nhiem vu", "dua ta di", "dua den noi",
                    # Từ khóa lựa chọn/hội thoại đặc trưng. Các từ chung như
                    # "đi đâu" tự nó không đủ phân biệt hội thoại với HUD.
                    "le phi", "lo phi", "3 luong", "luong thoi", "luung thii", "khong di dau",
                    # Từ khóa đối thoại NPC nhiệm vụ thông thường
                    "ta di tim", "di tim ngay", "ngay day", "ta di ngay",
                    "biet roi", "ta biet roi", "da hieu", "ta hieu", "ro roi",
                    "dong y", "tiep tuc", "xac nhan", "roi khoi", "duoc roi", "duoc ta se di",
                    "chua tim duoc", "nguoi chua", "khong the giup", "lam ruong chua",
                    "ruong chua do", "giup nguoi", "xin giup", "giup ta", "hay den",
                    "gap ta", "tim gap", "co viec gi", "chua mang ve", "ta se di",
                    "hang da den", "hang den", "ta den bao tin", "thong bao", "bao tin"
                )) or any(w in norm_ns for w in (
                    "lantruoc", "hoanthanhchua", "20nhiemvu", "dahoanthanh", "vanchua",
                    "xemtinhhinh", "huybo", "vitrieudinh", "conghien",
                    "phienduata", "dimotchuyen", "khongcandau", "cocandua",
                    # Từ khóa Xa Phu dạng liền khối chống nhiễu OCR
                    "lephi", "lophi", "3luong", "3luung", "3luprg", "3lueng", "luongthoi", "mufndi",
                    "khongdidau",
                    # Từ khóa đối thoại NPC thông thường dạng liền khối
                    "taditimngayday", "taditim", "ngayday", "tadingay", "bietroi",
                    "chuatimduoc", "nguoichua", "khongthegiup", "lamruong",
                    "hangdaden", "hangden", "tadenbaotin", "thongbao", "baotin"
                )):
                    return True

            if hasattr(self, "_find_finish_dialogue_button") and self._find_finish_dialogue_button(frame):
                return True
            if hasattr(self, "_find_dialogue_response_button") and self._find_dialogue_response_button(frame):
                return True
        except Exception:
            return False
        return False

    def _ensure_npc_dialog_closed(self, max_attempts: int = 2) -> bool:
        """
        Đảm bảo đóng hoàn toàn hộp thoại NPC nếu đang mở:
        - Bấm 'Kết thúc đối thoại' nếu có.
        - Hoặc bấm 'Không cần đâu' nếu có.
        - Hoặc bấm 'Nhiệm vụ vẫn chưa hoàn thành' nếu có (để mở ra 'Kết thúc đối thoại').
        - Hoặc bấm 'Không đi đâu cả' nếu là hộp thoại Xa Phu.
        - Fallback: Bấm (800, 200) hoặc phím Back (tối đa 2 lần thay vì 5 lần lặp).
        - Kiểm tra lại _is_npc_dialog_open() cho đến khi chắc chắn đã đóng.
        """
        try:
            if hasattr(self, "_is_map_open") and self._is_map_open():
                logger.info("_ensure_npc_dialog_closed: Phát hiện bản đồ đang mở thay vì NPC dialog, đóng bản đồ...")
                self._close_map()
                time.sleep(0.3)

            fallback_taps = 0
            for attempt in range(max_attempts):
                if not self._is_npc_dialog_open():
                    return True
                frame = self._screen.capture()
                if frame is None or not (hasattr(frame, "shape") and len(getattr(frame, "shape", ())) == 3):
                    break
                raw_boxes = self._text.read(frame, (150, 320, 1450, 850))
                # Cho phép vùng nút thấp của Xa Phu nhưng chỉ ở giữa màn hình;
                # vùng chat nằm sát mép dưới/trái và không được dùng làm nút.
                boxes = [
                    b for b in raw_boxes
                    if b.center_y < 750 and b.center_x >= 250
                ]
                tapped = False

                # 0. Tìm nút 'Rời khỏi' nếu là hộp thoại NPC hướng dẫn / NPC khác
                for b in boxes:
                    norm = normalize_game_text(b.text)
                    if "roi khoi" in norm or "roikhoi" in norm.replace(" ", ""):
                        logger.info(f"_ensure_npc_dialog_closed: Bấm nút '{b.text}' tại ({b.center_x}, {b.center_y})")
                        self._input.tap(b.center_x, b.center_y)
                        tapped = True
                        time.sleep(0.5)
                        break
                if tapped:
                    continue

                # 1. Chỉ bấm nút 'Kết thúc đối thoại' / 'Kết thúc'. Không
                # được coi mọi câu chứa chữ "đối thoại" là nút: banner như
                # "Cùng với Xa phu đối thoại, hoàn thành nhiệm vụ" chỉ là
                # thông báo giữa màn hình.
                for b in boxes:
                    norm = normalize_game_text(b.text)
                    norm_ns = norm.replace(" ", "")
                    # Chỉ chọn nhãn nút kết thúc đối thoại, hỗ trợ lỗi chính tả OCR
                    is_ket_thuc = any(k in norm for k in ("ket thuc", "ket thic", "ket thie")) or any(k in norm_ns for k in ("ketthuc", "ketthic", "ketthie"))
                    is_doi_thoai = any(k in norm for k in ("doi thoai", "doithoaf")) or any(k in norm_ns for k in ("doithoai", "doithoaf"))
                    if (
                        is_ket_thuc
                        and is_doi_thoai
                        and not any(k in norm for k in (
                            "chua hoan thanh", "tiep tuc co gang", "hoan thanh nhiem vu",
                            "nguoi chua", "nhiem vu ta giao",
                        ))
                    ):
                        logger.info(f"_ensure_npc_dialog_closed: Bấm '{b.text}' tại ({b.center_x}, {b.center_y})")
                        self._input.tap(b.center_x, b.center_y)
                        tapped = True
                        time.sleep(0.5)
                        break
                if tapped:
                    continue

                # 2. Tìm nút 'Không cần đâu' nếu là hộp thoại Chủ thương hội hỏi đưa đi
                for b in boxes:
                    norm = normalize_game_text(b.text)
                    norm_ns = norm.replace(" ", "")
                    if "khong can" in norm or "khongcandau" in norm_ns:
                        logger.info(f"_ensure_npc_dialog_closed: Bấm nút 'Không cần đâu' '{b.text}' tại ({b.center_x}, {b.center_y})")
                        self._input.tap(b.center_x, b.center_y)
                        tapped = True
                        time.sleep(0.5)
                        break
                if tapped:
                    continue

                # 2b. Tìm nút 'Không có việc gì!' nếu là hộp thoại NPC thông thường (như Vương Cẩn)
                for b in boxes:
                    norm = normalize_game_text(b.text)
                    norm_ns = norm.replace(" ", "")
                    if "khong co viec" in norm or "khongcoviecgi" in norm_ns or "khong co gi" in norm:
                        logger.info(f"_ensure_npc_dialog_closed: Bấm nút 'Không có việc gì!' '{b.text}' tại ({b.center_x}, {b.center_y})")
                        self._input.tap(b.center_x, b.center_y)
                        tapped = True
                        time.sleep(0.5)
                        break
                if tapped:
                    continue

                # 3. Tìm nút 'Không đi đâu cả' nếu là hộp thoại Xa Phu
                for b in boxes:
                    norm = normalize_game_text(b.text)
                    norm_ns = norm.replace(" ", "")
                    if "khong di" in norm or "khongdidau" in norm_ns:
                        logger.info(f"_ensure_npc_dialog_closed: Bấm nút Xa Phu '{b.text}' tại ({b.center_x}, {b.center_y})")
                        self._input.tap(b.center_x, b.center_y)
                        tapped = True
                        time.sleep(0.5)
                        break
                if tapped:
                    continue

                # OCR may miss the low "Không đi đâu cả" button. If the
                # distinctive Xa Phu fee prompt is visible, use its configured
                # safe exit coordinate instead of tapping empty space at 800,200.
                if self._has_xa_phu_dialog_evidence(boxes):
                    coords = self._loader.load_ui_coords() if hasattr(self, "_loader") else {}
                    commerce_cfg = coords.get("commerce_quest", {})
                    no_travel = commerce_cfg.get("xa_phu_no_travel_button", {"x": 300, "y": 780})
                    logger.info(
                        f"_ensure_npc_dialog_closed: OCR thấy hội thoại Xa Phu nhưng hụt nút thoát; "
                        f"bấm Không đi đâu cả tại ({no_travel['x']}, {no_travel['y']})"
                    )
                    self._input.tap(no_travel["x"], no_travel["y"])
                    tapped = True
                    time.sleep(0.5)
                if tapped:
                    continue

                # 4. Fallback: không tìm thấy nút đóng nào bằng OCR
                fallback_taps += 1
                if fallback_taps == 1:
                    logger.info("_ensure_npc_dialog_closed: Fallback tap an toàn tại (800, 200)...")
                    self._input.tap(800, 200)
                    time.sleep(0.4)
                else:
                    logger.info("_ensure_npc_dialog_closed: Fallback dùng phím Back để đóng đối thoại...")
                    if hasattr(self, "_input") and hasattr(self._input, "press_back"):
                        self._input.press_back()
                    time.sleep(0.4)
                    break

            return not self._is_npc_dialog_open()
        except Exception as e:
            logger.debug(f"_ensure_npc_dialog_closed error: {e}")
            return False

    def _is_npc_interact_bubble_present(self, frame: Optional[np.ndarray] = None) -> bool:
        """Kiểm tra xem icon bong bóng đối thoại nổi có đang xuất hiện trên màn hình hay không."""
        if frame is None and hasattr(self, "_screen") and self._screen is not None:
            frame = self._screen.capture()
        if frame is None:
            return False

        if hasattr(self, "_vision") and self._vision is not None:
            try:
                res = self._vision.find_template(
                    "npc_interact_bubble",
                    frame=frame,
                    region_override=(1140, 240, 1320, 380),
                )
                if res and res.found:
                    return True
            except Exception as e:
                logger.debug(f"is_npc_interact_bubble_present error: {e}")

        # Contour check: tìm đốm tròn đặc trưng của icon hội thoại nổi
        try:
            rx1, ry1, rx2, ry2 = 1150, 250, 1300, 360
            patch = frame[ry1:ry2, rx1:rx2]
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area < 300:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                if 20 <= w <= 65 and 20 <= h <= 65 and 0.7 <= (w / float(h)) <= 1.4:
                    perimeter = cv2.arcLength(c, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity >= 0.55:
                            return True
        except Exception:
            pass

        return False

    def _is_standing_next_to_xa_phu(self, frame: Optional[np.ndarray] = None) -> bool:
        """Require both the on-screen Xa phu label and interaction bubble."""
        if frame is None and hasattr(self, "_screen") and self._screen is not None:
            frame = self._screen.capture()
        if frame is None or not self._is_npc_interact_bubble_present(frame):
            return False
        try:
            boxes = self._text.read(frame, (100, 150, 1450, 800))
            return any(
                "xa phu" in normalize_game_text(box.text)
                and "thu kho" not in normalize_game_text(box.text)
                for box in boxes
            )
        except Exception:
            return False

    def _click_npc_interact_bubble(self, frame: Optional[np.ndarray] = None) -> bool:
        """
        Nhấn vào icon bong bóng đối thoại nổi xuất hiện khi nhân vật đến gần NPC
        (như Xa Phu, Chủ thương hội, NPC nhiệm vụ).
        Nếu không truyền frame (hoặc frame=None), bấm trực tiếp vào tọa độ mặc định
        (dùng khi nhân vật đã chắc chắn dừng lại tại NPC hoặc thử mở đối thoại).
        """
        found_pos = None
        if frame is not None:
            if hasattr(self, "_vision") and self._vision is not None:
                try:
                    res = self._vision.find_template(
                        "npc_interact_bubble",
                        frame=frame,
                        region_override=(1140, 240, 1320, 380),
                    )
                    if res and res.found:
                        found_pos = (res.center_x, res.center_y)
                except Exception as e:
                    logger.debug(f"Template match interact bubble error: {e}")

            # Contour check nếu template chưa tìm thấy
            if not found_pos:
                try:
                    rx1, ry1, rx2, ry2 = 1150, 250, 1300, 360
                    patch = frame[ry1:ry2, rx1:rx2]
                    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in contours:
                        area = cv2.contourArea(c)
                        if area < 300:
                            continue
                        x, y, w, h = cv2.boundingRect(c)
                        if 20 <= w <= 65 and 20 <= h <= 65 and 0.7 <= (w / float(h)) <= 1.4:
                            perimeter = cv2.arcLength(c, True)
                            if perimeter > 0:
                                circularity = 4 * np.pi * area / (perimeter * perimeter)
                                if circularity >= 0.55:
                                    found_pos = (rx1 + x + w // 2, ry1 + y + h // 2)
                                    break
                except Exception as e:
                    logger.debug(f"Contour bubble detection error: {e}")

        # Khi đã có frame mà vẫn không định vị được icon, bỏ qua để tránh click nhầm
        if frame is not None and not found_pos:
            logger.debug("Không định vị được icon đối thoại trên frame hiện tại; bỏ qua click an toàn.")
            return False

        coords = self._loader.load_ui_coords() if hasattr(self, "_loader") else {}
        bubble_cfg = coords.get("npc", {}).get("interact_bubble", {"x": 1220, "y": 299})
        bx = found_pos[0] if found_pos else bubble_cfg.get("x", 1220)
        by = found_pos[1] if found_pos else bubble_cfg.get("y", 299)

        logger.info(f"Bấm icon đối thoại NPC tại ({bx}, {by})...")
        if hasattr(self, "_input") and self._input is not None:
            self._input.tap(bx, by)
            time.sleep(0.3)
            return True
        return False

    def _open_commerce_chief_dialog(self, ctx: StateContext) -> bool:
        """
        Mở hộp thoại đối thoại với NPC Chủ thương hội:
        1. Nếu hộp thoại đã mở sẵn và đúng Chủ thương hội -> Trả về True ngay.
        2. Ưu tiên bấm ngay icon bong bóng đối thoại (nhanh ~10ms bằng template/contour, không tốn OCR).
        3. Nếu chưa mở: xử lý popup / tap NPC Chủ thương hội trên màn hình / nhích nhẹ.
        """
        # 1. Nếu hộp thoại đã mở sẵn:
        if self._is_npc_dialog_open():
            if self._verify_commerce_chief_dialog():
                logger.info(f"[{ctx.account_id}] Hộp thoại đối thoại với Chủ thương hội đã mở sẵn.")
                return True
            # Hộp thoại đang mở nhưng không phải menu Chủ thương hội (ví dụ bảng tóm tắt tiến độ hoặc đối thoại dở)
            logger.info(f"[{ctx.account_id}] Phát hiện hộp thoại khác/tiến độ đang che màn hình, đóng lại trước...")
            self._ensure_npc_dialog_closed()
            time.sleep(0.3)

        # 2. ƯU TIÊN SỐ 1: Bấm icon bong bóng đối thoại nổi (rất nhanh vì dùng template/contour, không tốn OCR)
        if self._click_npc_interact_bubble():
            time.sleep(0.4)
            if self._is_npc_dialog_open():
                if self._verify_commerce_chief_dialog():
                    logger.info(f"[{ctx.account_id}] Mở hộp thoại Chủ thương hội thành công bằng icon đối thoại!")
                    return True
                else:
                    logger.warning(f"[{ctx.account_id}] Bấm icon đối thoại nhưng mở ra KHÔNG PHẢI Chủ thương hội! Đóng lại...")
                    self._ensure_npc_dialog_closed()
                    time.sleep(0.3)
            else:
                self._dismiss_unexpected_popups()

        # 3. Kiểm tra nếu đang có thông báo trừ điểm danh vọng do vừa hủy nhiệm vụ -> click tắt thông báo trước
        chk_scr = self._screen.capture() if hasattr(self, "_screen") and self._screen is not None else None
        if chk_scr is not None and hasattr(self, "_text") and self._text is not None:
            chk_boxes = self._text.read(chk_scr, (150, 250, 1450, 750))
            if any(any(kw in normalize_game_text(b.text) for kw in ("bi tru 5 diem", "tru 5 diem", "danh vong vi huy", "huy nhiem vu!")) for b in chk_boxes):
                logger.info(f"[{ctx.account_id}] Phát hiện thông báo 'Bạn bị trừ 5 điểm danh vọng...' đang che màn hình, click đúp để tắt...")
                self._input.tap(303, 571)
                time.sleep(0.3)
                self._input.tap(303, 571)
                time.sleep(0.4)

        # Đóng mọi popup bất thường trước khi tương tác
        self._dismiss_unexpected_popups()
        time.sleep(0.2)

        # Hàm trợ giúp kiểm tra Chủ thương hội có đang hiển thị trên màn hình không
        def _find_chief_on_screen(scr_frame):
            if scr_frame is None or not hasattr(self, "_text") or self._text is None:
                return None
            # 1. Kiểm tra thanh mục tiêu trên cùng (450..650, 40..120)
            top_boxes = self._text.read(scr_frame, (450, 40, 650, 120))
            for b in top_boxes:
                norm = normalize_game_text(b.text)
                norm_ns = norm.replace(" ", "")
                if any(k in norm for k in ("nhiem vu", "truy sat", "tieu diet", "hoan thanh")):
                    continue
                if (
                    any(w in norm for w in ["thuong hoi", "chu thuong hoi"])
                    or any(w in norm_ns for w in ["thuonghoi", "chuthuonghoi"])
                ):
                    return (b.center_x, b.center_y)

            # 2. Kiểm tra tên NPC trên màn hình (hỗ trợ cả khi OCR viết liền 'Chu thuonghoi')
            body_boxes = self._text.read(scr_frame, (250, 150, 1350, 750))
            for b in body_boxes:
                norm = normalize_game_text(b.text)
                norm_ns = norm.replace(" ", "")
                if any(k in norm for k in (
                    "nhiem vu", "truy sat", "tieu diet", "hoan thanh",
                    "xem tinh hinh", "bat dau", "chuc mung", "tien do"
                )):
                    continue
                if (
                    any(w in norm for w in ("chu thuong hoi", "thuong hoi"))
                    or any(w in norm_ns for w in ("chuthuonghoi", "thuonghoi"))
                ):
                    return (b.center_x, b.center_y + 40)
            return None

        scr_init = self._screen.capture() if hasattr(self, "_screen") and self._screen is not None else None
        chief_target = _find_chief_on_screen(scr_init)

        # Nếu chưa thấy Chủ thương hội xuất hiện trên màn hình, nhích qua lại để game render NPC
        if not chief_target:
            logger.info(f"[{ctx.account_id}] Chưa thấy Chủ thương hội trên màn hình. Nhích tới lui để hiển thị NPC...")
            self._nudge_left_right_to_reveal_npc(ctx)
            time.sleep(0.4)
            scr_init = self._screen.capture() if hasattr(self, "_screen") and self._screen is not None else None
            chief_target = _find_chief_on_screen(scr_init)

        # Thử lại icon đối thoại sau khi nhích nếu có
        if self._click_npc_interact_bubble():
            time.sleep(0.4)
            if self._is_npc_dialog_open():
                if self._verify_commerce_chief_dialog():
                    logger.info(f"[{ctx.account_id}] Mở hộp thoại Chủ thương hội thành công bằng icon đối thoại!")
                    return True
                else:
                    self._ensure_npc_dialog_closed()
                    time.sleep(0.3)
            else:
                self._dismiss_unexpected_popups()

        for attempt in range(3):
            scr = self._screen.capture() if hasattr(self, "_screen") and self._screen is not None else None
            tapped = False
            chief_target = _find_chief_on_screen(scr)
            if chief_target:
                logger.info(f"[{ctx.account_id}] Tap vào mục tiêu/tên Chủ thương hội tại {chief_target}...")
                self._input.tap(chief_target[0], chief_target[1])
                tapped = True
            else:
                logger.info(f"[{ctx.account_id}] Không thấy Chủ thương hội (lần {attempt + 1}), nhích nhẹ nhân vật...")
                self._nudge_left_right_to_reveal_npc(ctx)
                time.sleep(0.3)
                if self._click_npc_interact_bubble():
                    time.sleep(0.4)
                    if self._is_npc_dialog_open():
                        if self._verify_commerce_chief_dialog():
                            return True
                        else:
                            self._ensure_npc_dialog_closed()
                            time.sleep(0.3)
                    else:
                        self._dismiss_unexpected_popups()

            time.sleep(0.4)
            if self._is_npc_dialog_open():
                if self._verify_commerce_chief_dialog():
                    logger.info(f"[{ctx.account_id}] Mở hộp thoại NPC Chủ thương hội thành công!")
                    return True
                else:
                    logger.warning(f"[{ctx.account_id}] Hộp thoại mở ra KHÔNG PHẢI Chủ thương hội! Đóng lại...")
                    self._ensure_npc_dialog_closed()
                    time.sleep(0.3)
            self._dismiss_unexpected_popups()

        logger.warning(f"[{ctx.account_id}] Không thể mở hộp thoại NPC Chủ thương hội sau 3 lần thử!")
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

    def _find_dialogue_response_button(self, frame=None) -> Optional[tuple[int, int]]:
        """
        Tìm các nút lựa chọn hội thoại của NPC khi không có 'Kết thúc đối thoại'.
        Ví dụ: 'Ta đi tìm ngay đây!', 'Ta đi ngay', 'Biết rồi', 'Đồng ý', 'Tiếp tục', 'Xác nhận'...
        Vùng quét: nửa dưới hộp thoại (x: 150..900, y: 430..750).
        """
        if frame is None and hasattr(self, "_screen") and self._screen is not None:
            frame = self._screen.capture()
        if frame is None or not hasattr(self, "_text") or self._text is None:
            return None

        try:
            boxes = self._text.read(frame, (150, 430, 900, 750))
            for box in boxes:
                if box.center_y >= 750:
                    continue
                norm = normalize_game_text(box.text)
                norm_ns = norm.replace(" ", "")
                if any(k in norm for k in (
                    "ta di tim", "di tim ngay", "ngay day", "ta di ngay",
                    "biet roi", "ta biet roi", "da hieu", "ta hieu", "ro roi",
                    "dong y", "tiep tuc", "xac nhan", "roi khoi", "duoc roi", "duoc ta se di",
                    "di thoi", "tim ngay", "chuyen thu", "gui thu", "dung vay",
                    "ta di", "ta se di",
                    "hang da den", "hang den", "ta den bao tin", "thong bao", "thuong hoi", "bao tin"
                )) or norm in ("duoc", "duoc roi") or any(k.replace(" ", "") in norm_ns for k in (
                    "taditimngayday", "taditim", "ngayday", "tadingay", "bietroi",
                    "dongy", "tieptuc", "xacnhan", "roikhoi", "dithoi",
                    "hangdaden", "hangden", "tadenbaotin", "thongbao", "thuonghoi", "baotin"
                )):
                    return (box.center_x, box.center_y)
        except Exception as exc:
            logger.debug(f"_find_dialogue_response_button error: {exc}")
        return None

    def _find_finish_dialogue_button(self, frame=None) -> Optional[tuple[int, int]]:
        """Tìm riêng nút ``Kết thúc đối thoại``, không lấy banner/nút màu bất kỳ."""
        if frame is None and hasattr(self, "_screen") and self._screen is not None:
            frame = self._screen.capture()
        if frame is None or not hasattr(self, "_text") or self._text is None:
            return None

        try:
            boxes = self._text.read(frame, (150, 320, 1450, 750))
            for box in boxes:
                if box.center_y >= 750:
                    continue
                norm = normalize_game_text(box.text)
                compact = norm.replace(" ", "")
                if (
                    "ket thuc" in norm
                    or "ket thic" in norm
                    or "ketthucdoithoai" in compact
                    or "ketthicdoithoaf" in compact
                ):
                    return (box.center_x, box.center_y)
        except Exception as exc:
            logger.debug(f"_find_finish_dialogue_button error: {exc}")
        return None

    def _press_buff_speed(self, ctx: Optional[StateContext] = None) -> None:
        """Nhấn nút Buff để tăng tốc độ di chuyển (tắt hoàn toàn khi đang trong nhiệm vụ đánh quái hoặc khi di chuyển)."""
        coords = self._loader.load_ui_coords() if hasattr(self, "_loader") and self._loader else {}
        commerce_cfg = coords.get("commerce_quest", {}) if coords else {}
        if getattr(self, "_disable_buff_on_move", commerce_cfg.get("disable_buff_on_move", False)):
            logger.debug("Bỏ qua buff tăng tốc khi di chuyển để tránh ngắt tự tìm đường / thông báo nhầm.")
            return
        if ctx is not None and getattr(ctx, "current_quest_type", None) == QuestType.COMBAT:
            logger.debug("Đang làm nhiệm vụ đánh quái -> Tắt buff tăng tốc.")
            return
        if getattr(self, "_in_combat", False):
            logger.debug("Đang đánh quái -> Tắt buff tăng tốc.")
            return
        try:
            buff_cfg = commerce_cfg.get("buff_button", {"x": 1308, "y": 160})
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
            try:
                self._adb.input_text_unicode(text)
                typed_ok = True
            except Exception as e:
                logger.warning(f"ADB input_text_unicode failed: {e}")
                norm = normalize_game_text(text)
                self._adb.input_text(norm.title())

    def _search_and_select_npc(
        self,
        ctx: StateContext,
        npc_name: str,
        *,
        allow_directional_xa_phu: bool = False,
    ) -> bool:
        """
        Tìm kiếm và chọn NPC trong tab 'Tìm NPC' trên bản đồ mini.
        1. Đảm bảo bản đồ đang mở và bấm tab 'Tìm NPC'.
        2. Quét trước danh sách NPC đang hiển thị trên trang đầu (nhiều NPC như Chủ Tiệm Nữ phục, Nam phục có sẵn).
        3. Nếu chưa thấy, dùng ô input tìm kiếm, gõ chính xác tên NPC có dấu và submit.
        4. Quét kết quả với cơ chế so khớp tên chuẩn, alias và phát âm OCR (tránh bắt nhầm ký tự rác/tiếng Trung).
        5. Đảm bảo bản đồ còn mở trước khi click để tránh click trúng màn hình chính.
        """
        coords = self._loader.load_ui_coords()
        nav_cfg = coords.get("navigation", {})
        find_npc_x = nav_cfg.get("map_tab_find_npc", {}).get("x", 376)
        find_npc_y = nav_cfg.get("map_tab_find_npc", {}).get("y", 161)
        search_x = nav_cfg.get("map_search_input", {}).get("x", 266)
        search_y = nav_cfg.get("map_search_input", {}).get("y", 222)

        # 0. Đảm bảo bản đồ đang mở
        if not self._is_map_open():
            self._ensure_map_open()
            time.sleep(0.3)

        # 1. Bấm tab Tìm NPC
        logger.info(f"[{ctx.account_id}] Bấm tab Tìm NPC ({find_npc_x}, {find_npc_y})")
        self._input.tap(find_npc_x, find_npc_y)
        time.sleep(0.4)

        # Hàm so khớp NPC chính xác, chống bắt nhầm text rác hoặc ký tự không dấu/tiếng Trung
        def _is_npc_match(box_text: str, target_name: str) -> bool:
            norm_b = normalize_game_text(box_text)
            norm_target = normalize_game_text(target_name)
            if not norm_b or len(norm_b) < 3 or not norm_target:
                return False

            # Nhiệm vụ đối thoại ở Tương Dương chỉ ghi chung là "Xa phu",
            # nhưng danh sách NPC hiển thị Xa phu Nam/Tây/Bắc. Cả ba đều hợp lệ.
            # Cờ này chỉ dành cho luồng nhiệm vụ đối thoại; luồng chuyển thành
            # phố qua Xa phu vẫn giữ quy tắc lựa chọn riêng.
            if allow_directional_xa_phu and norm_target == "xa phu":
                return any(
                    norm_b == f"xa phu {direction}"
                    for direction in ("nam", "tay", "bac")
                ) or norm_b == "xa phu"

            b_tokens = set(norm_b.split())
            t_tokens = set(norm_target.split())
            # Tránh nhầm lẫn các loại tiệm/cửa hiệu khác nhau (Vũ khí, Nam phục, Nữ phục, Kim hoàn, Dược...)
            shop_categories = (
                {"vu", "khi", "khl", "vk"},
                {"nu", "ni", "nt"},
                {"nam"},
                {"kim", "hoan"},
                {"duoc", "diem", "thuoc"},
                {"tap", "hoa"},
                {"tien", "trang"},
                {"tuu", "tiu", "lau"},
            )
            for cat in shop_categories:
                has_t = bool(t_tokens & cat)
                has_b = bool(b_tokens & cat)
                if has_t != has_b and any(w in norm_target or w in norm_b for w in ("tiem", "chu")):
                    return False

            # Tránh nhầm lẫn Nam và Nữ trong Tiệm phục
            if ("nu" in t_tokens or "ni" in t_tokens) and "nam" in b_tokens:
                return False
            if "nam" in t_tokens and ("nu" in b_tokens or "ni" in b_tokens):
                return False

            # Tránh nhầm lẫn Nạn dân với Nạn dân Giáp/Ất/Bính/Đinh
            t_has_refugee_sub = bool(t_tokens & {"giap", "at", "binh", "dinh"})
            b_has_refugee_sub = bool(b_tokens & {"giap", "at", "binh", "dinh"})
            if t_has_refugee_sub != b_has_refugee_sub and ("nan dan" in norm_target or "nan dan" in norm_b):
                return False

            # Exact match
            if norm_target == norm_b or norm_target in norm_b:
                return True

            # Aliases match
            from .quest_classifier import NPC_ALIASES
            target_aliases = [norm_target]
            for alias_norm, canonical in NPC_ALIASES.items():
                if normalize_game_text(canonical) == norm_target:
                    target_aliases.append(alias_norm)

            for alias in target_aliases:
                if alias == norm_b or alias in norm_b:
                    return True
                if norm_b in alias and len(norm_b) >= 8 and len(norm_b.split()) >= 2:
                    return True

            # Phonetic normalization (phyc -> phuc, ni -> nu, nt -> nu)
            def _phonetic(s: str) -> str:
                return s.replace("phyc", "phuc").replace("ni", "nu").replace("nt", "nu")

            ph_b = _phonetic(norm_b)
            ph_t = _phonetic(norm_target)
            if ph_t == ph_b or ph_t in ph_b:
                return True

            # Core words match
            stop_words = {"chu", "ong", "ba", "co", "tai", "ve", "dao", "nhan", "nguoi"}
            core_words = [w for w in norm_target.split() if w not in stop_words]
            if core_words and len(core_words) >= 2:
                ph_core = [_phonetic(w) for w in core_words]
                if all(w in ph_b for w in ph_core):
                    return True

            if fuzzy_score(ph_b, ph_t) >= 0.78:
                return True

            return False

        found_target = False
        result_pos = None

        # 2. Kiểm tra nhanh: nếu NPC đã có sẵn trên trang đầu của danh sách NPC bên trái (50..380, 240..840)
        scr = self._screen.capture()
        if scr is not None:
            boxes = self._text.read(scr, (50, 240, 380, 840))
            for b in boxes:
                if _is_npc_match(b.text, npc_name):
                    result_pos = (b.center_x, b.center_y)
                    found_target = True
                    logger.info(f"[{ctx.account_id}] Tìm thấy NPC '{b.text}' có sẵn trong danh sách tại {result_pos}")
                    break

        # 3. Nếu chưa thấy trên trang đầu, dùng ô input tìm kiếm
        if not found_target:
            logger.info(f"[{ctx.account_id}] Click vào ô input tìm kiếm NPC ({search_x}, {search_y})...")
            self._input.tap(search_x, search_y)
            time.sleep(0.25)

            # Xóa text cũ
            try:
                del_keys = " ".join(["67"] * 20)
                self._adb._run(["shell", f"input keyevent {del_keys}"])
            except Exception:
                pass
            time.sleep(0.15)

            # Gõ tên NPC
            logger.info(f"[{ctx.account_id}] Gõ chính xác tên NPC có dấu: '{npc_name}'...")
            self._type_unicode_text(ctx, npc_name)
            time.sleep(0.4)

            try:
                self._adb._run(["shell", "input keyevent 66"])
            except Exception:
                pass
            time.sleep(0.3)

            # Kiểm tra nếu map bị đóng ngoài ý muốn thì mở lại
            if not self._is_map_open():
                logger.warning(f"[{ctx.account_id}] Bản đồ bị đóng sau khi gõ tìm kiếm! Mở lại...")
                self._ensure_map_open()
                self._input.tap(find_npc_x, find_npc_y)
                time.sleep(0.3)

            # Quét vùng danh sách kết quả sau khi tìm kiếm
            scr = self._screen.capture()
            if scr is not None:
                boxes = self._text.read(scr, (50, 240, 380, 840))
                for b in boxes:
                    if _is_npc_match(b.text, npc_name):
                        result_pos = (b.center_x, b.center_y)
                        found_target = True
                        logger.info(f"[{ctx.account_id}] OCR nhận diện kết quả tìm kiếm NPC '{b.text}' tại {result_pos}")
                        break

        # 3.5. Nếu tên NPC bắt đầu bằng "Chủ " của cửa hiệu (ví dụ: "Chủ Tiệm Nam phục", "Chủ Tạp hóa")
        # mà không tìm thấy, thử tìm bằng tên ngắn bỏ "Chủ " (ví dụ: "Tiệm Nam phục", "Tạp hóa")
        # vì danh sách NPC trong game thường hiển thị không có chữ "Chủ".
        # KHÔNG áp dụng cho các NPC họ Chu (như "Chu Bân Cử", "Chu sứ giả").
        is_shop_owner_npc = any(
            npc_name.lower().startswith(prefix) for prefix in (
                "chủ tiệm", "chu tiem", "chủ dược", "chu duoc",
                "chủ tạp", "chu tap", "chủ kim", "chu kim",
                "chủ tiền", "chu tien", "chủ tửu", "chu tuu"
            )
        )
        if not found_target and is_shop_owner_npc:
            short_name = re.sub(r"^(?:Chủ|Chu)\s+", "", npc_name, flags=re.IGNORECASE).strip()
            if short_name:
                logger.info(f"[{ctx.account_id}] Thử tìm lại bằng tên ngắn bỏ chữ 'Chủ': '{short_name}'...")
                self._input.tap(search_x, search_y)
                time.sleep(0.15)
                try:
                    self._adb._run([
                        "shell",
                        "input keyevent 123; " + "; ".join(["input keyevent 67"] * 30),
                    ])
                except Exception:
                    pass
                self._type_unicode_text(ctx, short_name)
                time.sleep(0.35)
                try:
                    self._adb._run(["shell", "input keyevent 66"])
                except Exception:
                    pass
                time.sleep(0.25)
                if self._is_map_open():
                    scr = self._screen.capture()
                    if scr is not None:
                        boxes = self._text.read(scr, (50, 240, 380, 840))
                        for b in boxes:
                            if _is_npc_match(b.text, npc_name) or _is_npc_match(b.text, short_name):
                                result_pos = (b.center_x, b.center_y)
                                found_target = True
                                logger.info(f"[{ctx.account_id}] OCR nhận diện kết quả tìm kiếm '{short_name}': '{b.text}' tại {result_pos}")
                                break

        # 4. Nếu chưa thấy, thử cuộn nhẹ danh sách xuống
        # If the full name has no result, search the first word and choose the
        # closest candidate to the complete NPC name (for example: Tieu su -> Tieu).
        norm_target = normalize_game_text(npc_name)
        target_words = norm_target.split()
        display_words = str(npc_name).strip().split()
        if not found_target and len(target_words) >= 2 and display_words:
            first_word = display_words[0]
            norm_first = target_words[0]
            logger.info(
                f"[{ctx.account_id}] Khong thay ten day du '{npc_name}'; "
                f"tim lai bang tu dau '{first_word}' va chon ten giong nhat..."
            )
            self._input.tap(search_x, search_y)
            time.sleep(0.15)
            try:
                self._adb._run([
                    "shell",
                    "input keyevent 123; " + "; ".join(["input keyevent 67"] * 30),
                ])
            except Exception:
                pass
            self._type_unicode_text(ctx, first_word)
            time.sleep(0.35)
            try:
                self._adb._run(["shell", "input keyevent 66"])
            except Exception:
                pass
            time.sleep(0.25)

            if self._is_map_open():
                partial_frame = self._screen.capture()
                partial_boxes = (
                    self._text.read(partial_frame, (50, 240, 380, 840))
                    if partial_frame is not None else []
                )
                ranked = []
                for box in partial_boxes:
                    norm_box = normalize_game_text(box.text)
                    box_words = norm_box.split()
                    if not box_words:
                        continue
                    b_words = set(box_words)
                    t_words = set(target_words)
                    if "nam" in t_words and (b_words & {"nu", "ni", "nt"}):
                        continue
                    if (t_words & {"nu", "ni", "nt"}) and "nam" in b_words:
                        continue
                    first_score = max(fuzzy_score(norm_first, word) for word in box_words)
                    if norm_first not in box_words and first_score < 0.72:
                        continue
                    full_score = fuzzy_score(norm_target, norm_box)
                    if norm_box.startswith(norm_first):
                        full_score += 0.08
                    ranked.append((full_score, box))

                if ranked:
                    best_score, best_box = max(ranked, key=lambda item: item[0])
                    result_pos = (best_box.center_x, best_box.center_y)
                    found_target = True
                    logger.info(
                        f"[{ctx.account_id}] Tim '{first_word}': chon NPC giong nhat "
                        f"'{best_box.text}' (score={best_score:.2f}) tai {result_pos}."
                    )

        if not found_target:
            self._input.swipe(200, 700, 200, 350, duration=250)
            time.sleep(0.35)
            scr_scrolled = self._screen.capture()
            if scr_scrolled is not None:
                boxes_scrolled = self._text.read(scr_scrolled, (50, 240, 380, 840))
                for b in boxes_scrolled:
                    if _is_npc_match(b.text, npc_name):
                        result_pos = (b.center_x, b.center_y)
                        found_target = True
                        logger.info(f"[{ctx.account_id}] Sau khi cuộn, OCR nhận diện NPC '{b.text}' tại {result_pos}")
                        break

        # 5. Nếu vẫn không thấy trong danh sách, quét trực tiếp tên NPC trên bản đồ bên phải (450..1400, 150..800)
        if not found_target:
            scr_map = self._screen.capture()
            if scr_map is not None:
                map_boxes = self._text.read(scr_map, (450, 150, 1400, 800))
                for b in map_boxes:
                    if _is_npc_match(b.text, npc_name):
                        result_pos = (b.center_x, b.center_y)
                        found_target = True
                        logger.info(f"[{ctx.account_id}] Nhận diện vị trí NPC '{b.text}' trực tiếp trên bản đồ tại {result_pos}")
                        break

        # 6. Click chọn NPC (Bắt buộc kiểm tra bản đồ đang mở để tránh click nhầm màn hình chính)
        if found_target and result_pos:
            if not self._is_map_open():
                logger.error(f"[{ctx.account_id}] Bản đồ không mở, hủy click tọa độ {result_pos} để tránh click mù vào màn hình chính!")
                return False
            logger.info(f"[{ctx.account_id}] Click vào NPC '{npc_name}' tại {result_pos} trên bản đồ...")
            self._input.tap(*result_pos)
            time.sleep(0.5)
            return True

        logger.warning(f"[{ctx.account_id}] Không tìm thấy NPC '{npc_name}' trong danh sách tìm kiếm bản đồ!")
        return False

    def _search_tuyen_chau_xa_phu(self, ctx: StateContext) -> bool:
        """Chọn Xa phu trong danh sách minimap Tuyền Châu.

        Ngoài màn hình NPC mang nhãn ``Xa phu Tuyền Châu``, nhưng tab
        Tìm NPC liệt kê theo cổng. Luồng tự động chỉ dùng ``Xa phu Bắc``.
        """
        npc_name = "Xa phu Bắc"
        logger.info(
            f"[{ctx.account_id}] Tuyền Châu: tìm NPC minimap '{npc_name}'..."
        )
        return self._search_and_select_npc(ctx, npc_name)

    def _is_near_commerce_chief(self, frame: Optional[np.ndarray] = None) -> bool:
        """
        Kiểm tra xem nhân vật hiện tại có đang đứng cạnh NPC Chủ thương hội hay không:
        1. Hộp thoại của đúng Chủ thương hội đang mở sẵn (chứa từ khóa thương hội).
        2. Nhìn thấy tên NPC 'Chủ thương hội' trên màn hình xung quanh nhân vật.
        3. Tọa độ Minimap hiển thị gần (192, 182) tại Tuyền Châu.
        """
        try:
            scr = frame if frame is not None else (self._screen.capture() if hasattr(self, "_screen") and self._screen else None)
            if scr is None:
                return False

            # 0. Nếu bảng bản đồ lớn đang mở, tuyệt đối không coi là đang cạnh NPC
            if self._is_map_open(scr):
                return False

            # 1. Nếu có hộp thoại đang mở, phải kiểm tra đúng là của Chủ thương hội (tránh nhầm Chủ tiệm nam phục, Nha dịch...)
            if self._is_npc_dialog_open(scr) and self._verify_commerce_chief_dialog(scr):
                logger.info("_is_near_commerce_chief: Phát hiện hộp thoại Chủ thương hội đang mở sẵn!")
                return True

            # 2. Nhìn thấy tên NPC 'Chủ thương hội' trên màn hình
            boxes = self._text.read(scr, (350, 100, 1250, 650))
            for b in boxes:
                norm = normalize_game_text(b.text)
                norm_ns = norm.replace(" ", "")
                if (
                    "chu thuong hoi" in norm
                    or "chuthuonghoi" in norm_ns
                    or norm == "thuong hoi"
                ) and not any(k in norm for k in ("nhiem vu", "truy sat", "tieu diet", "hoan thanh")):
                    logger.info(f"_is_near_commerce_chief: Nhìn thấy NPC '{b.text}' tại ({b.center_x}, {b.center_y})!")
                    return True

            # 3. Kiểm tra tọa độ Minimap của Chủ thương hội (chỉ có tại Tuyền Châu 192, 182)
            cur_city, cur_coords = self._get_current_location_from_hud(scr)
            if cur_coords:
                # Chủ thương hội ở một số thành có điểm dừng auto-path riêng.
                # Dương Châu thường dừng tại (204,196), sát NPC nhưng OCR tên
                # có thể chập chờn; dùng vùng tọa độ cấu hình để xác nhận đã tới.
                if cur_city and hasattr(self, "_loader"):
                    chief_cfg = self._loader.load_ui_coords().get("commerce_quest", {}).get(
                        "commerce_chief_arrival_nudge", {}
                    )
                    for city_name, pos in chief_cfg.items():
                        if not is_same_city(cur_city, city_name):
                            continue
                        target_x = int(pos.get("x", 0))
                        target_y = int(pos.get("y", 0))
                        tolerance = int(pos.get("tolerance", 5))
                        city_dist = abs(cur_coords[0] - target_x) + abs(cur_coords[1] - target_y)
                        if city_dist <= tolerance:
                            logger.info(
                                f"_is_near_commerce_chief: Đã tới vùng Chủ thương hội {city_name} "
                                f"{cur_coords} (đích {target_x}/{target_y}, sai số {city_dist})."
                            )
                            return True

                    # Kiểm tra tọa độ Chủ thương hội ở các đại thành thị (Biện Kinh, Thành Đô, Tương Dương, Dương Châu...)
                    major_cities = self._loader.load_ui_coords().get("commerce_quest", {}).get("major_cities", [])
                    for mc in major_cities:
                        mc_name = mc.get("name", "")
                        if mc_name and is_same_city(cur_city, mc_name):
                            tx = int(mc.get("x", 0))
                            ty = int(mc.get("y", 0))
                            if tx > 0 and ty > 0 and (abs(cur_coords[0] - tx) + abs(cur_coords[1] - ty)) <= 2:
                                logger.info(f"_is_near_commerce_chief: Đang ở đúng tọa độ Chủ thương hội {cur_coords} tại {mc_name}!")
                                return True

                dist = abs(cur_coords[0] - 192) + abs(cur_coords[1] - 182)
                # Tại Tuyền Châu, Chủ thương hội đứng tại (192, 182). BẮT BUỘC cur_city phải là Tuyền Châu
                if dist <= 1 and cur_city and is_same_city(cur_city, "Tuyền Châu"):
                    logger.info(f"_is_near_commerce_chief: Đang ở đúng tọa độ Chủ thương hội {cur_coords} tại Tuyền Châu!")
                    return True

            return False
        except Exception as e:
            logger.debug(f"_is_near_commerce_chief error: {e}")
            return False

    def _check_quest_timeout(self, ctx: StateContext) -> bool:
        """Kiểm tra thời gian thực hiện nhiệm vụ hiện tại.
        Nếu vượt quá ngưỡng cấu hình (mặc định 15 phút), tự động quay về Chủ thương hội gần nhất để hủy nhiệm vụ.
        """
        try:
            start_t = ctx.extra.get("current_quest_start_time")
            if not start_t:
                return False

            timeout_min = 15
            try:
                opt = getattr(self._cfg, "commerce_optimization", None)
                if opt is not None:
                    timeout_min = int(getattr(opt, "quest_timeout_minutes", 15))
            except Exception:
                pass

            elapsed = time.time() - float(start_t)
            if elapsed >= timeout_min * 60:
                elapsed_min = int(elapsed // 60)
                logger.warning(
                    f"[{ctx.account_id}] ⚠️ [WATCHDOG TIMEOUT] Nhiệm vụ hiện tại đã chạy quá "
                    f"{timeout_min} phút (thực tế: {elapsed_min} phút)! "
                    "Tự động quay về Chủ thương hội gần nhất để hủy nhiệm vụ và nhận lượt mới..."
                )
                activity._write("⏱️", f"[{ctx.account_id}] NV qua {elapsed_min}p -> Quay ve Chu thuong hoi huy NV...")
                ctx.extra["abandon_combat_quest"] = True
                ctx.extra["abandon_combat_quest_reason"] = f"Quá {timeout_min} phút chưa hoàn thành ({elapsed_min} phút)"
                ctx.extra["returning_quest"] = True
                ctx.extra.pop("chief_nav_started", None)
                self._checkpoint.save(ctx)
                return True
        except Exception as e:
            logger.debug(f"_check_quest_timeout error: {e}")
        return False

    def _should_abandon_combat_immediately(self) -> bool:
        """Kiểm tra cấu hình có bật chế độ hủy ngay nhiệm vụ đánh quái để tối ưu thời gian không."""
        try:
            opt = getattr(self._cfg, "commerce_optimization", None)
            if opt is not None:
                return bool(getattr(opt, "abandon_combat_immediately", False))
        except Exception:
            pass
        return False

    def _nudge_to_reveal_commerce_chief(self, ctx: StateContext, frame=None) -> bool:
        """Nhích khỏi điểm dừng auto-path nếu NPC Chủ thương hội còn ngoài tầm nhìn."""
        try:
            cur_city, cur_coords = self._get_current_location_from_hud(frame)
            if not cur_city or not cur_coords:
                return False

            cfg = self._loader.load_ui_coords().get("commerce_quest", {})
            nudges = cfg.get("commerce_chief_arrival_nudge", {})
            city_cfg = next(
                (value for name, value in nudges.items() if is_same_city(cur_city, name)),
                None,
            )
            if not city_cfg:
                return False

            target_x = int(city_cfg.get("x", 0))
            target_y = int(city_cfg.get("y", 0))
            tolerance = int(city_cfg.get("tolerance", 5))
            if abs(cur_coords[0] - target_x) + abs(cur_coords[1] - target_y) > tolerance:
                return False

            attempts = int(ctx.extra.get("chief_arrival_nudge_attempts", 0))
            if attempts >= 2:
                return False

            dx = int(city_cfg.get("dx", 70))
            dy = int(city_cfg.get("dy", -70))
            duration = int(city_cfg.get("duration_ms", 500))
            joy_cfg = cfg.get("joystick_center", {"x": 220, "y": 680})
            joy_x, joy_y = int(joy_cfg.get("x", 220)), int(joy_cfg.get("y", 680))
            ctx.extra["chief_arrival_nudge_attempts"] = attempts + 1
            logger.info(
                f"[{ctx.account_id}] Đã tới vùng Chủ thương hội {cur_city} {cur_coords} nhưng chưa thấy NPC; "
                f"nhích một đoạn hướng Đông Bắc ({attempts + 1}/2) để đưa NPC vào tầm nhìn..."
            )
            self._input.swipe(joy_x, joy_y, joy_x + dx, joy_y + dy, duration=duration)
            time.sleep(0.5)
            return True
        except Exception as e:
            logger.debug(f"_nudge_to_reveal_commerce_chief error: {e}")
            return False

    def _nudge_left_right_to_reveal_npc(self, ctx: Optional[StateContext] = None) -> bool:
        """
        Nhích nhân vật qua lại để game cập nhật vị trí và render lại NPC (tránh lag hình ảnh).
        Luân phiên hướng (trái-phải, lên-xuống) nếu bị gọi nhiều lần liên tiếp.
        """
        try:
            coords = self._loader.load_ui_coords() if hasattr(self, "_loader") and self._loader else {}
            joy_cfg = coords.get("combat", {}).get("joystick_center", {"x": 220, "y": 680})
            joy_x = int(joy_cfg.get("x", 220))
            joy_y = int(joy_cfg.get("y", 680))
            step = 100
            hold_ms = 350

            # Đếm số lần nhích để luân phiên hướng di chuyển
            nudge_count = 0
            if ctx and hasattr(ctx, "extra"):
                nudge_count = int(ctx.extra.get("nudge_direction_cycle", 0))
                ctx.extra["nudge_direction_cycle"] = nudge_count + 1

            acc_str = f"[{ctx.account_id}] " if ctx else ""
            if hasattr(self, "_input") and hasattr(self._input, "swipe"):
                if nudge_count % 2 == 0:
                    logger.info(f"{acc_str}🔄 [Lag hình ảnh] Nhích trái rồi sang phải trở lại để game hiển thị lại NPC...")
                    activity._write("🔄", f"{acc_str}Nhich trai - phai hien NPC...")
                    self._input.swipe(joy_x, joy_y, joy_x - step, joy_y, duration=hold_ms)
                    time.sleep(0.3)
                    self._input.swipe(joy_x, joy_y, joy_x + step, joy_y, duration=hold_ms)
                else:
                    logger.info(f"{acc_str}🔄 [Lag hình ảnh] Nhích lên rồi xuống trở lại để game hiển thị lại NPC...")
                    activity._write("🔄", f"{acc_str}Nhich len - xuong hien NPC...")
                    self._input.swipe(joy_x, joy_y, joy_x, joy_y - step, duration=hold_ms)
                    time.sleep(0.3)
                    self._input.swipe(joy_x, joy_y, joy_x, joy_y + step, duration=hold_ms)
                time.sleep(0.4)
                return True
            return False
        except Exception as e:
            logger.debug(f"_nudge_left_right_to_reveal_npc error: {e}")
            return False

    def _verify_commerce_chief_dialog(self, frame=None) -> bool:
        """Xác nhận hộp thoại đang mở là của Chủ Thương Hội (không phải NPC khác).
        
        OCR quét nội dung hộp thoại tìm từ khóa đặc trưng:
        - 'thuong hoi' / 'chu thuong hoi' (tên NPC)
        - 'cong hien' / 'suc luc' (nội dung nhiệm vụ thương hội)
        - 'nhiem vu thuong hoi' (tiêu đề nhiệm vụ)
        
        QUAN TRỌNG: Icon đối thoại của TẤT CẢ NPC đều giống nhau,
        nên phải xác nhận bằng nội dung hộp thoại.
        """
        try:
            if frame is None and hasattr(self, "_screen") and self._screen is not None:
                frame = self._screen.capture()
            if frame is None or not hasattr(self, "_text") or self._text is None:
                return True

            from .text_recognition import normalize_game_text

            # Quét nội dung hộp thoại NPC (vùng rộng bao phủ dialog)
            dialog_boxes = self._text.read(frame, (150, 250, 1450, 850))
            dialog_text = " ".join(normalize_game_text(b.text) for b in dialog_boxes)
            
            commerce_keywords = (
                "chu thuong hoi", "thuong hoi", "nhiem vu thuong hoi",
                "vi trieu dinh", "cong hien suc luc", "cong hien", "suc luc", "suc lyc",
                "hoang thuong", "ban chieu", "giup suc trieu dinh", "gioi thieu ti mi",
                "khong co hung thu",
                "nhiem vu lan truoc", "lan truoc da hoan thanh chua",
                "trong ngay phai hoan thanh 20 nhiem vu", "nhiem vu da hoan thanh",
                "hoan thanh nhiem vu", "tra nhiem vu",
                "ta muon huy bo", "huy bo nhiem vu", "phien dua ta", "co can ta dua",
                "nhiem vu thuong hoi da hoan th", "tinh hinh nhiem vu",
                "chua hoan thanh", "mau di lam", "chua xong",
                "thuong doan", "giao nhiem vu", "ngay mai hay den", "khong con nhiem vu"
            )
            
            if any(kw in dialog_text for kw in commerce_keywords):
                logger.info(f"_verify_commerce_chief_dialog: ✅ Xác nhận là hộp thoại Chủ Thương Hội! (text: '{dialog_text[:100]}...')")
                return True
            
            logger.warning(f"_verify_commerce_chief_dialog: ❌ Hộp thoại KHÔNG PHẢI Chủ Thương Hội! (text: '{dialog_text[:100]}...')")
            return False
        except Exception as e:
            logger.debug(f"_verify_commerce_chief_dialog error: {e}")
            return False

    def _detect_current_city(self, frame=None) -> Optional[str]:
        """Đọc tên thành phố hiện tại từ góc trên phải (Minimap)."""
        city, _ = self._get_current_location_from_hud(frame)
        return city

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
        self._input_coordinate_field(x_pos[0], x_pos[1], target_x)

        # 2. Click TRỰC TIẾP vào ô input Y, xóa sạch và gõ Y
        logger.info(f"Click TRỰC TIẾP ô input Y tại {y_pos} để nhập {target_y}...")
        self._input_coordinate_field(y_pos[0], y_pos[1], target_y)

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

    def _input_search_text(self, text: str, telex: str = "", unicode_text: str = "", ctx: Optional[StateContext] = None) -> None:
        """
        Nhập nội dung tìm kiếm vào input:
        - Nếu có unicode_text (ví dụ: "Chủ Thương Hội"), dùng input_text_unicode để nhập có dấu.
        - Fallback: dùng ADB input text với chuỗi không dấu.
        """
        # Nếu chưa truyền unicode_text, tự động gán cho các trường hợp đặc biệt cần dấu và in hoa chính xác
        if not unicode_text:
            from .text_recognition import normalize_game_text
            norm_chk = normalize_game_text(text)
            if "chu thuong hoi" in norm_chk or "thuong hoi" in norm_chk:
                unicode_text = "Chủ thương hội"
            elif "tuyen chau" in norm_chk:
                unicode_text = "Tuyền Châu"

        # Ưu tiên nhập Unicode có dấu nếu được chỉ định
        if unicode_text:
            logger.info(f"Gõ tìm kiếm Unicode vào ô input: '{unicode_text}'")
            try:
                self._adb.input_text_unicode(unicode_text)
                time.sleep(0.4)
                try:
                    self._adb.key_event(66)  # Enter để kích hoạt lọc
                    time.sleep(0.4)
                except Exception:
                    pass
                return
            except Exception as e:
                logger.warning(f"Nhập Unicode thất bại: {e}, fallback sang ASCII...")

        from .text_recognition import normalize_game_text
        norm_ascii = normalize_game_text(text)
        escaped = norm_ascii.replace(" ", "%s")
        logger.info(f"Gõ tìm kiếm vào ô input: '{norm_ascii}'")
        self._adb._run(["shell", f"input text {escaped}"])
        time.sleep(0.4)
        try:
            self._adb.key_event(66)  # Enter để kích hoạt lọc
            time.sleep(0.4)
        except Exception:
            pass

    def _click_map_search_result(self, map_name: str) -> bool:
        """Quét và click kết quả bản đồ trong danh sách bên trái.
        
        Ưu tiên:
        1. Khớp chính xác hoàn toàn tên map (ví dụ: 'Đại Lý' == 'Đại Lý', không nhận nhầm 'Đại Lý Phủ').
        2. Loại bỏ các map ngoại thành / phủ / cổng thành / thôn dã ngoại nếu target không yêu cầu.
        """
        from .text_recognition import normalize_game_text
        coords = self._loader.load_ui_coords()
        nav_cfg = coords.get("navigation", {})
        region_cfg = nav_cfg.get("map_list_region", {"x1": 75, "y1": 250, "x2": 465, "y2": 865})
        region = (region_cfg.get("x1", 75), region_cfg.get("y1", 250), region_cfg.get("x2", 465), region_cfg.get("y2", 865))

        scr = self._screen.capture()
        if scr is not None:
            boxes = self._text.read(scr, region)
            norm_target = normalize_game_text(map_name)
            norm_target_nospace = norm_target.replace(" ", "")
            
            # 1. Ưu tiên TUYỆT ĐỐI: Khớp chính xác từng từ (Exact match: 'dai ly' == 'dai ly' hoặc OCR mất khoảng cách 'daily')
            exact_match = next((
                b for b in boxes 
                if normalize_game_text(b.text).replace(" ", "") == norm_target_nospace
            ), None)
            if exact_match:
                logger.info(f"Tìm thấy map chính xác '{exact_match.text}' tại ({exact_match.center_x}, {exact_match.center_y}), bấm...")
                self._input.tap(exact_match.center_x, exact_match.center_y)
                return True

            # 2. Fallback: Khớp chứa nhưng loại trừ 'phu', 'dong', 'tay', 'nam', 'bac', 'mon', 'thon'
            # Riêng với "Đại Lý": Tìm cứng chính xác "Đại lý", cấm tuyệt đối nhận nhầm "Đại Lý phủ" hay "Thành Đại Lý"
            if norm_target_nospace == "daily":
                logger.warning("Không tìm thấy map chính xác 'Đại Lý', từ chối fallback sang map khác (Đại Lý phủ / Thành Đại Lý).")
                return False

            bad_keywords = ("phu", "dong", "tay", "nam", "bac", "mon", "cong", "thon", "ngoai", "dong")
            for b in boxes:
                norm_b = normalize_game_text(b.text)
                norm_b_nospace = norm_b.replace(" ", "")
                if norm_target in norm_b or norm_target_nospace in norm_b_nospace:
                    if any(w in norm_b.split() for w in bad_keywords) and not any(w in norm_target.split() for w in bad_keywords):
                        continue
                    logger.info(f"Tìm thấy map tương tự '{b.text}' tại ({b.center_x}, {b.center_y}), bấm...")
                    self._input.tap(b.center_x, b.center_y)
                    return True
        return False

    def _click_npc_search_result(self, npc_name: str = "Chu thuong hoi") -> bool:
        """Quét và click kết quả NPC trong danh sách bên trái bằng nhận diện OCR."""
        from .text_recognition import normalize_game_text

        coords = self._loader.load_ui_coords() if hasattr(self, "_loader") else {}
        nav_cfg = coords.get("navigation", {})
        search_roi = (50, 250, 290, 820)

        scr = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
        if scr is None:
            return False

        norm_target = normalize_game_text(npc_name)

        def _match_in_boxes(boxes_list):
            if "xa phu" in norm_target:
                # 1. Ưu tiên tuyệt đối dòng Xa phu chính (không mang hướng Bắc, Nam, Đông, Tây, Môn, Kho)
                main_xa_phu = next((
                    b for b in boxes_list
                    if "xa phu" in normalize_game_text(b.text)
                    and not any(d in normalize_game_text(b.text).split() for d in ("bac", "nam", "dong", "tay", "mon", "ngoai", "kho"))
                ), None)
                if main_xa_phu:
                    return main_xa_phu, "Xa phu chính"

                # 2. Ưu tiên 'Xa phu Dong' nếu thành phố chỉ có Xa phu theo cổng (như Biện Kinh)
                dong = next((b for b in boxes_list if "xa phu dong" in normalize_game_text(b.text) and "kho" not in normalize_game_text(b.text)), None)
                if dong:
                    return dong, "Xa phu Đông"

                # 3. Fallback: bất kỳ Xa phu nào (phải có đủ chữ 'xa' và 'phu', tuyệt đối không phải 'thu kho')
                any_xa_phu = next((
                    b for b in boxes_list 
                    if "xa phu" in normalize_game_text(b.text) and "thu kho" not in normalize_game_text(b.text) and "kho" not in normalize_game_text(b.text).split()
                ), None)
                if any_xa_phu:
                    return any_xa_phu, "Xa phu"
                return None, ""

            for b in boxes_list:
                norm_b = normalize_game_text(b.text)
                if norm_target in norm_b or ("thuong hoi" in norm_target and any(k in norm_b for k in ("thuong hoi", "chu thuong"))):
                    return b, b.text
            return None, ""

        # Quét OCR tìm tên NPC trang đầu tiên
        raw_boxes = self._text.read(scr, search_roi)
        boxes = [b for b in raw_boxes if b.center_y <= 820]
        logger.info(f"Minimap NPC list OCR boxes: {[(b.text, b.center_x, b.center_y) for b in boxes]}")
        target, desc = _match_in_boxes(boxes)
        if target:
            logger.info(f"Tìm thấy NPC '{desc}' bằng OCR tại ({target.center_x}, {target.center_y}), bấm...")
            self._input.tap(target.center_x, target.center_y)
            return True

        # Cuộn danh sách xuống để tìm tiếp (tối đa 8 lần)
        for scroll_i in range(8):
            self._input.swipe(170, 700, 170, 350, duration_ms=300)
            time.sleep(0.4)
            scr = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
            if scr is None:
                continue
            raw_boxes = self._text.read(scr, search_roi)
            boxes = [b for b in raw_boxes if b.center_y <= 820]
            target, desc = _match_in_boxes(boxes)
            if target:
                logger.info(f"Tìm thấy NPC '{desc}' sau khi cuộn trang {scroll_i + 1} tại ({target.center_x}, {target.center_y}), bấm...")
                self._input.tap(target.center_x, target.center_y)
                return True

        logger.info(f"Không tìm thấy kết quả NPC '{npc_name}' trong danh sách minimap (vùng {search_roi}).")
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
        logger.info(f"[{ctx.account_id}] Đang kiểm tra vị trí hiện tại trước khi tìm Chủ thương hội...")

        # 0. Nếu đang ở thành phố khác (vd: Tương Dương, Biện Kinh...): nếu bật use_nearest_commerce_chief thì tìm Chủ thương hội ngay tại thành phố đó!
        cur_city, _ = self._get_current_location_from_hud()
        use_nearest = False
        try:
            opt = getattr(self._cfg, "commerce_optimization", None)
            if opt is not None:
                use_nearest = bool(getattr(opt, "use_nearest_commerce_chief", False))
        except Exception:
            pass

        target_city = "Tuyền Châu"
        is_at_target_city = False

        if cur_city:
            from .text_recognition import normalize_game_text, fuzzy_score
            norm_cur = normalize_game_text(cur_city)
            is_tuyen_chau = is_same_city(cur_city, "Tuyền Châu")
            matched_major = next((mc for mc in KNOWN_CITIES if is_same_city(cur_city, mc)), None)

            if use_nearest and matched_major:
                target_city = matched_major
                is_at_target_city = True
                ctx.extra["chief_city"] = target_city
                logger.info(f"[{ctx.account_id}] ⚡ [TỐI ƯU GẦN NHẤT] Đang ở đại thành thị '{target_city}', tìm Chủ thương hội ngay tại đây!")
            elif is_tuyen_chau:
                target_city = "Tuyền Châu"
                is_at_target_city = True
                ctx.extra["chief_city"] = "Tuyền Châu"
            else:
                target_city = "Tuyền Châu"
                ctx.extra["chief_city"] = "Tuyền Châu"
                if hasattr(self, "_switch_to_city"):
                    logger.info(f"[{ctx.account_id}] ⚡ Nhân vật đang ở '{cur_city}', chuyển map về {target_city}...")
                    activity._write("🐎", f"[{ctx.account_id}] Chuyen map ve {target_city}...")
                    switched = self._switch_to_city(
                        ctx,
                        target_city,
                        current_location=(cur_city, None),
                    )
                    time.sleep(0.5)
                    new_city, _ = self._get_current_location_from_hud()
                    if new_city and is_same_city(new_city, target_city):
                        is_at_target_city = True
                    else:
                        is_at_target_city = False
                        if switched:
                            logger.warning(
                                f"[{ctx.account_id}] Navigation returned success but HUD still shows "
                                f"'{new_city or cur_city}'; {target_city} arrival is unconfirmed."
                            )

        logger.info(f"[{ctx.account_id}] Bắt đầu quy trình tìm Chủ thương hội tại '{target_city}' (vị trí hiện tại: '{cur_city or 'chưa rõ'}')...")
        activity._write("🏃", f"[{ctx.account_id}] Tim Chu thuong hoi {target_city}...")

        coords = self._loader.load_ui_coords()
        nav_cfg = coords.get("navigation", {})
        search_x = nav_cfg.get("map_search_input", {}).get("x", 266)
        search_y = nav_cfg.get("map_search_input", {}).get("y", 222)
        find_npc_x = nav_cfg.get("map_tab_find_npc", {}).get("x", 376)
        find_npc_y = nav_cfg.get("map_tab_find_npc", {}).get("y", 161)

        # Bước 1: Mở bản đồ mini
        logger.info(f"[{ctx.account_id}] Bước 1: Mở bản đồ mini...")
        if not self._ensure_map_open():
            logger.warning(f"[{ctx.account_id}] Không mở được minimap; dừng quy trình tìm NPC để thử lại, không click mù.")
            return False
        time.sleep(0.2)

        if not is_at_target_city:
            # Bước 2: Nhấn tab 'Tìm map'
            logger.info(f"[{ctx.account_id}] Bước 2: Nhấn tab 'Tìm map'...")
            find_map_x = nav_cfg.get("map_tab_find_map", {}).get("x", 176)
            find_map_y = nav_cfg.get("map_tab_find_map", {}).get("y", 162)
            self._input.tap(find_map_x, find_map_y)
            time.sleep(0.25)

            # Bước 3: Nhấn vô input tìm map và xóa chữ cũ
            logger.info(f"[{ctx.account_id}] Bước 3: Nhấn vào ô input tìm map...")
            self._input.tap(search_x, search_y)
            time.sleep(0.15)
            try:
                delete_cmd = "input keyevent 123; " + "; ".join(["input keyevent 67"] * 30)
                self._adb._run(["shell", delete_cmd])
                time.sleep(0.1)
            except Exception:
                pass

            # Bước 4: Gõ tìm target_city
            logger.info(f"[{ctx.account_id}] Bước 4: Gõ tìm '{target_city}'...")
            from .text_recognition import normalize_game_text
            target_ascii = "Tuyen Chau" if target_city == "Tuyền Châu" else normalize_game_text(target_city).title()
            self._input_search_text(target_ascii, unicode_text=target_city, ctx=ctx)
            time.sleep(0.4)

            # Bước 5: Nhấn chọn target_city trong danh sách kết quả bản đồ
            logger.info(f"[{ctx.account_id}] Bước 5: Nhấn chọn '{target_city}' trong danh sách map...")
            map_clicked = self._click_map_search_result(target_city)
            if not map_clicked:
                first_res = nav_cfg.get("map_tuyen_chau", {"x": 115, "y": 315})
                self._input.tap(first_res.get("x", 115), first_res.get("y", 315))
            time.sleep(0.6)
            if not self._ensure_map_open():
                logger.warning(f"[{ctx.account_id}] Minimap không còn mở sau khi chọn {target_city}; sẽ thử lại quy trình.")
                return False
            time.sleep(0.2)
        else:
            logger.info(f"[{ctx.account_id}] Đã ở thành phố mục tiêu '{target_city}', bỏ qua bước tìm map, tìm thẳng NPC Chủ thương hội...")

        # Bước 6: Nhấn tab 'Tìm NPC'
        logger.info(f"[{ctx.account_id}] Bước 6: Nhấn tab 'Tìm NPC'...")
        self._input.tap(find_npc_x, find_npc_y)
        time.sleep(0.3)

        # Bước 7: Nhấn vào ô input tìm NPC (cùng vị trí ô search trên map)
        logger.info(f"[{ctx.account_id}] Bước 7: Nhấn vào ô input tìm NPC...")
        self._input.tap(search_x, search_y)
        time.sleep(0.15)
        # Xóa text cũ nếu có
        try:
            delete_cmd = "input keyevent 123; " + "; ".join(["input keyevent 67"] * 30)
            self._adb._run(["shell", delete_cmd])
            time.sleep(0.1)
        except Exception:
            pass

        # Bước 8: Gõ "Chủ thương hội" (bắt buộc có dấu tiếng Việt) vào ô input NPC
        logger.info(f"[{ctx.account_id}] Bước 8: Gõ tìm 'Chủ thương hội' (có dấu) vào ô input NPC...")
        self._input_search_text("Chu thuong hoi", unicode_text="Chủ thương hội", ctx=ctx)
        time.sleep(0.5)

        # Bước 9: Nhấn vào kết quả NPC 'Chủ thương hội' trong danh sách để xác nhận di chuyển
        logger.info(f"[{ctx.account_id}] Bước 9: Chọn và xác nhận NPC 'Chủ thương hội' từ kết quả tìm kiếm...")
        npc_clicked = self._click_npc_search_result("Chu thuong hoi")
        if not npc_clicked:
            logger.warning(f"[{ctx.account_id}] Không tìm thấy NPC Chủ thương hội trong kết quả tìm kiếm! Tuyệt đối không bấm tọa độ mù.")
            self._close_map()
            return False

        time.sleep(0.2)
        # Đóng bản đồ an toàn để nhân vật tự động chạy đến
        self._close_map()
        return True

    def _handle_find_commerce_chief(self, ctx: StateContext) -> BotState:
        """Tìm và di chuyển đến NPC Chủ thương hội tại Tuyền Châu theo đúng 8 bước chuẩn."""
        # Chỉ dò/đánh quái chặn đường khi nhiệm vụ hiện tại thực sự là COMBAT.
        # Với nhiệm vụ mua đồ/NPC, OCR tên người chơi (ví dụ "Han") từng bị
        # nhận nhầm là quái và làm bot lặp đánh sai hàng chục phút.
        # Chỉ dò/đánh quái chặn đường khi nhiệm vụ hiện tại thực sự là COMBAT và CHƯA hoàn thành/quay về.
        # Với nhiệm vụ mua đồ/NPC, hoặc khi đã hoàn thành/đang quay về trả nhiệm vụ: tuyệt đối không đánh quái!
        if (
            ctx.current_quest_type == QuestType.COMBAT
            and not ctx.extra.get("combat_objective_complete")
            and not ctx.extra.get("returning_quest")
            and not ctx.extra.get("abandon_combat_quest")
            and self._attack_monster_blocking_travel(ctx)
        ):
            ctx.extra.pop("chief_nav_started", None)
            time.sleep(0.5)
            return BotState.FIND_COMMERCE_CHIEF

        # 0. Nếu nhân vật ĐÃ ĐỨNG CẠNH Chủ Thương Hội:
        cur_city_at_start, _ = self._get_current_location_from_hud() if hasattr(self, "_get_current_location_from_hud") else (None, None)
        target_chief_city = ctx.extra.get("chief_city")
        is_at_chief_city = (
            not target_chief_city
            or cur_city_at_start is None
            or is_same_city(cur_city_at_start, target_chief_city)
            or is_same_city(cur_city_at_start, "Tuyền Châu")
            or any(is_same_city(cur_city_at_start, mc) for mc in KNOWN_CITIES)
        )
        if is_at_chief_city and self._is_near_commerce_chief():
            display_city = cur_city_at_start or target_chief_city or "Tuyền Châu"
            logger.info(f"[{ctx.account_id}] Nhân vật đã đứng cạnh Chủ thương hội tại '{display_city}'! Mở hộp thoại nhận nhiệm vụ ngay...")
            activity._write("💬", f"[{ctx.account_id}] Da o canh Chu thuong hoi ({display_city}), hoi thoai nhan NV...")
            if (
                not self._open_commerce_chief_dialog(ctx)
                or not self._verify_commerce_chief_dialog()
            ):
                failures = int(ctx.extra.get("chief_dialog_failures", 0)) + 1
                ctx.extra["chief_dialog_failures"] = failures
                logger.warning(
                    f"[{ctx.account_id}] Chưa xác nhận được hộp thoại Chủ thương hội tại {display_city} (lần {failures}/3); "
                    "không chuyển sang MEET_NPC."
                )
                self._ensure_npc_dialog_closed()
                time.sleep(0.3)

                # Do lag hình ảnh trong game, nhích tới nhích lui để hiển thị Chủ thương hội ra
                logger.info(f"[{ctx.account_id}] 🔄 [Lag hình ảnh] Nhích tới lui nhân vật để game hiển thị lại Chủ thương hội (lần {failures})...")
                activity._write("🔄", f"[{ctx.account_id}] Nhich toi lui hien Chu thuong hoi...")
                self._nudge_left_right_to_reveal_npc(ctx)
                time.sleep(0.5)

                if failures >= 3:
                    logger.warning(
                        f"[{ctx.account_id}] Đã nhích {failures} lần vẫn chưa mở được đối thoại Chủ thương hội! "
                        "Mở lại Minimap để tìm lại đường đến Chủ thương hội..."
                    )
                    activity._write("🗺️", f"[{ctx.account_id}] Tim lai Chu thuong hoi qua Minimap...")
                    ctx.extra.pop("chief_dialog_failures", None)
                    ctx.extra.pop("chief_nav_started", None)
                    self._navigate_to_tuyen_chau_commerce_chief(ctx)

                return BotState.FIND_COMMERCE_CHIEF

            ctx.extra.pop("chief_dialog_failures", None)
            ctx.extra.pop("chief_nav_started", None)
            ctx.extra.pop("chief_nav_failures", None)
            return BotState.MEET_NPC

        # Không mở icon Nhiệm vụ để suy đoán có nhiệm vụ hay chưa. Dù đang ở
        # thành hay map quái, luôn về gặp Chủ thương hội và dùng đối thoại làm
        # nguồn xác nhận duy nhất.

        # 1. Chỉ thực hiện quy trình tìm đường 8 bước nếu CHƯA bắt đầu di chuyển:
        if not ctx.extra.get("chief_nav_started"):
            navigation_started = self._navigate_to_tuyen_chau_commerce_chief(ctx)
            if not navigation_started:
                ctx.extra.pop("chief_nav_started", None)
                failures = int(ctx.extra.get("chief_nav_failures", 0)) + 1
                ctx.extra["chief_nav_failures"] = failures
                if failures >= 3:
                    logger.error(
                        f"[{ctx.account_id}] Không thể mở minimap/tìm Chủ thương hội "
                        f"sau {failures} lần liên tiếp. Chuyển RECOVERY thay vì lặp vô hạn."
                    )
                    activity._write("♻️", f"[{ctx.account_id}] Tim Chu thuong hoi that bai 3 lan, khoi dong lai game...")
                    ctx.extra.pop("chief_nav_failures", None)
                    return BotState.RECOVERY
                activity._write(
                    "🔄",
                    f"[{ctx.account_id}] Chua mo duoc minimap, thu lai {failures}/3...",
                )
                time.sleep(0.5)
                return BotState.FIND_COMMERCE_CHIEF
            ctx.extra["chief_nav_started"] = True
            ctx.extra.pop("chief_nav_failures", None)
        else:
            logger.info(f"[{ctx.account_id}] Nhân vật đã được lệnh di chuyển về Chủ thương hội từ trước, tiến hành chờ đến nơi...")

        # 2. Chờ nhân vật chạy đến đứng cạnh Chủ Thương Hội (tối đa 90s)
        target_chief_city = ctx.extra.get("chief_city")
        def _is_valid_chief_city(city: Optional[str]) -> bool:
            if not city:
                return True
            if target_chief_city:
                return is_same_city(city, target_chief_city)
            return is_same_city(city, "Tuyền Châu") or any(is_same_city(city, mc) for mc in KNOWN_CITIES)

        start_t = time.time()
        last_coords = None
        last_30s_check = time.time()
        coords_30s_ago = None
        last_popup_check = 0.0
        while time.time() - start_t < 90.0:
            time.sleep(0.25)
            frame = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
            if frame is None:
                continue

            # Chỉ kiểm tra đóng popup định kỳ mỗi 3s thay vì chạy OCR toàn màn hình liên tục mỗi 250ms
            now = time.time()
            if now - last_popup_check >= 3.0:
                self._dismiss_unexpected_popups(frame)
                last_popup_check = now

            # Confirm the city first; NPC labels/coordinates alone can match
            # similar points on another map.
            cur_city_hud, cur_coords = self._get_current_location_from_hud(frame)
            # Kiểm tra _is_near_commerce_chief trước (xác nhận đúng NPC bằng OCR/tọa độ)
            if (
                _is_valid_chief_city(cur_city_hud)
                and self._is_near_commerce_chief(frame)
            ):
                logger.info(f"[{ctx.account_id}] Nhân vật đã đến cạnh Chủ thương hội! Tiến hành mở hộp thoại...")
                ctx.extra.pop("chief_nav_started", None)
                # Chờ nhẹ 0.3s để game render bong bóng đối thoại NPC sau khi dừng chạy
                time.sleep(0.3)
                frame_now = self._screen.capture()
                self._click_npc_interact_bubble(frame_now)
                time.sleep(0.35)
                # Xác nhận trực tiếp hộp thoại Chủ Thương Hội bằng verify (tiết kiệm 1 lần OCR)
                if self._verify_commerce_chief_dialog():
                    ctx.extra.pop("chief_arrival_nudge_attempts", None)
                    ctx.extra.pop("chief_dialog_failures", None)
                    return BotState.MEET_NPC
                elif self._is_npc_dialog_open():
                    dialog_city, _ = self._get_current_location_from_hud()
                    if (
                        _is_valid_chief_city(dialog_city)
                        and self._verify_commerce_chief_dialog()
                    ):
                        ctx.extra.pop("chief_arrival_nudge_attempts", None)
                        ctx.extra.pop("chief_dialog_failures", None)
                        return BotState.MEET_NPC
                    else:
                        logger.warning(f"[{ctx.account_id}] Hộp thoại mở ra KHÔNG PHẢI Chủ Thương Hội! Đóng và thử lại...")
                        if hasattr(self, "_input") and hasattr(self._input, "press_back"):
                            self._input.press_back()
                        elif hasattr(self, "_adb") and hasattr(self._adb, "press_back"):
                            self._adb.press_back()
                        time.sleep(0.3)
                        continue
                # Nếu bấm bubble chưa ra dialog, thử mở bằng _open_commerce_chief_dialog
                if self._open_commerce_chief_dialog(ctx):
                    ctx.extra.pop("chief_arrival_nudge_attempts", None)
                    ctx.extra.pop("chief_dialog_failures", None)
                    return BotState.MEET_NPC
                    logger.warning(
                        f"[{ctx.account_id}] NPC dialog opened but not confirmed as the commerce chief "
                        f"(HUD: '{current_city_after_dialog or 'unknown'}'); closing it."
                    )
                    self._ensure_npc_dialog_closed()

                # Nếu đến nơi mà chưa mở được đối thoại (do lag hình ảnh chưa render NPC):
                logger.info(f"[{ctx.account_id}] 🔄 Đã tới cạnh Chủ thương hội nhưng chưa mở được đối thoại (do lag hình ảnh). Nhích tới lui để hiển thị NPC...")
                self._nudge_left_right_to_reveal_npc(ctx)
                time.sleep(0.5)

            # Nếu hộp thoại đã tự mở (vd: chạm vào NPC khi đi ngang)
            if self._is_npc_dialog_open(frame):
                if (
                    _is_valid_chief_city(cur_city_hud)
                    and self._verify_commerce_chief_dialog(frame)
                ):
                    logger.info(f"[{ctx.account_id}] Hộp thoại Chủ thương hội đã mở sẵn!")
                    ctx.extra.pop("chief_nav_started", None)
                    ctx.extra.pop("chief_arrival_nudge_attempts", None)
                    return BotState.MEET_NPC
                else:
                    logger.warning(f"[{ctx.account_id}] Hộp thoại mở ra KHÔNG PHẢI Chủ Thương Hội! Đóng...")
                    if hasattr(self, "_input") and hasattr(self._input, "press_back"):
                        self._input.press_back()
                    elif hasattr(self, "_adb") and hasattr(self._adb, "press_back"):
                        self._adb.press_back()
                    time.sleep(0.3)
                    continue

            # Theo dõi tọa độ HUD: nếu nhân vật đang chạy thì gia hạn thời gian
            cur_city_hud, cur_coords = self._get_current_location_from_hud(frame)
            if self._nudge_to_reveal_commerce_chief(ctx, frame):
                last_coords = cur_coords
                coords_30s_ago = cur_coords
                last_30s_check = time.time()
                continue
            if cur_coords and last_coords and cur_coords != last_coords:
                start_t = max(start_t, time.time() - 30.0)
            if cur_coords:
                last_coords = cur_coords
                if coords_30s_ago is None:
                    coords_30s_ago = cur_coords
                    last_30s_check = time.time()
                elif time.time() - last_30s_check >= 30.0:
                    dist_30s = abs(cur_coords[0] - coords_30s_ago[0]) + abs(cur_coords[1] - coords_30s_ago[1])
                    if dist_30s <= 1:
                        logger.warning(
                            f"[{ctx.account_id}] ⚠️ Đang tìm Chủ thương hội nhưng bị đứng yên tại chỗ 30s ở {cur_coords}! "
                            f"Bắt đầu tìm lại quá trình chạy..."
                        )
                        activity._write("🔄", f"[{ctx.account_id}] Dung yen 30s, tim lai duong Chu thuong hoi...")
                        ctx.extra.pop("chief_nav_started", None)

                        # Nếu đang ở map dã ngoại (như Vân Mộng Trạch), diệt sạch quái đang bu cắn nhân vật
                        if cur_city_hud and not _is_valid_chief_city(cur_city_hud):
                            self._clear_monsters_around_character(ctx)

                        # Nếu đang ở thành phố khác hoặc map dã ngoại: chuyển sang Tuyền Châu
                        if cur_city_hud and not _is_valid_chief_city(cur_city_hud):
                            logger.info(f"[{ctx.account_id}] Phát hiện đang ở thành/map '{cur_city_hud}', chuyển sang Tuyền Châu...")
                            is_major_city = any(is_same_city(cur_city_hud, mc) for mc in KNOWN_CITIES)
                            if is_major_city and not ctx.extra.get("out_of_money"):
                                switched = self._switch_to_city(ctx, "Tuyền Châu", force=True)
                                if switched:
                                    logger.info(f"[{ctx.account_id}] Đã tới Tuyền Châu qua Xa Phu; tìm đường tới Chủ thương hội ngay...")
                                    self._navigate_to_tuyen_chau_commerce_chief(ctx)
                                    ctx.extra["chief_nav_started"] = True
                                    return BotState.FIND_COMMERCE_CHIEF
                            else:
                                self._navigate_to_tuyen_chau_commerce_chief(ctx)
                                ctx.extra["chief_nav_started"] = True
                                return BotState.FIND_COMMERCE_CHIEF
                        break
                    else:
                        coords_30s_ago = cur_coords
                        last_30s_check = time.time()

        # Nếu sau thời gian chờ mà nhân vật VẪN CHƯA ĐẾN CẠNH Chủ thương hội:
        # Tuyệt đối không chuyển sang MEET_NPC để tránh log sai và click bừa vào đất
        final_city, _ = self._get_current_location_from_hud()
        final_is_wrong_city = bool(
            final_city and not _is_valid_chief_city(final_city)
        )
        final_is_chief_dialog = False
        if not final_is_wrong_city:
            has_dialog_visual = (
                (hasattr(self, "_find_dialog_buttons") and bool(self._find_dialog_buttons()))
                or (hasattr(self, "_vision") and self._vision.find_template("dialog_box").found)
            )
            if has_dialog_visual and self._is_npc_dialog_open():
                final_is_chief_dialog = self._verify_commerce_chief_dialog()

        if final_is_wrong_city or not (
            self._is_near_commerce_chief() or final_is_chief_dialog
        ):
            logger.warning(
                f"[{ctx.account_id}] ⚠️ Hết thời gian chờ nhưng nhân vật CHƯA ĐẾN Chủ thương hội "
                f"(HUD: {final_city or 'chưa rõ'}, tọa độ: {last_coords or 'chưa rõ'}); "
                "không báo đã tới và thử lại tìm đường..."
            )
            ctx.extra.pop("chief_nav_started", None)
            time.sleep(0.5)
            return BotState.FIND_COMMERCE_CHIEF

        if final_is_chief_dialog:
            logger.info(
                f"[{ctx.account_id}] HUD và nội dung hộp thoại xác nhận đúng Chủ thương hội tại Tuyền Châu."
            )
            return BotState.MEET_NPC

        ctx.extra.pop("chief_nav_started", None)
        logger.info(f"[{ctx.account_id}] Nhân vật đã đến cạnh Chủ thương hội, chuyển sang MEET_NPC.")
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
        quest_btn = coords.get("commerce_quest", {}).get("quest_button", {"x": 1460, "y": 765})
        self._input.tap(quest_btn.get("x", 1460), quest_btn.get("y", 765))   # Mở bảng nhiệm vụ
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
        if not self._is_near_commerce_chief() and not self._is_npc_dialog_open():
            logger.warning(f"[{ctx.account_id}] ⚠️ Chưa đứng ở vị trí Chủ thương hội! Quay lại FIND_COMMERCE_CHIEF...")
            return BotState.FIND_COMMERCE_CHIEF

        logger.info(f"[{ctx.account_id}] Đang tương tác với NPC Chủ thương hội...")
        time.sleep(0.15)

        # Không mở icon Nhiệm vụ trước đối thoại. Chủ thương hội là nguồn duy
        # nhất xác định có nhiệm vụ dở dang, hoàn thành hay chưa có nhiệm vụ.
        # 1. Đảm bảo hộp thoại NPC Chủ Thương Hội thực sự được mở
        dialog_scr = self._screen.capture()
        dialog_boxes = self._text.read(dialog_scr, (150, 250, 1450, 800))
        dialog_raw = " ".join(b.text for b in dialog_boxes)
        dialog_norm = normalize_game_text(dialog_raw)
        has_dialog = any(w in dialog_norm for w in (
            "chu thuong hoi", "thuong hoi", "vi trieu dinh", "cong hien", "hoan thanh",
            "huy bo", "nhiem vu", "lan truoc", "20 nhiem vu"
        ))
        if not has_dialog:
            if not self._open_commerce_chief_dialog(ctx):
                logger.warning(f"[{ctx.account_id}] Không mở được hộp thoại với Chủ thương hội, quay lại FIND_COMMERCE_CHIEF...")
                return BotState.FIND_COMMERCE_CHIEF
            dialog_scr = self._screen.capture()
            dialog_boxes = self._text.read(dialog_scr, (150, 250, 1450, 800))
            dialog_raw = " ".join(b.text for b in dialog_boxes)
            dialog_norm = normalize_game_text(dialog_raw)

        # Đồng bộ số nhiệm vụ đã hoàn thành hôm nay từ hộp thoại NPC nếu có
        # Ví dụ: "Hôm nay đã hoàn thành 8 lần nhiệm vụ. Còn có thể nhận 12 lần!"
        self._sync_quest_progress_from_text(ctx, dialog_raw, source="đối thoại Chủ thương hội")

        # Kiểm tra nếu đây là hộp thoại tóm tắt tiến độ ("Hôm nay đã hoàn thành X lần nhiệm vụ. Còn có thể nhận Y lần!"):
        is_progress_summary_dialog = (
            ("hoan thanh" in dialog_norm and any(k in dialog_norm for k in ("lan nhiem vu", "lan", "con co the nhan")))
            or ("con co the nhan" in dialog_norm)
        ) and not any(w in dialog_norm for w in ("vi trieu dinh", "cong hien", "tra nhiem vu", "nhiem vu da hoan thanh", "nhan 20 nhiem vu", "lan truoc da hoan thanh chua"))

        if is_progress_summary_dialog:
            logger.info(f"[{ctx.account_id}] 📊 Phát hiện hộp thoại tóm tắt tiến độ nhiệm vụ từ Chủ thương hội: '{dialog_raw}'")
            if ctx.total_quests_done >= 20:
                logger.info(f"[{ctx.account_id}] Đã đạt tối đa 20/20 nhiệm vụ Thương Hội hôm nay! Hoàn tất.")
                activity._write("🏆", f"[{ctx.account_id}] Da hoan thanh 20/20 nhiem vu Thuong Hoi! Dang xuat...")
                self._save_progress(ctx, status_text="Đã hoàn thành 20/20 🎉")
                return BotState.LOGOUT

            close_btn = None
            for b in dialog_boxes:
                # Nút 'Kết thúc đối thoại' nằm trong vùng thân/dưới hộp thoại (y >= 420) và bên trong modal (x >= 220)
                if b.center_y < 420 or b.center_x < 220:
                    continue
                b_norm = normalize_game_text(b.text)
                b_norm_ns = b_norm.replace(" ", "")
                if (
                    any(k in b_norm for k in ("ket thuc", "ket thic", "ket thie", "doi thoai", "doithoaf"))
                    or any(k in b_norm_ns for k in ("ketthuc", "ketthic", "ketthie", "doithoai", "doithoaf"))
                ) and not any(ex in b_norm for ex in ("dong bac", "tuyet son", "trong yem", "nhiem vu")):
                    close_btn = (b.center_x, b.center_y)
                    break

            btn_to_tap = close_btn or (348, 488)
            logger.info(f"[{ctx.account_id}] Bấm nút 'Kết thúc đối thoại' tại {btn_to_tap}...")
            self._input.tap(btn_to_tap[0], btn_to_tap[1])
            time.sleep(0.5)

            # Bắt buộc kiểm tra và đóng dứt điểm hộp thoại tiến độ trước khi chuyển sang thao tác mới:
            closed = False
            for attempt in range(3):
                scr_check = self._screen.capture()
                if not self._is_npc_dialog_open(scr_check):
                    logger.info(f"[{ctx.account_id}] Đã tắt xong hộp thoại tóm tắt tiến độ!")
                    closed = True
                    break
                logger.info(f"[{ctx.account_id}] Hộp thoại tiến độ vẫn còn mở, bấm lại tại (348, 488) (lần {attempt + 1})...")
                self._input.tap(348, 488)
                time.sleep(0.4)

            if not closed:
                self._ensure_npc_dialog_closed()

            time.sleep(0.3)
            ctx.extra["in_game_progress_synced"] = True
            return BotState.MEET_NPC

        # 0. TRẠNG THÁI CHƯA NHẬN NHIỆM VỤ NÀO:
        # Khi chưa nhận nhiệm vụ, Chủ thương hội hiển thị chiếu chỉ:
        # "Hoàng thượng vừa ban chiếu: Nhân sĩ nào giúp sức Triều đình sẽ được trọng thưởng! Mỗi ngày có thể nhận tối đa 20 lần nhiệm vụ!"
        # Kèm tùy chọn: "Vì triều đình cống hiến sức lực"
        contrib_btn = None
        for b in dialog_boxes:
            norm_b = normalize_game_text(b.text)
            norm_b_ns = norm_b.replace(" ", "")
            if (
                any(w in norm_b for w in ("vi trieu dinh", "cong hien suc luc", "cong hien", "suc luc", "suc lyc"))
                or any(w in norm_b_ns for w in ("vitrieudinh", "conghien", "sucluc", "suclyc"))
            ) and not any(w in norm_b for w in ("khong co hung thu", "tinh hinh", "gioi thieu")):
                contrib_btn = b
                break

        has_no_quest_dialog = (
            contrib_btn is not None
            or any(w in dialog_norm for w in ("hoang thuong", "ban chieu", "giup suc trieu dinh", "vi trieu dinh", "cong hien", "nhan si nao giup suc", "trong thuong"))
            or any(w in dialog_norm.replace(" ", "") for w in ("hoangthuong", "banchieu", "vitrieudinh", "conghien", "giupsuc", "20lannhiemvu", "sucluc", "suclyc"))
        )

        if has_no_quest_dialog:
            logger.info(
                f"[{ctx.account_id}] [Trạng thái: Chưa nhận nhiệm vụ nào] Phát hiện đối thoại nhận nhiệm vụ mới "
                f"(có nút 'Vì triều đình cống hiến sức lực'). Bấm nút để nhận nhiệm vụ..."
            )
            activity._write("📜", f"[{ctx.account_id}] Chua co NV. Nhan 'Vi trieu dinh cong hien suc luc'...")
            self._save_progress(ctx, status_text="Chưa có NV -> Nhận NV mới")
            for stale_flag in (
                "combat_monster", "combat_map", "combat_count", "combat_sect",
                "combat_objective_complete", "combat_avoided_directions", "combat_last_move_direction",
                "yellow_tokens", "yellow_item", "requested_shop_item", "quest_text", "quest_detail_text",
                "check_previous_quest_on_login", "returning_quest",
                "teleport_to_combat_quest", "abandon_combat_quest", "abandon_combat_quest_reason",
                "abandon_dialogue_quest", "quest_target_city", "quest_target_npc", "out_of_money",
                "merchant_found_city", "chief_nav_started", "no_quest_attempts", "has_active_quest",
            ):
                ctx.extra.pop(stale_flag, None)
            ctx.current_quest_type = None

            if contrib_btn:
                logger.info(f"[{ctx.account_id}] Bấm nút '{contrib_btn.text}' tại ({contrib_btn.center_x}, {contrib_btn.center_y}) để nhận nhiệm vụ...")
                self._input.tap(contrib_btn.center_x, contrib_btn.center_y)
            else:
                fallback = {"x": 330, "y": 580}
                if hasattr(self, "_loader") and self._loader is not None:
                    coords = self._loader.load_ui_coords()
                    fallback = coords.get("commerce_quest", {}).get("contribute_fallback", fallback)
                fx, fy = fallback.get("x", 330), fallback.get("y", 580)
                logger.info(f"[{ctx.account_id}] Dùng tọa độ fallback nút 'Vì triều đình cống hiến sức lực' ({fx}, {fy})...")
                self._input.tap(fx, fy)

            time.sleep(1.0)
            return BotState.GET_QUEST

        dialog_requires_completion_check = any(w in dialog_norm for w in (
            "lan truoc da hoan thanh chua", "nhiem vu lan truoc",
            "nhan 20 nhiem vu nhe", "van chua hoan thanh",
        )) or any(w in dialog_norm.replace(" ", "") for w in (
            "lantruocdahoanthanhchua", "nhiemvulantruoc",
        ))
        must_verify_completion = bool(
            ctx.extra.get("returning_quest")
            or ctx.extra.get("combat_objective_complete")
            or ctx.extra.get("check_previous_quest_on_login")
            or dialog_requires_completion_check
        )

        # Cờ hủy có thể còn từ lần phân loại sai trước đó. Trước
        # thao tác không thể hoàn tác, đối chiếu với thông tin nhiệm vụ
        # hiện tại; nếu là NPC thì xóa cache combat và tuyệt đối không hủy.
        # NGOẠI TRỪ khi có cờ hủy nhiệm vụ đối thoại (abandon_dialogue_quest như NPC Ấu đồng).
        if ctx.extra.get("abandon_combat_quest") and not ctx.extra.get("abandon_dialogue_quest"):
            # Nếu nhiệm vụ hiện tại rõ ràng là COMBAT (quest_type là COMBAT, có combat_map, hoặc timeout quá 10 phút)
            # thì đây là yêu cầu hủy hợp lệ của nhiệm vụ đánh quái, dọn sạch cờ đối thoại cũ còn sót lại.
            is_valid_combat_abandon = (
                ctx.current_quest_type == QuestType.COMBAT
                or bool(ctx.extra.get("combat_map"))
                or "quá" in str(ctx.extra.get("abandon_combat_quest_reason", "")).lower()
                or "10 phút" in str(ctx.extra.get("abandon_combat_quest_reason", ""))
            )
            if is_valid_combat_abandon:
                ctx.extra.pop("quest_target_city", None)
                ctx.extra.pop("quest_target_npc", None)
            else:
                current_quest_norm = normalize_game_text(ctx.extra.get("quest_text", ""))
                current_is_dialogue = bool(
                    ctx.extra.get("quest_target_city")
                    and ctx.extra.get("quest_target_npc")
                ) or any(w in current_quest_norm for w in (
                    "noi voi", "noi chuyen voi", "doi thoai voi", "thong bao cho",
                    "bao tin cho", "bao cho", "gap xa phu", "tim xa phu",
                ))
                if current_is_dialogue:
                    logger.warning(
                        f"[{ctx.account_id}] Cờ hủy combat cũ xung đột với nhiệm vụ "
                        "đối thoại hiện tại -> xóa cờ cũ, KHÔNG hủy nhiệm vụ."
                    )
                    for stale_key in (
                        "abandon_combat_quest", "abandon_combat_quest_reason",
                        "combat_map", "combat_monster", "combat_count", "combat_sect",
                        "combat_quest_start_time", "teleport_to_combat_quest",
                    ):
                        ctx.extra.pop(stale_key, None)
                    ctx.current_quest_type = QuestType.NPC
                    ctx.extra.pop("returning_quest", None)
                    self._checkpoint.save(ctx)

        # 0.1 NẾU CÓ CỜ HỦY NHIỆM VỤ (Đánh quái quá 10 phút HOẶC Nhiệm vụ đối thoại Ấu đồng)
        if ctx.extra.get("abandon_combat_quest") or ctx.extra.get("abandon_dialogue_quest"):
            logger.info(f"[{ctx.account_id}] Đang thực hiện quy trình hủy nhiệm vụ tại Chủ thương hội...")
            activity._write("❌", f"[{ctx.account_id}] Dang huy NV tai Chu thuong hoi...")

            coords = self._loader.load_ui_coords() if hasattr(self, "_loader") and self._loader else {}
            comm_cfg = coords.get("commerce_quest", {})
            opt_fallback = comm_cfg.get("abandon_quest_option", {"x": 1167, "y": 580})
            opt_shop_fallback = comm_cfg.get("abandon_quest_option_shop", {"x": 380, "y": 700})
            confirm_fallback = comm_cfg.get("abandon_quest_confirm", {"x": 303, "y": 571})
            finish_fallback = comm_cfg.get("abandon_quest_finish", {"x": 309, "y": 543})

            # BƯỚC 1: Chọn "Ta muốn hủy bỏ nhiệm vụ đang làm" trong đối thoại Chủ thương hội
            cancel_box = None
            cancel_keywords = (
                "muon huy bo", "tamuon huybo", "huy bo nhiem vu", "huy nhiem vu", "huy bo",
                "nhiem vu dang lam", "dang lam", "lan nay", "nhiemvudanglam", "muon huy", "huybo"
            )
            for b in dialog_boxes:
                if b.center_y < 420:
                    continue
                norm = normalize_game_text(b.text)
                norm_ns = norm.replace(" ", "")
                if any(kw in norm for kw in cancel_keywords) or any(kw in norm_ns for kw in ("muonhuybo", "tamuonhuybo", "huybonhiemvu", "huynhiemvu", "huybo", "nhiemvudanglam", "danglam")):
                    cancel_box = b
                    break

            # Nếu chưa tìm thấy bằng box ban đầu, chụp ảnh mới để đảm bảo UI đã load đủ các nút
            if not cancel_box:
                fresh_scr = self._screen.capture()
                if fresh_scr is not None:
                    fresh_boxes = self._text.read(fresh_scr, (150, 420, 1450, 800))
                    for b in fresh_boxes:
                        if b.center_y < 420:
                            continue
                        norm = normalize_game_text(b.text)
                        norm_ns = norm.replace(" ", "")
                        if any(kw in norm for kw in cancel_keywords) or any(kw in norm_ns for kw in ("muonhuybo", "tamuonhuybo", "huybonhiemvu", "huynhiemvu", "huybo", "nhiemvudanglam", "danglam")):
                            cancel_box = b
                            break

            is_shop_or_4_options = (
                getattr(ctx, "current_quest_type", None) == QuestType.SPECIAL_NPC_SHOP
                or any(w in dialog_norm for w in ("da hoan thanh", "xem tinh hinh", "hoan thanh!"))
                or any(w in dialog_norm.replace(" ", "") for w in ("dahoanthanh", "xemtinhhinh"))
            )

            tapped_x, tapped_y = None, None
            if cancel_box:
                tapped_x, tapped_y = cancel_box.center_x, cancel_box.center_y
                logger.info(f"[{ctx.account_id}] [Bước 1/3] Bấm nút '{cancel_box.text}' tại ({tapped_x}, {tapped_y})...")
                self._input.tap(tapped_x, tapped_y)
            elif any(w in dialog_norm for w in ("co can ta dua", "phien dua ta", "dua den noi")):
                # Đang ở hộp thoại hỏi đưa đi làm nhiệm vụ, không phải menu chính để hủy!
                logger.warning(f"[{ctx.account_id}] Đang ở hộp thoại hỏi dịch chuyển, không bấm fallback hủy NV để tránh bị dịch chuyển nhầm. Bấm từ chối...")
                self._input.tap(460, 280)
                time.sleep(1.0)
                return BotState.FIND_COMMERCE_CHIEF
            else:
                if is_shop_or_4_options:
                    tapped_x, tapped_y = opt_shop_fallback["x"], opt_shop_fallback["y"]
                    logger.info(f"[{ctx.account_id}] [Bước 1/3] Phát hiện đối thoại 4 nút (NV Mua đồ/Shop) -> Dùng tọa độ nút hủy tại ({tapped_x}, {tapped_y})...")
                else:
                    tapped_x, tapped_y = opt_fallback["x"], opt_fallback["y"]
                    logger.info(f"[{ctx.account_id}] [Bước 1/3] Đối thoại 3 nút (NV Đánh quái) -> Dùng tọa độ nút hủy tại ({tapped_x}, {tapped_y})...")
                self._input.tap(tapped_x, tapped_y)
            time.sleep(1.0)

            # BƯỚC 2: Hộp thoại xác nhận trừ 5 điểm danh vọng ("Hủy nhiệm vụ lần này sẽ bị trừ đi 5 điểm danh vọng! Bằng hữu hãy suy nghĩ kỹ nhé!")
            # Bấm nút "Hủy bỏ nhiệm vụ" (tọa độ chuẩn: 303, 571)
            step2_scr = self._screen.capture()
            confirmed_step2 = False
            has_confirm_popup = False
            if step2_scr is not None:
                step2_boxes = self._text.read(step2_scr, (150, 420, 1450, 800))
                for sb in step2_boxes:
                    sb_norm = normalize_game_text(sb.text)
                    if any(kw in sb_norm for kw in ("huybo nhiemvu", "huy bo nhiem vu", "huybonhiemvu", "huy bo", "dong y", "xac nhan")) and not any(kw in sb_norm for kw in ("suy nghi", "lai")):
                        logger.info(f"[{ctx.account_id}] [Bước 2/3] Bấm xác nhận hủy '{sb.text}' tại ({sb.center_x}, {sb.center_y})")
                        self._input.tap(sb.center_x, sb.center_y)
                        confirmed_step2 = True
                        has_confirm_popup = True
                        break
                    if any(kw in sb_norm for kw in ("tru 5 diem", "5 diem danh vong", "danh vong", "suy nghi ky")):
                        has_confirm_popup = True

            # Nếu chưa mở được popup xác nhận và bước 1 phải dùng tọa độ đoán (fallback):
            if not confirmed_step2 and not has_confirm_popup and not cancel_box and tapped_x is not None:
                alt_x, alt_y = (opt_fallback["x"], opt_fallback["y"]) if tapped_x < 800 else (opt_shop_fallback["x"], opt_shop_fallback["y"])
                logger.warning(f"[{ctx.account_id}] Chưa thấy popup xác nhận hủy sau khi bấm ({tapped_x}, {tapped_y}) -> Thử tọa độ đối ứng ({alt_x}, {alt_y})...")
                self._input.tap(alt_x, alt_y)
                time.sleep(0.8)
                alt_scr = self._screen.capture()
                if alt_scr is not None:
                    alt_boxes = self._text.read(alt_scr, (150, 420, 1450, 800))
                    for sb in alt_boxes:
                        sb_norm = normalize_game_text(sb.text)
                        if any(kw in sb_norm for kw in ("huybo nhiemvu", "huy bo nhiem vu", "huybonhiemvu", "huy bo", "dong y", "xac nhan")) and not any(kw in sb_norm for kw in ("suy nghi", "lai")):
                            logger.info(f"[{ctx.account_id}] [Bước 2/3] Bấm xác nhận hủy '{sb.text}' tại ({sb.center_x}, {sb.center_y})")
                            self._input.tap(sb.center_x, sb.center_y)
                            confirmed_step2 = True
                            has_confirm_popup = True
                            break
                        if any(kw in sb_norm for kw in ("tru 5 diem", "5 diem danh vong", "danh vong", "suy nghi ky")):
                            has_confirm_popup = True

            if not confirmed_step2:
                logger.info(f"[{ctx.account_id}] [Bước 2/3] Dùng tọa độ chuẩn nút 'Hủy bỏ nhiệm vụ' tại ({confirm_fallback['x']}, {confirm_fallback['y']})...")
                self._input.tap(confirm_fallback["x"], confirm_fallback["y"])
            time.sleep(1.0)

            # BƯỚC 3: Hộp thoại hoàn tất hủy ("Bạn bị trừ 5 điểm danh vọng vì hủy nhiệm vụ!" hoặc "Xong rồi! bây giờ bằng hữu có thể nhận nhiệm vụ mới!")
            # Người dùng yêu cầu: Nhấn vô chỗ hủy đó xong click cái nữa để tắt thông báo, r nhận nhiệm vụ kế.
            logger.info(f"[{ctx.account_id}] [Bước 3/3] Nhấn vô vị trí hủy tại ({confirm_fallback['x']}, {confirm_fallback['y']})...")
            self._input.tap(confirm_fallback["x"], confirm_fallback["y"])
            time.sleep(0.5)

            logger.info(f"[{ctx.account_id}] [Bước 3/3] Click thêm cái nữa tại ({confirm_fallback['x']}, {confirm_fallback['y']}) để tắt hoàn toàn thông báo hủy nhiệm vụ...")
            self._input.tap(confirm_fallback["x"], confirm_fallback["y"])
            time.sleep(0.8)

            # Kiểm tra nếu còn nút 'Kết thúc đối thoại' hoặc còn thông báo trên màn hình
            step3_scr = self._screen.capture()
            if step3_scr is not None:
                step3_boxes = self._text.read(step3_scr, (150, 300, 1450, 800))
                for sb in step3_boxes:
                    sb_norm = normalize_game_text(sb.text)
                    if any(kw in sb_norm for kw in ("ket thuc", "ket thic", "doi thoai", "doithoaf", "ket thic doithoaf", "ket thuc doi thoai")):
                        logger.info(f"[{ctx.account_id}] [Bước 3/3] Bấm '{sb.text}' tại ({sb.center_x}, {sb.center_y}) để kết thúc đối thoại hủy...")
                        self._input.tap(sb.center_x, sb.center_y)
                        time.sleep(0.6)
                        break
                    if any(kw in sb_norm for kw in ("bi tru 5 diem", "tru 5 diem", "danh vong vi huy", "huy nhiem vu!")):
                        logger.info(f"[{ctx.account_id}] Vẫn còn thông báo trừ điểm danh vọng, tap thêm lần nữa để đóng...")
                        self._input.tap(confirm_fallback["x"], confirm_fallback["y"])
                        time.sleep(0.6)
                        break
            time.sleep(0.5)

            # XÁC MINH HỦY THÀNH CÔNG: Nếu bước 1 không tìm thấy nút bằng OCR và không thấy popup xác nhận:
            if not cancel_box and not has_confirm_popup and not confirmed_step2:
                logger.error(
                    f"[{ctx.account_id}] ⚠️ XÁC MINH THẤT BẠI: Không mở được popup xác nhận hủy nhiệm vụ! "
                    "KHÔNG tăng tiến độ và GIỮ NGUYÊN cờ hủy để thử lại."
                )
                activity._write("⚠️", f"[{ctx.account_id}] Huy NV chua thanh cong, thu lai...")
                self._ensure_npc_dialog_closed()
                return BotState.FIND_COMMERCE_CHIEF

            # Game trừ 1 lượt làm nhiệm vụ trong ngày (nhiệm vụ bị hủy vẫn tính là 1 lượt trong tiến độ 20)
            ctx.total_quests_done += 1
            abandon_reason = ctx.extra.get("abandon_combat_quest_reason", "Hủy nhiệm vụ")
            q_type = ctx.current_quest_type.value if getattr(ctx, "current_quest_type", None) else "COMBAT"

            username = self._get_username(ctx.account_id) if hasattr(self, "_get_username") else ctx.account_id
            activity.quest_done(username, ctx.total_quests_done, f"{q_type} (Đã hủy)")
            if hasattr(self, "_checkpoint") and self._checkpoint:
                self._checkpoint.save(ctx)
            if hasattr(self, "_db") and self._db:
                self._db.log_quest(
                    account_id=ctx.account_id,
                    quest_type=q_type,
                    result="ABANDONED",
                )

            history = ctx.extra.setdefault("quest_history_list", [])
            history.append({
                "num": ctx.total_quests_done,
                "type": q_type,
                "detail": f"Hủy NV: {abandon_reason}",
                "status": "Đã hủy ❌ (tính 1 lượt)",
                "time": datetime.now().strftime("%H:%M:%S")
            })
            self._save_progress(ctx, status_text=f"Đã hủy NV #{ctx.total_quests_done}/20")
            logger.info(f"[{ctx.account_id}] ===> HỦY NHIỆM VỤ, TÍNH VÀO TIẾN ĐỘ: {ctx.total_quests_done}/20 <===")

            # Dọn dẹp sạch sẽ thông tin nhiệm vụ cũ
            ctx.extra.pop("abandon_combat_quest", None)
            ctx.extra.pop("abandon_combat_quest_reason", None)
            ctx.extra.pop("abandon_dialogue_quest", None)
            ctx.extra.pop("quest_target_city", None)
            ctx.extra.pop("quest_target_npc", None)
            ctx.extra.pop("combat_map", None)
            ctx.extra.pop("combat_monster", None)
            ctx.extra.pop("combat_count", None)
            ctx.extra.pop("combat_quest_start_time", None)
            ctx.extra.pop("current_quest_start_time", None)
            ctx.extra.pop("returning_quest", None)
            ctx.extra.pop("quest_text", None)
            ctx.extra.pop("teleport_to_combat_quest", None)
            ctx.extra.pop("dismounted_for_combat", None)
            ctx.current_quest_type = QuestType.UNKNOWN

            if ctx.total_quests_done >= 20:
                logger.info(f"[{ctx.account_id}] Đã đạt tối đa 20/20 nhiệm vụ Thương Hội hôm nay (tính cả hủy)! Hoàn tất.")
                activity._write("🏆", f"[{ctx.account_id}] Da dat 20/20 nhiem vu Thuong Hoi! Dang xuat...")
                self._save_progress(ctx, status_text="Đã hoàn thành 20/20 🎉")
                return BotState.LOGOUT

            logger.info(f"[{ctx.account_id}] ✅ Đã hủy xong nhiệm vụ! Mở lại đối thoại với Chủ thương hội để nhận nhiệm vụ mới...")
            activity._write("📋", f"[{ctx.account_id}] Da huy xong NV. Nhan NV tiep theo...")
            time.sleep(0.3)
            if not self._open_commerce_chief_dialog(ctx):
                return BotState.FIND_COMMERCE_CHIEF
            return self._select_contribute_to_receive_quest(ctx)

        # Kiểm tra nút "Đưa ta đến nơi làm nhiệm vụ" / "Phiền đưa ta đi một chuyến" (dành cho NV đánh quái có sẵn):
        teleport_btn = None
        for b in dialog_boxes:
            b_norm = normalize_game_text(b.text)
            # Bỏ qua câu hỏi của NPC (ví dụ: "Có cần ta đưa đến nơi làm nhiệm vụ không?")
            if "?" in b.text or any(q in b_norm for q in ("co can", "khong?", "khong ?")):
                continue
            b_norm_ns = b_norm.replace(" ", "")
            if any(kw in b_norm for kw in (
                "phien dua ta", "dua ta di", "di mot chuyen", "phien dua ta di",
                "dua ta toi noi", "dua ta den", "toi noi nhiem vu", "den noi nhiem vu",
                "di den noi", "dua ta toi", "den noi lam", "lam nhiem vu", "noi lam nhiem vu",
                "dua den noi"
            )) or any(kw in b_norm_ns for kw in (
                "phiendua", "duatadi", "dimotchuyen", "duataden", "duatatoi",
                "toinoi", "dennoi", "duadennoi", "lamnhiemvu"
            )):
                teleport_btn = (b.center_x, b.center_y)
                break

        if not teleport_btn and not must_verify_completion and (
            ctx.extra.get("teleport_to_combat_quest")
            or ctx.current_quest_type == QuestType.COMBAT
        ):
            teleport_btn = (764, 576)

        if teleport_btn and not must_verify_completion:
            logger.info(f"[{ctx.account_id}] Phát hiện nút dịch chuyển đến nơi làm nhiệm vụ tại {teleport_btn}, bấm chuyển map...")
            activity._write("🚀", f"[{ctx.account_id}] Bam 'Dua ta den noi lam nhiem vu'...")
            ctx.extra.pop("check_previous_quest_on_login", None)

            # Trích xuất thông tin nhiệm vụ đánh quái từ nội dung đối thoại hoặc chữ vàng trước khi dịch chuyển
            from .quest_classifier import extract_combat_target, extract_yellow_text_boxes, parse_yellow_entities
            c_map, c_monster, c_count = extract_combat_target(dialog_raw)
            if c_map:
                ctx.extra["combat_map"] = c_map
            if c_monster:
                ctx.extra["combat_monster"] = c_monster
            if c_count:
                ctx.extra["combat_count"] = c_count

            try:
                y_boxes = extract_yellow_text_boxes(dialog_scr, (150, 250, 1450, 800))
                y_toks = [b.text for b in y_boxes if not any(w in normalize_game_text(b.text) for w in ("thuong hoi", "tuan hoan", "nhiem vu"))]
                if y_toks:
                    ents = parse_yellow_entities(y_toks)
                    if ents.get("map_name"):
                        ctx.extra["combat_map"] = ents["map_name"]
                    if ents.get("monster"):
                        ctx.extra["combat_monster"] = ents["monster"]
                    if ents.get("count"):
                        ctx.extra["combat_count"] = ents["count"]
            except Exception as e:
                logger.debug(f"Yellow extraction in teleport check error: {e}")

            self._input.tap(*teleport_btn)
            time.sleep(3.0)
            ctx.extra.pop("teleport_to_combat_quest", None)
            ctx.current_quest_type = QuestType.COMBAT
            ctx.extra.setdefault("combat_quest_start_time", time.time())
            return BotState.NAVIGATE

        # Kiểm tra nếu Chủ thương hội báo đã hết nhiệm vụ hôm nay
        if any(w in dialog_norm for w in ("hoan thanh het", "ngay mai hay den", "khong con nhiem vu", "da hoan thanh 20", "da nhan toi da", "het nhiem vu")):
            logger.info(f"[{ctx.account_id}] Chủ thương hội báo đã hết nhiệm vụ hôm nay! Đăng xuất...")
            activity._write("🏁", f"[{ctx.account_id}] Chu thuong hoi bao da het nhiem vu hom nay. Dang xuat...")
            self._save_progress(ctx, status_text="Hết nhiệm vụ hôm nay 🏁")
            return BotState.LOGOUT


        # Tự động đồng bộ số lượng nhiệm vụ thực tế trong ngày từ NPC khi bắt đầu lượt chạy
        if not ctx.extra.get("in_game_progress_synced") and not ctx.extra.get("returning_quest"):
            sync_btn = None
            for b in dialog_boxes:
                b_norm = normalize_game_text(b.text)
                if ("xem tinh hinh" in b_norm or "tinh hinh" in b_norm) and ("da hoan thanh" in b_norm or "thuong hoi" in b_norm):
                    sync_btn = (b.center_x, b.center_y)
                    break
            if sync_btn:
                logger.info(f"[{ctx.account_id}] 🔄 Bấm 'Xem tình hình nhiệm vụ thương hội đã hoàn thành' tại {sync_btn} để đồng bộ tiến độ thực tế trong game...")
                self._input.tap(*sync_btn)
                time.sleep(0.8)
                ctx.extra["in_game_progress_synced"] = True
                return BotState.MEET_NPC

        # Kiểm tra nếu có nút "Xem tình hình nhiệm vụ" (dành cho NV đang làm dở, ưu tiên lấy thông tin từ NPC):
        # Lưu ý: Bỏ qua "Xem tình hình nhiệm vụ thương hội đã hoàn thành" vì đó là bảng thống kê NV đã hoàn thành.
        view_status_btn = None
        for b in dialog_boxes:
            b_norm = normalize_game_text(b.text)
            if any(kw in b_norm for kw in ("xem tinh hinh", "tinh hinh nhiem vu")) and not any(ex in b_norm for ex in ("da hoan thanh", "dahoanthanh", "da hoan th", "dahoanth", "hoan thanh", "hoanthanh", "hoanth")):
                view_status_btn = (b.center_x, b.center_y)
                break
        if view_status_btn and not ctx.extra.get("returning_quest"):
            logger.info(f"[{ctx.account_id}] Bấm nút 'Xem tình hình nhiệm vụ' tại {view_status_btn} để lấy thông tin chi tiết từ Chủ thương hội...")
            activity._write("📋", f"[{ctx.account_id}] Xem tinh hinh NV tai Chu thuong hoi...")
            self._input.tap(*view_status_btn)
            time.sleep(0.5)
            return BotState.GET_QUEST

        # Nếu có cờ teleport_to_combat_quest mà không có nút dịch chuyển trong hộp thoại:
        if ctx.extra.get("teleport_to_combat_quest") and not ctx.extra.get("returning_quest") and not ctx.extra.get("check_previous_quest_on_login"):
            ctx.extra.pop("teleport_to_combat_quest", None)
            logger.info(f"[{ctx.account_id}] Hộp thoại NPC không có nút dịch chuyển, đóng đối thoại để sang CLASSIFY_QUEST tự mở bản đồ...")
            self._ensure_npc_dialog_closed()
            return BotState.CLASSIFY_QUEST

        # Kiểm tra nếu là hộp thoại nhắc nhiệm vụ dở dang: "Nhiệm vụ lần trước đã hoàn thành chưa?"
        has_active_unfinished_notice = dialog_requires_completion_check

        # 2. XỬ LÝ NÚT HOÀN THÀNH NHIỆM VỤ (Kiểm tra xem nhiệm vụ trước đó đã hoàn thành chưa)
        complete_box = None
        for b in dialog_boxes:
            # Nút chọn trong hộp thoại NPC luôn nằm ở vùng options (y >= 500)
            if b.center_y < 500:
                continue
            norm = normalize_game_text(b.text)
            norm_ns = norm.replace(" ", "")
            # Tránh nhầm với lời nhắc chưa hoàn thành, nút khác, hoặc bảng thống kê nhiệm vụ hoàn thành
            if any(w in norm_ns for w in (
                "vanchua", "chuahoanthanh", "xemtinhhinh", "huybo", "truysat", "tieudiet", "lantruoc",
                "thuonghoi", "tinhhinh", "gioithieu", "khongcohungthu", "conghien", "vitrieudinh",
            )):
                continue
            if any(w in norm for w in ("hoan thanh nhiem vu", "tra nhiem vu", "giao nhiem vu", "giao tra", "nhiem vu da hoan thanh", "da hoan thanh")) or any(w in norm_ns for w in ("dahoanthanh", "hoanthanhnhiemvu", "tranhiemvu", "giaonhiemvu", "giaotra")):
                complete_box = b
                break

        # Nếu chưa tìm thấy bằng OCR nhưng có dấu hiệu nhiệm vụ cần hoàn thành:
        # TUYỆT ĐỐI không fallback hoàn thành nếu đối thoại có dấu hiệu chiếu chỉ nhận NV ("vi trieu dinh", "cong hien", "hoang thuong", "ban chieu")
        has_no_quest_markers = (
            any(w in dialog_norm for w in ("vi trieu dinh", "cong hien", "hoang thuong", "ban chieu", "giup suc trieu dinh", "nhan si nao giup suc"))
            or any(w in dialog_norm.replace(" ", "") for w in ("vitrieudinh", "conghien", "hoangthuong", "banchieu", "giupsuc", "sucluc", "suclyc", "20lannhiemvu"))
        )
        if not complete_box and not has_no_quest_markers and (has_active_unfinished_notice or ctx.extra.get("returning_quest") or ctx.extra.get("check_previous_quest_on_login")):
            fallback_pos = {"x": 340, "y": 590}
            if hasattr(self, "_loader") and self._loader is not None:
                coords = self._loader.load_ui_coords()
                fallback_pos = coords.get("commerce_quest", {}).get("complete_quest_fallback", fallback_pos)
            from src.text_recognition import TextBox
            complete_box = TextBox(text="Nhiệm vụ đã hoàn thành!", confidence=1.0, center_x=fallback_pos["x"], center_y=fallback_pos["y"])
            logger.info(f"[{ctx.account_id}] Dùng tọa độ nút 'Nhiệm vụ đã hoàn thành!' tại ({complete_box.center_x}, {complete_box.center_y})...")

        if complete_box:
            logger.info(
                f"[{ctx.account_id}] [Kiểm tra NV 1/4] Nhấn '{complete_box.text}' tại "
                f"({complete_box.center_x}, {complete_box.center_y}) để kiểm tra nhiệm vụ đã xong chưa..."
            )
            self._input.tap(complete_box.center_x, complete_box.center_y)
            time.sleep(0.8)

            # Kiểm tra trạng thái hộp thoại và phản hồi sau khi bấm:
            after_scr = self._screen.capture()
            is_dialog_still_open = self._is_npc_dialog_open(after_scr)

            resp_boxes = self._text.read(after_scr, (150, 250, 1450, 750)) if is_dialog_still_open else []
            resp_raw = " ".join(box.text for box in resp_boxes)
            resp_norm = normalize_game_text(resp_raw)

            # Đồng bộ tiến độ nhiệm vụ từ phản hồi của NPC nếu có
            if resp_raw:
                self._sync_quest_progress_from_text(ctx, resp_raw, source="phản hồi kiểm tra nhiệm vụ")

            # Kiểm tra xem có thông báo chưa hoàn thành không:
            is_unfinished = is_dialog_still_open and any(w in resp_norm for w in (
                "chua hoan thanh", "chua xong", "tien do", "nguoi chua", "chua the hoan thanh",
                "chua mang ve", "chua tim duoc", "mau di", "chua hoan tat", "khong the hoan thanh"
            ))
            # Theo hành vi thực tế của game: trả thành công thì hộp thoại tự
            # đóng. Vì vậy chỉ cần hộp thoại còn mở là chưa hoàn thành; không
            # dùng riêng chữ "đã hoàn thành" để suy đoán thành công.
            if is_unfinished or is_dialog_still_open:
                logger.warning(
                    f"[{ctx.account_id}] [Kiểm tra NV 2/4] ⚠️ Đối thoại vẫn mở -> nhiệm vụ CHƯA hoàn thành: "
                    f"'{resp_raw}'"
                )
                activity._write("⚠️", f"[{ctx.account_id}] NV chua xong, tiep tuc thuc hien...")
                ctx.extra.pop("returning_quest", None)
                ctx.extra.pop("check_previous_quest_on_login", None)
                current_num = getattr(ctx, "total_quests_done", 0) + 1
                self._save_progress(ctx, status_text=f"Đang làm NV #{current_num}/20")

                # Ưu tiên 1: Nếu hộp thoại phản hồi có sẵn nút 'Phiền đưa ta đi một chuyến' / 'Hãy đưa ta tới nơi nhiệm vụ'
                resp_teleport_btn = None
                for b in resp_boxes:
                    b_norm = normalize_game_text(b.text)
                    if "?" in b.text or any(q in b_norm for q in ("co can", "khong?", "khong ?")):
                        continue
                    if any(kw in b_norm for kw in (
                        "phien dua ta", "dua ta di", "di mot chuyen", "phien dua ta di", "dua ta den", "dua den noi", "dua ta toi", "toi noi nhiem vu"
                    )):
                        resp_teleport_btn = (b.center_x, b.center_y)
                        break
                if resp_teleport_btn:
                    logger.info(f"[{ctx.account_id}] Phát hiện nút dịch chuyển 'Phiền đưa ta đi một chuyến' tại {resp_teleport_btn}, bấm chuyển map ngay...")
                    self._input.tap(*resp_teleport_btn)
                    time.sleep(3.0)
                    ctx.current_quest_type = QuestType.COMBAT
                    return BotState.NAVIGATE

                # Ưu tiên 2: Nếu hộp thoại phản hồi có nút 'Xem tình hình nhiệm vụ', bấm vào để đọc chi tiết nhiệm vụ từ NPC
                resp_status_btn = None
                for b in resp_boxes:
                    b_norm = normalize_game_text(b.text)
                    if any(kw in b_norm for kw in ("xem tinh hinh", "tinh hinh nhiem vu", "nhiem vu van chua")):
                        resp_status_btn = (b.center_x, b.center_y)
                        break
                if resp_status_btn:
                    logger.info(f"[{ctx.account_id}] Bấm nút 'Xem tình hình nhiệm vụ' tại {resp_status_btn} để lấy thông tin từ Chủ thương hội...")
                    self._input.tap(*resp_status_btn)
                    time.sleep(0.5)
                    return BotState.GET_QUEST

                # Bước 2: Đóng đối thoại NPC Chủ thương hội, mở Bảng nhiệm vụ để đọc chi tiết bên phải
                dialog_closed = self._ensure_npc_dialog_closed()
                if not dialog_closed:
                    close_retries = ctx.extra.get("npc_dialog_close_retries", 0) + 1
                    ctx.extra["npc_dialog_close_retries"] = close_retries
                    if close_retries <= 2:
                        logger.warning(
                            f"[{ctx.account_id}] [Kiểm tra NV 3/4] Hộp thoại chưa đóng (lần {close_retries}/2); thử đóng lại..."
                        )
                        return BotState.MEET_NPC
                    logger.warning(
                        f"[{ctx.account_id}] [Kiểm tra NV 3/4] Đã thử đóng đối thoại {close_retries-1} lần, tiếp tục mở Bảng nhiệm vụ..."
                    )
                ctx.extra.pop("npc_dialog_close_retries", None)
                if has_no_quest_markers:
                    logger.info(
                        f"[{ctx.account_id}] [Kiểm tra NV 3/4] Đối thoại có dấu hiệu chưa nhận nhiệm vụ "
                        f"(Vì triều đình cống hiến sức lực). Tuyệt đối không mở icon Nhiệm vụ, quay lại MEET_NPC..."
                    )
                    return BotState.MEET_NPC

                logger.info(
                    f"[{ctx.account_id}] [Kiểm tra NV 3/4] Mở icon Nhiệm vụ để đọc lại "
                    "nội dung và tiếp tục thực hiện nhiệm vụ hiện tại..."
                )
                active = self._check_active_commerce_quest(ctx)
                if active:
                    ctx.extra["quest_text"] = active[0]
                    ctx.extra.pop("active_quest_check_retries", None)
                    current_num = getattr(ctx, "total_quests_done", 0) + 1
                    self._save_progress(
                        ctx,
                        status_text=f"Đang làm NV #{current_num}/20",
                        quest_detail=active[0][:100],
                    )
                    time.sleep(0.3)
                    self._press_buff_speed()
                    return BotState.CLASSIFY_QUEST

                # Nếu chưa đọc được: thử lại nếu bảng nhiệm vụ đã mở trên màn hình
                retries = ctx.extra.get("active_quest_check_retries", 0) + 1
                ctx.extra["active_quest_check_retries"] = retries
                if retries <= 2:
                    logger.warning(
                        f"[{ctx.account_id}] [Kiểm tra NV 3/4] Chưa lấy được nội dung Bảng nhiệm vụ (lần {retries}/2); "
                        "thử kiểm tra lại..."
                    )
                    time.sleep(0.5)
                    if hasattr(self, "_is_quest_panel_open") and self._is_quest_panel_open():
                        active_retry = self._check_active_commerce_quest(ctx)
                        if active_retry:
                            ctx.extra["quest_text"] = active_retry[0]
                            ctx.extra.pop("active_quest_check_retries", None)
                            time.sleep(0.3)
                            self._press_buff_speed()
                            return BotState.CLASSIFY_QUEST
                    return BotState.MEET_NPC

                ctx.extra.pop("active_quest_check_retries", None)
                logger.warning(
                    f"[{ctx.account_id}] [Kiểm tra NV 3/4] Không đọc được Bảng nhiệm vụ sau {retries} lần. "
                    "Đóng bảng và quay lại MEET_NPC để đối thoại lại với Chủ thương hội."
                )
                self._ensure_quest_panel_closed(ctx=ctx)
                return BotState.MEET_NPC

            # TRƯỜNG HỢP ĐÃ HOÀN THÀNH THÀNH CÔNG: Hộp thoại tự động đóng lại!
            logger.info(
                f"[{ctx.account_id}] [Kiểm tra NV 2/4] ✅ Game đã tự đóng đối thoại -> "
                "xác nhận nhiệm vụ trước đó ĐÃ HOÀN THÀNH!"
            )
            ctx.total_quests_done += 1
            ctx.extra.pop("returning_quest", None)
            ctx.extra.pop("check_previous_quest_on_login", None)
            ctx.extra.pop("combat_quest_start_time", None)
            ctx.extra.pop("current_quest_start_time", None)
            ctx.extra.pop("dismounted_for_combat", None)
            username = self._get_username(ctx.account_id) if hasattr(self, "_get_username") else ctx.account_id
            q_type = ctx.current_quest_type.value if ctx.current_quest_type else "NPC"
            activity.quest_done(username, ctx.total_quests_done, q_type)
            if hasattr(self, "_checkpoint") and self._checkpoint:
                self._checkpoint.save(ctx)
            if hasattr(self, "_db") and self._db:
                self._db.log_quest(
                    account_id=ctx.account_id,
                    quest_type=q_type,
                    result="COMPLETED",
                )

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

            # Dọn dữ liệu nhiệm vụ vừa trả trước khi nhận nhiệm vụ kế tiếp để
            # map/quái/vật phẩm cũ không ghi đè kết quả phân loại mới.
            for key in (
                "quest_text", "yellow_tokens", "yellow_item", "requested_shop_item",
                "combat_map", "combat_monster", "combat_count", "combat_sect",
                "quest_target_city", "quest_target_npc", "quest_detail_text",
                "teleport_to_combat_quest", "auto_walking", "chief_nav_started",
                "merchant_search_retries",
            ):
                ctx.extra.pop(key, None)
            ctx.current_quest_type = QuestType.UNKNOWN

            if ctx.total_quests_done >= 20:
                logger.info(f"[{ctx.account_id}] Đã đạt tối đa 20/20 nhiệm vụ Thương Hội hôm nay! Hoàn tất.")
                activity._write("🏆", f"[{ctx.account_id}] Da hoan thanh 20/20 nhiem vu Thuong Hoi! Dang xuat...")
                self._save_progress(ctx, status_text="Đã hoàn thành 20/20 🎉")
                return BotState.LOGOUT

            # "cần mở đối thoại tiếp để nhận nhiệm vụ tiếp"
            logger.info(
                f"[{ctx.account_id}] [Kiểm tra NV 4/4] Mở lại đối thoại với Chủ thương hội "
                f"để nhận nhiệm vụ thứ {ctx.total_quests_done + 1}/20..."
            )
            activity._write("💬", f"[{ctx.account_id}] Mo lai doi thoai nhan NV tiep theo...")
            time.sleep(0.1)
            if not self._open_commerce_chief_dialog(ctx):
                logger.warning(f"[{ctx.account_id}] Chưa mở lại được đối thoại, quay lại FIND_COMMERCE_CHIEF...")
                return BotState.FIND_COMMERCE_CHIEF
            time.sleep(0.2)
            return self._select_contribute_to_receive_quest(ctx)

        # 3. NẾU KHÔNG CÓ NÚT HOÀN THÀNH: Nhận nhiệm vụ mới ("Vì triều đình cống hiến sức lực")
        ctx.extra.pop("check_previous_quest_on_login", None)
        return self._select_contribute_to_receive_quest(ctx)

    def _select_contribute_to_receive_quest(self, ctx: StateContext) -> BotState:
        """Chọn 'Vì triều đình cống hiến sức lực' để nhận nhiệm vụ Thương Hội mới."""
        logger.info(f"[{ctx.account_id}] Chọn 'Vì triều đình cống hiến sức lực'...")
        found_contrib = False
        boxes = self._text.read(self._screen.capture(), (150, 450, 1450, 800))
        for b in boxes:
            norm = normalize_game_text(b.text)
            norm_ns = norm.replace(" ", "")
            if (
                "vi trieu dinh" in norm
                or "cong hien" in norm
                or "suc luc" in norm
                or "suc lyc" in norm
                or "vitrieudinh" in norm_ns
                or "conghien" in norm_ns
            ) and not any(w in norm for w in ("khong co hung thu", "tinh hinh", "gioi thieu")):
                logger.info(f"[{ctx.account_id}] Bấm nút '{b.text}' tại ({b.center_x}, {b.center_y})")
                self._input.tap(b.center_x, b.center_y)
                found_contrib = True
                break

        if not found_contrib:
            # Không dùng icon Nhiệm vụ để kiểm tra tồn tại. Nếu OCR bỏ sót nút
            # trong đối thoại, thử đúng vị trí nhận nhiệm vụ rồi để GET_QUEST
            # đọc phản hồi của Chủ thương hội.
            fallback = {"x": 330, "y": 580}
            if hasattr(self, "_loader") and self._loader is not None:
                coords = self._loader.load_ui_coords()
                fallback = coords.get("commerce_quest", {}).get("contribute_fallback", fallback)
            fx, fy = fallback.get("x", 330), fallback.get("y", 580)
            logger.info(f"[{ctx.account_id}] Bấm vị trí fallback 'Vì triều đình cống hiến sức lực' ({fx}, {fy})...")
            self._input.tap(fx, fy)

        time.sleep(0.8)
        return BotState.GET_QUEST

    def _handle_get_quest(self, ctx: StateContext) -> BotState:
        """
        Đọc nội dung nhiệm vụ được giao từ hộp thoại Chủ thương hội (vùng hộp thoại y: 200..750).
        Nếu hộp thoại không có nội dung quest hợp lệ:
        - Không mở icon Nhiệm vụ để suy đoán đã nhận hay chưa.
        - Thử đọc lại, sau đó quay lại đối thoại Chủ thương hội để xác nhận.
        """
        time.sleep(0.3)
        cap_frame = self._screen.capture()

        # Nếu hộp thoại NPC đã đóng: kiểm tra xem có nhiệm vụ Thương Hội thực sự trong bảng nhiệm vụ không.
        # Tuyệt đối không đọc text nền ngoài màn hình (như nhiệm vụ Tân thủ 'Phạm Trọng Yêm')!
        if cap_frame is not None and not self._is_npc_dialog_open(cap_frame):
            active = self._check_active_commerce_quest(ctx)
            if active:
                quest_text, _ = active
                logger.info(f"[{ctx.account_id}] Đã nhận nhiệm vụ Thương Hội thành công (đối thoại đã đóng): '{quest_text}'")
                ctx.extra["quest_text"] = quest_text
                ctx.last_action = "read_commerce_quest"
                self._checkpoint.save(ctx)
                time.sleep(0.3)
                self._press_buff_speed()
                return BotState.CLASSIFY_QUEST
            else:
                ctx.extra["no_quest_attempts"] = 0
                logger.warning(
                    f"[{ctx.account_id}] [GET_QUEST] Hộp thoại NPC đã đóng nhưng chưa nhận được nhiệm vụ Thương Hội! "
                    "Quay lại MEET_NPC đối thoại với Chủ thương hội và bấm 'Vì triều đình cống hiến sức lực'..."
                )
                return BotState.MEET_NPC

        # 1. Đọc vùng văn bản trong hộp thoại đối thoại của NPC (bao gồm cả dòng mô tả NV và các nút)
        # Vùng hộp thoại thực tế bắt đầu từ x=150 y=200 tới y=750 để bao trọn nút 'Vì triều đình cống hiến sức lực'
        region = (150, 200, 1450, 750)
        boxes = self._text.read(cap_frame, region)
        quest_raw = " ".join(box.text for box in boxes) if boxes else ""
        norm = normalize_game_text(quest_raw)

        # Đồng bộ số nhiệm vụ đã hoàn thành hôm nay từ hộp thoại NPC:
        # Ví dụ: "Hôm nay đã hoàn thành 8 lần nhiệm vụ. Còn có thể nhận 12 lần!"
        self._sync_quest_progress_from_text(ctx, quest_raw, source="hộp thoại nhận nhiệm vụ")

        explicit_combat_wording = any(w in norm for w in (
            "tieu diet", "danh bai", "ha guc", "giet", "danh quai",
            "di diet", "trung tri", "truing tri", "tring tri", "tao phan", "lam phan", "gay roi", "son tac",
            "di bat", "bat", "coc doc",
        ))
        # Không để dữ liệu của nhiệm vụ trước lấp vào nhiệm vụ mới.
        for key in ("combat_map", "combat_monster", "combat_count", "combat_sect", "combat_objective_complete", "returning_quest", "abandon_combat_quest", "chief_nav_started", "yellow_item", "requested_shop_item", "yellow_tokens"):
            ctx.extra.pop(key, None)
        logger.info(f"[{ctx.account_id}] Nội dung hộp thoại đọc được: '{quest_raw}'")

        # Trích xuất các thực thể chữ vàng trong hộp thoại nhận nhiệm vụ
        yellow_tokens = []
        try:
            from .quest_classifier import extract_yellow_text_boxes, parse_yellow_entities
            y_boxes = extract_yellow_text_boxes(cap_frame, region)
            yellow_tokens = [
                b.text for b in y_boxes
                if not any(w in normalize_game_text(b.text) for w in ("thuong hoi", "tuan hoan", "nhiem vu"))
            ]
            if yellow_tokens:
                logger.info(f"[{ctx.account_id}] Chữ vàng trong hộp thoại nhận NV: {yellow_tokens}")
                ctx.extra["yellow_tokens"] = yellow_tokens
                entities = parse_yellow_entities(yellow_tokens)
                has_yellow_combat = bool(
                    (entities.get("monster") and int(entities.get("count") or 0) > 0)
                    or (entities.get("map_name") and entities.get("monster"))
                )
                if entities.get("item") and not explicit_combat_wording and not has_yellow_combat:
                    ctx.extra["yellow_item"] = entities["item"]
                    ctx.extra["requested_shop_item"] = entities["item"]
                elif explicit_combat_wording or has_yellow_combat:
                    ctx.extra.pop("yellow_item", None)
                    ctx.extra.pop("requested_shop_item", None)
                if entities.get("map_name"):
                    ctx.extra["combat_map"] = entities["map_name"]
                if entities.get("monster"):
                    ctx.extra["combat_monster"] = entities["monster"]
                if entities.get("count"):
                    ctx.extra["combat_count"] = entities["count"]
                if entities.get("city") and not has_yellow_combat:
                    ctx.extra["quest_target_city"] = entities["city"]
                if entities.get("npc") and not has_yellow_combat:
                    ctx.extra["quest_target_npc"] = entities["npc"]
        except Exception as ex:
            logger.debug(f"Yellow extraction in get_quest failed: {ex}")

        # 2. Kiểm tra nếu màn hình vẫn còn là chiếu chỉ nhận nhiệm vụ ("Hoàng thượng vừa ban chiếu: ... Vì triều đình cống hiến sức lực")
        # chứng tỏ nút "Vì triều đình cống hiến sức lực" chưa được bấm trúng hoặc đối thoại chưa cập nhật.
        # Bấm lại nút "Vì triều đình cống hiến sức lực" để mở nhiệm vụ, TUYỆT ĐỐI không mở icon Nhiệm vụ!
        has_initial_edict = (
            any(w in norm for w in ("vi trieu dinh", "cong hien suc luc", "cong hien", "hoang thuong", "ban chieu", "giup suc trieu dinh", "nhan si nao giup suc"))
            or any(w in norm.replace(" ", "") for w in ("vitrieudinh", "conghien", "hoangthuong", "banchieu", "giupsuc", "20lannhiemvu", "sucluc", "suclyc"))
        )
        if has_initial_edict:
            logger.info(
                f"[{ctx.account_id}] [GET_QUEST] Vẫn còn hiển thị chiếu chỉ ban đầu (có 'Vì triều đình cống hiến sức lực'). "
                "Bấm lại nút để mở nội dung nhiệm vụ mới, không mở icon Nhiệm vụ..."
            )
            click_btn = None
            for b in boxes:
                nb = normalize_game_text(b.text)
                nb_ns = nb.replace(" ", "")
                if (
                    any(w in nb for w in ("vi trieu dinh", "cong hien", "suc luc", "suc lyc"))
                    or any(w in nb_ns for w in ("vitrieudinh", "conghien", "sucluc", "suclyc"))
                ) and not any(w in nb for w in ("khong co hung thu", "tinh hinh", "gioi thieu")):
                    click_btn = (b.center_x, b.center_y)
                    break
            target_pt = click_btn or (330, 580)
            logger.info(f"[{ctx.account_id}] [GET_QUEST] Bấm 'Vì triều đình cống hiến sức lực' tại {target_pt}...")
            self._input.tap(*target_pt)
            time.sleep(1.0)
            return BotState.GET_QUEST

        # 2.5. Kiểm tra xem có phải thông báo đã hết nhiệm vụ hôm nay không
        if any(w in norm for w in ("hoan thanh het", "ngay mai hay den", "khong con nhiem vu", "ngay mai")) or is_explicit_quest_exhausted(quest_raw):
            logger.info(f"[{ctx.account_id}] Chủ thương hội báo đã hết nhiệm vụ hôm nay! Đăng xuất...")
            activity._write("🏁", f"[{ctx.account_id}] Chu thuong hoi bao da het nhiem vu hom nay. Dang xuat...")
            ctx.total_quests_done = 20
            ctx.extra["quest_exhausted_today"] = True
            if hasattr(self, "_checkpoint") and self._checkpoint:
                self._checkpoint.save(ctx)
            self._save_progress(ctx, status_text="Đã hoàn thành 20/20 🎉")
            return BotState.LOGOUT

        # 3. Kiểm tra nếu có sẵn nhiệm vụ đang làm dở được Chủ thương hội nhắc
        # LƯU Ý: Không dùng "tinh hinh" đơn lẻ vì sẽ trùng với "Xem tình hình nhiệm vụ thương hội đã hoàn thành".
        is_unfinished_dialog = (
            any(w in norm for w in ("chua hoan thanh", "mau di", "tinh hinh the nao", "nhiem vu van chua", "lan truoc da hoan thanh chua", "nhan da den", "hang da den"))
            and not any(w in norm for w in ("da hoan thanh", "dahoanthanh", "vi trieu dinh", "cong hien", "hoang thuong", "ban chieu"))
        )
        if is_unfinished_dialog:
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
        valid_quest_type = self._classifier.classify(text=quest_raw, yellow_tokens=yellow_tokens)
        if explicit_combat_wording:
            valid_quest_type = QuestType.COMBAT
            ctx.extra.pop("yellow_item", None)
            ctx.extra.pop("requested_shop_item", None)
            from .quest_classifier import extract_combat_target
            combat_map, combat_monster, combat_count = extract_combat_target(
                quest_raw, yellow_tokens=yellow_tokens
            )
            if combat_map:
                ctx.extra["combat_map"] = combat_map
            if combat_monster:
                ctx.extra["combat_monster"] = combat_monster
            if combat_count:
                ctx.extra["combat_count"] = combat_count
        elif ctx.extra.get("yellow_item") or ctx.extra.get("requested_shop_item"):
            valid_quest_type = QuestType.SPECIAL_NPC_SHOP

        if valid_quest_type != QuestType.UNKNOWN:
            logger.info(f"[{ctx.account_id}] Nhận nhiệm vụ thành công từ hộp thoại ({valid_quest_type.value}): '{quest_raw}'")
            ctx.current_quest_type = valid_quest_type
            ctx.extra["quest_text"] = quest_raw
            ctx.last_action = "read_commerce_quest"
            self._checkpoint.save(ctx)

            # Ưu tiên kiểm tra nút "Phiền đưa ta đi một chuyến" / "Đưa ta đến nơi làm nhiệm vụ"
            teleport_btn = None
            for b in boxes:
                b_norm = normalize_game_text(b.text)
                # Bỏ qua câu hỏi của NPC (ví dụ: "Có cần ta đưa đến nơi làm nhiệm vụ không?")
                if "?" in b.text or any(q in b_norm for q in ("co can", "khong?", "khong ?")):
                    continue
                b_norm_ns = b_norm.replace(" ", "")
                if any(kw in b_norm for kw in (
                    "phien dua ta", "dua ta di", "di mot chuyen", "phien dua ta di",
                    "dua ta toi noi", "dua ta den", "toi noi nhiem vu", "den noi nhiem vu",
                    "di den noi", "dua ta toi", "den noi lam", "lam nhiem vu", "noi lam nhiem vu",
                    "dua den noi"
                )) or any(kw in b_norm_ns for kw in (
                    "phiendua", "duatadi", "dimotchuyen", "duataden", "duatatoi",
                    "toinoi", "dennoi", "duadennoi", "lamnhiemvu"
                )):
                    teleport_btn = (b.center_x, b.center_y)
                    break

            if not teleport_btn and any(w in norm for w in ("co can ta dua", "dua den noi", "phien dua ta")):
                teleport_btn = (330, 575)

            # TỐI ƯU TỐC ĐỘ: Nếu nhận nhiệm vụ đánh quái và bật tùy chọn hủy ngay
            if valid_quest_type == QuestType.COMBAT and self._should_abandon_combat_immediately():
                logger.info(
                    f"[{ctx.account_id}] ⚡ [TỐI ƯU TỐC ĐỘ] Nhận nhiệm vụ đánh quái -> "
                    "Hủy ngay tại Chủ thương hội để đổi sang nhiệm vụ đối thoại nhanh..."
                )
                activity._write("⚡", f"[{ctx.account_id}] Bo qua NV danh quai -> Huy de nhan NV doi thoai...")
                dismiss_btn = (460, 280) if any(w in norm for w in ("co can ta dua", "dua den noi", "phien dua ta")) else (280, 575)
                self._input.tap(*dismiss_btn)
                time.sleep(0.3)
                ctx.extra["abandon_combat_quest"] = True
                ctx.extra["abandon_combat_quest_reason"] = "Tối ưu tốc độ: Hủy nhiệm vụ đánh quái để nhận NV đối thoại"
                ctx.current_quest_type = QuestType.COMBAT
                return BotState.MEET_NPC

            # HỦY NHIỆM VỤ ĐỐI THOẠI ẤU ĐỒNG NGAY TẠI CHỦ THƯƠNG HỘI:
            from .quest_classifier import is_auto_abandon_dialogue_quest
            target_npc_chk = ctx.extra.get("quest_target_npc", "")
            if valid_quest_type == QuestType.NPC and is_auto_abandon_dialogue_quest(npc=target_npc_chk, text=quest_raw):
                logger.warning(
                    f"[{ctx.account_id}] ⚠️ Nhận nhiệm vụ gặp NPC '{target_npc_chk or 'Ấu đồng'}' -> "
                    "Hủy ngay tại chỗ tại Chủ thương hội..."
                )
                activity._write("❌", f"[{ctx.account_id}] Nhan NV {target_npc_chk or 'Au dong'} -> Huy ngay tai Chu thuong hoi.")
                dismiss_btn = (460, 280) if any(w in norm for w in ("co can ta dua", "dua den noi", "phien dua ta")) else (280, 575)
                self._input.tap(*dismiss_btn)
                time.sleep(0.3)
                ctx.extra["abandon_dialogue_quest"] = True
                ctx.extra["abandon_combat_quest"] = True
                ctx.extra["abandon_combat_quest_reason"] = "Nhiệm vụ gặp NPC Ấu đồng (tự động hủy theo yêu cầu)"
                ctx.current_quest_type = QuestType.NPC
                return BotState.MEET_NPC

            if teleport_btn and valid_quest_type == QuestType.COMBAT:
                logger.info(f"[{ctx.account_id}] Phát hiện nút dịch chuyển 'Phiền đưa ta đi một chuyến' tại {teleport_btn}, bấm chuyển map ngay...")
                activity._write("🚀", f"[{ctx.account_id}] Bam 'Phien dua ta di mot chuyen'...")

                # Trích xuất thông tin combat nếu chưa lưu
                from .quest_classifier import extract_combat_target
                c_map, c_monster, c_count = extract_combat_target(quest_raw, yellow_tokens=yellow_tokens)
                if c_map:
                    ctx.extra["combat_map"] = c_map
                if c_monster:
                    ctx.extra["combat_monster"] = c_monster
                if c_count:
                    ctx.extra["combat_count"] = c_count

                self._input.tap(*teleport_btn)
                time.sleep(3.0)
                ctx.extra.pop("teleport_to_combat_quest", None)
                ctx.current_quest_type = QuestType.COMBAT
                ctx.extra.setdefault("combat_quest_start_time", time.time())
                return BotState.NAVIGATE

            # Nếu không có nút teleport hoặc không phải combat: Bấm nút "Không cần đâu" hoặc "Kết thúc đối thoại"
            dismiss_btn = None
            for b in boxes:
                b_norm = normalize_game_text(b.text)
                if any(kw in b_norm for kw in ("khong can", "khong can dau", "ket thuc")):
                    dismiss_btn = (b.center_x, b.center_y)
                    break

            if not dismiss_btn:
                # Nếu không đọc được OCR, fallback: nếu có hỏi đưa đi thì nút "Không cần đâu" nằm bên phải (~460, 280), ngược lại (280, 575)
                dismiss_btn = (460, 280) if any(w in norm for w in ("co can ta dua", "dua den noi", "phien dua ta")) else (280, 575)

            logger.info(f"[{ctx.account_id}] Bấm nút đóng/từ chối đối thoại tại {dismiss_btn}...")
            self._input.tap(*dismiss_btn)
            time.sleep(0.3)
            return BotState.CLASSIFY_QUEST

        # 5. Không dùng icon để kết luận có nhiệm vụ. Nếu OCR đối thoại chưa
        # đọc được thì thử lại; quá số lần sẽ quay về đối thoại Chủ thương hội.
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
        yellow_tokens = ctx.extra.get("yellow_tokens") or []
        if not quest_text and not yellow_tokens:
            logger.warning(
                f"[{ctx.account_id}] [CLASSIFY_QUEST] Cả quest_text và yellow_tokens đều rỗng! "
                "Thử đọc lại từ Bảng nhiệm vụ..."
            )
            active = self._check_active_commerce_quest(ctx)
            if active:
                quest_text = active[0]
                ctx.extra["quest_text"] = quest_text
                yellow_tokens = ctx.extra.get("yellow_tokens") or []
            else:
                logger.warning(f"[{ctx.account_id}] Không đọc được nội dung nhiệm vụ; quay lại MEET_NPC.")
                return BotState.MEET_NPC

        if quest_text:
            self._sync_quest_progress_from_text(ctx, quest_text, source="nội dung nhiệm vụ phân loại")
        quest_type = self._classifier.classify(text=quest_text, yellow_tokens=yellow_tokens)
        norm_q = normalize_game_text(quest_text)
        has_active_npc_turnin = (
            "cung voi" in norm_q and any(
                k in norm_q for k in ("doi thoai", "hoan thanh", "hoan thanhnhiem vu")
            )
        ) or (
            "doi thoai" in norm_q and "hoan thanh nhiem vu" in norm_q
        )
        from .quest_classifier import parse_yellow_entities
        yellow_entities = parse_yellow_entities(yellow_tokens) if yellow_tokens else {}
        yellow_combat_evidence = bool(
            (yellow_entities.get("monster") and int(yellow_entities.get("count") or 0) > 0)
            or (yellow_entities.get("map_name") and yellow_entities.get("monster"))
        )
        explicit_dialogue_wording = any(w in norm_q for w in (
            "noi voi", "noi chuyen voi", "doi thoai voi", "thong bao cho",
            "bao tin cho", "bao cho", "gap xa phu", "tim xa phu",
        )) or has_active_npc_turnin
        # Commerce quests can contain noisy OCR from the surrounding quest
        # panel and be mislabeled NPC by a generic dialogue/icon rule. A
        # yellow-whitelisted item plus an explicit merchant/pickup instruction
        # is stronger evidence for the Western Merchant flow.
        has_shop_item = bool(
            yellow_entities.get("item")
            or ctx.extra.get("yellow_item")
            or ctx.extra.get("requested_shop_item")
        )
        has_shop_instruction = (
            "thuong nhan" in norm_q
            and any(word in norm_q for word in ("dat", "lay", "mua", "tim"))
        )
        is_shop_quest = has_shop_item and has_shop_instruction

        # Xóa metadata combat/hủy của nhiệm vụ trước khi nội dung
        # hiện tại đã xác định rõ là nhiệm vụ đối thoại.
        # Chữ vàng được đọc trong phần chi tiết nhiệm vụ hiện tại đáng tin hơn
        # fuzzy text OCR. Ví dụ thực tế: ['Nt Kiem khach', '10', '!!'].
        if yellow_combat_evidence:
            quest_type = QuestType.COMBAT
            ctx.extra["combat_monster"] = yellow_entities["monster"]
            if yellow_entities.get("count"):
                ctx.extra["combat_count"] = int(yellow_entities["count"])
            if yellow_entities.get("map_name"):
                ctx.extra["combat_map"] = yellow_entities["map_name"]
            if yellow_entities.get("sect"):
                ctx.extra["combat_sect"] = yellow_entities["sect"]
            ctx.extra.pop("yellow_item", None)
            ctx.extra.pop("requested_shop_item", None)
            ctx.extra.pop("quest_target_city", None)
            ctx.extra.pop("quest_target_npc", None)
        elif is_shop_quest:
            logger.info(
                f"[{ctx.account_id}] Ưu tiên nhiệm vụ mua đồ: vật phẩm vàng "
                f"'{yellow_entities.get('item') or ctx.extra.get('yellow_item') or ctx.extra.get('requested_shop_item')}' "
                "đi cùng chỉ dẫn Thương nhân; bỏ qua nhãn NPC chung."
            )
            quest_type = QuestType.SPECIAL_NPC_SHOP
        elif has_active_npc_turnin:
            logger.info(
                f"[{ctx.account_id}] Mục tiêu là đối thoại hoàn thành nhiệm vụ ('cùng với ... đối thoại / hoàn thành'); "
                "bỏ qua quái trong phần mô tả."
            )
            quest_type = QuestType.NPC

        if (
            not yellow_combat_evidence
            and not is_shop_quest
            and (
                has_active_npc_turnin
                or quest_type == QuestType.NPC
                or explicit_dialogue_wording
            )
        ):
            quest_type = QuestType.NPC
            for stale_key in (
                "combat_map", "combat_monster", "combat_count", "combat_sect",
                "combat_quest_start_time", "teleport_to_combat_quest",
                "abandon_combat_quest", "abandon_combat_quest_reason",
            ):
                ctx.extra.pop(stale_key, None)

        # Nếu có yellow_item hoặc requested_shop_item thì chắc chắn là SPECIAL_NPC_SHOP
        if quest_type == QuestType.NPC:
            pass
        elif ctx.extra.get("yellow_item") or ctx.extra.get("requested_shop_item"):
            quest_type = QuestType.SPECIAL_NPC_SHOP
        elif (
            quest_type == QuestType.UNKNOWN
            and ctx.extra.get("combat_monster")
            and ctx.extra.get("combat_map")
            and ctx.extra.get("combat_count")
        ):
            quest_type = QuestType.COMBAT

        # Nếu UNKNOWN -> Kiểm tra lại từ bảng nhiệm vụ ()
        if quest_type == QuestType.UNKNOWN:
            active = self._check_active_commerce_quest(ctx)
            if active and active[0] != quest_text:
                quest_text = active[0]
                ctx.extra["quest_text"] = quest_text
                yellow_tokens = ctx.extra.get("yellow_tokens") or []
                quest_type = self._classifier.classify(text=quest_text, yellow_tokens=yellow_tokens)
                if quest_type == QuestType.NPC:
                    pass
                elif ctx.extra.get("yellow_item") or ctx.extra.get("requested_shop_item"):
                    quest_type = QuestType.SPECIAL_NPC_SHOP
                elif (
                    quest_type == QuestType.UNKNOWN
                    and ctx.extra.get("combat_monster")
                    and ctx.extra.get("combat_map")
                    and ctx.extra.get("combat_count")
                ):
                    quest_type = QuestType.COMBAT

        # Kiểm tra xác định chính xác theo từ khóa nếu còn phân vân hoặc bị nhận nhầm sang NPC
        explicit_combat_wording = any(w in norm_q for w in (
            "tieu diet", "danh bai", "ha guc", "giet", "danh quai",
            "di diet", "trung tri", "tao phan", "gay roi",
        ))
        is_merchant_direction = "de tim ong ta" in norm_q or sum(1 for c in ("bien kinh", "thanh do", "tuyen chau", "tuong duong", "duong chau", "dai ly") if c in norm_q) >= 2
        whitelisted_shop_item = (
            match_western_merchant_item(ctx.extra.get("yellow_item", ""))
            or match_western_merchant_item(ctx.extra.get("requested_shop_item", ""))
            or match_western_merchant_item(quest_text)
        )
        is_western_merchant = bool(whitelisted_shop_item) and (
            any(w in norm_q for w in ("thuong nhan tay vuc", "thuong nhan phuong tay", "thuong nhan tay ho", "linh duoc", "tay vuc", "dat mua"))
            or (any(w in norm_q for w in ("mua", "dat", "lay")) and any(w in norm_q for w in ("thuong nhan", "tay vuc", "binh", "chiec")))
            or is_merchant_direction
        )
        if explicit_combat_wording:
            quest_type = QuestType.COMBAT
            ctx.extra.pop("yellow_item", None)
            ctx.extra.pop("requested_shop_item", None)
        elif is_western_merchant:
            quest_type = QuestType.SPECIAL_NPC_SHOP
        elif any(w in norm_q for w in ("tieu diet", "danh bai", "giet", "quai", "trung tri", "tao phan")):
            quest_type = QuestType.COMBAT
        elif quest_type == QuestType.UNKNOWN:
            if ctx.extra.get("auto_walking") or any(w in norm_q for w in ("den", "gap", "noi", "hoi", "ve binh", "dao nhan", "sinh", "nan dan", "thuong hoi")):
                quest_type = QuestType.NPC
            else:
                logger.warning(f"[{ctx.account_id}] Không xác định được nhiệm vụ nào từ text: '{quest_text}'! Quay lại gặp Chủ thương hội...")
                activity._write("⚠️", f"[{ctx.account_id}] Khong co NV, quay lai Chu thuong hoi...")
                return BotState.MEET_NPC

        ctx.current_quest_type = quest_type
        ctx.last_action = f"classify:{quest_type.value}"
        ctx.extra.setdefault("current_quest_start_time", time.time())
        self._checkpoint.save(ctx)

        from .quest_classifier import extract_dialogue_target, extract_shop_item, extract_combat_target

        logger.info(f"[{ctx.account_id}] Phân loại nhiệm vụ: {quest_type.value.upper()}")

        # ==================== LOẠI 1: ĐỐI THOẠI NPC ====================
        if quest_type == quest_type.__class__.NPC:
            for k in (
                "combat_map", "combat_monster", "combat_count", "combat_sect",
                "combat_quest_start_time", "teleport_to_combat_quest",
                "yellow_item", "requested_shop_item", "combat_objective_complete",
                "combat_avoided_directions", "combat_last_move_direction", "combat_reverse_vector",
            ):
                ctx.extra.pop(k, None)

            # Ưu tiên 1: Lấy trực tiếp từ chữ màu vàng đã trích xuất (từ Bảng nhiệm vụ hoặc hộp thoại)
            city = ctx.extra.get("quest_target_city", "")
            npc = ctx.extra.get("quest_target_npc", "")

            # Ưu tiên 2: Nếu thiếu Thành phố hoặc Tên NPC, mở Bảng nhiệm vụ quét chữ vàng chính xác
            if not city or not npc:
                active = self._check_active_commerce_quest(ctx)
                if active:
                    city = ctx.extra.get("quest_target_city", "") or city
                    npc = ctx.extra.get("quest_target_npc", "") or npc
                    quest_text = active[0] or quest_text

            # Ưu tiên 3: Fallback trích xuất từ nội dung chuỗi mô tả
            if not city or not npc:
                y_tokens = ctx.extra.get("yellow_tokens")
                c, n = extract_dialogue_target(quest_text, with_accents=True, yellow_tokens=y_tokens)
                city = city or c
                npc = npc or n

            # Nếu không tìm thấy thành phố cụ thể (ví dụ: 'Cùng với Chủ tửu lầu đối thoại'),
            # lấy thành phố hiện tại của nhân vật từ HUD
            if not city:
                hud_loc = self._get_current_location_from_hud()
                if hud_loc and hud_loc[0]:
                    city = hud_loc[0]
                    norm_c = normalize_game_text(city)
                    from .quest_classifier import SPECIAL_CITY_ALIASES
                    for alias, canon in SPECIAL_CITY_ALIASES.items():
                        if alias == norm_c or alias in norm_c:
                            city = canon
                            break

            # Nhiệm vụ này là một chuỗi bốn mục tiêu, không phải đối thoại đơn.
            # Luôn khóa đúng thứ tự để checkpoint có thể tiếp tục sau khi bot
            # khởi động lại và không trả nhiệm vụ sau Nạn dân đầu tiên.
            if self._is_four_refugee_quest(quest_text):
                route = list(self.FOUR_REFUGEE_NPCS)
                saved_route = ctx.extra.get("multi_npc_route")
                if saved_route != route:
                    ctx.extra["multi_npc_route"] = route
                    ctx.extra["multi_npc_index"] = 0
                    ctx.extra.pop("multi_npc_completed", None)
                index = max(0, min(int(ctx.extra.get("multi_npc_index", 0)), len(route) - 1))
                city = "Thành Đô"
                npc = route[index]

            ctx.extra["quest_target_city"] = city
            ctx.extra["quest_target_npc"] = npc
            if ctx.extra.get("multi_npc_route"):
                index = int(ctx.extra.get("multi_npc_index", 0))
                detail = f"Đối thoại Nạn dân {index + 1}/4: {city} -> {npc}"
            else:
                detail = f"Đối thoại: {city} -> {npc}" if (city or npc) else "Đối thoại NPC"
            ctx.extra["quest_detail_text"] = detail

            # Người dùng yêu cầu: Gặp nhiệm vụ NPC Ấu đồng thì tự động hủy bỏ
            from .quest_classifier import is_auto_abandon_dialogue_quest
            if is_auto_abandon_dialogue_quest(npc=npc, text=quest_text):
                logger.warning(
                    f"[{ctx.account_id}] ⚠️ Phát hiện nhiệm vụ gặp NPC '{npc}' tại '{city}' "
                    "(thuộc danh sách tự động hủy theo yêu cầu). "
                    "Tiến hành quay về Chủ thương hội để hủy nhiệm vụ..."
                )
                activity._write("❌", f"[{ctx.account_id}] NV gap {npc} -> Quay ve Chu thuong hoi huy NV.")
                ctx.extra["abandon_dialogue_quest"] = True
                ctx.extra["abandon_combat_quest"] = True
                ctx.extra["abandon_combat_quest_reason"] = f"Nhiệm vụ gặp NPC {npc} (tự động hủy theo yêu cầu)"
                ctx.extra["returning_quest"] = True
                ctx.extra.pop("chief_nav_started", None)
                navigation_started = self._return_to_nearest_city_commerce_chief(ctx)
                if navigation_started:
                    ctx.extra["chief_nav_started"] = True
                self._checkpoint.save(ctx)
                return BotState.RETURN_QUEST
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
            item = ctx.extra.get("yellow_item", "") or ctx.extra.get("requested_shop_item", "")
            if item:
                item = match_western_merchant_item(item) or clean_shop_item_name(item)
            if not item or not match_western_merchant_item(item):
                item = extract_shop_item(quest_text)
            if item:
                item = match_western_merchant_item(item) or clean_shop_item_name(item)

            # Nếu item vẫn chưa hợp lệ theo danh mục Thương nhân Tây Vực, mở lại bảng nhiệm vụ để đọc chữ vàng
            if not item or not match_western_merchant_item(item):
                logger.warning(
                    f"[{ctx.account_id}] Vật phẩm shop '{item}' không hợp lệ hoặc chưa nhận dạng đúng trong '{quest_text}'! "
                    "Mở lại bảng nhiệm vụ để trích xuất chữ vàng..."
                )
                ctx.extra.pop("yellow_item", None)
                ctx.extra.pop("requested_shop_item", None)
                active = self._check_active_commerce_quest(ctx)
                if active:
                    item = ctx.extra.get("yellow_item", "") or ctx.extra.get("requested_shop_item", "")
                    if item:
                        item = match_western_merchant_item(item) or clean_shop_item_name(item)

            if not item or not match_western_merchant_item(item):
                logger.warning(f"[{ctx.account_id}] Nhận diện Mua đồ nhưng không có vật phẩm hợp lệ trong '{quest_text}'! Quay lại MEET_NPC...")
                active = self._check_active_commerce_quest()
                if active and active[0] != quest_text:
                    ctx.extra["quest_text"] = active[0]
                    return BotState.CLASSIFY_QUEST
                return BotState.MEET_NPC

            item = match_western_merchant_item(item) or item
            ctx.extra["requested_shop_item"] = item
            ctx.extra["yellow_item"] = item
            detail = f"Mua hàng Tây Vực: {item}"
            ctx.extra["quest_detail_text"] = detail
            self._save_progress(ctx, status_text=f"Làm NV #{ctx.total_quests_done + 1}: Mua đồ", quest_detail=detail)
            logger.info(f"[{ctx.account_id}] [Loại 2 - Shop] Món đồ cần mua: '{item}'")
            activity._write("🛍️", f"[{ctx.account_id}] NV Mua do: {item}")
            self._checkpoint.save(ctx)

            if self._find_western_merchant(ctx):
                return BotState.BUY_ITEM

            logger.warning(f"[{ctx.account_id}] Chưa tìm thấy Thương nhân Tây Vực ở các điểm spawn!")
            activity._write("⚠️", f"[{ctx.account_id}] Chua tim thay Thuong nhan Tay Vuc o cac thanh!")

            if self._check_quest_timeout(ctx):
                return BotState.FIND_COMMERCE_CHIEF

            # Thương nhân xuất hiện luân phiên nên không được coi hai vòng quét
            # thất bại là nhiệm vụ đã kết thúc. Giữ nguyên nhiệm vụ/vật phẩm và
            # tiếp tục quét cho tới khi thật sự tìm thấy rồi mua xong.
            retries = int(ctx.extra.get("merchant_search_retries", 0)) + 1
            ctx.extra["merchant_search_retries"] = retries
            self._checkpoint.save(ctx)
            logger.info(
                f"[{ctx.account_id}] Tiếp tục tìm Thương nhân Tây Vực "
                f"(vòng quét kế tiếp: {retries + 1}); không quay về Chủ thương hội."
            )
            activity._write(
                "🏃",
                f"[{ctx.account_id}] Chua thay Thuong nhan, tiep tuc quet vong {retries + 1}...",
            )
            self._save_progress(
                ctx,
                status_text=f"Đang tiếp tục tìm Thương nhân Tây Vực (vòng {retries + 1})",
                quest_detail=detail,
            )
            time.sleep(2.0)
            return BotState.CLASSIFY_QUEST

        # ==================== LOẠI 3: ĐÁNH QUÁI ====================
        if quest_type == quest_type.__class__.COMBAT:
            if self._should_abandon_combat_immediately():
                logger.info(
                    f"[{ctx.account_id}] ⚡ [TỐI ƯU TỐC ĐỘ] Nhận nhiệm vụ đánh quái -> "
                    "Hủy ngay tại Chủ thương hội để đổi sang nhiệm vụ đối thoại nhanh."
                )
                activity._write(
                    "⚡",
                    f"[{ctx.account_id}] Bo qua NV danh quai -> Huy de nhan NV doi thoai...",
                )
                ctx.extra["abandon_combat_quest"] = True
                ctx.extra["abandon_combat_quest_reason"] = "Tối ưu tốc độ: Hủy nhiệm vụ đánh quái để nhận NV đối thoại"
                ctx.extra["returning_quest"] = True
                ctx.extra.pop("chief_nav_started", None)
                self._checkpoint.save(ctx)
                return BotState.FIND_COMMERCE_CHIEF

            saved_map = ctx.extra.get("combat_map", "")
            saved_monster = ctx.extra.get("combat_monster", "")
            saved_count = ctx.extra.get("combat_count", 0)

            y_tokens = ctx.extra.get("yellow_tokens")
            map_name, monster, count = extract_combat_target(quest_text, yellow_tokens=y_tokens)
            map_name = saved_map or map_name
            monster = saved_monster or monster
            count = saved_count or count or 0

            sect = ctx.extra.get("combat_sect", "")
            if not sect and map_name:
                from .quest_classifier import parse_sect_and_combat_map
                parsed_sect, clean_m = parse_sect_and_combat_map(map_name, with_accents=True)
                if parsed_sect:
                    sect = parsed_sect
                if clean_m:
                    map_name = clean_m

            missing_fields = []
            if not str(map_name or "").strip():
                missing_fields.append("địa điểm/map đánh quái")
            if not str(monster or "").strip():
                missing_fields.append("tên quái")
            try:
                valid_count = int(count) > 0
            except (TypeError, ValueError):
                valid_count = False
            if not valid_count:
                missing_fields.append("số lượng quái")

            if missing_fields:
                reason = ", ".join(missing_fields)
                logger.warning(
                    f"[{ctx.account_id}] ❌ Nhiệm vụ đánh quái thiếu {reason}; "
                    "không thể thực hiện an toàn -> quay về Chủ thương hội hủy nhiệm vụ và nhận nhiệm vụ khác."
                )
                activity._write(
                    "❌",
                    f"[{ctx.account_id}] NV danh quai thieu thong tin ({reason}) -> Huy NV, nhan NV khac.",
                )
                ctx.extra["abandon_combat_quest"] = True
                ctx.extra["abandon_combat_quest_reason"] = reason
                ctx.extra["returning_quest"] = True
                ctx.extra.pop("chief_nav_started", None)
                self._checkpoint.save(ctx)
                return BotState.FIND_COMMERCE_CHIEF

            for k in (
                "quest_target_city", "quest_target_npc", "yellow_item",
                "requested_shop_item", "multi_npc_route", "multi_npc_index",
                "multi_npc_completed", "multi_npc_needs_navigation",
            ):
                ctx.extra.pop(k, None)

            ctx.extra["combat_map"] = map_name
            ctx.extra["combat_monster"] = monster
            ctx.extra["combat_count"] = count
            ctx.extra.setdefault("combat_quest_start_time", time.time())
            # Một nhiệm vụ combat mới/chưa xong không được thừa hưởng trạng thái
            # hoàn thành từ lần chạy trước hoặc từ luồng phục hồi điều hướng.
            ctx.extra.pop("combat_objective_complete", None)
            ctx.extra.pop("combat_avoided_directions", None)
            ctx.extra.pop("combat_last_move_direction", None)
            if sect:
                ctx.extra["combat_sect"] = sect

            sect_info = f" ({sect})" if sect else ""
            detail = f"Đánh quái: {monster} ({count} con) tại {map_name}{sect_info}"
            ctx.extra["quest_detail_text"] = detail
            self._save_progress(ctx, status_text=f"Làm NV #{ctx.total_quests_done + 1}: Đánh quái", quest_detail=detail)
            logger.info(f"[{ctx.account_id}] [Loại 3 - Combat] Địa bàn: '{sect or 'Không rõ'}', Map: '{map_name}', Quái: '{monster}', SL: {count}")
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
        """
        Di chuyển đến map quái của nhiệm vụ Thương hội:
        Map quái KHÔNG có trên bản đồ thế giới.
        Cần đối thoại với Chủ thương hội và chọn dòng 'Đưa ta đến nơi làm nhiệm vụ'.
        """
        from .text_recognition import normalize_game_text, fuzzy_score

        logger.info(f"[{ctx.account_id}] Tiến hành chuyển đến map quái '{map_name}' thông qua Chủ thương hội (không tìm bản đồ thế giới)...")
        self._dismiss_unexpected_popups()
        time.sleep(0.2)

        # 1. Kiểm tra nếu nhân vật đã ở đúng map quái (ngoài thành phố lớn)
        cur_city = None
        for _ in range(3):
            cur_city, _ = self._get_current_location_from_hud()
            if cur_city:
                break
            time.sleep(0.3)
        sect_name = ctx.extra.get("combat_sect", "") if hasattr(ctx, "extra") else ""

        # Kiểm tra map nhiệm vụ trước khi đoán đây có phải thành lớn hay không.
        # Ví dụ "Tường Vân động" có độ giống chuỗi cao với "Tương Dương" và
        # từng bị nhận nhầm là thành phố, khiến bot quay ngược về Tuyền Châu.
        if cur_city and map_name and self._is_same_combat_map(cur_city, map_name, sect=sect_name):
            logger.info(
                f"[{ctx.account_id}] Nhân vật đã ở đúng map nhiệm vụ '{cur_city}', "
                "bắt đầu săn quái ngay (không quay về Chủ thương hội)!"
            )
            return True

        major_city_names = [c["name"] for c in self._loader.load_ui_coords().get("commerce_quest", {}).get("major_cities", [])] or [
            "Tuyền Châu", "Biện Kinh", "Thành Đô", "Tương Dương", "Dương Châu", "Đại Lý", "Phượng Tường", "Lâm An"
        ]
        is_outside = bool(cur_city and not any(fuzzy_score(cur_city, mc) >= 0.65 for mc in major_city_names))
        if is_outside:
            if not map_name:
                logger.info(f"[{ctx.account_id}] Nhân vật đã ở map quái '{cur_city}', sẵn sàng săn quái (không cần quay về Chủ thương hội)!")
                if not ctx.extra.get("combat_map"):
                    ctx.extra["combat_map"] = cur_city
                return True

        # 2. Đảm bảo mở ĐÚNG hộp thoại Chủ thương hội.  Một popup thường cũng có
        # thể làm _is_npc_dialog_open() trả True, nên không được dùng tín hiệu đó
        # một mình rồi bấm tọa độ chuyển map cố định.
        dialog_open = self._is_npc_dialog_open()
        chief_dialog_open = bool(dialog_open and self._verify_commerce_chief_dialog())
        if dialog_open and not chief_dialog_open:
            logger.warning(
                f"[{ctx.account_id}] Có hộp thoại/popup đang mở nhưng không phải Chủ thương hội; "
                "đóng popup trước khi tìm đường về NPC."
            )
            self._dismiss_unexpected_popups()
            try:
                self._input.press_back()
            except Exception:
                pass
            time.sleep(0.3)

        if not chief_dialog_open:
            is_near = self._is_near_commerce_chief()
            if not is_near:
                logger.info(f"[{ctx.account_id}] Nhân vật chưa đứng cạnh Chủ thương hội -> Chạy tới Chủ thương hội...")
                self._return_to_nearest_city_commerce_chief(ctx)
                start_wait = time.time()
                last_30s_check = time.time()
                coords_30s_ago = None
                stuck_retries = 0
                wait_map = None
                while time.time() - start_wait < 45.0:
                    time.sleep(0.8)
                    if (
                        wait_map
                        and is_same_city(wait_map, "Tuyền Châu")
                        and self._is_near_commerce_chief()
                    ):
                        is_near = True
                        break

                    wait_map, cur_c = self._get_current_location_from_hud()
                    # Có thể lệnh di chuyển cũ/teleport trễ đưa nhân vật trở lại
                    # map quái trong lúc đang tìm đường về Chủ thương hội. Khi đó
                    # phải hủy luồng quay về ngay, không tiếp tục chạy sang Tuyền Châu.
                    if wait_map and self._is_same_combat_map(wait_map, map_name, sect=sect_name):
                        logger.info(
                            f"[{ctx.account_id}] Nhân vật đã trở lại đúng map nhiệm vụ '{wait_map}' "
                            "trong lúc phục hồi; hủy quay về Chủ thương hội và tiếp tục đánh quái."
                        )
                        return True
                    if self._nudge_to_reveal_commerce_chief(ctx):
                        coords_30s_ago = cur_c
                        last_30s_check = time.time()
                        continue
                    if cur_c:
                        if coords_30s_ago is None:
                            coords_30s_ago = cur_c
                            last_30s_check = time.time()
                        elif time.time() - last_30s_check >= 30.0:
                            dist_30s = abs(cur_c[0] - coords_30s_ago[0]) + abs(cur_c[1] - coords_30s_ago[1])
                            if dist_30s <= 1:
                                stuck_retries += 1
                                logger.warning(
                                    f"[{ctx.account_id}] ⚠️ Đang chạy đến Chủ thương hội nhưng bị đứng yên tại chỗ 30s ({cur_c[0]}/{cur_c[1]})! "
                                    f"Bắt đầu tìm lại quá trình chạy đối với NV đánh quái..."
                                )
                                coords_30s_ago = cur_c
                                last_30s_check = time.time()
                            else:
                                coords_30s_ago = cur_c
                                last_30s_check = time.time()

            if not is_near and not self._is_npc_dialog_open() and not self._is_near_commerce_chief():
                logger.warning(f"[{ctx.account_id}] ⚠️ Hết thời gian chờ nhưng nhân vật vẫn chưa đến cạnh Chủ thương hội (vẫn ở xa)! Không thể mở đối thoại.")
                return False

            logger.info(f"[{ctx.account_id}] Đã đến cạnh Chủ thương hội, mở hộp thoại...")
            opened = self._open_commerce_chief_dialog(ctx)
            time.sleep(0.5)
        else:
            opened = True

        if not opened:
            logger.warning(f"[{ctx.account_id}] ⚠️ Hộp thoại NPC chưa mở, không thể bấm 'Đưa ta đến nơi làm nhiệm vụ'!")
            return False

        # 3. Quét tìm nút 'Đưa ta đến nơi làm nhiệm vụ' trong hộp thoại
        scr = self._screen.capture()
        btn_pos = None
        if scr is not None:
            boxes = self._text.read(scr, (200, 480, 1400, 750))
            for b in boxes:
                norm = normalize_game_text(b.text)
                if any(k in norm for k in [
                    "dua ta den", "den noi lam", "lam nhiem vu", "dua ta toi", "noi lam nhiem vu",
                    "phien dua ta", "dua ta di", "di mot chuyen", "phien dua ta di", "dua den noi"
                ]):
                    btn_pos = (b.center_x, b.center_y)
                    logger.info(f"[{ctx.account_id}] Tìm thấy nút '{b.text}' tại {btn_pos}")
                    break

        # Chỉ dùng tọa độ dự phòng sau khi đã xác minh đúng hội thoại Chủ thương
        # hội. Nếu không, tọa độ này có thể bấm thẳng xuống mặt đất/UI thành phố.
        if not btn_pos:
            if not self._verify_commerce_chief_dialog(scr):
                logger.warning(
                    f"[{ctx.account_id}] ⚠️ Không tìm thấy nút 'Đưa ta đến nơi làm nhiệm vụ' và "
                    "không xác minh được hội thoại Chủ thương hội; không bấm tọa độ dự phòng."
                )
                return False
            btn_pos = (764, 576)
            logger.info(f"[{ctx.account_id}] Dùng tọa độ chuẩn cho nút 'Đưa ta đến nơi làm nhiệm vụ': {btn_pos}")

        self._input.tap(btn_pos[0], btn_pos[1])
        logger.info(f"[{ctx.account_id}] Đã bấm 'Đưa ta đến nơi làm nhiệm vụ' tại {btn_pos}. Đang chờ dịch chuyển map...")
        activity._write("🚀", f"[{ctx.account_id}] Bam 'Dua ta den noi lam nhiem vu' -> {map_name}...")
        # Không coi cú bấm là thành công cho tới khi HUD thực sự đổi sang map
        # nhiệm vụ. Điều này ngăn vòng combat tiếp tục tuần tra trong thành.
        for _ in range(12):
            time.sleep(0.5)
            arrived_map, _ = self._get_current_location_from_hud()
            if arrived_map and self._is_same_combat_map(arrived_map, map_name, sect=sect_name):
                logger.info(
                    f"[{ctx.account_id}] Xác nhận đã dịch chuyển thành công tới map nhiệm vụ '{arrived_map}'."
                )
                return True

        current_after, _ = self._get_current_location_from_hud()
        logger.warning(
            f"[{ctx.account_id}] ⚠️ Đã bấm chuyển map nhưng HUD vẫn ở "
            f"'{current_after or 'chưa xác định'}', chưa tới '{map_name}'; không bắt đầu tìm quái."
        )
        return False

    def _navigate_to_dialogue_npc(self, ctx: StateContext, city: str, npc: str) -> bool:
        """Select the quest area, then the NPC, directly from the minimap."""
        logger.info(
            f"[{ctx.account_id}] NV doi thoai: minimap -> Tim map '{city}' -> "
            f"Tim NPC '{npc}'..."
        )

        # 1. Tối ưu thời gian: Nếu đang ở thành phố khác, di chuyển nhanh sang thành phố đích qua Xa Phu trước!
        # Tuyệt đối không để game chạy bộ xuyên lục địa (cross-map walk) mất 8-10 phút.
        cur_city, _ = self._get_current_location_from_hud()
        if city and cur_city and not is_same_city(cur_city, city):
            logger.info(
                f"[{ctx.account_id}] ⚡ [TỐI ƯU ĐỐI THOẠI] Hiện tại ở '{cur_city}', "
                f"chuyển nhanh sang thành phố đích '{city}' qua Xa Phu trước khi tìm NPC '{npc}'..."
            )
            activity._write("🐎", f"[{ctx.account_id}] Di Xa Phu sang {city}...")
            switched = self._switch_to_city(ctx, city, force=False)
            if not switched:
                logger.warning(
                    f"[{ctx.account_id}] Không thể chuyển sang thành phố '{city}' qua Xa Phu!"
                )
                return False
            cur_city, _ = self._get_current_location_from_hud()
            if cur_city and not is_same_city(cur_city, city):
                logger.warning(
                    f"[{ctx.account_id}] Chưa tới đúng thành phố '{city}' "
                    f"(HUD: '{cur_city}'); chưa tìm/chọn NPC '{npc}'."
                )
                return False

        coords = self._loader.load_ui_coords()
        nav_cfg = coords.get("navigation", {})
        find_map = nav_cfg.get("map_tab_find_map", {"x": 172, "y": 175})
        search = nav_cfg.get("map_search_input", {"x": 266, "y": 222})

        if not self._ensure_map_open():
            logger.warning(f"[{ctx.account_id}] Khong mo duoc minimap de tim '{city}'.")
            return False
        self._input.tap(find_map.get("x", 172), find_map.get("y", 175))
        time.sleep(0.2)
        self._input.tap(search.get("x", 266), search.get("y", 222))
        time.sleep(0.1)
        try:
            self._adb._run([
                "shell",
                "input keyevent 123; " + "; ".join(["input keyevent 67"] * 30),
            ])
        except Exception:
            pass
        self._type_unicode_text(ctx, city)
        time.sleep(0.35)
        if not self._click_map_search_result(city):
            logger.warning(
                f"[{ctx.account_id}] Khong tim/click duoc dia ban '{city}'; "
                f"khong chon NPC '{npc}'."
            )
            if self._is_map_open():
                self._close_map()
            return False
        time.sleep(0.25)

        # Never continue if selecting the area unexpectedly closed minimap;
        # otherwise NPC coordinates would land on the game/chat UI.
        if not self._is_map_open():
            logger.warning(
                f"[{ctx.account_id}] Minimap da dong sau khi chon '{city}'; "
                f"khong click NPC '{npc}' len man hinh game."
            )
            return False

        norm_city = normalize_game_text(city)
        norm_npc = normalize_game_text(npc)
        allow_directional_xa_phu = (
            "tuong duong" in norm_city and norm_npc == "xa phu"
        )
        clicked = self._search_and_select_npc(
            ctx,
            npc,
            allow_directional_xa_phu=allow_directional_xa_phu,
        )
        if not clicked:
            ctx.extra["npc_nav_started"] = False
            logger.warning(
                f"[{ctx.account_id}] Khong tim/click duoc NPC '{npc}' "
                f"trong dia ban '{city}'."
            )
            if self._is_map_open():
                self._close_map()
            return False

        ctx.extra["npc_nav_started"] = True
        self._close_map()
        time.sleep(0.2)
        self._press_buff_speed()
        logger.info(
            f"[{ctx.account_id}] Da chon NPC '{npc}' tai '{city}'; "
            "dang auto-path toi noi de doi thoai."
        )
        return True

    def _navigate_to_dialogue_npc_legacy(self, ctx: StateContext, city: str, npc: str) -> bool:
        """Chuyển bản đồ sang thành phố mục tiêu qua Tìm map, sau đó tìm NPC. Tuyệt đối không dùng ô nhiệm vụ góc trái."""
        logger.info(f"[{ctx.account_id}] Điều hướng đến thành phố '{city}', gặp NPC '{npc}' qua bản đồ mini (không dùng ô nhiệm vụ góc trái)...")
        self._dismiss_unexpected_popups()
        time.sleep(0.15)

        # 1. Chuyển sang thành phố mục tiêu qua Tìm map. Không được
        # tiếp tục tìm NPC đích khi việc chuyển thành chưa hoàn tất:
        # minimap có thể đã nạp map đích trong khi nhân vật vẫn ở
        # cạnh Xa phu của thành hiện tại.
        switched = self._switch_to_city(ctx, city)
        current_city, _ = self._get_current_location_from_hud()
        if not switched or not current_city or not is_same_city(current_city, city):
            logger.warning(
                f"[{ctx.account_id}] Chưa tới đúng thành phố '{city}' "
                f"(HUD: '{current_city or 'chưa xác định'}'); "
                f"chưa tìm/chọn NPC '{npc}'."
            )
            return False

        # 2. Mở bản đồ thành phố để tìm NPC
        self._ensure_map_open()
        time.sleep(0.3)

        # 3. Tìm và chọn NPC bằng ô input tìm kiếm (không dùng ô nhiệm vụ góc trái)
        norm_city = normalize_game_text(city)
        norm_npc = normalize_game_text(npc)
        allow_directional_xa_phu = (
            "tuong duong" in norm_city and norm_npc == "xa phu"
        )
        if allow_directional_xa_phu:
            logger.info(
                f"[{ctx.account_id}] NV đối thoại tại Tương Dương: "
                "chấp nhận Xa phu Nam, Xa phu Tây hoặc Xa phu Bắc."
            )
        clicked = self._search_and_select_npc(
            ctx,
            npc,
            allow_directional_xa_phu=allow_directional_xa_phu,
        )
        if not clicked:
            logger.warning(f"[{ctx.account_id}] Không thể tìm/chọn NPC '{npc}' trên bản đồ '{city}'!")
            self._close_map()
            return False

        # 4. Đóng bản đồ an toàn để nhân vật bắt đầu chạy
        self._close_map()
        time.sleep(0.2)
        self._press_buff_speed()
        logger.info(f"[{ctx.account_id}] Nhân vật đang tự chạy đến NPC '{npc}' tại '{city}'...")
        return True

    def is_forbidden_ui_coordinate(
        self, x: int, y: int, w: int = 1600, h: int = 900,
        allow_quest_button: bool = True,
    ) -> bool:
        """
        Kiểm tra xem tọa độ (x, y) có rơi vào các khu vực nút UI/menu nhạy cảm hay không.
        Tuyệt đối cấm click vào các vùng này trong quá trình đánh quái/di chuyển tự do
        để tránh hiện các bảng popup/giao diện không cần thiết (Nhiệm vụ, Nhân vật, Cài đặt, Map...).
        """
        rx = x / w if w else x / 1600.0
        ry = y / h if h else y / 900.0

        # Nút Nhiệm vụ là ngoại lệ duy nhất trong cụm phải-dưới. Dùng một vùng
        # nhỏ quanh tâm chuẩn (1450, 770), tránh mở nhầm các nút Tổ đội/Cài đặt
        # nằm sát bên cạnh. Tọa độ được chuẩn hóa để dùng được ở mọi độ phân giải.
        quest_dx = (rx - (1450 / 1600.0)) / (65 / 1600.0)
        quest_dy = (ry - (770 / 900.0)) / (55 / 900.0)
        is_quest_button = (quest_dx * quest_dx + quest_dy * quest_dy) <= 1.0

        # 1. Cụm nút tính năng góc dưới bên phải (Võ công, Nhân vật, Đấu trường,
        # Bang hội, Tổ đội, Cài đặt, Hảo hữu). Chỉ chừa nút Nhiệm vụ ở trên.
        if rx >= 0.76 and ry >= 0.53:
            return not (allow_quest_button and is_quest_button)

        # 2. Dải nút & HUD đỉnh màn hình (Avatar, Máu/Mana, Mở khóa, Luyện công, Phúc lợi, Hạng, Ngự các)
        if ry <= 0.19 and rx <= 0.78:
            return True

        # 3. Góc trên bên phải: Bản đồ con Minimap & Dải nút điều khiển nhanh (Buff, Auto, Nhặt, Đổi, Túi, Ngựa)
        if rx >= 0.81 and ry <= 0.27:
            return True
        if rx >= 0.75 and 0.18 <= ry <= 0.53:
            return True

        # 4. Góc bên trái: Bảng theo dõi Nhiệm vụ Hướng dẫn Tân thủ (Dã Tẩu...)
        if rx <= 0.27 and 0.17 <= ry <= 0.62:
            return True

        # 5. Cột nút chức năng dọc mép trái dưới (Linh phách, Động tác, Đi, Thiền)
        if rx <= 0.11 and ry >= 0.58:
            return True

        # 6. Toàn bộ dải giữa-dưới: khung chat và dàn nút Nhanh/Tùy chọn.
        # Vùng này bắt đầu cao hơn thanh đáy vì chat có thể đang mở rộng.
        if rx <= 0.76 and ry >= 0.82:
            return True

        return False

    def clamp_to_safe_combat_ground(self, x: int, y: int, w: int = 1600, h: int = 900) -> tuple[int, int]:
        """
        Ép tọa độ click di chuyển/tuần tra về vùng mặt đất chiến trường hoàn toàn an toàn
        (không dính bất kỳ nút UI hay banner nào).
        """
        x_min = int(w * 0.28) if w else 450
        x_max = int(w * 0.74) if w else 1180
        y_min = int(h * 0.22) if h else 200
        y_max = int(h * 0.82) if h else 740

        clamped_x = max(x_min, min(x, x_max))
        clamped_y = max(y_min, min(y, y_max))
        return (clamped_x, clamped_y)

    def _move_combat_patrol(self, point: dict, w: int = 1600, h: int = 900) -> tuple[int, int]:
        """Di chuyển tuần tra thật bằng joystick, kèm tap mặt đất làm dự phòng."""
        coords = self._loader.load_ui_coords() if hasattr(self, "_loader") else {}
        combat_cfg = coords.get("combat", {})
        joystick = combat_cfg.get("joystick_center", {"x": 220, "y": 680})
        # Nhích joystick ngắn để quét quái kế tiếp sớm. Giới hạn an toàn tránh
        # cấu hình sai làm giữ phím quá lâu hoặc tạo thao tác quá ngắn.
        hold_ms = int(combat_cfg.get("patrol_joystick_hold_ms", 350) or 350)
        hold_ms = max(150, min(5000, hold_ms))

        scale_x = w / 1600.0 if w else 1.0
        scale_y = h / 900.0 if h else 1.0
        joy_x = int(joystick.get("x", 220) * scale_x)
        joy_y = int(joystick.get("y", 680) * scale_y)
        dx = int(point.get("dx", 0) * scale_x)
        dy = int(point.get("dy", 0) * scale_y)

        ground_x, ground_y = self.clamp_to_safe_combat_ground(
            point.get("x", 800), point.get("y", 600), w, h
        )

        if dx or dy:
            self._input.swipe(joy_x, joy_y, joy_x + dx, joy_y + dy, duration=hold_ms)
            time.sleep(0.02)
        self._input.tap(ground_x, ground_y)
        return ground_x, ground_y

    def _find_monster_on_screen(
        self,
        frame: np.ndarray,
        monster_name: str,
        blocked_positions: Optional[list[dict]] = None
    ) -> Optional[tuple[int, int]]:
        """Quét tìm vị trí nhãn tên con quái xuất hiện trên màn hình thực chiến để tap tấn công trực tiếp."""
        if frame is None:
            return None
        # Nếu bảng bản đồ đang mở thì không tìm quái (để tránh click nhầm tên map trên bảng đồ)
        if hasattr(self, "_is_map_open") and self._is_map_open(frame):
            return None
        # Nếu Bảng nhiệm vụ đang mở thì không tìm quái (để tránh click nhầm chữ trên bảng)
        if hasattr(self, "_is_quest_panel_open") and self._is_quest_panel_open(frame):
            return None

        from difflib import SequenceMatcher
        from .text_recognition import normalize_game_text, fuzzy_score
        h_f, w_f = frame.shape[:2]
        # Vùng thực chiến mở rộng giữa màn hình: loại bỏ cột UI trái (quest tracker), HUD trên, dock nút góc dưới phải
        min_x = max(int(w_f * 0.28), 440)
        max_x = min(int(w_f * 0.75), 1200)
        min_y = max(int(h_f * 0.20), 180)
        max_y = min(int(h_f * 0.82), 740)
        combat_roi = (min_x, min_y, max_x, max_y)

        # Nhện trong Tường Vân động có phần thân xanh neon rất rõ,
        # nhưng nhãn chữ "Nhện" quá nhỏ nên OCR thường bỏ sót. Nhận
        # diện blob thân xanh trước OCR để right-click ngay tâm thân.
        # Chỉ bật fallback này cho quái Nhện, tránh nhầm tên xanh của NPC.
        norm_requested = normalize_game_text(monster_name)
        if "nhen" in norm_requested:
            try:
                roi = frame[min_y:max_y, min_x:max_x]
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                green_mask = cv2.inRange(
                    hsv,
                    np.array([35, 100, 80], dtype=np.uint8),
                    np.array([92, 255, 255], dtype=np.uint8),
                )
                # Nối các chân/thân Nhện thành một blob, trong khi chữ
                # tên mỏng vẫn có chiều cao quá thấp và sẽ bị loại.
                green_mask = cv2.morphologyEx(
                    green_mask,
                    cv2.MORPH_CLOSE,
                    np.ones((5, 5), dtype=np.uint8),
                )
                count, _, stats, centroids = cv2.connectedComponentsWithStats(green_mask, 8)
                green_bodies: list[tuple[int, int]] = []
                for idx in range(1, count):
                    x, y, bw, bh, area = stats[idx]
                    aspect = bw / max(bh, 1)
                    if not (
                        80 <= area <= 5000
                        and 18 <= bw <= 120
                        and 14 <= bh <= 100
                        and aspect <= 2.6
                    ):
                        continue
                    # Dùng tâm bounding-box của toàn bộ thân/chân thay vì
                    # centroid màu (thường bị kéo lệch về một cụm chân xanh).
                    cx = min_x + x + bw // 2
                    cy = min_y + y + bh // 2
                    if self.is_forbidden_ui_coordinate(cx, cy, w_f, h_f, allow_quest_button=False):
                        continue
                    green_bodies.append((cx, cy))
                if green_bodies:
                    char_x, char_y = int(w_f * 0.50), int(h_f * 0.53)
                    # Nhân vật/hiệu ứng cưỡi ngựa cũng có các mảng xanh ngay
                    # giữa màn hình. Log thực tế từng khóa sai tại (805,478),
                    # chỉ cách tâm nhân vật 5px. Loại hẳn vùng này trước khi
                    # chọn Nhện gần nhất.
                    # Vùng 45-50px vẫn lọt hiệu ứng/tên xanh sát người chơi;
                    # log thực tế false-positive tại (851,509), cách tâm 60px.
                    # Nâng vùng loại trừ lên tối thiểu 75px. Nếu Nhện thật áp
                    # sát nhân vật, nhánh OCR tên quái phía dưới vẫn nhận được.
                    self_exclusion_radius = max(75, int(min(w_f, h_f) * 0.08))
                    green_bodies = [
                        pt for pt in green_bodies
                        if math.hypot(pt[0] - char_x, pt[1] - char_y)
                        >= self_exclusion_radius
                    ]
                if green_bodies:
                    green_bodies.sort(key=lambda pt: math.hypot(pt[0] - char_x, pt[1] - char_y))
                    logger.debug(f"Phát hiện thân Nhện xanh trực tiếp tại {green_bodies[0]} (không cần OCR nhãn).")
                    return green_bodies[0]
            except Exception as exc:
                logger.debug(f"Green spider-body detection failed: {exc}")

        boxes = self._text.read(frame, combat_roi)

        # Unselected monsters use a very small green name label. RapidOCR can
        # miss that label on the normal game frame (especially "Nhện"), even
        # though it is clearly visible to a person. Run one additional OCR pass
        # with green pixels converted to black text on a white background. Keep
        # the image at the original size so returned coordinates stay valid.
        try:
            roi = frame[min_y:max_y, min_x:max_x]
            if roi.size:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                green_mask = cv2.inRange(
                    hsv,
                    np.array([32, 65, 65], dtype=np.uint8),
                    np.array([95, 255, 255], dtype=np.uint8),
                )
                green_mask = cv2.morphologyEx(
                    green_mask,
                    cv2.MORPH_CLOSE,
                    np.ones((2, 2), dtype=np.uint8),
                )
                if cv2.countNonZero(green_mask) >= 12:
                    green_text_frame = np.full_like(frame, 255)
                    green_text_frame[min_y:max_y, min_x:max_x] = cv2.cvtColor(
                        255 - green_mask, cv2.COLOR_GRAY2BGR
                    )
                    green_boxes = self._text.read(green_text_frame, combat_roi)
                    seen = {(b.text, b.center_x, b.center_y) for b in boxes}
                    boxes.extend(
                        b for b in green_boxes
                        if (b.text, b.center_x, b.center_y) not in seen
                    )
        except Exception as exc:
            logger.debug(f"Green monster-label OCR fallback failed: {exc}")
        norm_m = normalize_game_text(monster_name)
        m_tokens = set(norm_m.split()) if norm_m else set()
        norm_m_c = norm_m.replace(" ", "") if norm_m else ""

        # Mở rộng danh sách alias của quái nếu là quái đã biết (chịu lỗi font game và OCR viết tắt)
        m_aliases = {norm_m} if norm_m else set()
        try:
            from .quest_classifier import KNOWN_MONSTER_ALIASES
            for alias_k, canon_v in KNOWN_MONSTER_ALIASES.items():
                if canon_v == monster_name or normalize_game_text(canon_v) == norm_m:
                    m_aliases.add(alias_k)
        except Exception:
            pass

        IGNORED_COMBAT_UI = (
            "hk0", "acc", "cap", "lv", "hp", "mp", "nhan vat", "thuong hoi",
            "nhiem vu", "auto", "buff", "nhat", "ngua", "vo cong", "dau truong",
            "bang hoi", "hao huu", "cai dat", "tuy chon", "nhanh", "tong hop",
            "cua hieu", "bay ban", "dong gia", "cua hang", "shop", "sap",
            "linh phach", "dong tac", "thien", "luyen cong", "mo khoa", "doi",
            "tuyen chau", "thanh do", "bien kinh", "tuong duong", "duong chau",
            "long tuyen", "ban do", "the gioi", "tim map", "tim npc",
            "ne", "hut", "chi mang", "bao kich", "hut mau", "mien dich", "sat thuong", "khang",
            "phuc loi", "hang", "ngu cac",
            # Các từ trong nhiệm vụ tân thủ/NPC tránh click nhầm làm nhân vật chạy theo nhiệm vụ khác:
            "huong dan", "tan thu", "da tau", "khau chuan", "pham trong", "vuong nghiep",
            "cai bang", "duong chu", "de tu", "gia ve", "ha hau", "chuyen thu", "thu thap",
            "thao duoc", "thao duong"
        )

        candidates = []
        char_x, char_y = int(w_f * 0.50), int(h_f * 0.53)

        # Ghép các box nằm liền kề trên cùng một dòng (chênh lệch y <= 18px, khoảng cách x giữa 2 box <= 120px)
        sorted_boxes = sorted([b for b in boxes if b.center_x >= min_x], key=lambda b: (b.center_y // 20, b.center_x))
        n = len(sorted_boxes)
        compound_candidates = []
        for i in range(n):
            b1 = sorted_boxes[i]
            compound_candidates.append((b1.text, b1.center_x, b1.center_y))
            for j in range(i + 1, min(i + 4, n)):
                b2 = sorted_boxes[j]
                if abs(b1.center_y - b2.center_y) <= 18 and 0 < (b2.center_x - b1.center_x) <= 120:
                    compound_text = f"{b1.text} {b2.text}"
                    compound_candidates.append((compound_text, (b1.center_x + b2.center_x) // 2, (b1.center_y + b2.center_y) // 2))

        for text, cx, cy in compound_candidates:
            norm_b = normalize_game_text(text)
            if not norm_b:
                continue
            if any(ignored in norm_b or norm_b == ignored for ignored in IGNORED_COMBAT_UI):
                continue

            is_match = False
            norm_b_c = norm_b.replace(" ", "")
            b_tokens = set(norm_b.split())
            matching_tokens = [t for t in (m_tokens & b_tokens) if len(t) >= 3]

            # 1. Khớp chuỗi đầy đủ hoặc chuỗi con (bao gồm cả các biến thể trong m_aliases)
            if norm_m and (norm_m in norm_b or norm_b in norm_m or any(al in norm_b or norm_b in al for al in m_aliases)):
                is_match = True
            # 2. Khớp dạng compact (xử lý dính chữ như 'nguoilun' vs 'nguoi lun')
            elif norm_m_c and (norm_m_c in norm_b_c or norm_b_c in norm_m_c or any(al.replace(" ", "") in norm_b_c or norm_b_c in al.replace(" ", "") for al in m_aliases)):
                is_match = True
            # 3. Khớp token
            elif len(m_tokens) >= 2 and len(matching_tokens) >= 2:
                is_match = True
            elif len(m_tokens) == 1 and matching_tokens:
                is_match = True
            # 4. Fuzzy match trên compact text (chịu lỗi OCR như 'barnh nuong' vs 'ban nuong')
            elif norm_m_c and SequenceMatcher(None, norm_m_c, norm_b_c).ratio() >= 0.75:
                is_match = True
            # 5. Fuzzy match thông thường
            elif norm_m and len(norm_b.split()) >= max(1, len(m_tokens) - 1) and fuzzy_score(norm_b, norm_m) >= 0.70:
                is_match = True
            # 6. Nếu tên quái có từ đặc trưng (độ dài >= 4) và xuất hiện trong nhãn quái
            elif any(len(t) >= 4 and t in norm_b for t in m_tokens):
                is_match = True

            if is_match:
                # Tâm thân quái chỉ nằm khoảng 40-55px dưới tâm nhãn tên
                # trên khung 1600x900. Offset cũ 0.11 (~99px) rơi xuống mặt đất,
                # đặc biệt với quái thấp như Nhện. Giới hạn offset để vừa
                # không bấm lên chữ, vừa không vượt qua thân quái.
                body_offset = max(32, min(55, int(h_f * 0.055))) if h_f else 48
                body_y = min(cy + body_offset, frame.shape[0] - 20)
                # Tuyệt đối cấm click nếu vị trí thân quái rơi vào các nút UI nhạy cảm (tránh bật bảng Nhiệm vụ, Cài đặt, Nhân vật...)
                if self.is_forbidden_ui_coordinate(
                    cx, body_y, w_f, h_f, allow_quest_button=False
                ):
                    logger.debug(f"Bỏ qua ứng viên quái tại ({cx}, {body_y}) do rơi vào vùng nút UI cấm.")
                    continue
                candidates.append((cx, body_y))

        if not candidates:
            return None

        # Lọc bỏ các vị trí quái bị kẹt tường (blocked_positions)
        filtered_candidates = []
        now_t = time.time()
        for pt in candidates:
            if blocked_positions:
                is_blocked = False
                for bp in blocked_positions:
                    if bp.get("expires_at", 0) > now_t:
                        b_pos = bp.get("pos", (0, 0))
                        if math.hypot(pt[0] - b_pos[0], pt[1] - b_pos[1]) < 85:
                            is_blocked = True
                            break
                if is_blocked:
                    continue
            filtered_candidates.append(pt)

        if not filtered_candidates:
            return None

        # Sắp xếp và ưu tiên tấn công con quái mục tiêu ở gần nhân vật nhất
        filtered_candidates.sort(key=lambda pt: math.hypot(pt[0] - char_x, pt[1] - char_y))
        return filtered_candidates[0]

    def _attack_monster_blocking_travel(self, ctx: StateContext) -> bool:
        """Diệt quái đang áp sát/tấn công làm gián đoạn tự tìm đường hoặc dịch chuyển quay về.

        Hỗ trợ tìm quái từ:
        1. Tên quái nhiệm vụ trong ctx (combat_monster, vd: Cóc độc).
        2. Vùng mục tiêu đang tấn công/bị khóa ở HUD góc dưới trái.
        3. Thanh mục tiêu HUD phía trên (nếu có).
        """
        if not ctx:
            return False

        # Nếu đã hoàn thành mục tiêu nhiệm vụ hoặc đang trên đường quay về / hủy nhiệm vụ: KHÔNG ĐƯỢC đánh quái chặn đường!
        if (
            ctx.extra.get("combat_objective_complete")
            or ctx.extra.get("returning_quest")
            or ctx.extra.get("abandon_combat_quest")
        ):
            return False

        # Nếu đã ở thành lớn an toàn (ví dụ Tuyền Châu) thì không cần kiểm tra quái
        if hasattr(self, "_get_current_location_from_hud"):
            loc = self._get_current_location_from_hud()
            cur_city = loc[0] if (loc and isinstance(loc, (list, tuple)) and len(loc) >= 1) else None
            if cur_city and is_same_city(cur_city, "Tuyền Châu"):
                return False

        if not hasattr(self, "_screen") or self._screen is None:
            return False

        try:
            frame = self._screen.capture()
            if frame is None:
                return False

            # Nếu phát hiện banner hoàn thành nhiệm vụ trên màn hình thì lập tức ghi nhận và không đánh quái nữa
            if hasattr(self, "_text") and self._text:
                banner_boxes = self._text.read(frame, (200, 300, 1400, 800))
                banner_text = " ".join(b.text for b in banner_boxes)
                banner_norm = normalize_game_text(banner_text)
                if ("truy sat" in banner_norm or "thuong hoi" in banner_norm or "tieu diet" in banner_norm) and (
                    "da hoan thanh" in banner_norm or "hoan thanh" in banner_norm or "da du" in banner_norm
                ):
                    logger.info(
                        f"[{ctx.account_id}] 🎯 Phát hiện banner hoàn thành nhiệm vụ ('{banner_text}'), "
                        "lập tức dừng đánh quái và chuyển sang quay về trả nhiệm vụ!"
                    )
                    ctx.extra["combat_objective_complete"] = True
                    ctx.extra["returning_quest"] = True
                    return False

            if hasattr(self, "_is_map_open") and self._is_map_open(frame):
                self._close_map()
                time.sleep(0.2)
                frame = self._screen.capture()

            # Không dò quái qua một lớp giao diện đang che thế giới
            if hasattr(self, "_is_bag_open") and self._is_bag_open(frame):
                logger.info(
                    f"[{ctx.account_id}] Hành trang đang mở; đóng trước khi kiểm tra quái chặn đường."
                )
                self._close_bag()
                return False

            candidate_monster_names = []

            # 1. Quét từ thanh mục tiêu HUD phía trên (quái đang tấn công/bị khóa vào người chơi)
            coords = self._loader.load_ui_coords() if hasattr(self, "_loader") else {}
            hud = coords.get("combat", {}).get(
                "target_hud_region",
                {"x1": 450, "y1": 60, "x2": 1100, "y2": 130},
            )
            if hasattr(self, "_text") and self._text:
                hud_boxes = self._text.read(
                    frame, (hud["x1"], hud["y1"], hud["x2"], hud["y2"])
                )
                ignored = (
                    "npc", "cap", "level", "lv", "hp", "mp", "phuc loi",
                    "hang", "ngu cac", "auto", "buff", "choi qua",
                    "loai", "nang", "gia", "ban tiem", "giao dich", "vut bo",
                    "hanh trang", "sap xep", "thong tin", "tach", "bay ban",
                )
                for box in hud_boxes:
                    norm = normalize_game_text(box.text).strip()
                    compact = norm.replace(" ", "")
                    if len(compact) < 3 or compact.isdigit():
                        continue
                    if any(word in norm for word in ignored):
                        continue
                    if box.text.strip() not in candidate_monster_names:
                        candidate_monster_names.append(box.text.strip())

            # 2. Quét từ vùng target mục tiêu ở góc dưới trái (vùng x: 10..220, y: 680..850)
            if hasattr(self, "_text") and self._text:
                btm_boxes = self._text.read(frame, (10, 680, 220, 850))
                for box in btm_boxes:
                    norm = normalize_game_text(box.text).strip()
                    if any(m in norm for m in ("coc doc", "nhen doc", "soi", "quai", "ha dong su", "bat phu")):
                        if box.text.strip() not in candidate_monster_names:
                            candidate_monster_names.append(box.text.strip())

            # 3. Quái nhiệm vụ trong ctx (nếu đang làm dở nhiệm vụ đánh quái)
            c_monster = ctx.extra.get("combat_monster", "")
            if c_monster and c_monster not in candidate_monster_names:
                candidate_monster_names.append(c_monster)

            monster_pos = None
            target_name = ""
            for m_name in candidate_monster_names:
                candidate_pos = self._find_monster_on_screen(frame, m_name)
                if candidate_pos:
                    target_name = m_name
                    monster_pos = candidate_pos
                    break

            if not monster_pos:
                return False

            mx, my = monster_pos
            logger.warning(
                f"[{ctx.account_id}] ⚔️ Bị quái '{target_name}' chặn/tấn công; "
                f"chuột phải vào thân quái ({mx}, {my}) để diệt trước khi đi tiếp..."
            )
            activity._write("⚔️", f"[{ctx.account_id}] Quai '{target_name}' chan duong -> Right-click diet quai...")
            if hasattr(self._input, "right_click"):
                self._input.right_click(mx, my, count=1)
            else:
                self._input.double_tap(mx, my)
            time.sleep(1.2)
            return True
        except Exception as exc:
            logger.debug(f"[{ctx.account_id}] Kiem tra quai chan duong that bai: {exc}")
            return False

    def _clear_monsters_around_character(self, ctx: StateContext, max_monsters: int = 4, timeout: float = 8.0) -> bool:
        """
        Quét và tiêu diệt sạch các quái đang ở gần hoặc tấn công nhân vật
        để không bị ngắt quãng khi dịch chuyển/chạy bộ quay về trả nhiệm vụ.
        """
        logger.info(f"[{ctx.account_id}] ⚔️ Kiểm tra dọn sạch quái xung quanh nhân vật trước khi rút lui...")
        start_t = time.time()
        killed_any = False
        monsters_killed = 0

        while (time.time() - start_t) < timeout and monsters_killed < max_monsters:
            if not self._attack_monster_blocking_travel(ctx):
                break
            killed_any = True
            monsters_killed += 1
            time.sleep(1.2)

        if killed_any:
            logger.info(f"[{ctx.account_id}] ✅ Đã diệt {monsters_killed} quái cản đường, sẵn sàng quay về.")
        else:
            logger.info(f"[{ctx.account_id}] ✅ Xung quanh nhân vật đã an toàn, không có quái cản đường.")
        return killed_any

    def _find_horse_button(self, frame: Optional[np.ndarray] = None) -> tuple[int, int]:
        """Tìm tọa độ nút 'Ngựa' trên màn hình bằng OCR hoặc fallback cấu hình (1553, 370)."""
        coords = self._loader.load_ui_coords() if hasattr(self, "_loader") and self._loader else {}
        h_cfg = coords.get("combat", {}).get("horse_button", {})
        if h_cfg and "x" in h_cfg and "y" in h_cfg:
            return (h_cfg["x"], h_cfg["y"])
        if frame is None and hasattr(self, "_screen") and self._screen is not None:
            frame = self._screen.capture()
        if frame is not None and hasattr(self, "_text") and self._text is not None:
            h, w = frame.shape[:2]
            boxes = self._text.read(frame, (int(w * 0.88), int(h * 0.32), w, int(h * 0.55)))
            for b in boxes:
                norm = normalize_game_text(b.text)
                if any(k in norm for k in ["ngua", "ngya", "ngu a"]):
                    return (b.center_x, max(0, b.center_y - 14))

        return (1553, 370)

    def _is_character_mounted_on_horse(self, frame: Optional[np.ndarray] = None) -> bool:
        """
        Kiểm tra nhân vật có đang cưỡi ngựa không.
        - Frame None -> False
        - Kiểm tra vùng tâm nhân vật (y: 400..500, x: 750..850)
        """
        if frame is None:
            return False
        try:
            h, w = frame.shape[:2]
            y1, y2 = int(h * 400 / 900), int(h * 500 / 900)
            x1, x2 = int(w * 750 / 1600), int(w * 850 / 1600)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return False
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask_brown = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([20, 255, 255]))
            mask_white = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 45, 255]))
            total_px = roi.shape[0] * roi.shape[1]
            if np.count_nonzero(mask_brown) / total_px > 0.15 or np.count_nonzero(mask_white) / total_px > 0.25:
                return True
        except Exception as e:
            logger.debug(f"_is_character_mounted_on_horse error: {e}")
        return False

    def _ensure_dismounted_from_horse(self, ctx: Optional[StateContext] = None, frame: Optional[np.ndarray] = None) -> bool:
        """
        Đảm bảo nhân vật xuống ngựa khi vào map đánh quái / chuẩn bị tấn công.
        Nếu phát hiện đang cưỡi ngựa hoặc mới vào map chưa xuống ngựa:
        Bấm nút 'Ngựa' để xuống ngựa.
        """
        # Do not click the horse button during combat. The former fallback
        # treated a missing context flag as proof that the character was
        # mounted, while the colour heuristic also confused scenery/clothes
        # with a horse. A right-click on the monster is sufficient for combat.
        return True

    def _ensure_combat_skills_visible(self, frame: Optional[np.ndarray] = None) -> bool:
        """
        Không tự động bấm nút ``Tùy chọn`` trong lúc chiến đấu.

        Các nhãn Võ công/Nhân vật/Nhiệm vụ/Cài đặt vốn có thể xuất hiện trong
        giao diện bình thường, nên OCR không đủ tin cậy để kết luận menu đang
        mở. Cơ chế cũ còn có thể dùng frame cũ và bấm lần hai, vô tình mở lại
        chính menu vừa đóng. Vùng click chiến đấu đã được giới hạn riêng bởi
        ``is_forbidden_ui_coordinate`` và ``clamp_to_safe_combat_ground``.
        """
        return True

    def _handle_navigate(self, ctx: StateContext) -> BotState:
        from .text_recognition import normalize_game_text, fuzzy_score

        logger.info(f"[{ctx.account_id}] Navigating / auto-walking to target...")
        # Cờ này bảo vệ cửa sổ crash giữa lúc đổi sang NPC kế tiếp và state
        # machine kịp chuyển từ INTERACT_NPC sang NAVIGATE.
        ctx.extra.pop("multi_npc_needs_navigation", None)
        ctx.last_action = "navigate"
        self._checkpoint.save(ctx)

        # Kiểm tra watchdog timeout nhiệm vụ (mặc định 15 phút)
        if self._check_quest_timeout(ctx):
            return BotState.FIND_COMMERCE_CHIEF

        # Trường hợp NHIỆM VỤ ĐÁNH QUÁI
        if ctx.current_quest_type == ctx.current_quest_type.__class__.COMBAT:
            # 0. Nếu mục tiêu đánh quái đã hoàn thành hoặc đang trên đường quay về trả/hủy nhiệm vụ:
            if ctx.extra.get("combat_objective_complete") or ctx.extra.get("returning_quest") or ctx.extra.get("abandon_combat_quest"):
                logger.info(
                    f"[{ctx.account_id}] Mục tiêu đánh quái đã hoàn thành hoặc đang quay về Chủ thương hội..."
                )
                if hasattr(self, "_is_near_commerce_chief") and self._is_near_commerce_chief():
                    logger.info(f"[{ctx.account_id}] Đã ở cạnh Chủ thương hội, chuyển sang đối thoại (MEET_NPC)...")
                    return BotState.MEET_NPC
                return BotState.FIND_COMMERCE_CHIEF

            # 0.05 KIỂM TRA TIMEOUT 10 PHÚT: Nếu quá 10 phút (600s) từ lúc bắt đầu nhiệm vụ đánh quái -> quay về hủy NV
            combat_start_time = ctx.extra.get("combat_quest_start_time")
            if combat_start_time and (time.time() - combat_start_time >= 600):
                self._in_combat = False
                elapsed_min = int((time.time() - combat_start_time) / 60)
                logger.warning(
                    f"[{ctx.account_id}] ❌ Nhiệm vụ đánh quái vượt quá 10 phút ({elapsed_min} phút); "
                    "quay về Chủ thương hội hủy nhiệm vụ."
                )
                activity._write(
                    "❌",
                    f"[{ctx.account_id}] NV danh quai qua 10 phut ({elapsed_min}p) -> Quay ve Chu thuong hoi huy NV.",
                )
                ctx.extra["abandon_combat_quest"] = True
                ctx.extra["abandon_combat_quest_reason"] = f"Quá 10 phút chưa hoàn thành ({elapsed_min} phút)"
                ctx.extra["returning_quest"] = True
                ctx.extra.pop("chief_nav_started", None)
                navigation_started = self._return_to_nearest_city_commerce_chief(ctx)
                if navigation_started:
                    ctx.extra["chief_nav_started"] = True
                self._checkpoint.save(ctx)
                return BotState.RETURN_QUEST

            # 0.1 Kiểm tra banner hoàn thành trên màn hình nếu có (phục hồi khi bot vừa khởi động lại hoặc sau khi diệt quái)
            chk_frame = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
            if chk_frame is not None and hasattr(self, "_text") and self._text:
                screen_boxes = self._text.read(chk_frame, (200, 300, 1400, 800))
                screen_text = " ".join(sb.text for sb in screen_boxes)
                sb_norm = normalize_game_text(screen_text)
                if "da tau" not in sb_norm and ("truy sat" in sb_norm or "thuong hoi" in sb_norm or "tieu diet" in sb_norm) and (
                    "da hoan thanh" in sb_norm or "hoan thanh" in sb_norm or "da du" in sb_norm
                ):
                    logger.info(
                        f"[{ctx.account_id}] 🎯 Phát hiện banner nhiệm vụ đã hoàn thành trên màn hình ('{screen_text}'), "
                        "đánh dấu hoàn tất và quay về Chủ thương hội trả nhiệm vụ!"
                    )
                    ctx.extra["combat_objective_complete"] = True
                    ctx.extra["returning_quest"] = True
                    if hasattr(self, "_is_near_commerce_chief") and self._is_near_commerce_chief():
                        return BotState.MEET_NPC
                    return BotState.FIND_COMMERCE_CHIEF

            detail_dict = getattr(ctx, "current_quest_detail", {})
            if not isinstance(detail_dict, dict):
                detail_dict = {}
            combat_map = ctx.extra.get("combat_map", "") or detail_dict.get("map_name", "") or detail_dict.get("map", "")
            monster = ctx.extra.get("combat_monster", "") or detail_dict.get("monster", "")
            count = ctx.extra.get("combat_count", 0) or detail_dict.get("count", 0) or 0

            cur_city, _ = self._get_current_location_from_hud()
            major_city_names = [c["name"] for c in self._loader.load_ui_coords().get("commerce_quest", {}).get("major_cities", [])] or [
                "Tuyền Châu", "Biện Kinh", "Thành Đô", "Tương Dương", "Dương Châu", "Đại Lý", "Phượng Tường", "Lâm An"
            ]
            is_outside_major_cities = bool(cur_city and not any(fuzzy_score(cur_city, mc) >= 0.65 for mc in major_city_names))

            try:
                valid_count = int(count) > 0
            except (TypeError, ValueError):
                valid_count = False

            # An toàn: nếu nhận diện được tên quái và map quái rõ ràng nhưng thiếu số lượng,
            # mặc định 10 con thay vì hủy nhiệm vụ và chạy bộ ngược về Chủ thương hội.
            if not valid_count and monster and combat_map:
                logger.warning(
                    f"[{ctx.account_id}] ⚠️ Nhiệm vụ đánh quái ({monster} tại {combat_map}) thiếu số lượng cụ thể, "
                    "mặc định mục tiêu là 10 con thay vì hủy nhiệm vụ."
                )
                count = 10
                ctx.extra["combat_count"] = 10
                valid_count = True

            if not monster or not combat_map or not valid_count:
                missing = []
                if not monster:
                    missing.append("tên quái")
                if not combat_map:
                    missing.append("địa điểm")
                if not valid_count:
                    missing.append("số lượng")
                reason = ", ".join(missing)
                logger.warning(
                    f"[{ctx.account_id}] Nhiệm vụ đánh quái thiếu {reason}; hủy nhiệm vụ, không suy đoán dữ liệu."
                )
                ctx.extra["abandon_combat_quest"] = True
                ctx.extra["abandon_combat_quest_reason"] = reason
                ctx.extra["returning_quest"] = True
                self._checkpoint.save(ctx)
                return BotState.FIND_COMMERCE_CHIEF

            # Đảm bảo nhân vật đã thực sự di chuyển đến combat_map
            norm_cur = normalize_game_text(cur_city) if cur_city else ""
            norm_target = normalize_game_text(combat_map) if combat_map else ""
            sect_name = ctx.extra.get("combat_sect", "") if hasattr(ctx, "extra") else ""
            norm_sect = normalize_game_text(sect_name) if sect_name else ""

            if not combat_map:
                is_at_target_map = is_outside_major_cities
            else:
                is_at_target_map = bool(
                    cur_city
                    and self._is_same_combat_map(cur_city, combat_map, sect=sect_name)
                )

            if combat_map and not is_at_target_map:
                if not cur_city:
                    time.sleep(0.5)
                    cur_city, _ = self._get_current_location_from_hud()
                    if cur_city:
                        is_at_target_map = bool(self._is_same_combat_map(cur_city, combat_map, sect=sect_name))

            if combat_map and not is_at_target_map:
                logger.info(
                    f"[{ctx.account_id}] Nhân vật đang ở map '{cur_city or 'Chưa xác định'}', chưa tới map quái '{combat_map}' -> "
                    f"Quay về Chủ thương hội (Tuyền Châu) để nhờ đưa đến đúng map quái..."
                )
                activity._write("🏃", f"[{ctx.account_id}] Tu {cur_city or 'ngoai'} -> Ve Chu thuong hoi Tuyen Chau...")
                nav_ok = self._navigate_to_combat_map(ctx, combat_map)
                time.sleep(2.0)
                cur_check, _ = self._get_current_location_from_hud()
                check_matched = bool(
                    cur_check
                    and self._is_same_combat_map(cur_check, combat_map, sect=sect_name)
                )
                if not nav_ok or (cur_check and not check_matched):
                    logger.warning(
                        f"[{ctx.account_id}] ⚠️ Chưa thể chuyển sang map quái '{combat_map}' (hiện tại: '{cur_check or cur_city}', nav_ok={nav_ok})! "
                        f"Tuyệt đối không tuần tra ở map ngoài/thành thị. Thử lại..."
                    )
                    return BotState.NAVIGATE

            logger.info(f"[{ctx.account_id}] Bắt đầu làm NV đánh quái: {monster} tại {combat_map} (yêu cầu {count} con)...")
            self._in_combat = True

            # Chờ ngắn để nhân vật load map hoàn tất
            time.sleep(0.2)

            # Đảm bảo Bảng nhiệm vụ đã đóng hoàn toàn trước khi vào chiến đấu (chỉ kiểm tra nếu trước đó từng mở)
            if ctx.extra.get("quest_panel_opened") and hasattr(self, "_ensure_quest_panel_closed"):
                self._ensure_quest_panel_closed(ctx=ctx)

            # Đảm bảo bảng chiêu thức võ công hiển thị (không bị menu che)
            self._ensure_combat_skills_visible()

            # Lấy cấu hình combat & tuần tra
            coords = self._loader.load_ui_coords()
            combat_cfg = coords.get("combat", {})
            combat_scan_interval = float(combat_cfg.get("combat_scan_interval", 0.05) or 0.05)
            combat_scan_interval = max(0.04, min(0.8, combat_scan_interval))
            target_attack_window = float(combat_cfg.get("target_attack_window", 2.0) or 2.0)
            target_attack_window = max(1.5, min(6.0, target_attack_window))
            auto_btn_cfg = combat_cfg.get("auto_button", {"x": 1306, "y": 240})
            auto_btn = (auto_btn_cfg["x"], auto_btn_cfg["y"])
            attack_btn_cfg = combat_cfg.get("attack_button")
            attack_btn = (attack_btn_cfg["x"], attack_btn_cfg["y"]) if (attack_btn_cfg and isinstance(attack_btn_cfg, dict) and "x" in attack_btn_cfg) else None
            patrol_points = combat_cfg.get("patrol_ground_points") or combat_cfg.get("patrol_minimap_points", [
                {"x": 1150, "y": 550, "name": "Đông - Sâu trong hang", "dx": 130, "dy": 40},
                {"x": 1100, "y": 620, "name": "Đông Nam - Khoảng trống", "dx": 120, "dy": 80},
                {"x": 800,  "y": 680, "name": "Nam - Thoát vách đá", "dx": 0, "dy": 120},
                {"x": 500,  "y": 620, "name": "Tây Nam - Cửa động", "dx": -120, "dy": 80},
                {"x": 480,  "y": 540, "name": "Tây - Dọc hành lang", "dx": -130, "dy": 40},
            ])
            # Trong map mê cung hoặc hang động, loại bỏ điểm tuần tra chạy về Cửa động/Portal để không bị chuyển nhầm sang tầng khác
            if any(k in normalize_game_text(combat_map or "") for k in ("me cung", "dong", "tang")):
                safe_patrol = [p for p in patrol_points if not any(w in normalize_game_text(p.get("name", "")) for w in ("cua dong", "cua hang", "loi ra", "cua", "portal"))]
                if safe_patrol:
                    patrol_points = safe_patrol

            def direction_sector(dx: int, dy: int) -> str:
                sx = 0 if abs(dx) < 25 else (1 if dx > 0 else -1)
                sy = 0 if abs(dy) < 25 else (1 if dy > 0 else -1)
                return f"{sx},{sy}"

            avoided_directions = set(ctx.extra.get("combat_avoided_directions", []))
            if avoided_directions:
                alternate_patrol = [
                    p for p in patrol_points
                    if direction_sector(int(p.get("dx", 0)), int(p.get("dy", 0))) not in avoided_directions
                ]
                if alternate_patrol:
                    logger.info(
                        f"[{ctx.account_id}] Tránh các hướng từng làm văng khỏi map: "
                        f"{sorted(avoided_directions)}; dùng {len(alternate_patrol)} hướng tuần tra khác."
                    )
                    patrol_points = alternate_patrol
            joy_cfg = combat_cfg.get("joystick_center", {"x": 220, "y": 680})
            joy_x, joy_y = joy_cfg.get("x", 220), joy_cfg.get("y", 680)

            target_hud_roi = combat_cfg.get("target_hud_region", {"x1": 450, "y1": 60, "x2": 1100, "y2": 130})
            hud_roi = (target_hud_roi["x1"], target_hud_roi["y1"], target_hud_roi["x2"], target_hud_roi["y2"])
            tracker_roi_cfg = combat_cfg.get("left_tracker_region", {"x1": 50, "y1": 250, "x2": 350, "y2": 500})
            tracker_roi = (tracker_roi_cfg["x1"], tracker_roi_cfg["y1"], tracker_roi_cfg["x2"], tracker_roi_cfg["y2"])

            # 1. Chuẩn bị trước khi tuần tra
            init_frame = self._screen.capture()
            if init_frame is not None:
                # Thu gọn thanh chat dưới đáy màn hình để tránh bấm nhầm khi tuần tra/đánh quái
                self._collapse_chat_bar(init_frame)
                # Đảm bảo nhân vật xuống ngựa khi vào map đánh quái
                self._ensure_dismounted_from_horse(ctx, init_frame)
                # Đảm bảo cụm phím kỹ năng hiển thị (không bị menu tính năng che)
                self._ensure_combat_skills_visible(init_frame)

            # 1.1 Thoát cổng chuyển map: Nếu vừa bị văng sang map khác ở lần trước, ngay khi
            # vừa dịch chuyển tới vị trí này, lập tức chạy theo hướng ngược lại để rời xa cổng!
            rev_vector = ctx.extra.pop("combat_reverse_vector", None)
            if rev_vector:
                rev_dx, rev_dy = rev_vector
                logger.warning(
                    f"[{ctx.account_id}] ⚠️ Vừa dịch chuyển vào lại map quái sau khi bị văng map! "
                    f"Lập tức di chuyển theo hướng ngược lại ({rev_dx}, {rev_dy}) để rời xa cổng chuyển map..."
                )
                activity._write("🔄", f"[{ctx.account_id}] Vao lai map, chay nguoc lai ({rev_dx},{rev_dy}) tranh cong...")
                try:
                    self._input.swipe(joy_x, joy_y, joy_x + rev_dx, joy_y + rev_dy, duration=2000)
                    time.sleep(0.3)
                    w_init = init_frame.shape[1] if init_frame is not None else 1600
                    h_init = init_frame.shape[0] if init_frame is not None else 900
                    char_x, char_y = int(w_init * 0.50), int(h_init * 0.53)
                    safe_x, safe_y = self.clamp_to_safe_combat_ground(char_x + rev_dx, char_y + rev_dy, w_init, h_init)
                    self._input.tap(safe_x, safe_y)
                    time.sleep(0.5)
                except Exception as ex:
                    logger.debug(f"Reverse vector escape error: {ex}")

            # 2. Vòng lặp tuần tra map và diệt quái cho đến khi đủ số lượng
            max_duration = max(30, min(300, count * 15))
            start_combat = time.time()
            ctx.extra.setdefault("combat_quest_start_time", time.time())
            # HUD can keep showing a dead target. Only trust it briefly after
            # this bot has actually issued a right-click attack.
            last_target_seen = 0.0
            last_map_check_time = time.time()
            last_obstruction_check_time = time.time()
            last_completion_check_time = 0.0
            last_popup_check_time = time.time()
            patrol_idx = 0
            patrol_step_count = 0
            last_30s_combat_check = time.time()
            coords_30s_combat_ago = None
            complete_found = False
            is_auto_active = False
            is_targeting_monster = False
            last_coords: Optional[tuple[int, int]] = None
            stuck_count = 0
            last_patrol_dx = int(ctx.extra.get("combat_last_patrol_dx", 0))
            last_patrol_dy = int(ctx.extra.get("combat_last_patrol_dy", 0))
            coord_history: list[tuple[int, int]] = []
            escape_dir_idx = 0
            ESCAPE_DIRECTIONS = [
                ("Đông Nam (Thoát vách đá)", 130, 130, 1150, 680),
                ("Tây (Chạy dạt sang trái)", -150, 0, 460, 480),
                ("Tây Nam (Thoát xuống góc trái)", -140, 110, 480, 680),
                ("Tây Bắc (Thoát ngược lên trái)", -140, -110, 480, 250),
                ("Bắc (Chạy ngược lên trên)", 0, -140, 800, 220),
                ("Đông Bắc (Chạy lên góc phải)", 140, -110, 1150, 250),
                ("Nam (Chạy thẳng xuống dưới)", 0, 140, 800, 700),
                ("Đông (Chạy thẳng sang phải)", 150, 0, 1150, 480),
            ]
            alternate_escape = [
                d for d in ESCAPE_DIRECTIONS
                if direction_sector(d[1], d[2]) not in avoided_directions
            ]
            if alternate_escape:
                ESCAPE_DIRECTIONS = alternate_escape
            blocked_monsters: list[dict] = []
            wall_stuck_target_count = 0
            last_targeted_monster_pos: Optional[tuple[int, int]] = None
            repeated_unconfirmed_target_count = 0
            wall_follow_idx = 0
            WALL_FOLLOW_VECTORS = [
                ("Đông Nam (Dọc bờ tường / hành lang)", 140, 120, 0.70, 0.75),
                ("Tây Bắc (Dọc bờ tường / hành lang)", -140, -120, 0.30, 0.28),
                ("Tây Nam (Dọc bờ tường / hành lang)", -140, 120, 0.30, 0.75),
                ("Đông Bắc (Dọc bờ tường / hành lang)", 140, -120, 0.70, 0.28),
            ]
            logger.info(f"[{ctx.account_id}] Bắt đầu chạy tuần tra tìm quái '{monster}' (tối đa {max_duration}s)...")

            dismount_attempts = 0
            while time.time() - start_combat < max_duration:
                time.sleep(combat_scan_interval)
                frame = self._screen.capture()
                if frame is None:
                    continue

                # KIỂM TRA TIMEOUT 10 PHÚT: Nếu quá 10 phút (600s) chưa hoàn thành NV đánh quái -> quay về hủy NV
                combat_start_time = ctx.extra.get("combat_quest_start_time", start_combat)
                if time.time() - combat_start_time >= 600:
                    self._in_combat = False
                    elapsed_min = int((time.time() - combat_start_time) / 60)
                    logger.warning(
                        f"[{ctx.account_id}] ❌ Nhiệm vụ đánh quái vượt quá 10 phút ({elapsed_min} phút); "
                        "quay về Chủ thương hội hủy nhiệm vụ."
                    )
                    activity._write(
                        "❌",
                        f"[{ctx.account_id}] NV danh quai qua 10 phut ({elapsed_min}p) -> Quay ve Chu thuong hoi huy NV.",
                    )
                    ctx.extra["abandon_combat_quest"] = True
                    ctx.extra["abandon_combat_quest_reason"] = f"Quá 10 phút chưa hoàn thành ({elapsed_min} phút)"
                    ctx.extra["returning_quest"] = True
                    ctx.extra.pop("chief_nav_started", None)
                    navigation_started = self._return_to_nearest_city_commerce_chief(ctx)
                    if navigation_started:
                        ctx.extra["chief_nav_started"] = True
                    self._checkpoint.save(ctx)
                    return BotState.RETURN_QUEST

                # Đảm bảo Bảng nhiệm vụ hoặc Bản đồ lớn không che khuất màn hình thực chiến
                if time.time() - last_obstruction_check_time >= 10.0:
                    last_obstruction_check_time = time.time()
                    if hasattr(self, "_is_map_open") and self._is_map_open(frame):
                        logger.info(f"[{ctx.account_id}] Phát hiện Bảng bản đồ lớn đang mở che màn hình chiến đấu -> Đóng ngay...")
                        self._close_map()
                        time.sleep(0.3)
                        continue
                # =========================================================================
                # 0. KIỂM TRA HOÀN THÀNH NHIỆM VỤ (ƯU TIÊN HÀNG ĐẦU - TRƯỚC TẤT CẢ CÁC BƯỚC KHÁC)
                # =========================================================================
                now_t = time.time()
                # Chỉ kiểm tra hoàn thành định kỳ 3s và sau khi đã vào map > 5s (tránh nghẽn 5s OCR ngay khi vừa vào map)
                if now_t - start_combat >= 5.0 and now_t - last_completion_check_time >= 3.0:
                    last_completion_check_time = now_t
                    # A. Banner hoàn thành nằm khá thấp (khoảng y=680..720 trên
                    # 1600x900). Ghép toàn bộ mảnh OCR vì recognizer thường chia
                    # câu "Nhiệm vụ thương hội ... đã hoàn thành" thành nhiều box.
                    banner_boxes = self._text.read(frame, (200, 380, 1400, 800))
                    banner_text = " ".join(b.text for b in banner_boxes)
                    banner_norm = normalize_game_text(banner_text)
                    monster_norm = normalize_game_text(monster) if monster else ""
                    completion_words = any(w in banner_norm for w in (
                        "hoan thanh", "da hoan", "hoan tat", "da xong", "da du",
                    ))
                    combat_context = any(w in banner_norm for w in (
                        "nhiem vu thuong hoi", "thuong hoi", "truy sat",
                        "tieu diet", "diet quai",
                    )) or (monster_norm and fuzzy_score(banner_norm, monster_norm) >= 0.45)
                    if (
                        f"{count}/{count}" in banner_text
                        or (completion_words and combat_context)
                    ):
                        logger.info(
                            f"[{ctx.account_id}] 🎯 Phát hiện banner đã đánh đủ quái: "
                            f"'{banner_text}' -> Quay về Chủ thương hội trả nhiệm vụ ngay!"
                        )
                        activity._write(
                            "✅",
                            f"[{ctx.account_id}] Da danh du {count} {monster}. Quay ve tra nhiem vu!",
                        )
                        complete_found = True

                    # B. Kiểm tra thanh tracker nhiệm vụ bên trái
                    if not complete_found:
                        tracker_boxes = self._text.read(frame, tracker_roi)
                        tracker_text = " ".join(b.text for b in tracker_boxes)
                        norm_tracker = normalize_game_text(tracker_text)
                        if "thuong hoi" in norm_tracker:
                            if f"{count}/{count}" in tracker_text or any(w in norm_tracker for w in ("da hoan thanh", "tieu diet xong", f"{count} {count}")):
                                complete_found = True

                        # Tìm tiến độ X/Y gắn liền với Thương Hội hoặc quái mục tiêu
                        progress_m = re.search(r"(\d+)\s*[/]\s*(\d+)", tracker_text)
                        if progress_m:
                            cur_killed = int(progress_m.group(1))
                            req_killed = int(progress_m.group(2))
                            if req_killed == count and cur_killed >= req_killed:
                                if "thuong hoi" in norm_tracker or any(w in norm_tracker for w in ("tieu diet", "diet", "quai")) or (monster and normalize_game_text(monster) in norm_tracker):
                                    logger.debug(f"[{ctx.account_id}] Tiến độ diệt quái: {cur_killed}/{req_killed}")
                                    complete_found = True

                if complete_found:
                    logger.info(f"[{ctx.account_id}] ✅ Đã đánh đủ {count} {monster}! Lập tức quay về Chủ thương hội trả nhiệm vụ...")
                    activity._write("⚔️", f"[{ctx.account_id}] Da danh du {count} {monster}! Quay ve tra nhiem vu.")
                    break

                # Dọn dẹp danh sách quái kẹt tường đã hết hạn 15s
                blocked_monsters = [b for b in blocked_monsters if b.get("expires_at", 0) > now_t]

                # 0.1 Đảm bảo cụm phím kỹ năng luôn hiển thị (không bị menu tính năng che)
                self._ensure_combat_skills_visible(frame)

                # 0.3 Tự động đóng các popup/hộp thoại bất thường che màn hình (định kỳ 30s hoặc khi bị kẹt, tránh quét OCR toàn màn hình mỗi khung hình)
                if now_t - last_popup_check_time >= 30.0:
                    last_popup_check_time = now_t
                    if self._dismiss_unexpected_popups(frame):
                        time.sleep(0.3)
                        continue

                # 0.4 Đảm bảo bản đồ lớn KHÔNG bị mở trong lúc đang tuần tra tìm quái (nếu đang mở thì đóng ngay)
                if hasattr(self, "_is_map_open") and self._is_map_open(frame):
                    logger.info(f"[{ctx.account_id}] 🛑 Phát hiện bảng bản đồ đang mở trong lúc tìm quái -> Đóng bản đồ ngay...")
                    self._close_map()
                    time.sleep(0.3)
                    continue

                # 0.5 Định kỳ 5s kiểm tra nếu nhân vật bị văng ra khỏi map quái (về thành, chết hồi sinh ở Nga My, chạy ra cửa động, v.v.)
                if combat_map and (now_t - last_map_check_time >= 5.0):
                    last_map_check_time = now_t
                    cur_map, _ = self._get_current_location_from_hud(frame)
                    if cur_map:
                        is_target = self._is_same_combat_map(cur_map, combat_map, sect=ctx.extra.get("combat_sect"))
                        if not is_target:
                            norm_cur = normalize_game_text(cur_map)
                            cur_floor_m = re.search(r"(?:tang\s*(\d+)|me\s*cung\s*(\d+)|\b(\d+)\b)", norm_cur)
                            cur_floor = (cur_floor_m.group(1) or cur_floor_m.group(2) or cur_floor_m.group(3)) if cur_floor_m else None
                            target_floor_m = re.search(r"(?:tang\s*(\d+)|me\s*cung\s*(\d+)|\b(\d+)\b)", normalize_game_text(combat_map or ""))
                            target_floor = (target_floor_m.group(1) or target_floor_m.group(2) or target_floor_m.group(3)) if target_floor_m else None
                            is_wrong_floor = bool(target_floor and cur_floor and target_floor != cur_floor)

                            known_external = ["tuyen chau", "bien kinh", "thanh do", "tuong duong", "duong chau", "dai ly", "phuong tuong", "lam an"]
                            is_external_map = any(ext in norm_cur or fuzzy_score(norm_cur, ext) >= 0.65 for ext in known_external)

                            # Nếu chưa có combat_map mà đang ở ngoài thành lớn (như động, hang): tự động nhận diện map hiện tại
                            if not combat_map and not is_external_map:
                                combat_map = cur_map
                                ctx.extra["combat_map"] = combat_map
                                logger.info(f"[{ctx.account_id}] Đang trong map quái '{combat_map}', tự động gán combat_map!")
                                continue

                            # HUD đã xác nhận một map khác map nhiệm vụ: ghi nhớ
                            # ngay hướng tuần tra vừa dùng, kể cả map mới cũng là
                            # map quái (trước đây chỉ xử lý sai tầng/thành lớn).
                            if combat_map:
                                failed_direction = ctx.extra.get("combat_last_move_direction")
                                failed_dx = int(ctx.extra.get("combat_last_patrol_dx", 0))
                                failed_dy = int(ctx.extra.get("combat_last_patrol_dy", 0))
                                if failed_dx == 0 and failed_dy == 0 and failed_direction:
                                    try:
                                        sx_s, sy_s = failed_direction.split(",")
                                        failed_dx = int(sx_s) * 140
                                        failed_dy = int(sy_s) * 140
                                    except Exception:
                                        pass

                                if failed_dx != 0 or failed_dy != 0:
                                    rev_dx = -failed_dx
                                    rev_dy = -failed_dy
                                    ctx.extra["combat_reverse_vector"] = (rev_dx, rev_dy)
                                    logger.warning(
                                        f"[{ctx.account_id}] ⚠️ Văng khỏi map khi di chuyển vector ({failed_dx}, {failed_dy})! "
                                        f"Ghi nhận vector ngược lại ({rev_dx}, {rev_dy}) để lập tức thoát cổng khi vào lại map."
                                    )

                                if failed_direction:
                                    avoided = list(ctx.extra.get("combat_avoided_directions", []))
                                    if failed_direction not in avoided:
                                        avoided.append(failed_direction)
                                    # Giữ tối đa 4 hướng gần nhất để luôn còn hướng thay thế.
                                    ctx.extra["combat_avoided_directions"] = avoided[-4:]
                                    logger.warning(
                                        f"[{ctx.account_id}] Ghi nhớ hướng '{failed_direction}' đã làm nhân vật "
                                        "rời map; lần tuần tra sau sẽ không đi hướng này nữa."
                                    )
                                    ctx.extra.pop("combat_last_move_direction", None)
                                    if hasattr(self, "_checkpoint") and self._checkpoint:
                                        self._checkpoint.save(ctx)
                                if is_wrong_floor:
                                    reason = f"sai tầng (đang ở '{cur_map}', yêu cầu '{combat_map}')"
                                elif is_external_map:
                                    reason = f"bị văng về thành ('{cur_map}' != '{combat_map}')"
                                else:
                                    reason = f"đi nhầm sang map khác ('{cur_map}' != '{combat_map}')"
                                logger.warning(
                                    f"[{ctx.account_id}] ⚠️ Nhân vật đang ở {reason}! "
                                    f"Tiến hành quay về Chủ thương hội để nhờ đưa lại tới đúng map quái..."
                                )
                                activity._write("⚠️", f"[{ctx.account_id}] {reason} -> Ve Chu thuong hoi...")
                                nav_ok = self._navigate_to_combat_map(ctx, combat_map)
                                time.sleep(2.0)
                                last_map_check_time = time.time()
                                if not nav_ok:
                                    logger.warning(f"[{ctx.account_id}] ⚠️ Chưa thể chuyển sang map quái '{combat_map}' -> Dừng tuần tra để quay lại tìm đường...")
                                    return BotState.NAVIGATE
                                # Khởi tạo lại vòng combat để danh sách hướng tuần tra
                                # được lọc theo hướng vừa làm nhân vật văng khỏi map.
                                return BotState.NAVIGATE
                            else:
                                logger.debug(f"[{ctx.account_id}] Đang ở map '{cur_map}', tiếp tục tuần tra đánh quái...")
                        else:
                            logger.debug(f"[{ctx.account_id}] Định kỳ: Đang trong map làm nhiệm vụ '{cur_map}'. Tiếp tục đánh quái...")

                # C. Kiểm tra thanh mục tiêu. Không đọc hoặc thao tác nút Auto;
                # cơ chế chiến đấu chỉ dùng chuột phải vào quái.
                h_f, w_f = frame.shape[:2]
                is_auto_active = False

                hud_boxes = self._text.read(frame, hud_roi)
                is_targeting_monster = False
                if monster:
                    norm_monster = normalize_game_text(monster)
                    for b in hud_boxes:
                        b_norm = normalize_game_text(b.text)
                        # Loại bỏ các nút icon UI sát HUD (Phúc lợi, Hạng, Ngự các, Luyện công, Mở khóa)
                        if any(ui_word in b_norm for ui_word in ("phuc loi", "hang", "ngu cac", "luyen cong", "mo khoa", "choiqua")):
                            continue
                        if norm_monster in b_norm or (len(b_norm) >= 3 and b_norm in norm_monster) or (len(b_norm) >= 3 and fuzzy_score(b_norm, norm_monster) >= 0.70):
                            is_targeting_monster = True
                            break
                if not is_targeting_monster and not monster:
                    is_targeting_monster = any(w in normalize_game_text(b.text) for b in hud_boxes for w in ("hp", "lv", "quai"))

                # D. Kiểm tra xem có quái xuất hiện trên màn hình không
                try:
                    monster_pos = self._find_monster_on_screen(frame, monster, blocked_positions=blocked_monsters)
                except TypeError:
                    monster_pos = self._find_monster_on_screen(frame, monster)
                if monster_pos:
                    char_x, char_y = int(w_f * 0.50), int(h_f * 0.53)
                    mx, my = monster_pos
                    dist = math.hypot(mx - char_x, my - char_y)

                    # Kiểm tra tọa độ xem có bị kẹt bờ tường khi tiếp cận con quái này không
                    _, cur_coords = self._get_current_location_from_hud(frame)
                    is_same_target = (
                        last_targeted_monster_pos is not None
                        and math.hypot(mx - last_targeted_monster_pos[0], my - last_targeted_monster_pos[1]) < 70
                    )
                    if is_same_target and not is_targeting_monster and dist <= 100:
                        repeated_unconfirmed_target_count += 1
                    elif not is_same_target or dist > 100:
                        repeated_unconfirmed_target_count = 0

                    # Một blob xanh giả thường đứng nguyên tại cùng một điểm
                    # trong khi HUD không hề có mục tiêu. Không để điểm đó giữ
                    # nhân vật đứng yên và click vô hạn; tạm chặn rồi tuần tra.
                    if repeated_unconfirmed_target_count >= 2:
                        logger.warning(
                            f"[{ctx.account_id}] Điểm xanh ({mx}, {my}) lặp lại nhưng HUD không có "
                            f"mục tiêu '{monster}' -> coi là nhận nhầm, bỏ qua 20s và tiếp tục tuần tra."
                        )
                        blocked_monsters.append({"pos": (mx, my), "expires_at": time.time() + 20.0})
                        repeated_unconfirmed_target_count = 0
                        last_targeted_monster_pos = None
                        patrol_idx = (patrol_idx + 1) % len(patrol_points)
                        recovery_pt = patrol_points[patrol_idx]
                        ctx.extra["combat_last_move_direction"] = direction_sector(
                            int(recovery_pt.get("dx", 0)), int(recovery_pt.get("dy", 0))
                        )
                        self._move_combat_patrol(recovery_pt, w_f, h_f)
                        time.sleep(0.15)
                        continue
                    if is_same_target and cur_coords and last_coords:
                        dist_moved = math.hypot(cur_coords[0] - last_coords[0], cur_coords[1] - last_coords[1])
                        # Quái ở gần (dist <= 220px) là cự ly áp sát cận chiến, tuyệt đối không coi là kẹt tường.
                        # Chỉ đếm kẹt tường khi quái ở xa (> 220px), tọa độ không đổi và chưa khóa được mục tiêu trên HUD.
                        if dist_moved <= 1.0 and dist > 220 and not is_targeting_monster:
                            wall_stuck_target_count += 1
                        else:
                            wall_stuck_target_count = 0
                    elif not is_same_target:
                        wall_stuck_target_count = 0

                    if cur_coords:
                        last_coords = cur_coords

                    # Chỉ coi là kẹt bờ tường khi quái ở xa (> 220px) và bị đứng yên liên tiếp >= 2 chu kỳ
                    if wall_stuck_target_count >= 2 and dist > 220 and not is_targeting_monster:
                        wf_name, wf_dx, wf_dy, wf_rx, wf_ry = WALL_FOLLOW_VECTORS[wall_follow_idx % len(WALL_FOLLOW_VECTORS)]
                        wall_follow_idx += 1
                        logger.warning(
                            f"[{ctx.account_id}] ⚠️ Quái '{monster}' tại ({mx}, {my}) bị kẹt sau bờ tường "
                            f"(khoảng cách {int(dist)}px, tọa độ {cur_coords} không đổi sau {wall_stuck_target_count} chu kỳ)! "
                            f"Tạm bỏ qua quái này trong 30s và đi dọc theo bờ tường hướng {wf_name} để tìm đường/quái tiếp..."
                        )
                        activity._write("🧱", f"[{ctx.account_id}] Quai bi ket tuong tai {cur_coords}! Di doc bo tuong...")
                        # Thêm vào danh sách tạm bỏ qua
                        blocked_monsters.append({"pos": (mx, my), "expires_at": time.time() + 30.0})
                        wall_stuck_target_count = 0
                        last_targeted_monster_pos = None

                        # 3. Đi dọc bờ tường: dùng joystick vuốt dứt khoát và tap điểm xa dọc hành lang
                        try:
                            hold_ms = int(combat_cfg.get("wall_follow_hold_ms", 3000) or 3000)
                            hold_ms = max(500, min(5000, hold_ms))
                            self._input.swipe(
                                joy_x, joy_y, joy_x + wf_dx, joy_y + wf_dy,
                                duration=hold_ms,
                            )
                            ctx.extra["combat_last_move_direction"] = direction_sector(wf_dx, wf_dy)
                        except Exception as e:
                            logger.debug(f"Wall follow error: {e}")

                        time.sleep(0.15)
                        continue

                    last_targeted_monster_pos = (mx, my)

                    # Đảm bảo nhân vật đã xuống ngựa để có thể xuất chiêu
                    self._ensure_dismounted_from_horse(ctx, frame)

                    # Đảm bảo bảng kỹ năng đang hiển thị (không bị menu tính năng che)
                    self._ensure_combat_skills_visible(frame)

                    # Khi phát hiện quái trên màn hình:
                    # Gửi tap trực tiếp vào thân quái để Android nhận diện chạm mục tiêu, đồng thời gửi chuột phải
                    logger.info(f"[{ctx.account_id}] ⚔️ Nhìn thấy quái '{monster}' tại thân quái ({mx}, {my}) (khoảng cách {int(dist)}px) -> Tap chọn quái và tấn công...")
                    self._input.tap(mx, my)
                    if hasattr(self._input, "right_click"):
                        self._input.right_click(mx, my, count=1)
                    else:
                        self._input.double_tap(mx, my)
                    time.sleep(0.15)

                    # Bật Auto tự đánh tại auto_button để nhân vật xuất chiêu combo tiêu diệt quái
                    if auto_btn and not ctx.extra.get("combat_auto_active"):
                        logger.info(f"[{ctx.account_id}] ⚔️ Bật Auto tự đánh tại {auto_btn} để nhân vật xuất chiêu...")
                        self._input.tap(auto_btn[0], auto_btn[1])
                        ctx.extra["combat_auto_active"] = True

                    time.sleep(0.1)
                    last_target_seen = time.time()
                    patrol_step_count = 0

                elif is_targeting_monster and (time.time() - last_target_seen) < 4.0:
                    last_targeted_monster_pos = None
                    wall_stuck_target_count = 0
                    patrol_step_count = 0
                    # Đã khóa mục tiêu: nếu Auto chưa bật thì bật để tiếp tục xuất chiêu
                    if auto_btn and not ctx.extra.get("combat_auto_active"):
                        logger.info(f"[{ctx.account_id}] ⚔️ Đang khóa mục tiêu quái, bật Auto tự đánh tại {auto_btn}...")
                        self._input.tap(auto_btn[0], auto_btn[1])
                        ctx.extra["combat_auto_active"] = True

                else:
                    last_targeted_monster_pos = None
                    wall_stuck_target_count = 0
                    # Nếu đang bật Auto mà không có quái xung quanh thì tắt Auto để joystick tuần tra mượt mà
                    if auto_btn and ctx.extra.get("combat_auto_active"):
                        logger.info(f"[{ctx.account_id}] Không còn quái trong tầm, tắt Auto tại {auto_btn} để tuần tra...")
                        self._input.tap(auto_btn[0], auto_btn[1])
                        ctx.extra["combat_auto_active"] = False

                    # E. Không tìm thấy quái cần đánh:
                    # Giữ nguyên Auto để hệ thống tự động tìm quái trong tầm, không tắt Auto sau mỗi 3s

                    # Kiểm tra tọa độ hiện tại trên HUD để phát hiện kẹt tường / chạy lắc qua lắc lại tại 1 chỗ
                    _, cur_coords = self._get_current_location_from_hud(frame)
                    is_oscillating = False
                    if cur_coords:
                        if last_coords:
                            dist_moved = math.hypot(cur_coords[0] - last_coords[0], cur_coords[1] - last_coords[1])
                            if dist_moved <= 1.0:
                                stuck_count += 1
                            else:
                                stuck_count = 0
                        last_coords = cur_coords

                        # Kiểm tra nếu 3 bước quay lại chỗ cũ (chạy lắc qua lắc lại)
                        if len(coord_history) >= 2:
                            for past_c in coord_history[-4:]:
                                if math.hypot(cur_coords[0] - past_c[0], cur_coords[1] - past_c[1]) <= 2.0:
                                    is_oscillating = True
                                    break
                        coord_history.append(cur_coords)
                        if len(coord_history) > 8:
                            coord_history.pop(0)

                        # KIỂM TRA ĐỊNH KỲ 10S: Nếu nhân vật bị đứng yên tại chỗ -> đi lùi hướng ngược lại để thoát góc
                        if coords_30s_combat_ago is None:
                            coords_30s_combat_ago = cur_coords
                            last_30s_combat_check = time.time()
                        elif time.time() - last_30s_combat_check >= 10.0:
                            dist_30s = math.hypot(cur_coords[0] - coords_30s_combat_ago[0], cur_coords[1] - coords_30s_combat_ago[1])
                            if dist_30s <= 1.5 and not is_targeting_monster:
                                # Tính hướng ngược lại với hướng vừa di chuyển vào góc
                                rev_dx = -last_patrol_dx if last_patrol_dx else -100
                                rev_dy = -last_patrol_dy if last_patrol_dy else -100
                                mag = math.hypot(rev_dx, rev_dy)
                                if mag > 0:
                                    sw_dx = int((rev_dx / mag) * 150)
                                    sw_dy = int((rev_dy / mag) * 150)
                                else:
                                    sw_dx, sw_dy = -130, -130
                                logger.warning(
                                    f"[{ctx.account_id}] ⚠️ Đứng yên tại chỗ 10s ở {cur_coords}! "
                                    f"Đi lùi hướng ngược lại ({sw_dx}, {sw_dy}) để thoát khỏi góc..."
                                )
                                activity._write("🔄", f"[{ctx.account_id}] Dung yen 10s, di lui huong nguoc ({sw_dx},{sw_dy}) thoat goc...")
                                try:
                                    ctx.extra["combat_last_move_direction"] = direction_sector(sw_dx, sw_dy)
                                    self._input.swipe(joy_x, joy_y, joy_x + sw_dx, joy_y + sw_dy, duration=800)
                                    time.sleep(0.15)
                                    char_x, char_y = int(w_f * 0.50), int(h_f * 0.53)
                                    safe_gr_x, safe_gr_y = self.clamp_to_safe_combat_ground(char_x + sw_dx, char_y + sw_dy, w_f, h_f)
                                    self._input.tap(safe_gr_x, safe_gr_y)
                                    time.sleep(0.3)
                                except Exception as e:
                                    logger.debug(f"Escape stuck error: {e}")

                                # Kiểm tra xem có bị văng khỏi map quái về thành thị lớn không
                                cur_chk_city, _ = self._get_current_location_from_hud(frame)
                                if cur_chk_city and combat_map:
                                    is_in_major = any(fuzzy_score(cur_chk_city, mc) >= 0.65 for mc in major_city_names)
                                    if is_in_major and not self._is_same_combat_map(cur_chk_city, combat_map, sect=sect_name):
                                        logger.info(f"[{ctx.account_id}] Đã ra ngoài map quái (đang ở thành '{cur_chk_city}'), dịch chuyển lại map quái...")
                                        self._navigate_to_combat_map(ctx, combat_map)

                                # Đánh dấu tránh hướng vừa kẹt
                                fwd_sector = direction_sector(last_patrol_dx, last_patrol_dy)
                                if fwd_sector and fwd_sector != "0,0":
                                    avoided_directions.add(fwd_sector)
                                    ctx.extra["combat_avoided_directions"] = list(avoided_directions)[-4:]

                                # Đổi sang điểm tuần tra không hướng vào góc kẹt
                                valid_next = [
                                    (idx, p) for idx, p in enumerate(patrol_points)
                                    if (int(p.get("dx", 0)) * last_patrol_dx + int(p.get("dy", 0)) * last_patrol_dy) <= 0
                                    and direction_sector(int(p.get("dx", 0)), int(p.get("dy", 0))) not in avoided_directions
                                ]
                                if valid_next:
                                    patrol_idx = valid_next[0][0]
                                else:
                                    patrol_idx = (patrol_idx + 2) % len(patrol_points)

                                stuck_count = 0
                                patrol_step_count = 0
                                coord_history.clear()
                                coords_30s_combat_ago = cur_coords
                                last_30s_combat_check = time.time()
                                continue
                            else:
                                coords_30s_combat_ago = cur_coords
                                last_30s_combat_check = time.time()

                    if stuck_count >= 2 or is_oscillating:
                        reason = "3 bước quay lại chỗ cũ (chạy lắc qua lắc lại)" if is_oscillating else f"{stuck_count} chu kỳ đứng yên"
                        # Ưu tiên đi hướng ngược lại với hướng vừa di chuyển vào góc kẹt
                        rev_dx = -last_patrol_dx if last_patrol_dx else 0
                        rev_dy = -last_patrol_dy if last_patrol_dy else 0
                        mag = math.hypot(rev_dx, rev_dy)
                        if mag > 0:
                            joy_dx = int((rev_dx / mag) * 150)
                            joy_dy = int((rev_dy / mag) * 150)
                            dir_name = f"Ngược hướng kẹt ({joy_dx}, {joy_dy})"
                        else:
                            escape_dir = ESCAPE_DIRECTIONS[escape_dir_idx % len(ESCAPE_DIRECTIONS)]
                            escape_dir_idx += 1
                            dir_name, joy_dx, joy_dy, _, _ = escape_dir

                        logger.warning(
                            f"[{ctx.account_id}] ⚠️ Phát hiện kẹt tại {cur_coords or 'chưa rõ'} ({reason})! "
                            f"Đi lùi hướng ngược lại để thoát góc: {dir_name}..."
                        )
                        activity._write("🔄", f"[{ctx.account_id}] Ket goc ({reason}), di lui ({joy_dx},{joy_dy}) thoat goc...")
                        # Thoát kẹt: Dùng joystick vuốt dứt khoát theo hướng ngược lại
                        try:
                            ctx.extra["combat_last_move_direction"] = direction_sector(joy_dx, joy_dy)
                            self._input.swipe(joy_x, joy_y, joy_x + joy_dx, joy_y + joy_dy, duration=1200)
                            time.sleep(0.15)
                            char_x, char_y = int(w_f * 0.50), int(h_f * 0.53)
                            safe_gr_x, safe_gr_y = self.clamp_to_safe_combat_ground(char_x + joy_dx, char_y + joy_dy, w_f, h_f)
                            self._input.tap(safe_gr_x, safe_gr_y)
                            time.sleep(0.3)
                        except Exception as e:
                            logger.debug(f"Escape stuck error: {e}")

                        # Tránh hướng vừa kẹt
                        fwd_sector = direction_sector(last_patrol_dx, last_patrol_dy)
                        if fwd_sector and fwd_sector != "0,0":
                            avoided_directions.add(fwd_sector)
                            ctx.extra["combat_avoided_directions"] = list(avoided_directions)[-4:]

                        # Chọn điểm tuần tra tiếp theo không hướng vào góc kẹt
                        valid_next = [
                            (idx, p) for idx, p in enumerate(patrol_points)
                            if (int(p.get("dx", 0)) * last_patrol_dx + int(p.get("dy", 0)) * last_patrol_dy) <= 0
                            and direction_sector(int(p.get("dx", 0)), int(p.get("dy", 0))) not in avoided_directions
                        ]
                        if valid_next:
                            patrol_idx = valid_next[0][0]
                        else:
                            patrol_idx = (patrol_idx + 2) % len(patrol_points)

                        stuck_count = 0
                        patrol_step_count = 0
                        coord_history.clear()
                        coords_30s_combat_ago = cur_coords
                        last_30s_combat_check = time.time()
                        continue

                    # Di chuyển dò đường liên tục theo danh sách patrol_points:
                    # Giữ nguyên hướng di chuyển liên tục trong 3 chu kỳ (~2.5s) để nhân vật chạy đường dài mượt mà,
                    # không bị đổi hướng giật cục hoặc ngập ngừng sau mỗi vài bước chân
                    patrol_step_count += 1
                    if patrol_step_count >= 2:
                        patrol_idx = (patrol_idx + 1) % len(patrol_points)
                        patrol_step_count = 0

                    pt = patrol_points[patrol_idx % len(patrol_points)]
                    last_patrol_dx = int(pt.get("dx", 0))
                    last_patrol_dy = int(pt.get("dy", 0))
                    raw_x = pt.get("x", 800)
                    raw_y = pt.get("y", 600)
                    pt_x, pt_y = self.clamp_to_safe_combat_ground(raw_x, raw_y, w_f, h_f)
                    if patrol_step_count <= 1:
                        logger.info(
                            f"[{ctx.account_id}] 🏃 Điều khiển joystick tuần tra tìm quái '{monster}', "
                            f"hướng '{pt.get('name', 'bước tuần tra')}' "
                            f"(vector {last_patrol_dx}/{last_patrol_dy}, điểm dự phòng {pt_x}/{pt_y})..."
                        )
                    ctx.extra["combat_last_move_direction"] = direction_sector(last_patrol_dx, last_patrol_dy)
                    ctx.extra["combat_last_patrol_dx"] = last_patrol_dx
                    ctx.extra["combat_last_patrol_dy"] = last_patrol_dy
                    self._move_combat_patrol(pt, w_f, h_f)
                    patrol_delay = float(combat_cfg.get("patrol_step_delay", 0.05))
                    time.sleep(patrol_delay)

            if not complete_found:
                combat_start_time = ctx.extra.get("combat_quest_start_time", start_combat)
                if time.time() - combat_start_time >= 600:
                    self._in_combat = False
                    if auto_btn and ctx.extra.get("combat_auto_active"):
                        self._input.tap(auto_btn[0], auto_btn[1])
                        ctx.extra["combat_auto_active"] = False
                    elapsed_min = int((time.time() - combat_start_time) / 60)
                    logger.warning(
                        f"[{ctx.account_id}] ❌ Nhiệm vụ đánh quái vượt quá 10 phút ({elapsed_min} phút); "
                        "quay về Chủ thương hội hủy nhiệm vụ."
                    )
                    activity._write(
                        "❌",
                        f"[{ctx.account_id}] NV danh quai qua 10 phut ({elapsed_min}p) -> Quay ve Chu thuong hoi huy NV.",
                    )
                    ctx.extra["abandon_combat_quest"] = True
                    ctx.extra["abandon_combat_quest_reason"] = f"Quá 10 phút chưa hoàn thành ({elapsed_min} phút)"
                    ctx.extra["returning_quest"] = True
                    ctx.extra.pop("chief_nav_started", None)
                    navigation_started = self._return_to_nearest_city_commerce_chief(ctx)
                    if navigation_started:
                        ctx.extra["chief_nav_started"] = True
                    self._checkpoint.save(ctx)
                    return BotState.RETURN_QUEST

                logger.warning(f"[{ctx.account_id}] ⚠️ Chưa diệt đủ số lượng quái yêu cầu! Tiếp tục tuần tra...")
                return BotState.NAVIGATE

            # 4. Quay về trả nhiệm vụ. Tiêu diệt sạch quái cản đường xung quanh nhân vật trước.
            self._in_combat = False
            if auto_btn and ctx.extra.get("combat_auto_active"):
                self._input.tap(auto_btn[0], auto_btn[1])
                ctx.extra["combat_auto_active"] = False
            logger.info(f"[{ctx.account_id}] ✅ Hoàn tất đánh quái! Dọn sạch quái xung quanh trước khi rút lui...")
            self._clear_monsters_around_character(ctx)

            ctx.extra["combat_objective_complete"] = True
            ctx.extra["returning_quest"] = True
            ctx.extra.pop("chief_nav_started", None)
            navigation_started = self._return_to_nearest_city_commerce_chief(ctx)
            if navigation_started:
                ctx.extra["chief_nav_started"] = True
            return BotState.RETURN_QUEST

        # Trường hợp NHIỆM VỤ ĐỐI THOẠI NPC
        # Trước khi chờ di chuyển, đảm bảo đóng hộp thoại cũ nếu còn sót lại
        # Popup Cảnh Báo thường xuất hiện ngay sau khi qua map. Phải
        # đóng nó trước; nếu không OCR không thấy HUD/NPC và NAVIGATE
        # sẽ hết timeout rồi khởi động lại đường đi vô hạn.
        self._dismiss_unexpected_popups()
        self._ensure_npc_dialog_closed()
        target_city = ctx.extra.get("quest_target_city", "")
        target_npc = ctx.extra.get("quest_target_npc", "")
        # Chờ nhân vật di chuyển (tối đa 30s-40s). Chọn NPC trên minimap kích
        # hoạt chạy tới cạnh NPC; khi nhân vật dừng lại hoặc thấy tên NPC thì
        # bấm bong bóng tương tác để mở đối thoại.
        # TUYỆT ĐỐI KHÔNG bấm bong bóng khi nhân vật còn đang chạy qua các NPC ven đường.
        stationary_count = 0
        last_cur_coords = None
        last_frame = None
        stable_since = None
        max_wait_iterations = 60
        i = 0
        npc_nudge_attempts = 0
        unseen_npc_bubble_clicks = 0
        while i < max_wait_iterations:
            i += 1
            time.sleep(0.3)
            frame = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
            # Chỉ kiểm tra popup che màn hình định kỳ (mỗi 15 vòng) hoặc khi nhân vật đã đứng yên (stationary_count >= 5)
            # TUYỆT ĐỐI không gọi _dismiss_unexpected_popups (chạy full OCR 1600x900) mỗi vòng 0.5s gây nghẽn CPU và giật lag game!
            if frame is not None and (stationary_count >= 5 or (i % 15 == 0)):
                if self._dismiss_unexpected_popups(frame):
                    logger.info(
                        f"[{ctx.account_id}] Đã đóng popup che màn hình trong lúc di chuyển; "
                        "tiếp tục kiểm tra đích, không khởi động lại auto-path."
                    )
                    continue
            current_city, cur_coords = self._get_current_location_from_hud(frame)
            at_target_city = bool(
                not target_city
                or (current_city and is_same_city(current_city, target_city))
            )

            # Kiểm tra trạng thái dừng di chuyển:
            # Ưu tiên so sánh tọa độ HUD (nếu HUD có tọa độ).
            # Fallback: So sánh chuyển động hình ảnh giữa hai frame liên tiếp nếu HUD không có số tọa độ.
            is_stopped = False
            frame_is_stationary = False
            if frame is not None and last_frame is not None:
                try:
                    h_f, w_f = frame.shape[:2]
                    y1, y2 = int(h_f * 0.35), int(h_f * 0.65)
                    x1, x2 = int(w_f * 0.25), int(w_f * 0.75)
                    g_curr = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    g_prev = cv2.cvtColor(last_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(g_curr, g_prev)
                    mean_diff = float(np.mean(diff))
                    if mean_diff < 4.0:
                        frame_is_stationary = True
                except Exception:
                    pass
            last_frame = frame

            now = time.time()
            if cur_coords:
                if cur_coords == last_cur_coords:
                    is_stopped = True
                else:
                    is_stopped = False
                    if i >= max_wait_iterations - 5 and max_wait_iterations < 80:
                        max_wait_iterations += 5
                last_cur_coords = cur_coords
            elif frame_is_stationary:
                is_stopped = True

            if is_stopped:
                stationary_count += 1
                if stable_since is None:
                    stable_since = now
            else:
                stationary_count = 0
                stable_since = None

            stable_for = (now - stable_since) if stable_since is not None else 0.0

            # Nếu hộp thoại đã mở (kiểm tra nhanh bằng template/buttons khi đang chạy, tránh OCR toàn màn hình):
            dialog_open = False
            if hasattr(self, "_find_dialog_buttons") and self._find_dialog_buttons(frame):
                dialog_open = True
            elif hasattr(self, "_vision") and self._vision.find_template("dialog_box", frame=frame).found:
                dialog_open = True
            elif stationary_count >= 1 and self._is_npc_dialog_open(frame):
                dialog_open = True

            if dialog_open and at_target_city:
                # Kiểm tra xem có mở nhầm hộp thoại Xa Phu khi tìm NPC khác không
                bxs = self._text.read(frame, (150, 320, 1450, 850)) if hasattr(self, "_text") and self._text and frame is not None else []
                has_xa_phu = self._has_xa_phu_dialog_evidence(bxs, cur_city=current_city)
                is_target_xa_phu = bool(target_npc and "xa phu" in normalize_game_text(target_npc))
                if has_xa_phu and not is_target_xa_phu:
                    logger.warning(
                        f"[{ctx.account_id}] Mở nhầm hộp thoại Xa Phu khi đang tìm '{target_npc}'; đóng lại để tiếp tục."
                    )
                    self._ensure_npc_dialog_closed()
                else:
                    logger.info(f"[{ctx.account_id}] Hộp thoại NPC đã mở khi đang di chuyển!")
                    return BotState.INTERACT_NPC

            # Kiểm tra xem có thấy tên NPC đích hiển thị trên màn hình không
            has_target_npc = not bool(target_npc)
            target_npc_box = None
            bubble_present = self._is_npc_interact_bubble_present(frame) if frame is not None else False
            if target_npc and hasattr(self, "_text") and self._text and frame is not None:
                # Chỉ quét tên NPC khi nhân vật đã bắt đầu dừng lại (tránh nghẽn CPU khi đang chạy qua các NPC ven đường)
                if stationary_count >= 1:
                    body_boxes = self._text.read(frame, (100, 150, 1450, 700))
                    norm_target = normalize_game_text(target_npc)
                    has_target_npc = False
                    for b in body_boxes:
                        norm_b = normalize_game_text(b.text)
                        if not norm_b:
                            continue
                        if b.center_y < 120 and b.center_x < 350:
                            continue
                        if norm_target in norm_b or (len(norm_b) >= 4 and norm_b in norm_target and len(norm_b) >= len(norm_target) * 0.6):
                            has_target_npc = True
                            target_npc_box = b
                            break

            # Tương tác với NPC khi:
            # 1. Đang ở đúng thành phố đích.
            # 2. VÀ một trong các điều kiện:
            #    a) Thấy tên NPC đích hiển thị trên màn hình (has_target_npc), HOẶC
            #    b) Nhân vật đã thực sự dừng di chuyển tại đích sau khi auto-path (nav_started và stationary_count >= 2 và stable_for >= 0.4s).
            # -> TUYỆT ĐỐI KHÔNG tương tác khi nhân vật còn đang chạy qua các NPC ven đường hoặc khi chưa khởi động auto-path tới NPC!
            nav_started = ctx.extra.get("npc_nav_started", True)
            arrived_at_target = has_target_npc or (nav_started and stationary_count >= 2 and stable_for >= 0.4)
            if at_target_city and arrived_at_target:
                opened_dialog = False
                # Ưu tiên 1: Bấm bong bóng đối thoại nếu phát hiện
                if bubble_present:
                    logger.info(
                        f"[{ctx.account_id}] Đã tới cạnh NPC nhiệm vụ '{target_npc}' "
                        f"(đã dừng={stationary_count >= 2}, thấy tên={has_target_npc}), "
                        "bấm bong bóng đối thoại..."
                    )
                    self._click_npc_interact_bubble(frame)
                    time.sleep(0.4)
                    frame_after = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
                    if (
                        self._find_dialog_buttons(frame_after)
                        or self._is_npc_dialog_open(frame_after)
                        or self._find_finish_dialogue_button(frame_after)
                    ):
                        opened_dialog = True

                # Ưu tiên 2: Nếu chưa mở được và nhìn thấy nhãn tên NPC trên màn hình -> Click trực tiếp vào NPC!
                if not opened_dialog and target_npc_box is not None:
                    tap_x = target_npc_box.center_x
                    tap_y = min(target_npc_box.center_y + 35, 750)
                    logger.info(
                        f"[{ctx.account_id}] Thấy nhãn NPC '{target_npc}' tại ({tap_x}, {tap_y}), "
                        "click trực tiếp vào NPC để mở đối thoại..."
                    )
                    if hasattr(self, "_input") and self._input:
                        self._input.tap(tap_x, tap_y)
                        time.sleep(0.5)
                        frame_after = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
                        if (
                            self._find_dialog_buttons(frame_after)
                            or self._is_npc_dialog_open(frame_after)
                            or self._find_finish_dialogue_button(frame_after)
                        ):
                            opened_dialog = True

                # Ưu tiên 3: Nếu vẫn chưa mở được và nhân vật đã dừng hẳn tại đích (stationary_count >= 2) mà chưa có bubble:
                # Thử bấm tọa độ mặc định của icon đối thoại (1220, 299 / 1268, 335)
                if not opened_dialog and not bubble_present and stationary_count >= 2:
                    logger.info(
                        f"[{ctx.account_id}] Đã dừng yên tại đích '{target_npc}', "
                        "thử bấm icon đối thoại tại tọa độ mặc định..."
                    )
                    self._click_npc_interact_bubble()
                    time.sleep(0.5)
                    frame_after = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
                    if (
                        self._find_dialog_buttons(frame_after)
                        or self._is_npc_dialog_open(frame_after)
                        or self._find_finish_dialogue_button(frame_after)
                    ):
                        opened_dialog = True

                if opened_dialog:
                    chk_frame = frame_after if frame_after is not None else (self._screen.capture() if hasattr(self, "_screen") and self._screen else None)
                    if target_npc and "xa phu" not in normalize_game_text(target_npc) and self._is_wrong_xa_phu_dialog(chk_frame):
                        logger.warning(
                            f"[{ctx.account_id}] Mở nhầm hộp thoại Xa Phu khi đang tìm '{target_npc}'; đóng lại để tiếp tục."
                        )
                        self._ensure_npc_dialog_closed()
                        opened_dialog = False
                    else:
                        logger.info(f"[{ctx.account_id}] Mở đối thoại NPC thành công.")
                        return BotState.INTERACT_NPC
                elif not has_target_npc:
                    unseen_npc_bubble_clicks += 1
            elif bubble_present and at_target_city:
                logger.debug(
                    f"[{ctx.account_id}] Thấy bong bóng tương tác nhưng nhân vật vẫn đang di chuyển "
                    f"(stationary_count={stationary_count}), bỏ qua NPC ven đường..."
                )

            # Cơ chế chống hack của game: Không hiển thị/render NPC nếu nhân vật vừa chạy tới hoặc đứng yên tại chỗ.
            # Cần nhích trái một chút rồi nhích phải trở lại để game render hiển thị NPC:
            if (
                at_target_city
                and arrived_at_target
                and not has_target_npc
                and npc_nudge_attempts < 3
                and (unseen_npc_bubble_clicks >= 1 or stationary_count >= 4)
            ):
                logger.info(
                    f"[{ctx.account_id}] ⚠️ Đã tới điểm đến nhưng game chưa hiển thị NPC '{target_npc}' "
                    f"(đã thử bấm={unseen_npc_bubble_clicks}, đứng yên={stationary_count} lần, nudge {npc_nudge_attempts + 1}/3). "
                    "Tiến hành nhích trái rồi nhích phải để hiển thị NPC..."
                )
                self._nudge_left_right_to_reveal_npc(ctx)
                npc_nudge_attempts += 1
                unseen_npc_bubble_clicks = 0
                stationary_count = 0
                stable_since = time.time()
                time.sleep(0.4)
                continue

            stopped_at_wrong_city_npc = (
                current_city
                and target_city
                and not at_target_city
                and (
                    self._is_npc_interact_bubble_present(frame)
                    or self._is_npc_dialog_open()
                )
            )
            if stopped_at_wrong_city_npc:
                self._ensure_npc_dialog_closed()
                if ctx.extra.get("out_of_money"):
                    logger.warning(
                        f"[{ctx.account_id}] Đang hết tiền, dừng tại Xa phu '{current_city}' -> "
                        f"tiếp tục chạy bộ sang '{target_city}'."
                    )
                    self._walk_intercity_via_world_map(ctx, target_city, retry_local_xa_phu=False)
                else:
                    logger.warning(
                        f"[{ctx.account_id}] Đường đi dừng tại Xa phu trung gian '{current_city}', "
                        f"mở hộp thoại Xa phu để đi '{target_city}'."
                    )
                    switched = self._switch_to_city(
                        ctx,
                        target_city,
                        force=True,
                        current_location=(current_city, None),
                    )
                    if switched:
                        self._navigate_to_dialogue_npc(ctx, target_city, target_npc)
                return BotState.NAVIGATE

            if current_city and target_city and not at_target_city and i % 10 == 0:
                logger.info(
                    f"[{ctx.account_id}] Đang ở thành phố trung gian '{current_city}', "
                    f"chưa phải đích '{target_city}'; không đối thoại với Xa phu tại đây."
                )

            # Chỉ tăng tốc khi chưa tới cạnh NPC, tránh bấm thêm nút đúng lúc
            # bong bóng đối thoại vừa xuất hiện.
            if i % 6 == 0 and not bubble_present:
                self._press_buff_speed()

        current_city, _ = self._get_current_location_from_hud()
        if target_city and current_city and not is_same_city(current_city, target_city):
            logger.warning(
                f"[{ctx.account_id}] Hết thời gian chờ nhưng vẫn ở '{current_city}', "
                f"chưa tới '{target_city}'. Tiếp tục chuyển thành, không mở đối thoại sai NPC."
            )
            if ctx.extra.get("out_of_money"):
                self._walk_intercity_via_world_map(ctx, target_city, retry_local_xa_phu=False)
            else:
                switched = self._switch_to_city(
                    ctx,
                    target_city,
                    force=True,
                    current_location=(current_city, None),
                )
                if switched:
                    self._navigate_to_dialogue_npc(ctx, target_city, target_npc)
            return BotState.NAVIGATE

        # Nếu đã ở thành phố đích nhưng chưa mở được đối thoại sau thời gian chờ:
        frame_end = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
        if not (self._find_dialog_buttons(frame_end) or self._is_npc_dialog_open(frame_end) or self._find_finish_dialogue_button(frame_end)):
            logger.warning(
                f"[{ctx.account_id}] Hết thời gian chờ NAVIGATE nhưng chưa mở được đối thoại với '{target_npc}' "
                f"(đứng yên={stationary_count >= 2}). Điều hướng lại tới NPC..."
            )
            self._ensure_npc_dialog_closed()
            if target_city and target_npc:
                self._navigate_to_dialogue_npc(ctx, target_city, target_npc)
            return BotState.NAVIGATE

        return BotState.INTERACT_NPC

    def _is_wrong_xa_phu_dialog(self, frame: Optional[np.ndarray] = None) -> bool:
        """Kiểm tra xem hộp thoại hiện tại có phải là hộp thoại của Xa Phu hay không."""
        if frame is None:
            if hasattr(self, "_screen") and self._screen:
                frame = self._screen.capture()
            else:
                return False
        if not hasattr(self, "_text") or not self._text:
            return False
        boxes = self._text.read(frame, (100, 150, 1450, 750))
        if hasattr(self, "_has_xa_phu_dialog_evidence") and self._has_xa_phu_dialog_evidence(boxes):
            return True
        xa_phu_keywords = ("xa phu", "le phi", "3 luong", "roi khoi", "khong di dau", "chuyen di", "ngoixe", "hayngoixe")
        joined = " ".join(normalize_game_text(b.text) for b in boxes)
        return any(kw in joined for kw in xa_phu_keywords)

    def _handle_interact_npc(self, ctx: StateContext) -> BotState:
        """Đã đến gặp NPC nhiệm vụ, đối thoại và chuẩn bị quay về trả nhiệm vụ."""
        logger.info(f"[{ctx.account_id}] Đã đến gặp NPC nhiệm vụ, tiến hành đối thoại...")
        time.sleep(0.3)

        if self._check_quest_timeout(ctx):
            self._ensure_npc_dialog_closed()
            return BotState.FIND_COMMERCE_CHIEF

        if ctx.extra.get("multi_npc_needs_navigation"):
            target_city = ctx.extra.get("quest_target_city", "Thành Đô")
            target_npc = ctx.extra.get("quest_target_npc", "")
            logger.info(
                f"[{ctx.account_id}] Khôi phục chuỗi 4 Nạn dân: tiếp tục đi tới '{target_npc}'."
            )
            self._navigate_to_dialogue_npc(ctx, target_city, target_npc)
            return BotState.NAVIGATE

        # Bong bóng Xa phu có thể xuất hiện ở một thành phố trung gian trên
        # đường liên thành. Không được coi đó là NPC đích của nhiệm vụ.
        target_city = ctx.extra.get("quest_target_city", "")
        target_npc = ctx.extra.get("quest_target_npc", "")
        current_city, _ = self._get_current_location_from_hud()
        if target_city and current_city and not is_same_city(current_city, target_city):
            logger.warning(
                f"[{ctx.account_id}] Từ chối đối thoại nhiệm vụ tại '{current_city}': "
                f"đích thật là '{target_city}' -> tiếp tục tìm đường."
            )
            self._ensure_npc_dialog_closed()
            self._navigate_to_dialogue_npc(ctx, target_city, target_npc)
            return BotState.NAVIGATE

        # Nếu target_npc không phải Xa Phu nhưng hộp thoại đang mở lại là của Xa Phu:
        if target_npc and "xa phu" not in normalize_game_text(target_npc):
            frame_check = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
            if self._is_wrong_xa_phu_dialog(frame_check):
                logger.warning(
                    f"[{ctx.account_id}] Mở nhầm hộp thoại Xa Phu (cần gặp '{target_npc}'); "
                    "đóng hộp thoại và điều hướng lại."
                )
                self._ensure_npc_dialog_closed()
                self._navigate_to_dialogue_npc(ctx, target_city, target_npc)
                return BotState.NAVIGATE

        # Nếu hộp thoại chưa mở, ưu tiên bấm icon bong bóng đối thoại nổi.
        # Thử lại vài lần vì icon có thể xuất hiện chậm sau khi nhân vật dừng.
        frame_cur = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
        buttons = self._find_dialog_buttons(frame_cur)
        finish_btn = self._find_finish_dialogue_button(frame_cur)
        if not buttons and not self._is_npc_dialog_open(frame_cur) and not finish_btn:
            for attempt in range(3):
                frame = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
                bubble_present = self._is_npc_interact_bubble_present(frame)
                logger.info(
                    f"[{ctx.account_id}] Hộp thoại NPC chưa mở, "
                    f"thử bấm bong bóng đối thoại lần {attempt + 1}/3 "
                    f"(phát hiện icon={bubble_present})..."
                )
                if bubble_present:
                    self._click_npc_interact_bubble(frame)
                else:
                    self._click_npc_interact_bubble()
                time.sleep(0.3)
                frame_after = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
                finish_btn = self._find_finish_dialogue_button(frame_after)
                if finish_btn:
                    break
                buttons = self._find_dialog_buttons(frame_after)
                if buttons or self._is_npc_dialog_open(frame_after):
                    break

                # Nếu bấm bong bóng chưa mở, quét màn hình xem có tên NPC để click trực tiếp không
                if target_npc and hasattr(self, "_text") and self._text and frame_after is not None:
                    body_boxes = self._text.read(frame_after, (100, 150, 1450, 700))
                    norm_target = normalize_game_text(target_npc)
                    for b in body_boxes:
                        norm_b = normalize_game_text(b.text)
                        if norm_target in norm_b or (len(norm_b) >= 4 and norm_b in norm_target and len(norm_b) >= len(norm_target) * 0.6):
                            tap_x = b.center_x
                            tap_y = min(b.center_y + 35, 750)
                            logger.info(f"[{ctx.account_id}] Thử click trực tiếp vào NPC '{target_npc}' tại ({tap_x}, {tap_y})...")
                            if hasattr(self, "_input") and self._input:
                                self._input.tap(tap_x, tap_y)
                                time.sleep(0.4)
                            break
                    frame_after = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
                    if (
                        self._find_finish_dialogue_button(frame_after)
                        or self._find_dialog_buttons(frame_after)
                        or self._is_npc_dialog_open(frame_after)
                    ):
                        break

                if attempt == 0:
                    self._nudge_left_right_to_reveal_npc(ctx)

        # 1. Tìm nút hội thoại (ưu tiên 'Kết thúc đối thoại', sau đó đến lựa chọn hội thoại, sau đó đến nút đồ họa)
        frame_action = self._screen.capture() if hasattr(self, "_screen") and self._screen else frame_cur
        target_btn = finish_btn or self._find_finish_dialogue_button(frame_action)
        btn_desc = "Kết thúc đối thoại"

        if not target_btn:
            target_btn = self._find_dialogue_response_button(frame_action)
            if target_btn:
                btn_desc = "Lựa chọn đối thoại"

        if not target_btn:
            visual_buttons = self._find_dialog_buttons(frame_action)
            safe_buttons = [
                pos for pos in visual_buttons
                if pos[0] < 900 and pos[1] >= 430
            ]
            if safe_buttons:
                target_btn = min(
                    safe_buttons,
                    key=lambda pos: math.hypot(pos[0] - 309, pos[1] - 543),
                )
                btn_desc = "Nút hội thoại NPC đồ họa"

        if target_btn:
            # Kiểm tra phòng vệ: Tuyệt đối không bấm lựa chọn đối thoại nếu vô tình mở nhầm Xa Phu
            target_npc = ctx.extra.get("quest_target_npc", "")
            if target_npc and "xa phu" not in normalize_game_text(target_npc):
                if self._is_wrong_xa_phu_dialog(frame_action):
                    logger.warning(
                        f"[{ctx.account_id}] Phát hiện hộp thoại Xa Phu khi đang tìm '{target_npc}'; "
                        "đóng lại ngay để tránh dịch chuyển nhầm sang thành phố khác!"
                    )
                    self._ensure_npc_dialog_closed()
                    target_city = ctx.extra.get("quest_target_city", "")
                    self._navigate_to_dialogue_npc(ctx, target_city, target_npc)
                    return BotState.NAVIGATE

            logger.info(
                f"[{ctx.account_id}] Bấm '{btn_desc}' tại {target_btn}..."
            )
            self._input.tap(*target_btn)
            time.sleep(0.4)

            # Kiểm tra xem sau khi bấm hộp thoại đã đóng chưa
            check_frame = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
            if not self._is_npc_dialog_open(check_frame):
                next_multi_state = self._continue_four_refugee_quest(ctx)
                if next_multi_state is not None:
                    return next_multi_state

                logger.info(f"[{ctx.account_id}] Hoàn thành đối thoại với NPC (hộp thoại đã đóng)! Chuẩn bị quay về trả nhiệm vụ...")
                # Tắt banner thông báo nổi trên khung chat theo hướng dẫn của user để tránh OCR nhận diện nhầm
                self._dismiss_dialogue_completion_banner()
                ctx.extra.pop("npc_finish_state_failures", None)
                activity._write("✅", f"[{ctx.account_id}] Da doi thoai xong voi NPC. Quay ve tra nhiem vu.")
                ctx.extra["returning_quest"] = True
                time.sleep(0.1)
                return BotState.RETURN_QUEST

        if not target_btn and not self._is_npc_dialog_open(frame_action):
            logger.warning(
                f"[{ctx.account_id}] Hộp thoại NPC không mở sau các lần bấm; "
                "điều hướng lại tới NPC."
            )
            self._navigate_to_dialogue_npc(ctx, target_city, target_npc)
            return BotState.NAVIGATE

        # Nếu chưa tìm thấy nút hoặc bấm xong hộp thoại vẫn còn mở:
        # Tăng số lần thử đối thoại (nhiều NPC cần bấm thêm vài lựa chọn hội thoại nữa mới đóng)
        failures = int(ctx.extra.get("npc_finish_state_failures", 0)) + 1
        ctx.extra["npc_finish_state_failures"] = failures
        logger.warning(
            f"[{ctx.account_id}] Hộp thoại NPC vẫn còn mở (lượt đối thoại {failures}/2)..."
        )
        if failures >= 2:
            ctx.extra.pop("npc_finish_state_failures", None)
            self._ensure_npc_dialog_closed()
            if ctx.extra.get("multi_npc_route") and not ctx.extra.get("multi_npc_completed"):
                logger.warning(
                    f"[{ctx.account_id}] Đối thoại '{target_npc}' chưa đóng; "
                    "đi lại NPC này, không quay về trả nhiệm vụ sớm."
                )
                self._navigate_to_dialogue_npc(ctx, target_city, target_npc)
                return BotState.NAVIGATE
            ctx.extra["returning_quest"] = True
            logger.warning(
                f"[{ctx.account_id}] Hộp thoại không đóng sau 2 vòng đối thoại -> "
                "quay về Chủ thương hội để kiểm tra trạng thái thật."
            )
            self._dismiss_dialogue_completion_banner()
            return BotState.RETURN_QUEST
        return BotState.INTERACT_NPC

        next_multi_state = self._continue_four_refugee_quest(ctx)
        if next_multi_state is not None:
            return next_multi_state

        logger.info(f"[{ctx.account_id}] Hoàn thành đối thoại với NPC! Chuẩn bị quay về trả nhiệm vụ...")
        self._dismiss_dialogue_completion_banner()
        ctx.extra.pop("npc_finish_state_failures", None)
        activity._write("✅", f"[{ctx.account_id}] Da doi thoai xong voi NPC. Quay ve tra nhiem vu.")
        ctx.extra["returning_quest"] = True
        time.sleep(0.2)
        return BotState.RETURN_QUEST

    @staticmethod
    def _match_known_city(text: str) -> Optional[str]:
        """Nhận diện tên thành phố từ text OCR HUD (chịu lỗi OCR như 'TupenChau' -> 'Tuyền Châu')."""
        return match_known_city(text)

    def _get_current_location_from_hud(self, frame: Optional[np.ndarray] = None) -> tuple[Optional[str], Optional[tuple[int, int]]]:
        """
        Đọc tên map và tọa độ hiện tại từ widget Minimap góc trên bên phải (khi chưa click mở bản đồ lớn).
        Vùng đọc: (1300, 0, 1600, 140).
        Hỗ trợ nhận diện chuẩn xác cả thành phố lớn, thôn trấn và các map môn phái / mê cung tầng.
        """
        import re
        if frame is None:
            if hasattr(self, "_is_map_open") and self._is_map_open():
                self._close_map()
                time.sleep(0.2)
            if hasattr(self, "_screen") and self._screen is not None:
                frame = self._screen.capture()
            else:
                return None, None
        if frame is None or not hasattr(self, "_text") or self._text is None:
            return None, None
        h_f, w_f = frame.shape[:2]
        hud_crop = (int(w_f * 0.80), 0, w_f, int(h_f * 0.16)) if (w_f != 1600 or h_f != 900) else (1300, 0, 1600, 140)
        boxes = self._text.read(frame, hud_crop)
        city = None
        coords = None

        # 1. Quét tọa độ (x/y)
        coord_box = None
        for b in boxes:
            m = re.search(r'(\d+)\s*[/\\|\s]\s*(\d+)', b.text)
            if m:
                coords = (int(m.group(1)), int(m.group(2)))
                coord_box = b
                break

        # 2. Lọc bỏ các box nút hệ thống, nút HUD bên ngoài minimap hoặc tên NPC/cảnh báo
        IGNORED_MAP_WORDS = (
            "phut", "ngay", "suc khoe", "anh huong", "choi qua", "180", "buff", "auto", "ngucac",
            "xa phu", "xaphu", "chu thuong hoi", "chu tiem", "duoc diem", "tap hoa", "thu kho", "da tau",
            "vo cong", "nhan vat", "cai dat", "dau truong", "bang hoi", "nhiem vu", "to doi",
            "phuc loi", "hang", "ngu cac", "doi", "tui", "tui do", "nhat", "ngua", "nhanh", "tuy chon",
            "dong tac", "binh phach", "thien", "hoang lien", "thiet moc"
        )

        max_title_y = (coord_box.center_y - 10) if coord_box else int(h_f * 0.08 if (w_f != 1600 or h_f != 900) else 65)
        min_title_x = int(w_f * 0.88) if (w_f != 1600 or h_f != 900) else 1410

        title_boxes = []
        for b in boxes:
            if b.center_y > max_title_y:
                continue
            if b.center_x < min_title_x:
                continue
            if re.search(r'^\d+$', b.text.strip()) or re.search(r'(\d+)\s*[/\\|\s]\s*(\d+)', b.text):
                continue
            norm = normalize_game_text(b.text)
            if not norm or any(bad in norm for bad in IGNORED_MAP_WORDS):
                continue
            title_boxes.append(b)

        # Nếu không có box nào x >= min_title_x (do lệch khung), nới lỏng x
        if not title_boxes:
            for b in boxes:
                if b.center_y > max_title_y:
                    continue
                if re.search(r'^\d+$', b.text.strip()) or re.search(r'(\d+)\s*[/\\|\s]\s*(\d+)', b.text):
                    continue
                norm = normalize_game_text(b.text)
                if not norm or any(bad in norm for bad in IGNORED_MAP_WORDS):
                    continue
                title_boxes.append(b)

        # Sắp xếp các box tiêu đề từ trái qua phải theo dòng
        title_boxes.sort(key=lambda b: (b.center_y // 25, b.center_x))

        from .quest_classifier import parse_sect_and_combat_map

        # A. Ghép toàn bộ các từ tiêu đề minimap thành chuỗi hoàn chỉnh (tránh bị cắt cụt từ đơn)
        if title_boxes:
            combined_raw = " ".join(b.text for b in title_boxes).strip()
            # Bỏ hậu tố kênh phân tuyến (-1, -2, _1 hoặc số ở đuôi, nhưng giữ lại số tầng mê cung)
            clean_title = re.sub(r'[-_]\s*\d+$', '', combined_raw).strip()
            if clean_title == combined_raw:
                clean_title = re.sub(r'(?<!tang)(?<!me cung)\s+\d+$', '', combined_raw, flags=re.IGNORECASE).strip()

            # 1. Kiểm tra 8 Đại thành thị
            matched = match_known_city(clean_title)
            if matched:
                city = matched

            # 2. Kiểm tra Thôn Trấn
            if not city:
                matched_town = match_known_town(clean_title)
                if matched_town:
                    city = matched_town

            # 3. Kiểm tra map quái / mê cung môn phái
            if not city:
                sect, c_map = parse_sect_and_combat_map(clean_title, with_accents=True)
                if c_map:
                    city = c_map

            # 4. Fallback: Nếu không khớp map đã biết, dùng toàn bộ tên chuỗi clean_title (loại bỏ từ đơn mơ hồ)
            if not city:
                norm_c = normalize_game_text(clean_title)
                if norm_c and norm_c not in AMBIGUOUS_CITY_WORDS and len(norm_c) >= 3:
                    city = clean_title

        # B. Fallback quét từng box (chỉ chấp nhận map/thôn chuẩn, tuyệt đối không lấy từ đơn mơ hồ)
        if not city:
            for b in title_boxes:
                matched = match_known_city(b.text)
                if matched:
                    city = matched
                    break
                matched_town = match_known_town(b.text)
                if matched_town:
                    city = matched_town
                    break
                sect, c_map = parse_sect_and_combat_map(b.text, with_accents=True)
                if c_map:
                    city = c_map
                    break

        return city, coords

    def _is_same_combat_map(self, current_map: Optional[str], target_map: Optional[str], sect: Optional[str] = None) -> bool:
        """
        Kiểm tra xem nhân vật có đang ở đúng bản đồ quái yêu cầu không:
        - So sánh chính xác tầng (tầng 1 != tầng 2 != tầng 3).
        - So sánh tên map quái hoặc địa bàn môn phái (hỗ trợ OCR sai sót như ngi doc -> ngu doc).
        - Loại trừ tuyệt đối thành phố lớn nếu map yêu cầu là map quái ngoài thành.
        """
        if not current_map or not target_map:
            return False
        import re
        from .text_recognition import normalize_game_text, fuzzy_score
        from .quest_classifier import parse_sect_and_combat_map, KNOWN_SECTS

        # 1. Thử phân tích chuẩn hóa qua parse_sect_and_combat_map
        s_cur, m_cur = parse_sect_and_combat_map(current_map, with_accents=True)
        s_tgt, m_tgt = parse_sect_and_combat_map(target_map, with_accents=True)
        if m_cur and m_tgt:
            norm_mc = normalize_game_text(m_cur)
            norm_mt = normalize_game_text(m_tgt)
            if norm_mc == norm_mt:
                return True

        norm_cur = normalize_game_text(current_map)
        norm_target = normalize_game_text(target_map)
        if not norm_cur or not norm_target:
            return False

        # Tên thành lớn và tên map quái có thể vô tình đạt ngưỡng fuzzy thấp
        # (ví dụ "Dương Châu" ~ "Vân Mộng trạch" > 0.50).  Không bao giờ
        # cho phép một thành lớn khớp với map quái khác chỉ vì độ giống chuỗi.
        major_cities = (
            "Tuyền Châu", "Biện Kinh", "Thành Đô", "Tương Dương",
            "Dương Châu", "Đại Lý", "Phượng Tường", "Lâm An",
        )
        current_major = next(
            (normalize_game_text(city) for city in major_cities
             if fuzzy_score(norm_cur, normalize_game_text(city)) >= 0.80),
            None,
        )
        if current_major and not (
            current_major in norm_target
            or norm_target in current_major
            or fuzzy_score(norm_target, current_major) >= 0.80
        ):
            return False

        # Kiểm tra số tầng (tầng 1 vs tầng 2 vs tầng 3)
        cur_floor_m = re.search(r"(?:tang\s*(\d+)|me\s*cung\s*(\d+)|\b(\d+)\b)", norm_cur)
        target_floor_m = re.search(r"(?:tang\s*(\d+)|me\s*cung\s*(\d+)|\b(\d+)\b)", norm_target)
        cur_floor = (cur_floor_m.group(1) or cur_floor_m.group(2) or cur_floor_m.group(3)) if cur_floor_m else None
        target_floor = (target_floor_m.group(1) or target_floor_m.group(2) or target_floor_m.group(3)) if target_floor_m else None

        # Nếu cả 2 đều có tầng hoặc target có tầng mà tầng không khớp -> Chắc chắn KHÔNG PHẢI map này (sai tầng)!
        if target_floor and cur_floor and target_floor != cur_floor:
            return False

        # Tách bỏ phần tầng để so sánh tên map cơ sở
        norm_cur_base = re.sub(r"(?:tang\s*\d+|me\s*cung\s*\d+|\b\d+\b)", "", norm_cur).strip()
        norm_target_base = re.sub(r"(?:tang\s*\d+|me\s*cung\s*\d+|\b\d+\b)", "", norm_target).strip()

        # Canonicalize môn phái nếu có
        norm_cur_canon = normalize_game_text(KNOWN_SECTS.get(norm_cur_base, norm_cur_base))
        norm_target_canon = normalize_game_text(KNOWN_SECTS.get(norm_target_base, norm_target_base))
        norm_sect = normalize_game_text(sect or "")

        base_matched = (
            norm_target_base in norm_cur_base
            or norm_cur_base in norm_target_base
            or norm_cur_canon == norm_target_canon
            or fuzzy_score(norm_cur_base, norm_target_base) >= 0.50
            or (norm_sect and (norm_sect in norm_cur_base or norm_cur_base in norm_sect or fuzzy_score(norm_cur_base, norm_sect) >= 0.60))
        )
        if not base_matched:
            return False

        if target_floor and cur_floor and target_floor != cur_floor:
            return False
        return True

    def _is_bag_open(self, frame=None) -> bool:
        """Kiểm tra xem bảng Hành trang (Túi đồ) có đang mở hay không."""
        try:
            if frame is None and hasattr(self, "_screen") and self._screen is not None:
                frame = self._screen.capture()
            if frame is None or not hasattr(self, "_text") or not self._text:
                return False
            boxes = self._text.read(frame, (800, 50, 1600, 200))
            norm_texts = [normalize_game_text(b.text) for b in boxes]
            return any("hanh trang" in t or "tui do" in t for t in norm_texts)
        except Exception:
            return False

    def _open_bag(self) -> bool:
        """Mở bảng Hành trang (Túi đồ)."""
        if self._is_bag_open():
            return True
        coords = self._loader.load_ui_coords()
        bag_pos = coords.get("main_menu", {}).get("bag_button", {"x": 1470, "y": 325})
        self._input.tap(bag_pos["x"], bag_pos["y"])
        time.sleep(0.4)
        for _ in range(3):
            if self._is_bag_open():
                return True
            time.sleep(0.2)
        return self._is_bag_open()

    def _close_bag(self) -> bool:
        """Đóng bảng Hành trang (Túi đồ)."""
        coords = self._loader.load_ui_coords()
        shop_cfg = coords.get("shop", {})
        close_inv_pos = shop_cfg.get("inventory_close", {"x": 1565, "y": 95})
        self._input.tap(close_inv_pos["x"], close_inv_pos["y"])
        time.sleep(0.2)
        return not self._is_bag_open()

    def _is_inventory_full_or_overweight(self, frame=None) -> bool:
        """Kiểm tra xem có cảnh báo Hành trang quá nặng hoặc Hành trang đã đầy không."""
        try:
            if frame is None and hasattr(self, "_screen") and self._screen is not None:
                frame = self._screen.capture()
            if frame is None or not hasattr(self, "_text") or not self._text:
                return False
            boxes = self._text.read(frame)
            norm_texts = [normalize_game_text(b.text).replace(" ", "") for b in boxes]
            full_keywords = (
                "quanang", "hanhtrangquanang", "hanhtrangday", "dayhanhtrang",
                "khongthechua", "khongthemua", "quatai", "hanhtrangcuabanquanang",
                "hanhtrangcuabanday"
            )
            return any(any(k in t for k in full_keywords) for t in norm_texts)
        except Exception:
            return False

    def _sort_inventory(self, ctx: Optional[StateContext] = None) -> bool:
        """
        Thực hiện sắp xếp gọn gàng hành trang khi bị đầy hoặc quá nặng:
        1. Bấm nút 'Sắp xếp' trên bảng Hành trang (ưu tiên OCR 'Sap xep', fallback 1485, 260).
        2. Bấm nút 'Đồng ý' trong popup xác nhận 'Bạn đồng ý dùng chế độ tự động sắp xếp không?' (OCR 'Dong y', fallback 635, 600).
        3. Đợi 0.6s để hệ thống gom gộp và sắp xếp lại vật phẩm.
        """
        tag = f"[{ctx.account_id}] " if ctx else ""
        logger.info(f"{tag}📦 Bắt đầu tự động sắp xếp hành trang...")
        activity._write("📦", f"{tag}Tu dong sap xep hanh trang...")

        coords = self._loader.load_ui_coords()
        shop_cfg = coords.get("shop", {})
        sort_btn_pos = shop_cfg.get("sort_button", {"x": 1485, "y": 260})
        sort_confirm_pos = shop_cfg.get("sort_confirm_button", {"x": 635, "y": 600})

        # 1. Tìm và bấm nút 'Sắp xếp' trên bảng Hành trang
        frame = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
        sort_box = None
        if frame is not None and hasattr(self, "_text") and self._text:
            boxes = self._text.read(frame, (1350, 180, 1580, 350))
            for b in boxes:
                norm = normalize_game_text(b.text).replace(" ", "")
                if "sapxep" in norm or "sap" in norm:
                    sort_box = b
                    break

        if sort_box:
            logger.info(f"{tag}Tìm thấy nút 'Sắp xếp' tại ({sort_box.center_x}, {sort_box.center_y}), bấm...")
            self._input.tap(sort_box.center_x, sort_box.center_y)
        else:
            logger.info(f"{tag}Bấm nút 'Sắp xếp' tại tọa độ ({sort_btn_pos['x']}, {sort_btn_pos['y']})...")
            self._input.tap(sort_btn_pos["x"], sort_btn_pos["y"])

        time.sleep(0.4)

        # 2. Tìm và bấm nút 'Đồng ý' trong popup xác nhận 'Bạn đồng ý dùng chế độ tự động sắp xếp không?'
        frame_popup = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
        confirm_box = None
        if frame_popup is not None and hasattr(self, "_text") and self._text:
            boxes = self._text.read(frame_popup, (350, 450, 950, 750))
            for b in boxes:
                norm = normalize_game_text(b.text).replace(" ", "")
                if "dongy" in norm or norm == "dong":
                    confirm_box = b
                    break

        if confirm_box:
            logger.info(f"{tag}Bấm nút 'Đồng ý' sắp xếp tại ({confirm_box.center_x}, {confirm_box.center_y})...")
            self._input.tap(confirm_box.center_x, confirm_box.center_y)
        else:
            logger.info(f"{tag}Bấm nút 'Đồng ý' sắp xếp tại tọa độ ({sort_confirm_pos['x']}, {sort_confirm_pos['y']})...")
            self._input.tap(sort_confirm_pos["x"], sort_confirm_pos["y"])

        time.sleep(0.6)
        logger.info(f"{tag}✅ Đã sắp xếp hành trang thành công!")
        activity._write("✅", f"{tag}Da sap xep hanh trang thanh cong!")
        return True

    def _check_xa_phu_out_of_money(self, frame=None, boxes=None) -> bool:
        """Kiểm tra xem có thông báo hết tiền / không đủ ngân lượng khi đi Xa Phu không."""
        try:
            if boxes is None:
                if frame is None and hasattr(self, "_screen") and self._screen is not None:
                    frame = self._screen.capture()
                if frame is None or not hasattr(self, "_text") or not self._text:
                    return False
                boxes = self._text.read(frame, (150, 320, 1450, 850))
            if not boxes:
                return False
            norm_texts = [normalize_game_text(b.text).replace(" ", "") for b in boxes]
            out_of_money_keywords = (
                "khongdunganluong", "khongdutien", "khongdulephi", "nganluongkhongdu",
                "thieunganluong", "thieutien", "khongcodunganluong", "khongdu3luong",
                "thieulephi", "cotienthihayngoixe", "hayngoixe", "cotienthi",
                "ngoixe"
            )
            raw_texts = [normalize_game_text(b.text) for b in boxes]
            if any(any(k in t for k in out_of_money_keywords) for t in norm_texts):
                return True
            if any("co tien thi hay ngoi xe" in t or "ngoi xe" in t for t in raw_texts):
                return True
            return False
        except Exception:
            return False

    @staticmethod
    def _has_xa_phu_dialog_evidence(boxes: list, cur_city: Optional[str] = None) -> bool:
        """Chỉ xác nhận dialog Xa phu bằng cụm từ đặc trưng trong vùng hộp thoại.

        Loại bỏ vùng chat thế giới phía dưới (y > 745), tracker bên trái (x < 300),
        HUD phía trên (y < 350) và bên phải (x > 1380).
        """
        strong_markers = (
            "lephi", "lophi", "3luong", "3luung", "3luprg", "3lueng",
            "khongdidau", "khongdidauca", "muondidau", "khongdi",
            "cotienthihayngoixe", "hayngoixe", "cotienthi", "ngoixe",
        )
        norm_cur_city = normalize_game_text(cur_city) if cur_city else ""
        all_cities = ("bien kinh", "thanh do", "tuyen chau", "tuong duong", "duong chau", "dai ly", "phuong tuong", "phung tuong")

        # 1. Khung đối thoại Xa Phu có các từ khóa đặc trưng (Lệ phí 3 lượng, Muốn đi đâu, Không đi đâu cả)
        for box in boxes or []:
            cx = getattr(box, "center_x", 0)
            cy = getattr(box, "center_y", 9999)
            if cx < 250 or cx > 1380 or cy < 350 or cy > 820:
                continue
            norm = normalize_game_text(getattr(box, "text", ""))
            norm_ns = norm.replace(" ", "")
            if any(marker in norm_ns for marker in strong_markers):
                return True

        # 2. Nếu OCR hụt cụm từ đặc trưng, chỉ chấp nhận khi thấy ít nhất 3 thành phố
        # ĐÍCH ĐẾN KHÁC NHAU nằm đúng trong lưới nút bấm chuyển thành (y: 500..750).
        # Tuyệt đối không đếm lặp 1 tên thành phố, không tính thành phố hiện tại
        # và không lấy chữ rơi vào khung chat hoặc tên NPC/người chơi xung quanh.
        distinct_dest_cities = set()
        for box in boxes or []:
            cx = getattr(box, "center_x", 0)
            cy = getattr(box, "center_y", 9999)
            if 250 <= cx <= 1380 and 500 <= cy <= 750:
                norm = normalize_game_text(getattr(box, "text", ""))
                for c in all_cities:
                    if c in norm:
                        if norm_cur_city and (c in norm_cur_city or norm_cur_city in c):
                            continue
                        distinct_dest_cities.add(c)

        if len(distinct_dest_cities) >= 3:
            return True

        return False

    @staticmethod
    def _has_other_npc_dialog_evidence(boxes: list) -> bool:
        """Detect another NPC dialog from its central text, ignoring HUD/quest text.

        Quest trackers and chat can contain generic words such as "võ công" or
        "thủ khố" while the Xa Phu city picker is open. Those left-side labels
        must never trigger the code that closes the current dialog.
        """
        other_npc_markers = (
            "thu kho", "cat giu", "gui tien", "rut tien", "kho hang",
            "vi bang huu", "roi khoi", "vo cong",
        )
        for box in boxes or []:
            # NPC conversation text is centered. The left edge contains the
            # quest tracker and the lower edge contains chat/HUD overlays.
            # Bottom right contains action buttons (x > 1350).
            cx = getattr(box, "center_x", 0)
            cy = getattr(box, "center_y", 9999)
            if cx < 350 or cx > 1350 or cy > 700:
                continue
            norm = normalize_game_text(getattr(box, "text", ""))
            if any(marker in norm for marker in other_npc_markers):
                return True
        return False

    def _try_teleport_with_than_hanh_phu(self, ctx: StateContext, city_name: str) -> bool:
        """
        Ưu tiên sử dụng Thần Hành Phù trong Hành trang để dịch chuyển tức thì sang thành phố đích.
        Nếu không có hoặc thất bại, trả về False để chuyển sang đi Xa Phu.
        """
        try:
            logger.info(f"[{ctx.account_id}] [Thần Hành Phù] Kiểm tra và ưu tiên sử dụng Thần Hành Phù để chuyển sang '{city_name}'...")
            activity._write("⚡", f"[{ctx.account_id}] Thu dung Than Hanh Phu sang {city_name}...")
            
            # 1. Mở Hành trang
            if not self._open_bag():
                logger.info(f"[{ctx.account_id}] Không mở được Hành trang để tìm Thần Hành Phù.")
                return False

            time.sleep(0.3)
            # Tọa độ các tab túi đồ (Tab 1, Tab 2, Tab 3)
            tab_positions = [(838, 270), (838, 470), (838, 670)]
            
            # Grid các ô đồ trong Hành trang (5 cột x 8 hàng)
            grid_x = [565, 622, 680, 737, 795]
            grid_y = [220, 276, 332, 388, 444, 500, 556, 612]

            found_item = False
            for tab_idx, (tx, ty) in enumerate(tab_positions):
                self._input.tap(tx, ty)
                time.sleep(0.2)
                
                for gx in grid_x:
                    for gy in grid_y:
                        self._input.tap(gx, gy)
                        time.sleep(0.15)
                        
                        frame = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
                        if frame is None or not hasattr(self, "_text") or not self._text:
                            continue
                            
                        boxes = self._text.read(frame, (500, 100, 1100, 650))
                        norm_texts = [normalize_game_text(b.text) for b in boxes]
                        
                        is_than_hanh = any(
                            any(k in t for k in ("than hanh phu", "than hanh", "phu truyen tong", "dich chuyen den"))
                            for t in norm_texts
                        )
                        
                        if is_than_hanh:
                            logger.info(f"[{ctx.account_id}] Phát hiện Thần Hành Phù tại Tab {tab_idx+1}, ô ({gx}, {gy})!")
                            use_btn = next((b for b in boxes if any(k in normalize_game_text(b.text) for k in ("su dung", "dung", "su dung vat pham")) and b.center_y > 350), None)
                            if use_btn:
                                logger.info(f"[{ctx.account_id}] Bấm nút '{use_btn.text}' tại ({use_btn.center_x}, {use_btn.center_y})...")
                                self._input.tap(use_btn.center_x, use_btn.center_y)
                            else:
                                self._input.tap(gx + 120, gy + 100)
                            
                            time.sleep(0.5)
                            found_item = True
                            break
                    if found_item:
                        break
                if found_item:
                    break

            if not found_item:
                logger.info(f"[{ctx.account_id}] Không tìm thấy Thần Hành Phù trong Hành trang.")
                self._close_bag()
                return False

            # 2. Quét danh sách thành phố trong hộp thoại Thần Hành Phù
            norm_target = normalize_game_text(city_name)
            time.sleep(0.4)
            for attempt in range(3):
                frame = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
                if frame is None or not hasattr(self, "_text") or not self._text:
                    time.sleep(0.2)
                    continue

                boxes = self._text.read(frame, (200, 200, 1400, 750))
                target_box = next((b for b in boxes if is_same_city(b.text, city_name) or norm_target in normalize_game_text(b.text)), None)
                if target_box:
                    logger.info(f"[{ctx.account_id}] Chọn điểm đến '{target_box.text}' trên Thần Hành Phù tại ({target_box.center_x}, {target_box.center_y})...")
                    self._input.tap(target_box.center_x, target_box.center_y)
                    time.sleep(0.4)
                    
                    chk_frame = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
                    if chk_frame is not None and hasattr(self, "_text") and self._text:
                        chk_boxes = self._text.read(chk_frame, (350, 300, 1250, 750))
                        confirm_btn = self._find_travel_confirm_button(chk_boxes)
                        if confirm_btn:
                            self._input.tap(confirm_btn.center_x, confirm_btn.center_y)
                            time.sleep(0.3)
                    break
                time.sleep(0.3)

            # 3. Đóng Hành trang nếu còn mở và chờ load map mới
            self._close_bag()
            logger.info(f"[{ctx.account_id}] Đang chờ dịch chuyển đến '{city_name}' qua Thần Hành Phù...")
            for _ in range(8):
                time.sleep(0.4)
                cur, _ = self._get_current_location_from_hud()
                if cur and is_same_city(cur, city_name):
                    logger.info(f"[{ctx.account_id}] ✅ Đã dịch chuyển thành công sang '{city_name}' bằng Thần Hành Phù (HUD: '{cur}')!")
                    activity._write("⚡", f"[{ctx.account_id}] Den {city_name} bang Than Hanh Phu!")
                    return True

            logger.warning(f"[{ctx.account_id}] Dịch chuyển qua Thần Hành Phù chưa thành công, fallback sang Xa Phu...")
            return False
        except Exception as e:
            logger.warning(f"[{ctx.account_id}] Lỗi khi sử dụng Thần Hành Phù: {e}")
            self._close_bag()
            return False

    def _trigger_intercity_autopath(self, ctx: StateContext, city_name: str) -> bool:
        """
        Kích hoạt tự động tìm đường chạy bộ liên thành qua chức năng Tìm map:
        1. Mở bản đồ mini (nếu chưa mở).
        2. Tìm và chọn đúng thành phố mục tiêu.
        3. Khi bản đồ đích vẫn đang mở, chuyển sang tab Tìm NPC.
        4. Chọn Xa phu trong thành phố đích để bắt đầu chạy bộ liên bản đồ.
        5. Đóng bản đồ và lên ngựa.
        """
        coords = self._loader.load_ui_coords()
        nav_cfg = coords.get("navigation", {})
        find_npc_x = nav_cfg.get("map_tab_find_npc", {}).get("x", 372)
        find_npc_y = nav_cfg.get("map_tab_find_npc", {}).get("y", 175)

        # 1. Mở bản đồ mini
        self._ensure_map_open()
        time.sleep(0.3)

        # 2. Chọn bản đồ đích bằng tab Tìm map. Nút "Bản đồ thế giới" có thể
        # không mở do giao diện đang ở trạng thái khác; khi đó tọa độ thành phố
        # thế giới bị click nhầm ngay trên bản đồ thành phố hiện tại.
        find_map = nav_cfg.get("map_tab_find_map", {"x": 172, "y": 175})
        search = nav_cfg.get("map_search_input", {"x": 266, "y": 222})
        logger.info(
            f"[{ctx.account_id}] [Chạy liên thành] Tìm bản đồ đích '{city_name}' "
            "bằng tab Tìm map..."
        )
        self._input.tap(find_map.get("x", 172), find_map.get("y", 175))
        time.sleep(0.25)
        self._input.tap(search.get("x", 266), search.get("y", 222))
        time.sleep(0.15)
        try:
            self._adb._run(["shell", "input keyevent 123"])
            self._adb._run(["shell", "input keyevent " + " ".join(["67"] * 20)])
        except Exception:
            pass
        self._type_unicode_text(ctx, city_name)
        time.sleep(0.4)
        if not self._click_map_search_result(city_name):
            logger.warning(
                f"[{ctx.account_id}] [Chạy liên thành] Không chọn được kết quả map '{city_name}'."
            )
            self._close_map()
            return False
        time.sleep(0.25)
        # Chọn dòng kết quả đã nạp bản đồ đích ngay trong cửa sổ minimap.
        # Không bấm nút "Xác nhận" ở đây: nút đó đóng minimap và bắt đầu đi tới
        # tọa độ đang chọn. Nếu tiếp tục bấm tab Tìm NPC sau đó, các tọa độ sẽ rơi
        # lên màn hình game và OCR sẽ đọc nhầm bảng nhiệm vụ thành danh sách NPC.
        if not self._is_map_open():
            logger.warning(
                f"[{ctx.account_id}] [Chạy liên thành] Minimap đã đóng sau khi chọn '{city_name}'; "
                "không thể mở danh sách NPC đích."
            )
            return False
        time.sleep(0.2)

        # 4. Chuyển sang tab 'Tìm NPC' và click NPC trong thành phố để kích hoạt chạy bộ liên bản đồ
        logger.info(f"[{ctx.account_id}] [Chạy liên thành] Chuyển sang tab Tìm NPC ({find_npc_x}, {find_npc_y}) và chọn NPC tại '{city_name}'...")
        self._input.tap(find_npc_x, find_npc_y)
        time.sleep(0.4)
        # Ưu tiên chọn Xa Phu trong danh sách NPC để luôn chạy ra đường phố ngoài trời của thành phố (tránh chui vào Dinh Phủ)
        # Tuy nhiên: Nếu nhân vật đã đứng ngay sát Xa phu hiện tại (tọa độ ~ 194, 202) thì không bấm lại Xa phu
        cur_c, cur_pt = self._get_current_location_from_hud()
        already_at_xa_phu = False
        if cur_pt and abs(cur_pt[0] - 194) <= 6 and abs(cur_pt[1] - 202) <= 6:
            already_at_xa_phu = True

        xa_phu_clicked = False
        if not already_at_xa_phu:
            # Quét tìm trực tiếp Xa Phu trong danh sách NPC của thành đích.
            # Không gọi _search_tuyen_chau_xa_phu vì thao tác gõ/Enter sẽ làm đóng minimap
            # và khi mở lại minimap sẽ bị văng về map dã ngoại hiện tại.
            xa_phu_clicked = self._click_npc_search_result("Xa phu")
        if not xa_phu_clicked:
            logger.warning(
                f"[{ctx.account_id}] [Chạy liên thành] Danh sách NPC của '{city_name}' "
                "chưa hiện Xa phu; dùng điểm map đã chọn để bắt đầu chạy liên thành."
            )
            # Các map phụ như Nam Thành Đô không có Xa Phu. Sau khi đã chọn
            # đúng kết quả ở tab Tìm map, nút xác nhận sẽ đóng minimap và bắt
            # đầu auto-path tới điểm đã chọn. Không được đóng bản đồ bằng nút X,
            # vì như vậy chỉ hủy lựa chọn rồi vòng giám sát sẽ chờ giả.
            confirm = nav_cfg.get("map_confirm_button", {"x": 1015, "y": 778})
            self._input.tap(confirm.get("x", 1015), confirm.get("y", 778))
            time.sleep(0.5)
            if self._is_map_open():
                logger.warning(
                    f"[{ctx.account_id}] [Chạy liên thành] Bấm xác nhận map nhưng minimap vẫn mở; "
                    "chưa thể khởi động auto-path."
                )
                self._close_map()
                return False
            self._press_buff_speed()
            return True
        time.sleep(0.4)

        # 5. Đóng bản đồ và lên ngựa
        self._close_map()
        time.sleep(0.2)
        self._press_buff_speed()
        return True

    def _walk_intercity_via_world_map(
        self,
        ctx: StateContext,
        city_name: str,
        max_wait: float = 240.0,
        retry_local_xa_phu: bool = True,
    ) -> bool:
        """
        Chạy bộ liên thành (xuyên các map trung gian) khi hết tiền đi Xa Phu hoặc không có Thần Hành Phù.
        Đánh giá tốc độ và trạng thái di chuyển thực tế theo thời gian thực (tọa độ HUD) để tính toán thời gian chính xác.
        """
        logger.info(f"[{ctx.account_id}] 🏃 [Chạy bộ liên thành] Bắt đầu chạy bộ sang '{city_name}' (thời gian chờ tối đa {int(max_wait)}s)...")
        activity._write("🏃", f"[{ctx.account_id}] Chay bo lien thanh sang {city_name}...")

        # Kích hoạt tìm đường liên thành
        # Prefer the Xa phu already beside the character. Searching after
        # loading the destination map inspects that destination's NPC list,
        # not the visible Xa phu in the current city.
        current_location = self._get_current_location_from_hud()
        if isinstance(current_location, tuple) and len(current_location) >= 2:
            cur_city, cur_coords = current_location[0], current_location[1]
        else:
            cur_city, cur_coords = None, None
        cur_city_start = cur_city
        initial_frame = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
        is_major_city = bool(
            cur_city and any(is_same_city(cur_city, mc) for mc in KNOWN_CITIES)
        )
        if (
            is_major_city
            and cur_city
            and not is_same_city(cur_city, city_name)
            and initial_frame is not None
            and retry_local_xa_phu
            and not ctx.extra.get("out_of_money")
            and self._is_standing_next_to_xa_phu(initial_frame)
        ):
            logger.info(
                f"[{ctx.account_id}] Dang dung canh Xa phu tai '{cur_city}' "
                f"{cur_coords or ''}; mo hoi thoai de di '{city_name}', "
                "khong tim Xa phu trong map dich."
            )
            return self._switch_to_city(
                ctx,
                city_name,
                force=True,
                current_location=(cur_city, cur_coords),
            )

        # If selection of the destination NPC has already switched maps, this
        # was a map transition, not a completed walk. Confirm using a second
        # HUD sample and report the correct method.
        if cur_city and is_same_city(cur_city, city_name):
            time.sleep(1.0)
            confirm_city, confirm_coords = self._get_current_location_from_hud()
            if confirm_city and is_same_city(confirm_city, city_name):
                logger.info(
                    f"[{ctx.account_id}] ✅ HUD xác nhận đã chuyển tới '{city_name}' "
                    f"(địa điểm: {confirm_city} {confirm_coords}); không ghi nhận nhầm là chạy bộ."
                )
                activity._write("📍", f"[{ctx.account_id}] Da xac nhan den {city_name}.")
                return True

        # From a major city, cross-city travel starts at the Xa phu in the
        # current city. Searching the destination city's NPC list only selects
        # a destination marker; it does not walk the character to the local
        # Xa phu and can leave them wandering in the source city indefinitely.
        if (
            is_major_city
            and cur_city
            and not is_same_city(cur_city, city_name)
            and not ctx.extra.get("out_of_money")
        ):
            if retry_local_xa_phu:
                logger.info(
                    f"[{ctx.account_id}] [Chạy liên thành] Vẫn ở thành nguồn '{cur_city}'; "
                    f"đi tới Xa phu tại thành này trước khi thử đối thoại."
                )
                return self._switch_to_city(
                    ctx,
                    city_name,
                    force=True,
                    current_location=(cur_city, cur_coords),
                    allow_world_map_fallback=False,
                )
            logger.warning(
                f"[{ctx.account_id}] [Chạy liên thành] Không tới được Xa phu tại '{cur_city}'; "
                "dừng lượt này, không chọn NPC trên bản đồ thành đích."
            )
            return False

        if not self._trigger_intercity_autopath(ctx, city_name):
            logger.warning(
                f"[{ctx.account_id}] Không khởi động được auto-path sang '{city_name}'; "
                "không vào vòng chờ đứng yên giả."
            )
            return False

        # Giám sát chuyển động thực tế theo tọa độ HUD
        start_walk = time.time()
        last_log_sec = 0
        last_coords = None
        stationary_ticks = 0
        last_check_t = time.time()
        last_blocking_monster_attack = 0.0

        while time.time() - start_walk < max_wait:
            time.sleep(1.0)
            now = time.time()
            dt = max(now - last_check_t, 0.1)
            last_check_t = now

            cur_city, cur_coords = self._get_current_location_from_hud()
            elapsed = int(now - start_walk)

            # 1. Đã tới đúng thành phố đích
            if cur_city and is_same_city(cur_city, city_name):
                # A selected NPC may trigger an immediate teleport, before the
                # first monitoring sample. Do not label that as a 1-second
                # walk: require observable travel progress or a non-source
                # location before reporting intercity walking success.
                if elapsed >= 2 and (not cur_city_start or not is_same_city(cur_city, cur_city_start)):
                    logger.info(f"[{ctx.account_id}] ✅ Đã chạy bộ đến thành công '{city_name}' sau {elapsed}s (HUD: '{cur_city}' {cur_coords})!")
                    activity._write("📍", f"[{ctx.account_id}] Da den {city_name} (chay bo {elapsed}s)!")
                    return True
                if elapsed >= 2 and cur_city_start and is_same_city(cur_city, cur_city_start):
                    logger.warning(
                        f"[{ctx.account_id}] HUD vẫn báo thành nguồn '{cur_city_start}' dù trùng tên đích '{city_name}'; "
                        "chưa xác nhận chuyển thành."
                    )
                    return False

            # 2. Đánh giá trạng thái di chuyển thực tế qua thay đổi tọa độ
            is_moving = False
            speed = 0.0
            if cur_coords and last_coords:
                import math
                dx = cur_coords[0] - last_coords[0]
                dy = cur_coords[1] - last_coords[1]
                dist_step = math.hypot(dx, dy)
                if dist_step > 0.5:
                    is_moving = True
                    speed = dist_step / dt
                    stationary_ticks = 0
                    # Gia hạn thời gian nếu nhân vật vẫn đang chạy liên tục qua các map
                    start_walk = max(start_walk, now - (max_wait - 30.0))
                else:
                    stationary_ticks += 1
            elif cur_coords:
                last_coords = cur_coords

            if cur_coords:
                last_coords = cur_coords

            # Tự tìm đường có thể bị quái đang tấn công ngắt. Nếu đứng yên
            # liên tục 3 giây trong map dã ngoại, diệt con quái nhiệm vụ
            # nhìn thấy gần nhất trước, sau đó mới kích hoạt lại đường đi.
            is_major_map = cur_city and any(is_same_city(cur_city, mc) for mc in KNOWN_CITIES)
            # Auto-path liên thành dừng cạnh Xa phu chứ không tự chọn
            # thành trong hộp thoại. Khi đã đứng yên và bong bóng
            # xuất hiện ở một thành trung gian, chuyển ngay sang luồng
            # Xa phu thay vì chờ 10 giây rồi mở lại minimap vô hạn.
            if is_major_map and stationary_ticks >= 2 and not ctx.extra.get("out_of_money"):
                arrival_frame = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
                if arrival_frame is not None and self._is_standing_next_to_xa_phu(arrival_frame):
                    logger.info(
                        f"[{ctx.account_id}] [Chạy liên thành] Đã dừng cạnh Xa phu "
                        f"tại '{cur_city}' {cur_coords}; mở hộp thoại để đi '{city_name}'."
                    )
                    return self._switch_to_city(
                        ctx,
                        city_name,
                        force=True,
                        current_location=(cur_city, cur_coords),
                    )
            if (
                cur_coords
                and stationary_ticks >= 3
                and not is_major_map
                and now - last_blocking_monster_attack >= 4.0
            ):
                last_blocking_monster_attack = now
                if self._attack_monster_blocking_travel(ctx):
                    stationary_ticks = 0
                    # Cho game đủ thời gian tiếp cận/ra đòn. Vòng sau nếu còn
                    # quái khác chặn đường thì sẽ tiếp tục chuột phải con kế tiếp.
                    time.sleep(2.0)
                    continue

            # 3. Xử lý khi nhân vật bị đứng yên (ngưỡng 10s)
            if stationary_ticks >= 10:
                logger.warning(
                    f"[{ctx.account_id}] ⚠️ [Chạy liên thành] Phát hiện nhân vật ĐỨNG YÊN tại '{cur_city}' {cur_coords or ''} "
                    f"sau {stationary_ticks}s! Kích hoạt lại tìm đường sang '{city_name}'..."
                )
                activity._write("⚠️", f"[{ctx.account_id}] Nhan vat dung yen {stationary_ticks}s tai {cur_city or 'map'}...")
                
                # Nếu đang hết tiền: Tiếp tục kích hoạt lại auto-path chạy bộ cho đến khi kết thúc nhiệm vụ!
                if ctx.extra.get("out_of_money"):
                    logger.info(f"[{ctx.account_id}] Đang hết tiền, tiếp tục kích hoạt lại auto-path chạy bộ sang '{city_name}'...")
                    self._trigger_intercity_autopath(ctx, city_name)
                    stationary_ticks = 0
                    time.sleep(1.0)
                    continue

                is_major = cur_city and any(is_same_city(cur_city, mc) for mc in KNOWN_CITIES)

                if (
                    is_major
                    and cur_city
                    and not is_same_city(cur_city, city_name)
                    and not ctx.extra.get("out_of_money")
                ):
                    logger.warning(
                        f"[{ctx.account_id}] [Chạy liên thành] Nhân vật vẫn đứng tại thành nguồn "
                        f"'{cur_city}'; dừng lệnh auto-path sai hướng để thử lại Xa phu địa phương."
                    )
                    return self._switch_to_city(
                        ctx,
                        city_name,
                        force=True,
                        current_location=(cur_city, cur_coords),
                        allow_world_map_fallback=False,
                    )

                if not self._trigger_intercity_autopath(ctx, city_name):
                    logger.warning(
                        f"[{ctx.account_id}] Kích hoạt lại auto-path sang '{city_name}' thất bại; "
                        "dừng lượt chờ để state machine thử lại."
                    )
                    return False
                stationary_ticks = 0
                continue

            # 4. Ghi log tiến độ thực tế có đánh giá tốc độ và trạng thái di chuyển
            if elapsed - last_log_sec >= 8:
                if is_moving:
                    logger.info(
                        f"[{ctx.account_id}] 🏃 Đang chạy bộ liên thành tới '{city_name}' "
                        f"(vị trí: '{cur_city or 'Đang qua map'}' {cur_coords or ''} | tốc độ ~{speed:.1f} đv/s | đã chạy {elapsed}s)..."
                    )
                    activity._write("🏃", f"[{ctx.account_id}] Chay sang {city_name}: {cur_city or 'Qua map'} ({elapsed}s)...")
                    self._press_buff_speed()
                else:
                    status_desc = "đang load map..." if not cur_coords else f"đứng yên {stationary_ticks}s"
                    logger.warning(
                        f"[{ctx.account_id}] ⚠️ Nhân vật {status_desc} tại '{cur_city or 'Đang qua map'}' {cur_coords or ''} "
                        f"(mục tiêu: '{city_name}' | đã chờ {elapsed}s)..."
                    )
                    activity._write("⚠️", f"[{ctx.account_id}] {cur_city or 'Map'}: {status_desc} ({elapsed}s)...")
                last_log_sec = elapsed

        cur_final, final_coords = self._get_current_location_from_hud()
        if cur_final and is_same_city(cur_final, city_name):
            return True

        logger.warning(f"[{ctx.account_id}] Hết thời gian chờ chạy bộ ({int(max_wait)}s) sang '{city_name}' (vị trí hiện tại: '{cur_final}' {final_coords})!")
        return False

    @staticmethod
    def _find_travel_confirm_button(boxes):
        """Tìm đúng nút xác nhận dịch chuyển, không khớp chữ ``di`` trong tên người chơi."""
        exact_labels = {
            "dong y", "dongy", "xac nhan", "xacnhan",
            "chap nhan", "chapnhan", "di", "co",
        }
        for box in boxes or []:
            if box.center_y <= 450:
                continue
            norm = normalize_game_text(box.text).strip()
            compact = norm.replace(" ", "")
            # Các nút có nhãn rất ngắn. So khớp toàn bộ nhãn thay vì dùng
            # ``"di" in text`` vốn nhận nhầm BaoTriDienLanh và tên người chơi.
            if norm in exact_labels or compact in exact_labels:
                return box
        return None

    def _switch_to_city(
        self,
        ctx: StateContext,
        city_name: str,
        force: bool = False,
        current_location: Optional[tuple[str, Optional[tuple[int, int]]]] = None,
        allow_world_map_fallback: bool = True,
    ) -> bool:
        """
        Di chuyển thực sự sang thành phố chỉ định:
        1. Kiểm tra nếu đã ở sẵn thành phố mục tiêu -> bỏ qua.
        2. Nếu đi Xa Phu (có tiền): thời gian chờ load nhanh (~5-7s).
        3. Nếu hết tiền / hết Thần Hành Phù: tự động chuyển sang Chạy bộ liên thành (thời gian chờ 90s).
        """
        if current_location is not None:
            cur_city, cur_coords = current_location
        else:
            cur_city, cur_coords = self._get_current_location_from_hud()
        if cur_city is None:
            time.sleep(0.2)
            cur_city, cur_coords = self._get_current_location_from_hud()

        norm_target = normalize_game_text(city_name)
        # Kiểm tra tuyệt đối: Nếu đã ở thành phố mục tiêu thì dừng ngay, TUYỆT ĐỐI không đi Xa Phu (bất kể force=True/False)
        if cur_city and is_same_city(cur_city, city_name):
            logger.info(
                f"[{ctx.account_id}] ✅ Kiểm tra HUD trước khi đi Xa Phu: "
                f"Đã ở sẵn thành phố '{city_name}' (HUD: '{cur_city}', Tọa độ: {cur_coords}), không cần chuyển map qua Xa Phu."
            )
            activity._write("📍", f"[{ctx.account_id}] Da o san {city_name}, bo qua Xa Phu.")
            return True

        # Đảm bảo đóng bảng Hành trang nếu đang mở để không che màn hình
        if self._is_bag_open():
            self._close_bag()
            time.sleep(0.2)

        # Chỉ dùng Thần Hành Phù ở lượt đầu. Khi đang retry tiếp cận Xa phu
        # địa phương, không rẽ sang một đường đi khác rồi nhầm là đã tới NPC.
        if allow_world_map_fallback and ctx.extra.get("use_than_hanh_phu"):
            try:
                if hasattr(self, "_try_teleport_with_than_hanh_phu") and self._try_teleport_with_than_hanh_phu(ctx, city_name):
                    logger.info(f"[{ctx.account_id}] Đã chuyển sang '{city_name}' bằng Thần Hành Phù.")
                    return True
            except Exception as e:
                logger.debug(f"Try than hanh phu check: {e}")

        # Nếu đã biết tài khoản hết tiền đi Xa Phu: Chạy bộ liên thành trực tiếp, không tới Xa Phu nữa
        if ctx.extra.get("out_of_money"):
            logger.info(f"[{ctx.account_id}] 💸 Đã ghi nhận hết tiền đi Xa Phu (out_of_money=True), chuyển thẳng sang chạy bộ liên thành sang '{city_name}'...")
            activity._write("🏃", f"[{ctx.account_id}] Het tien Xa Phu -> Chay bo sang {city_name}...")
            return self._walk_intercity_via_world_map(
                ctx, city_name, max_wait=90.0,
                retry_local_xa_phu=False,
            )

        # Nếu đang ở map dã ngoại / hang động / môn phái (không phải 8 đại thành thị):
        # Trong map dã ngoại/hang quái hoàn toàn không có Xa Phu.
        # Chuyển thẳng sang chạy bộ liên thành qua Bản đồ thế giới để chạy ngay từ lần 1 (không mất 25s cuộn tìm Xa Phu rồi đứng yên)!
        is_major_city = cur_city and any(is_same_city(cur_city, mc) for mc in KNOWN_CITIES)
        if cur_city and not is_major_city:
            logger.info(f"[{ctx.account_id}] Nhân vật đang ở map dã ngoại/hang động '{cur_city}', không có Xa Phu. Chạy bộ liên thành sang '{city_name}' qua Bản đồ thế giới...")
            activity._write("🏃", f"[{ctx.account_id}] Map ngoai {cur_city} khong co Xa Phu -> Chay bo sang {city_name}...")
            return self._walk_intercity_via_world_map(
                ctx, city_name, max_wait=90.0,
                retry_local_xa_phu=allow_world_map_fallback,
            )

        # Di chuyển trực tiếp qua Xa Phu trên Minimap (nhanh chóng, ổn định, không mở Hành trang)
        logger.info(f"[{ctx.account_id}] Đang di chuyển sang thành phố '{city_name}' qua Xa Phu (HUD hiện tại: '{cur_city}')...")
        activity._write("🐎", f"[{ctx.account_id}] Den Xa Phu di sang {city_name}...")
        if not allow_world_map_fallback:
            logger.info(
                f"[{ctx.account_id}] Chế độ thử lại Xa phu địa phương; bỏ qua Thần Hành Phù "
                "để không rẽ sang auto-path bản đồ đích."
            )
        self._ensure_map_open()
        time.sleep(0.15)

        coords = self._loader.load_ui_coords()
        nav_cfg = coords.get("navigation", {})
        find_npc_x = nav_cfg.get("map_tab_find_npc", {}).get("x", 376)
        find_npc_y = nav_cfg.get("map_tab_find_npc", {}).get("y", 161)

        # 1. Bấm tab 'Tìm NPC'
        self._input.tap(find_npc_x, find_npc_y)
        time.sleep(0.3)

        # 2. Tại Tuyền Châu, luồng tự động chỉ dùng Xa phu Bắc.
        # Nhãn "Xa phu Tuyền Châu" chỉ xuất hiện ngoài màn hình.
        if cur_city and is_same_city(cur_city, "Tuyền Châu"):
            npc_selected = self._search_tuyen_chau_xa_phu(ctx)
            xa_phu_search_name = "Xa phu Bắc"
        else:
            xa_phu_search_name = "Xa phu"
            npc_selected = self._click_npc_search_result("Xa phu")

        if not npc_selected:
            logger.warning(f"[{ctx.account_id}] Không tìm thấy '{xa_phu_search_name}' trên minimap (có thể đang ở map dã ngoại). Chuyển sang chạy bộ liên thành qua Bản đồ thế giới...")
            if self._is_map_open():
                self._close_map()
            if not allow_world_map_fallback:
                return False
            return self._walk_intercity_via_world_map(
                ctx, city_name, max_wait=90.0,
                retry_local_xa_phu=allow_world_map_fallback,
            )

        time.sleep(0.1)
        if self._is_map_open():
            self._close_map()
            time.sleep(0.1)

        logger.info(f"[{ctx.account_id}] Đang chạy bộ đến gặp Xa Phu...")
        activity._write("🏃", f"[{ctx.account_id}] Dang chay den Xa Phu...")

        # 4. Chờ nhân vật chạy đến Xa Phu và hộp thoại NPC mở ra.
        # Auto-path có thể dừng tạm rồi "nhích" thêm một đoạn, vì vậy không
        # được coi hai frame có cùng tọa độ là đã đến nơi. Chỉ mở hộp thoại
        # sau khi HUD đứng yên liên tục và frame MỚI vẫn nhìn thấy đúng Xa phu.
        dialog_opened = False
        start_wait = time.time()
        last_cur_coords = None
        stationary_count = 0
        stable_since = None
        xa_phu_label_streak = 0
        # Không bấm vào tên/thân NPC sau khi auto-path kết thúc: cú bấm có thể
        # rơi xuống mặt đất và tạo lệnh đi bộ mới, làm nhân vật lệch khỏi Xa Phu.
        # Chỉ bấm bong bóng tương tác, tối đa hai lần trong cả lượt chờ.
        max_xa_phu_nudges = 2
        tuyen_chau_arrival = (coords.get("xa_phu_arrival", {}) or {}).get("Tuyền Châu", {})
        is_tuyen_chau_xa_phu = bool(cur_city and is_same_city(cur_city, "Tuyền Châu"))
        if is_tuyen_chau_xa_phu and tuyen_chau_arrival:
            max_xa_phu_nudges = int(tuyen_chau_arrival.get("max_interact_clicks", 1))
        xa_phu_nudge_count = 0
        at_fixed_tuyen_chau_xa_phu = False
        body_xa_phu = None
        while time.time() - start_wait < 25.0:
            time.sleep(0.25)
            frame = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
            if frame is None:
                continue

            # Nút chat cố định nằm trong vùng từng dùng để dò bong bóng NPC và
            # có thể bị nhận nhầm. Nếu bảng Kênh chat đã mở thì đóng ngay, không
            # tiếp tục dùng frame này để quyết định tương tác với Xa phu.
            if self._dismiss_unexpected_popups(frame):
                continue

            if hasattr(self, "_text") and self._text:
                body_boxes = self._text.read(frame, (100, 150, 1450, 700))
                body_xa_phu = next((
                    b for b in body_boxes 
                    if "xa phu" in normalize_game_text(b.text) and "thu kho" not in normalize_game_text(b.text)
                ), None)
                if body_xa_phu:
                    xa_phu_label_streak += 1
                    logger.debug(
                        f"[{ctx.account_id}] Đã thấy Xa Phu; chờ bong bóng tương tác, "
                        "không bấm vào thân NPC."
                    )
                else:
                    xa_phu_label_streak = 0

            # Cập nhật trạng thái di chuyển TRƯỚC khi quyết định bấm. Cách cũ
            # kiểm tra bấm trước rồi mới cập nhật tọa độ, nên có thể dùng
            # nhầm trạng thái "đã dừng" của frame trước.
            _, cur_coords = self._get_current_location_from_hud(frame)
            at_fixed_tuyen_chau_xa_phu = False
            if is_tuyen_chau_xa_phu and tuyen_chau_arrival and cur_coords:
                arrival_x = int(tuyen_chau_arrival.get("x", 188))
                arrival_y = int(tuyen_chau_arrival.get("y", 179))
                arrival_tol = int(tuyen_chau_arrival.get("tolerance", 3))
                at_fixed_tuyen_chau_xa_phu = (
                    abs(cur_coords[0] - arrival_x) <= arrival_tol
                    and abs(cur_coords[1] - arrival_y) <= arrival_tol
                )
                if at_fixed_tuyen_chau_xa_phu:
                    logger.debug(
                        f"[{ctx.account_id}] Đã vào điểm cố định Xa phu Tuyền Châu "
                        f"{cur_coords}; khóa mọi điều chỉnh vị trí, chỉ cho phép click bong bóng."
                    )
            now = time.time()
            if cur_coords:
                if cur_coords == last_cur_coords:
                    stationary_count += 1
                    if stable_since is None:
                        stable_since = now
                else:
                    stationary_count = 0
                    stable_since = now
                last_cur_coords = cur_coords
            else:
                stationary_count = 0
                stable_since = None

            stable_for = (now - stable_since) if stable_since is not None else 0.0

            # Chỉ đóng hội thoại khác khi có nhãn NPC/hành động đặc trưng.
            # Các câu chung như "võ công" dễ xuất hiện ở HUD hoặc hộp thoại Xa Phu.
            bxs = self._text.read(frame, (150, 320, 1450, 850)) if hasattr(self, "_text") and self._text else []
            has_xa_phu_dialog = self._has_xa_phu_dialog_evidence(bxs, cur_city=cur_city)
            wrong_dialog = self._has_other_npc_dialog_evidence(bxs)
            if wrong_dialog and not has_xa_phu_dialog:
                logger.warning(f"[{ctx.account_id}] Mở nhầm hộp thoại NPC khác (không phải Xa Phu), đóng ngay để tiếp tục tới Xa Phu...")
                self._ensure_npc_dialog_closed()
                self._dismiss_unexpected_popups()
                time.sleep(0.4)
                continue

            if has_xa_phu_dialog:
                dialog_opened = True
                break

            # CHỈ click đối thoại khi đúng Xa Phu:
            # Tuyệt đối không click icon bong bóng khi đang chạy bộ trên đường!
            # Nhãn Xa Phu + icon tương tác trên cùng frame đã đủ chứng minh
            # nhân vật ở sát đúng NPC. Bấm ngay, không chờ đứng yên 1.5 giây
            # hoặc hai frame vì auto-path có thể chạy vượt qua NPC.
            has_xa_phu_name = (body_xa_phu is not None) or any(
                "xa phu" in normalize_game_text(b.text) for b in (body_boxes if 'body_boxes' in locals() else [])
            )
            # Minimap đã chọn đúng Xa Phu. Nếu tên NPC bị OCR hụt, cho phép
            # tương tác khi nhân vật đã đứng yên tại điểm auto-path và bong bóng
            # đang hiện; tuyệt đối không bấm bong bóng khi còn đang chạy.
            arrived_at_target = stable_for >= 0.8 and stationary_count >= 2
            if (
                xa_phu_nudge_count < max_xa_phu_nudges
                and (has_xa_phu_name or arrived_at_target or at_fixed_tuyen_chau_xa_phu)
                and self._is_npc_interact_bubble_present(frame)
            ):
                logger.info(
                    f"[{ctx.account_id}] Đã tới điểm Xa Phu"
                    f"{' (OCR tên NPC)' if has_xa_phu_name else ' (đã đứng yên, OCR hụt tên)'}; "
                    "bấm bong bóng tương tác..."
                )
                self._click_npc_interact_bubble(frame)
                xa_phu_nudge_count += 1
                time.sleep(0.4)
                post_frame = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
                if post_frame is not None and hasattr(self, "_text") and self._text:
                    post_bxs = self._text.read(post_frame, (150, 320, 1450, 850))
                    if self._has_xa_phu_dialog_evidence(post_bxs, cur_city=cur_city):
                        dialog_opened = True
                        break

        if not dialog_opened:
            # Thử lại vài lần nhưng chỉ được bấm khi OCR nhìn thấy chính xác nhãn
            # Xa phu hoặc bong bóng tương tác. Tuyệt đối không bấm tọa độ mặt đất
            # cố định vì nhân vật đang đứng cạnh NPC sẽ bị kéo đi xa.
            for retry_idx in range(3):
                frame_target = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
                if frame_target is not None and hasattr(self, "_text") and self._text:
                    body_boxes = self._text.read(frame_target, (100, 150, 1450, 700))
                    body_xa_phu = next((b for b in body_boxes if "xa phu" in normalize_game_text(b.text) and "thu kho" not in normalize_game_text(b.text)), None)
                    _, retry_coords = self._get_current_location_from_hud(frame_target)
                    retry_stable = (
                        retry_coords is not None
                        and retry_coords == last_cur_coords
                        and stable_since is not None
                        and time.time() - stable_since >= 1.5
                    )
                    bubble_on_retry = self._is_npc_interact_bubble_present(frame_target)
                    if (
                        xa_phu_nudge_count < max_xa_phu_nudges
                        and (retry_stable or at_fixed_tuyen_chau_xa_phu or body_xa_phu)
                        and bubble_on_retry
                    ):
                        logger.info(
                            f"[{ctx.account_id}] Đã tới điểm Xa Phu; "
                            f"bấm bong bóng tương tác trong lượt thử lại ({retry_idx + 1}/3)..."
                        )
                        self._click_npc_interact_bubble(frame_target)
                        xa_phu_nudge_count += 1
                    elif (
                        xa_phu_nudge_count < max_xa_phu_nudges
                        and (retry_stable or at_fixed_tuyen_chau_xa_phu or body_xa_phu)
                        and not bubble_on_retry
                        and retry_idx == 0
                    ):
                        # Nếu đã tới Xa Phu mà chưa thấy bong bóng, nhích trái-phải để game hiển thị NPC
                        self._nudge_left_right_to_reveal_npc(ctx)
                time.sleep(0.4)

                frame = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
                if frame is not None:
                    bxs = self._text.read(frame, (150, 320, 1450, 850)) if hasattr(self, "_text") and self._text else []
                    bxs = [bx for bx in bxs if bx.center_y < 950 and not (1100 <= bx.center_x <= 1250 and 300 <= bx.center_y <= 450)]
                    dialog_opened = self._has_xa_phu_dialog_evidence(bxs, cur_city=cur_city)
                    if dialog_opened:
                        break

        if not dialog_opened:
            logger.warning(
                f"[{ctx.account_id}] Chưa xác nhận được hội thoại Xa phu sau "
                f"{int(time.time() - start_wait)}s; không coi là đã tới NPC."
            )
            if not allow_world_map_fallback:
                logger.warning(
                    f"[{ctx.account_id}] Chưa tới được Xa phu tại '{cur_city}'; "
                    "không chuyển sang chọn NPC trên bản đồ đích."
                )
                return False
            return self._walk_intercity_via_world_map(
                ctx, city_name, max_wait=90.0,
                retry_local_xa_phu=allow_world_map_fallback,
            )

        logger.info(f"[{ctx.account_id}] Đã mở hộp thoại Xa Phu! Đang tìm tùy chọn chuyển đến '{city_name}'...")
        activity._write("🐎", f"[{ctx.account_id}] Chon {city_name} tai Xa Phu...")

        # Kiểm tra ngay nếu hộp thoại Xa Phu báo không đủ tiền ("Có tiền thì hãy ngồi xe!")
        if self._check_xa_phu_out_of_money(boxes=bxs):
            logger.warning(
                f"[{ctx.account_id}] 💸 KHÔNG ĐỦ TIỀN ĐI XA PHU ('Có tiền thì hãy ngồi xe!'). "
                f"Tự động chuyển hẳn sang CHẠY BỘ LIÊN THÀNH sang '{city_name}' cho tới hết nhiệm vụ..."
            )
            activity._write("💸", f"[{ctx.account_id}] Het tien Xa Phu -> Chay bo sang {city_name}...")
            ctx.extra["out_of_money"] = True
            if hasattr(self, "_checkpoint"):
                self._checkpoint.save(ctx)
            self._ensure_npc_dialog_closed()
            return self._walk_intercity_via_world_map(
                ctx, city_name, max_wait=120.0,
                retry_local_xa_phu=False,
            )

        # 5. Quét các tùy chọn trong hộp thoại Xa Phu (vùng nút: x: 320..1380, y: 480..720)
        teleport_clicked = False
        for attempt in range(3):
            if attempt == 0 and bxs:
                boxes = [b for b in bxs if 380 <= b.center_y <= 820 and b.center_x >= 180]
            else:
                frame = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
                if frame is None:
                    time.sleep(0.2)
                    continue

                raw_boxes = self._text.read(frame, (180, 380, 1420, 800)) if hasattr(self, "_text") and self._text else []
                boxes = [b for b in raw_boxes if 380 <= b.center_y <= 820 and b.center_x >= 180]
            target_box = None
            category_box = None
            tokens_target = norm_target.split()
            first_token = tokens_target[0] if tokens_target else ""

            for b in boxes:
                norm_b = normalize_game_text(b.text)
                b_clean = norm_b.replace(" ", "")
                # Riêng với "Đại Lý": Tìm cứng "Đại lý", cấm tuyệt đối nhận nhầm "Đại Lý phủ"
                if norm_target == "dai ly" and "phu" in norm_b.split():
                    continue
                # City choices occupy the button grid only. OCR text above the
                # grid (e.g. dialogue/HUD fragments like "uong cha") must not
                # be accepted just because its fuzzy score is high.
                is_city_choice_area = 240 <= b.center_x <= 1380 and 460 <= b.center_y <= 780
                if is_city_choice_area:
                    if is_same_city(b.text, city_name) or norm_target in b_clean or norm_target in norm_b:
                        target_box = b
                        break
                    # City-specific high-confidence tokens:
                    b_words = norm_b.split()
                    if norm_target == "tuyen chau" and any(w in b_words for w in ("tuyen", "tupen", "tuyenchau")):
                        target_box = b
                        break
                    if norm_target == "bien kinh" and "bien" in b_words:
                        target_box = b
                        break
                    if norm_target == "thanh do" and ("thanh do" in norm_b or ("thanh" in b_words and "do" in b_words)):
                        target_box = b
                        break
                    if norm_target in ("phung tuong", "phuong tuong") and any(w in b_words for w in ("phung", "phuong")):
                        target_box = b
                        break

                    ratio = SequenceMatcher(None, norm_target, b_clean).ratio()
                    if (first_token and first_token in norm_b and ratio >= 0.50) or ratio >= 0.70:
                        target_box = b
                        break
                if any(w in norm_b for w in ("thanh thi", "dai thanh", "cac dai thanh", "thanh pho", "noi khac", "tiep theo", "trang sau", "den noi khac", "trang truoc")):
                    category_box = b

            if not target_box:
                logger.debug(f"[{ctx.account_id}] Xa Phu attempt {attempt} OCR boxes: {[(b.text, b.center_x, b.center_y) for b in boxes]}")

            if target_box:
                logger.info(f"[{ctx.account_id}] Tìm thấy tùy chọn '{target_box.text}' tại ({target_box.center_x}, {target_box.center_y}), bấm...")
                self._input.tap(target_box.center_x, target_box.center_y)
                teleport_clicked = True
                time.sleep(0.4)
                
                # Kiểm tra nếu xuất hiện bảng thông báo hết tiền/không đủ ngân lượng
                chk_frame = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
                if self._check_xa_phu_out_of_money(chk_frame):
                    logger.warning(f"[{ctx.account_id}] 💸 KHÔNG ĐỦ TIỀN ĐI XA PHU (cần 3 lượng). Tự động chuyển sang CHẠY BỘ LIÊN THÀNH sang '{city_name}'...")
                    activity._write("💸", f"[{ctx.account_id}] Het tien Xa Phu -> Chay bo sang {city_name}...")
                    ctx.extra["out_of_money"] = True
                    self._checkpoint.save(ctx)
                    self._ensure_npc_dialog_closed()
                    return self._walk_intercity_via_world_map(
                        ctx, city_name, max_wait=90.0,
                        retry_local_xa_phu=False,
                    )

                # Kiểm tra nếu xuất hiện bảng thông báo xác nhận chuyển map
                if chk_frame is not None and hasattr(self, "_text") and self._text:
                    chk_boxes = self._text.read(chk_frame, (350, 300, 1250, 750))
                    confirm_btn = self._find_travel_confirm_button(chk_boxes)
                    if confirm_btn:
                        logger.info(f"[{ctx.account_id}] Phát hiện nút xác nhận chuyển map '{confirm_btn.text}' tại ({confirm_btn.center_x}, {confirm_btn.center_y}), bấm...")
                        self._input.tap(confirm_btn.center_x, confirm_btn.center_y)
                        time.sleep(0.3)
                break
            elif category_box and attempt < 2:
                logger.info(f"[{ctx.account_id}] Chưa thấy trực tiếp {city_name}, bấm menu chuyển trang '{category_box.text}' tại ({category_box.center_x}, {category_box.center_y})...")
                self._input.tap(category_box.center_x, category_box.center_y)
                time.sleep(0.55)
            elif attempt == 2:
                xa_phu_cfg = coords.get("commerce_quest", {}).get("xa_phu_cities", {}) or coords.get("xa_phu_cities", {})
                for c_name, c_pos in xa_phu_cfg.items():
                    norm_c = normalize_game_text(c_name)
                    if norm_c == norm_target or norm_target in norm_c or norm_c in norm_target:
                        logger.info(f"[{ctx.account_id}] Sử dụng tọa độ nút '{c_name}' cố định tại Xa Phu ({c_pos['x']}, {c_pos['y']})...")
                        self._input.tap(c_pos["x"], c_pos["y"])
                        teleport_clicked = True
                        time.sleep(0.4)
                        chk_frame = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
                        
                        if self._check_xa_phu_out_of_money(chk_frame):
                            logger.warning(f"[{ctx.account_id}] 💸 KHÔNG ĐỦ TIỀN ĐI XA PHU (cần 3 lượng). Tự động chuyển sang CHẠY BỘ LIÊN THÀNH sang '{city_name}'...")
                            activity._write("💸", f"[{ctx.account_id}] Het tien Xa Phu -> Chay bo sang {city_name}...")
                            ctx.extra["out_of_money"] = True
                            self._checkpoint.save(ctx)
                            self._ensure_npc_dialog_closed()
                            return self._walk_intercity_via_world_map(
                                ctx, city_name, max_wait=90.0,
                                retry_local_xa_phu=False,
                            )

                        if chk_frame is not None and hasattr(self, "_text") and self._text:
                            chk_boxes = self._text.read(chk_frame, (350, 300, 1250, 750))
                            confirm_btn = self._find_travel_confirm_button(chk_boxes)
                            if confirm_btn:
                                logger.info(f"[{ctx.account_id}] Phát hiện nút xác nhận chuyển map '{confirm_btn.text}' tại ({confirm_btn.center_x}, {confirm_btn.center_y}), bấm...")
                                self._input.tap(confirm_btn.center_x, confirm_btn.center_y)
                                time.sleep(0.3)
                        break
                if teleport_clicked:
                    break
                time.sleep(0.2)

        if not teleport_clicked:
            logger.warning(f"[{ctx.account_id}] Không bấm được tùy chọn sang '{city_name}' trong Xa Phu. Chuyển sang chạy bộ liên thành...")
            self._ensure_npc_dialog_closed()
            return self._walk_intercity_via_world_map(
                ctx, city_name, max_wait=90.0,
                retry_local_xa_phu=allow_world_map_fallback,
            )

        # 6. Chờ chuyển cảnh và chỉ xác nhận thành công khi HUD thật sự đổi map.
        # Máy chậm/Remote LDPlayer có thể mất hơn 2.5 giây; timeout cũ khiến bot
        # tưởng dịch chuyển thất bại rồi phát lệnh chạy bộ trong lúc game còn load.
        xa_phu_load_timeout = 15.0
        logger.info(
            f"[{ctx.account_id}] Đã chọn '{city_name}' tại Xa Phu; "
            f"đang chờ HUD chuyển từ '{cur_city or 'map hiện tại'}' sang '{city_name}' "
            f"(tối đa {int(xa_phu_load_timeout)}s)..."
        )
        activity._write("⏳", f"[{ctx.account_id}] Dang load vao {city_name}...")
        time.sleep(0.4)
        load_started = time.time()
        while time.time() - load_started < xa_phu_load_timeout:
            cur, _ = self._get_current_location_from_hud()
            if cur and is_same_city(cur, city_name):
                logger.info(f"[{ctx.account_id}] ✅ Đã đến thành công {city_name} qua Xa Phu (HUD: '{cur}')!")
                activity._write("📍", f"[{ctx.account_id}] Da den {city_name}!")
                time.sleep(0.2)
                return True
            # Trong lúc chuyển cảnh OCR có thể vẫn giữ map cũ hoặc trả None.
            # Không phát thêm lệnh di chuyển cho tới hết timeout này.
            time.sleep(0.5)

        cur_fail, _ = self._get_current_location_from_hud()
        chk_money_frame = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
        if self._check_xa_phu_out_of_money(chk_money_frame):
            logger.warning(f"[{ctx.account_id}] 💸 Không load được '{city_name}' qua Xa Phu (HUD vẫn ở: '{cur_fail}') -> Tài khoản không đủ tiền! Đánh dấu out_of_money và chuyển ngay sang CHẠY BỘ LIÊN THÀNH...")
            activity._write("💸", f"[{ctx.account_id}] Khong du tien Xa Phu -> Chay bo sang {city_name}...")
            ctx.extra["out_of_money"] = True
            self._checkpoint.save(ctx)
            self._ensure_npc_dialog_closed()
            return self._walk_intercity_via_world_map(
                ctx, city_name, max_wait=90.0,
                retry_local_xa_phu=False,
            )

        logger.warning(f"[{ctx.account_id}] Chưa dịch chuyển được sang '{city_name}' qua Xa Phu (HUD vẫn ở: '{cur_fail}'), đóng hộp thoại và chuyển sang chạy bộ...")
        self._ensure_npc_dialog_closed()
        return self._walk_intercity_via_world_map(
            ctx, city_name, max_wait=90.0,
            retry_local_xa_phu=allow_world_map_fallback,
        )

    def _input_coordinate_field(self, x_field: int, y_field: int, value: int, idx: Optional[int] = None) -> None:
        """
        Click vào ô tọa độ (X hoặc Y), xóa sạch toàn bộ tọa độ cũ, sau đó nhập giá trị mới.
        Dùng trực tiếp ADB input text vì tọa độ là số thuần ASCII, tương thích 100% không phụ thuộc Android IME.
        """
        self._input.tap(x_field, y_field)
        time.sleep(0.1)
        self._input.tap(x_field + 20, y_field)
        time.sleep(0.08)

        # Xóa số cũ bằng Backspace và Del nhanh gọn
        clear_cmds = [
            "input keyevent 123",          # MOVE_END
        ] + ["input keyevent 67"] * 6 + [  # Backspace 6 lần
            "input keyevent 122",          # MOVE_HOME
        ] + ["input keyevent 112"] * 6     # Forward Del 6 lần
        
        try:
            self._adb._run(["shell", "; ".join(clear_cmds)])
        except Exception as e:
            logger.debug(f"Xóa ô tọa độ ({x_field}, {y_field}) gặp lỗi: {e}")

        time.sleep(0.08)

        try:
            self._adb.input_text(str(value))
        except Exception:
            try:
                self._adb._run(["shell", f"input text {value}"])
            except Exception as e:
                logger.warning(f"Không thể nhập tọa độ {value} qua ADB: {e}")
        time.sleep(0.12)

    def _go_to_city_coordinate(self, ctx: StateContext, city: str, x: int, y: int) -> bool:
        """Đến vị trí của thành phố trước, sau đó mở bản đồ nhập tọa độ (x, y) của thương nhân."""
        self._dismiss_unexpected_popups()
        if hasattr(self, "_ensure_npc_dialog_closed"):
            self._ensure_npc_dialog_closed()
        time.sleep(0.1)

        # 1. Đi tới thành phố mục tiêu qua Tìm map (chỉ chuyển map nếu chưa ở trong thành phố này)
        if not self._switch_to_city(ctx, city, force=False):
            logger.warning(f"[{ctx.account_id}] Không thể chuyển sang thành phố '{city}' qua Xa Phu! Hủy nhập tọa độ ({x}/{y}) tại thành phố này.")
            return False

        # 2. Xác nhận chắc chắn nhân vật đã thực sự ở thành phố mục tiêu
        cur_city, _ = self._get_current_location_from_hud()
        if cur_city:
            is_there = is_same_city(cur_city, city)
            # Riêng với "Đại Lý": Tuyệt đối không chấp nhận nếu đang ở trong Đại Lý Phủ
            norm_cur = normalize_game_text(cur_city)
            if city == "Đại Lý" and "phu" in norm_cur.split():
                is_there = False
            if not is_there:
                logger.error(f"[{ctx.account_id}] ❌ Hiện tại đang ở '{cur_city}', KHÔNG PHẢI thành phố '{city}'! Hủy nhập tọa độ ({x}/{y})!")
                activity._write("⚠️", f"[{ctx.account_id}] Vẫn ở {cur_city}, hủy nhập tọa độ {city}!")
                return False

        # 3. Sau khi đã ở trong thành, mở bản đồ thành phố để nhập tọa độ
        if not self._ensure_map_open():
            logger.error(f"[{ctx.account_id}] Không thể mở bản đồ thành phố '{city}'! Hủy nhập tọa độ ({x}/{y}) để tránh bấm nhầm thanh chat.")
            self._dismiss_unexpected_popups()
            return False
        time.sleep(0.2)

        coords = self._loader.load_ui_coords(reload=True)
        commerce = coords.get("commerce_quest", {})
        field_x = commerce.get("map_coordinate_x", {"x": 667, "y": 780})
        field_y = commerce.get("map_coordinate_y", {"x": 804, "y": 780})
        confirm = commerce.get("map_coordinate_confirm", {"x": 1018, "y": 783})

        idx = self._get_instance_index(ctx)

        # 1. Nhập tọa độ X (xóa sạch số cũ rồi gõ số mới)
        logger.info(f"[{ctx.account_id}] Xóa cũ và nhập tọa độ X: {x}...")
        self._input_coordinate_field(field_x["x"], field_x["y"], x, idx=idx)

        # 2. Nhập tọa độ Y (xóa sạch số cũ rồi gõ số mới)
        logger.info(f"[{ctx.account_id}] Xóa cũ và nhập tọa độ Y: {y}...")
        self._input_coordinate_field(field_y["x"], field_y["y"], y, idx=idx)

        # 3. Nhấn "Xác nhận"
        logger.info(f"[{ctx.account_id}] Xác nhận tọa độ ({x}/{y}) tại {city}...")
        self._input.tap(confirm["x"], confirm["y"])
        time.sleep(0.25)

        # Khi bấm Xác nhận thành công, nếu bản đồ chưa tự đóng, chủ động đóng bản đồ để nhân vật bắt đầu chạy
        if self._is_map_open():
            time.sleep(0.3)
            self._input.tap(confirm["x"], confirm["y"])
            time.sleep(0.3)
            if self._is_map_open():
                logger.info(f"[{ctx.account_id}] Đã bấm Xác nhận ({x}/{y}) tại {city}, đóng bản đồ để nhân vật chạy...")
                self._close_map()
                time.sleep(0.2)

        self._press_buff_speed()
        logger.info(f"[{ctx.account_id}] Bắt đầu chạy bộ đến tọa độ {city} ({x}/{y})")
        activity._write("🏃", f"[{ctx.account_id}] Chay den toa do {city} ({x}/{y})...")
        return True

    def _is_western_merchant_dialog(self, frame: Optional[np.ndarray] = None) -> bool:
        """Kiểm tra xem hộp thoại mua hàng của Thương nhân Tây Vực ('Đồng ý mua') có đang mở hay không."""
        if frame is None and hasattr(self, "_screen") and self._screen:
            frame = self._screen.capture()
        if frame is None or not hasattr(self, "_text") or not self._text:
            return False
        boxes = self._text.read(frame, (150, 200, 1450, 850))
        for b in boxes:
            norm = normalize_game_text(b.text).replace(" ", "")
            if "dongymua" in norm or ("dongy" in norm and "mua" in norm):
                return True
        return False

    def _detect_and_interact_merchant(self, ctx: StateContext, city_name: str, frame: Optional[np.ndarray], aliases: list[str], allow_bubble: bool = False) -> bool:
        """Kiểm tra và click tương tác với Thương nhân Tây Vực nếu xuất hiện trên màn hình."""
        if frame is None and hasattr(self, "_screen") and self._screen is not None:
            frame = self._screen.capture()
        if frame is None or not hasattr(self, "_text") or self._text is None:
            return False

        # 0. Kiểm tra nếu hộp thoại 'Đồng ý mua' đã mở sẵn
        if self._is_western_merchant_dialog(frame):
            logger.info(f"[{ctx.account_id}] Hộp thoại mua hàng của Thương nhân Tây Vực đã mở sẵn!")
            ctx.extra["merchant_found_city"] = city_name
            self._checkpoint.save(ctx)
            return True

        # 1. Kiểm tra thanh mục tiêu NPC ở đỉnh màn hình (450..1100, 10..130)
        top_boxes = [b for b in self._text.read(frame, (450, 10, 1100, 130)) if b.center_y <= 130]
        for tb in top_boxes:
            norm_tb = normalize_game_text(tb.text).replace(" ", "")
            if "thuongnhan" in norm_tb or "tayvuc" in norm_tb or ("thuong" in norm_tb and "tay" in norm_tb):
                logger.info(f"[{ctx.account_id}] Phát hiện Thương nhân Tây Vực trên thanh mục tiêu đỉnh màn hình! Bấm tại ({tb.center_x}, {tb.center_y})...")
                self._input.tap(tb.center_x, tb.center_y)
                time.sleep(0.5)
                for _ in range(6):
                    f_chk = self._screen.capture()
                    if self._is_western_merchant_dialog(f_chk):
                        logger.info(f"[{ctx.account_id}] Đã mở hộp thoại Thương nhân Tây Vực từ thanh mục tiêu!")
                        ctx.extra["merchant_found_city"] = city_name
                        self._checkpoint.save(ctx)
                        return True
                    if self._is_npc_dialog_open(f_chk) and not self._is_western_merchant_dialog(f_chk):
                        logger.warning(f"[{ctx.account_id}] Mở nhầm NPC khác khi target Thương nhân ({city_name}), đóng ngay!")
                        self._ensure_npc_dialog_closed()
                    if hasattr(self, "_click_npc_interact_bubble") and self._is_npc_interact_bubble_present(f_chk):
                        self._click_npc_interact_bubble(f_chk)
                        time.sleep(0.4)
                        if self._is_western_merchant_dialog():
                            ctx.extra["merchant_found_city"] = city_name
                            self._checkpoint.save(ctx)
                            return True
                    time.sleep(0.4)
                ctx.extra["merchant_found_city"] = city_name
                self._checkpoint.save(ctx)
                return True

        # 2. Quét vùng trung tâm màn hình tìm tên Thương nhân (Bỏ qua khung chat đáy màn hình và icon chat nhanh)
        raw_boxes = self._text.read(frame, (200, 150, 1400, 850))
        boxes = [b for b in raw_boxes if b.center_y < 950 and not (1100 <= b.center_x <= 1250 and 300 <= b.center_y <= 450)]
        merchant_boxes = [box for box in boxes if is_western_merchant_name(box.text)]
        match = self._text.best_match(merchant_boxes, aliases, minimum=0.50)
        if not match:
            for b in merchant_boxes:
                norm_b = normalize_game_text(b.text).replace(" ", "")
                if ("thuongnhan" in norm_b and ("tay" in norm_b or "vuc" in norm_b)) or "tayvuc" in norm_b:
                    match = (0.9, b, b.text)
                    break

        if match:
            score, box, _ = match
            logger.info(f"[{ctx.account_id}] Đã phát hiện Thương nhân Tây Vực tại {city_name}: '{box.text}' ({box.center_x}, {box.center_y})!")
            activity._write("🛍️", f"[{ctx.account_id}] Gap Thuong nhan tai {city_name}!")
            # Tap vào NPC Thương nhân để nhân vật chạy lại gần
            self._input.tap(box.center_x, box.center_y + 35)

            # Chờ nhân vật chạy lại gần Thương nhân (tối đa ~3s), KHÔNG bấm bong bóng ngay lập tức
            # để tránh bấm nhầm bong bóng của NPC khác đứng kế bên (như Vương Cẩn):
            for step in range(8):
                time.sleep(0.4)
                f_chk = self._screen.capture()
                if f_chk is None:
                    continue

                # Nếu hộp thoại 'Đồng ý mua' của Thương nhân đã mở ra:
                if self._is_western_merchant_dialog(f_chk):
                    logger.info(f"[{ctx.account_id}] Hộp thoại 'Đồng ý mua' của Thương nhân Tây Vực đã mở ra thành công!")
                    ctx.extra["merchant_found_city"] = city_name
                    self._checkpoint.save(ctx)
                    return True

                # Nếu lỡ mở nhầm hộp thoại NPC khác (như Vương Cẩn "Tìm ta có việc gì?"):
                if self._is_npc_dialog_open(f_chk) and not self._is_western_merchant_dialog(f_chk):
                    logger.warning(f"[{ctx.account_id}] Mở nhầm hộp thoại NPC khác ({city_name}), đóng ngay để tiếp tục tới Thương nhân!")
                    self._ensure_npc_dialog_closed()
                    self._input.tap(box.center_x, box.center_y + 35)
                    continue

                # Kiểm tra xem đã target Thương nhân hoặc tới gần và có bong bóng:
                top_boxes = [b for b in self._text.read(f_chk, (450, 10, 1100, 130)) if b.center_y <= 130]
                is_merchant_targeted = any(
                    "thuongnhan" in normalize_game_text(tb.text).replace(" ", "")
                    or "tayvuc" in normalize_game_text(tb.text).replace(" ", "")
                    for tb in top_boxes
                )

                if is_merchant_targeted and hasattr(self, "_click_npc_interact_bubble") and self._is_npc_interact_bubble_present(f_chk):
                    logger.info(f"[{ctx.account_id}] Đã target Thương nhân Tây Vực và có bong bóng đối thoại, bấm tương tác...")
                    self._click_npc_interact_bubble(f_chk)
                    time.sleep(0.4)
                    f_after = self._screen.capture()
                    if self._is_western_merchant_dialog(f_after):
                        logger.info(f"[{ctx.account_id}] Mở thành công hộp thoại Thương nhân Tây Vực!")
                        ctx.extra["merchant_found_city"] = city_name
                        self._checkpoint.save(ctx)
                        return True
                    elif self._is_npc_dialog_open(f_after):
                        logger.warning(f"[{ctx.account_id}] Mở nhầm hộp thoại NPC khác khi bấm bong bóng ({city_name}), đóng ngay!")
                        self._ensure_npc_dialog_closed()

            # Sau khi chờ chạy lại gần, kiểm tra lần cuối:
            f_end = self._screen.capture()
            if f_end is not None:
                if self._is_western_merchant_dialog(f_end):
                    ctx.extra["merchant_found_city"] = city_name
                    self._checkpoint.save(ctx)
                    return True
                top_boxes = [b for b in self._text.read(f_end, (450, 10, 1100, 130)) if b.center_y <= 130]
                if any("thuongnhan" in normalize_game_text(tb.text).replace(" ", "") or "tayvuc" in normalize_game_text(tb.text).replace(" ", "") for tb in top_boxes):
                    if hasattr(self, "_click_npc_interact_bubble") and self._is_npc_interact_bubble_present(f_end):
                        self._click_npc_interact_bubble(f_end)
                        time.sleep(0.4)
                        if self._is_western_merchant_dialog():
                            ctx.extra["merchant_found_city"] = city_name
                            self._checkpoint.save(ctx)
                            return True

            ctx.extra["merchant_found_city"] = city_name
            self._checkpoint.save(ctx)
            return True

        return False

    def _find_western_merchant(self, ctx: StateContext) -> bool:
        """
        Tìm Thương nhân Tây Vực:
        1. Ưu tiên kiểm tra tọa độ tại thành hiện tại trước tiên.
        2. Đi thẳng tới tọa độ thiết lập trong Cài đặt mà không cần điều khiển tay tuần tra lăng quăng.
        3. Nếu không có ở thành hiện tại, tiếp tục kiểm tra lần lượt các thành còn lại.
        """
        commerce = self._loader.load_ui_coords(reload=True).get("commerce_quest", {})
        spawns = list(commerce.get("western_merchant_spawns", [
            {"city": "Tuyền Châu", "x": 187, "y": 191},
            {"city": "Biện Kinh", "x": 180, "y": 185},
            {"city": "Thành Đô", "x": 225, "y": 225},
            {"city": "Tương Dương", "x": 187, "y": 180},
            {"city": "Dương Châu", "x": 206, "y": 194},
            {"city": "Đại Lý", "x": 178, "y": 187},
        ]))
        aliases = ["Thương nhân Tây Vực", "Thương nhân Tây Hồ", "Thuong nhan Tay Vuc", "Thương nhân phương tây", "thuong nhan", "tay vuc", "thuong nhan tay"]

        # Nhận diện thành phố hiện tại
        cur_city, _ = self._get_current_location_from_hud()
        if not cur_city:
            cur_city = self._detect_current_city()

        # Ưu tiên thành phố được chỉ dẫn trong nhiệm vụ nếu có (ví dụ: "Có thể đến Biện Kinh")
        suggested_city = None
        quest_text = ctx.extra.get("quest_text", "")
        if quest_text:
            norm_q = normalize_game_text(quest_text)
            for c in ("Biện Kinh", "Thành Đô", "Tương Dương", "Dương Châu", "Đại Lý", "Tuyền Châu", "Phượng Tường"):
                norm_c = normalize_game_text(c)
                if f"den {norm_c}" in norm_q or f"tai {norm_c}" in norm_q or f"o {norm_c}" in norm_q or f"o cho {norm_c}" in norm_q:
                    suggested_city = c
                    break

        if suggested_city:
            sugg_spawns = [sp for sp in spawns if is_same_city(suggested_city, sp.get("city", ""))]
            rest_spawns = [sp for sp in spawns if not is_same_city(suggested_city, sp.get("city", ""))]
            if sugg_spawns:
                logger.info(f"[{ctx.account_id}] Nhiệm vụ gợi ý đến '{suggested_city}'! Ưu tiên tìm Thương nhân tại thành này trước...")
                activity._write("📍", f"[{ctx.account_id}] NV goi y den {suggested_city}! Kiem tra truoc...")
                spawns = sugg_spawns + rest_spawns
        elif cur_city:
            curr_spawns = []
            other_spawns = []
            for sp in spawns:
                if is_same_city(cur_city, sp.get("city", "")):
                    curr_spawns.append(sp)
                else:
                    other_spawns.append(sp)

            if curr_spawns:
                logger.info(f"[{ctx.account_id}] Đang ở thành hiện tại: {cur_city}. Ưu tiên kiểm tra tọa độ Thương nhân Tây Vực tại thành này trước!")
                activity._write("📍", f"[{ctx.account_id}] Kiem tra {cur_city} (thanh hien tai) truoc...")
                spawns = curr_spawns + other_spawns

        for spawn in spawns:
            if self._check_quest_timeout(ctx):
                return False
            city_name = spawn["city"]
            target_x, target_y = int(spawn["x"]), int(spawn["y"])
            logger.info(f"[{ctx.account_id}] Tìm Thương nhân Tây Vực tại {city_name} ({target_x}/{target_y})...")

            if not self._go_to_city_coordinate(ctx, city_name, target_x, target_y):
                # Chuyển thành thất bại nghĩa là nhân vật vẫn ở thành cũ. Nếu
                # tiếp tục duyệt spawn kế tiếp, log sẽ nói đang đi nhiều thành
                # khác nhau trong khi HUD không hề đổi. Dừng lượt tìm hiện tại
                # để state machine thử lại từ vị trí thật ở vòng sau.
                actual_city, actual_coords = self._get_current_location_from_hud()
                logger.error(
                    f"[{ctx.account_id}] Không thể chuyển tới '{city_name}'; "
                    f"HUD vẫn ở '{actual_city or 'không xác định'}' {actual_coords or ''}. "
                    "Dừng luân phiên thành phố, không phát thêm lệnh chuyển thành."
                )
                activity._write(
                    "⚠️",
                    f"[{ctx.account_id}] Van o {actual_city or 'thanh cu'}, dung chuyen thanh de thu lai...",
                )
                return False

            # Chờ nhân vật chạy bộ thẳng tới tọa độ đích (tối đa 40 giây, kiểm tra định kỳ 30s)
            start_walk = time.time()
            found_merchant = False
            last_coords = None
            stationary_count = 0
            initial_coords = None
            arrived_near_target = False
            last_30s_check = time.time()
            coords_30s_ago = None

            while time.time() - start_walk < 40.0:
                time.sleep(0.25)
                frame = self._screen.capture()
                if frame is None:
                    continue

                # 1. Đọc tọa độ hiện tại từ HUD góc trên bên phải
                _, cur_coords = self._get_current_location_from_hud(frame)
                dist = 999
                if cur_coords:
                    if initial_coords is None:
                        initial_coords = cur_coords
                    if coords_30s_ago is None:
                        coords_30s_ago = cur_coords
                        last_30s_check = time.time()

                    # Kiểm tra định kỳ mỗi 30s: nếu nhân vật bị đứng yên tại chỗ -> bỏ qua thành này, tìm thương nhân tiếp theo
                    if time.time() - last_30s_check >= 30.0:
                        dist_moved_30s = abs(cur_coords[0] - coords_30s_ago[0]) + abs(cur_coords[1] - coords_30s_ago[1])
                        if dist_moved_30s <= 1:
                            logger.warning(
                                f"[{ctx.account_id}] ⚠️ Trong quá trình di chuyển 30s nhân vật bị đứng yên tại chỗ ({cur_coords[0]}/{cur_coords[1]}), "
                                f"không thể chạy tới tọa độ ({target_x}/{target_y}) tại {city_name}! "
                                f"Bỏ qua thành này và tìm thương nhân ở thành tiếp theo..."
                            )
                            activity._write("⚠️", f"[{ctx.account_id}] Dung yen 30s tai {city_name}, tim thuong nhan tiep theo...")
                            break
                        else:
                            coords_30s_ago = cur_coords
                            last_30s_check = time.time()

                    dist = abs(cur_coords[0] - target_x) + abs(cur_coords[1] - target_y)
                    if dist <= 3:
                        logger.info(f"[{ctx.account_id}] Đã chạy thẳng tới gần tọa độ thiết lập {city_name} ({cur_coords[0]}/{cur_coords[1]}), khoảng cách={dist}")
                        arrived_near_target = True
                        if self._detect_and_interact_merchant(ctx, city_name, frame, aliases, allow_bubble=True):
                            found_merchant = True
                            break
                        if cur_coords == last_coords:
                            stationary_count += 1
                            if stationary_count >= 2:
                                break
                    elif cur_coords == last_coords:
                        stationary_count += 1
                        if stationary_count >= 2 and dist <= 5:
                            logger.info(f"[{ctx.account_id}] Nhân vật đã dừng lại tại tọa độ mục tiêu ({cur_coords[0]}/{cur_coords[1]})")
                            arrived_near_target = True
                            break
                        if stationary_count >= 4 and cur_coords == initial_coords and dist > 8:
                            logger.warning(f"[{ctx.account_id}] Nhân vật chưa di chuyển khỏi ({cur_coords[0]}/{cur_coords[1]}), re-trigger đi tới tọa độ...")
                            self._go_to_city_coordinate(ctx, city_name, target_x, target_y)
                            stationary_count = 0
                    else:
                        stationary_count = 0
                        start_walk = max(start_walk, time.time() - 10.0)
                    last_coords = cur_coords

                # 2. Quét xem Thương nhân Tây Vực đã xuất hiện chưa
                if not found_merchant and self._detect_and_interact_merchant(ctx, city_name, frame, aliases):
                    found_merchant = True
                    break

            if found_merchant:
                return True

            # Sau khi tới đúng tọa độ thiết lập (hoặc dừng hẳn tại đích), kiểm tra nhanh tại chỗ
            # Thử 2 lần (mỗi lần 0.3s) để game kịp render NPC và text
            if arrived_near_target or dist <= 4:
                activity._write("🔍", f"[{ctx.account_id}] Kiem tra Thuong nhan tai {city_name}...")
                bubble_failed_once = False
                for attempt in range(2):
                    time.sleep(0.3)
                    frame_final = self._screen.capture()
                    if self._detect_and_interact_merchant(ctx, city_name, frame_final, aliases, allow_bubble=True):
                        return True
                    # Nếu đứng sát tọa độ đích (dist <= 3) và thấy icon bong bóng đối thoại NPC:
                    if not bubble_failed_once and dist <= 3 and hasattr(self, "_is_npc_interact_bubble_present") and self._is_npc_interact_bubble_present(frame_final):
                        logger.info(f"[{ctx.account_id}] Đang ở sát tọa độ Thương nhân Tây Vực {city_name} (dist={dist}) và có bong bóng đối thoại! Bấm tương tác...")
                        if self._click_npc_interact_bubble():
                            time.sleep(0.4)
                            scr_after = self._screen.capture()
                            if self._is_western_merchant_dialog(scr_after):
                                logger.info(f"[{ctx.account_id}] Mở thành công hộp thoại mua hàng của Thương nhân Tây Vực qua bong bóng đối thoại!")
                                ctx.extra["merchant_found_city"] = city_name
                                self._checkpoint.save(ctx)
                                return True
                            elif self._is_npc_dialog_open(scr_after):
                                logger.warning(f"[{ctx.account_id}] Bong bóng đối thoại là của NPC khác kế bên ({city_name}), đóng ngay để không che màn hình!")
                                self._ensure_npc_dialog_closed()
                            bubble_failed_once = True

            if arrived_near_target or dist <= 4:
                logger.info(f"[{ctx.account_id}] Đã tới đúng tọa độ ({target_x}/{target_y}) tại {city_name}: KHÔNG CÓ Thương nhân Tây Vực. Tiếp tục kiểm tra thành tiếp theo...")
            else:
                logger.warning(f"[{ctx.account_id}] Hết thời gian chờ nhưng chưa tới được tọa độ ({target_x}/{target_y}) tại {city_name} (cách {dist}). Tiếp tục kiểm tra thành tiếp theo...")
            activity._write("🏃", f"[{ctx.account_id}] {city_name} khong co Thuong nhan, sang thanh tiep theo...")

        logger.warning(f"[{ctx.account_id}] Đã chạy tới kiểm tra tọa độ tất cả các thành nhưng không tìm thấy Thương nhân Tây Vực!")
        return False

    def _handle_buy_item(self, ctx: StateContext) -> BotState:
        """Hội thoại với Thương nhân Tây Vực (chọn 'Đồng ý mua'), mở Tạp hóa chọn vật phẩm, bấm Mua và xác nhận."""
        if self._check_quest_timeout(ctx):
            self._ensure_npc_dialog_closed()
            return BotState.FIND_COMMERCE_CHIEF

        if self._app_cfg.mode.dry_run:
            logger.info(f"[{ctx.account_id}] DRY_RUN: Bỏ qua mua vật phẩm")
            ctx.extra["returning_quest"] = True
            return BotState.RETURN_QUEST

        requested_item = ctx.extra.get("requested_shop_item", "")
        if not requested_item:
            logger.warning(f"[{ctx.account_id}] ⚠️ Chưa xác định được vật phẩm cần mua trong ctx.extra! Quay lại CLASSIFY_QUEST.")
            return BotState.CLASSIFY_QUEST
        logger.info(f"[{ctx.account_id}] Tiến hành giao dịch với Thương nhân Tây Vực mua món đồ: '{requested_item}'...")
        ctx.last_action = "buy_requested_item"

        coords = self._loader.load_ui_coords(reload=True)
        shop_cfg = coords.get("shop", {})

        # Đảm bảo không bị hộp thoại NPC khác (như Vương Cẩn) che khuất màn hình
        f_init = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
        if f_init is not None and self._is_npc_dialog_open(f_init) and not self._is_western_merchant_dialog(f_init):
            logger.warning(f"[{ctx.account_id}] Phát hiện hộp thoại NPC khác che màn hình khi chuẩn bị mua đồ, đóng ngay...")
            self._ensure_npc_dialog_closed()
            time.sleep(0.3)

        # Bước 1: Hội thoại với Thương nhân Tây Vực -> bấm tùy chọn "Đồng ý mua"
        dialog_clicked = False
        for attempt in range(3):
            frame = self._screen.capture()
            if frame is None:
                time.sleep(0.4)
                continue
            boxes = self._text.read(frame)
            opt_box = None
            for b in boxes:
                norm = normalize_game_text(b.text).replace(" ", "")
                if "dongymua" in norm or ("dongy" in norm and "mua" in norm):
                    opt_box = b
                    break
            if opt_box:
                logger.info(f"[{ctx.account_id}] Bấm tùy chọn '{opt_box.text}' tại ({opt_box.center_x}, {opt_box.center_y})...")
                self._input.tap(opt_box.center_x, opt_box.center_y)
                dialog_clicked = True
                break
            else:
                # Nếu hộp thoại chưa mở, click vào Thương nhân để mở đối thoại
                logger.info(f"[{ctx.account_id}] Hộp thoại chưa mở (lần {attempt + 1}), tap tương tác Thương nhân Tây Vực...")
                top_boxes = [b for b in boxes if 700 <= b.center_x <= 1050 and b.center_y <= 130]
                target_top = [b for b in top_boxes if any(w in normalize_game_text(b.text) for w in ("npc", "thuong nhan", "tay vuc"))]
                if target_top:
                    self._input.tap(target_top[0].center_x, target_top[0].center_y)
                else:
                    npc_boxes = [b for b in boxes if any(w in normalize_game_text(b.text) for w in ("thuong nhan", "tay vuc")) and b.center_y > 200]
                    if npc_boxes:
                        self._input.tap(npc_boxes[0].center_x, npc_boxes[0].center_y + 40)
                    else:
                        self._input.tap(870, 480)
                time.sleep(0.3)

        if not dialog_clicked:
            logger.warning(
                f"[{ctx.account_id}] Không xác nhận được hội thoại/mục 'Đồng ý mua' của "
                f"Thương nhân Tây Vực cho '{requested_item}'. Hủy giao dịch, không bấm tọa độ dự phòng."
            )
            activity._write(
                "⚠️",
                f"[{ctx.account_id}] Chua xac nhan duoc hoi thoai Thuong nhan; khong mua {requested_item}.",
            )
            ctx.extra.pop("merchant_found_city", None)
            return BotState.CLASSIFY_QUEST
        time.sleep(0.35)

        # Kiểm tra xem giao diện Tạp hóa / Cửa hiệu có thực sự mở ra không trước khi duyệt các ô
        frame_shop = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
        chk_boxes = (
            self._text.read(frame_shop)
            if frame_shop is not None and hasattr(self, "_text") and self._text
            else []
        )
        norm_texts = [normalize_game_text(b.text) for b in chk_boxes]
        has_shop = any(
            any(k in t for k in ("tap hoa", "cua hieu", "gia tri", "so luong mua", "mua nhanh", "roi khoi", "giam dinh", "sua chua"))
            for t in norm_texts
        )
        if not has_shop:
            logger.warning(f"[{ctx.account_id}] ⚠️ Chưa xác nhận giao diện Tạp hóa đã mở. Hủy giao dịch, không chọn vật phẩm.")
            activity._write("⚠️", f"[{ctx.account_id}] Chua xac nhan Tap hoa; khong mua {requested_item}.")
            if hasattr(self, "_ensure_npc_dialog_closed"):
                self._ensure_npc_dialog_closed()
            ctx.extra.pop("merchant_found_city", None)
            return BotState.CLASSIFY_QUEST

        # Bước 2: Cửa hàng "Tạp hóa" mở ra -> click vô từng vật phẩm xem có đúng tên vật phẩm yêu cầu mua không
        grid_slots = shop_cfg.get("grid_slots", [])
        if not grid_slots:
            if "item_slot_linh_duoc" in shop_cfg:
                grid_slots = [shop_cfg["item_slot_linh_duoc"]]
            else:
                cols = [285, 380, 475]
                rows = [210, 305, 400, 495, 590, 685, 780]
                grid_slots = [{"x": c, "y": r} for r in rows for c in cols]

        from .quest_classifier import clean_shop_item_name
        requested_item_cleaned = clean_shop_item_name(requested_item)
        if not requested_item_cleaned:
            logger.warning(f"[{ctx.account_id}] ⚠️ Tên vật phẩm rỗng sau khi làm sạch! Hủy mua để tránh mua nhầm.")
            return BotState.CLASSIFY_QUEST
        logger.info(f"[{ctx.account_id}] Đang xác định vị trí vật phẩm '{requested_item_cleaned}' (ưu tiên tọa độ cố định)...")

        found_target_slot = False
        target_slot_pos = None
        best_slot = (grid_slots[0]["x"], grid_slots[0]["y"])
        best_score = -1.0
        best_item_name = ""

        title_roi = shop_cfg.get("tooltip_title_region", {"x1": 670, "y1": 40, "x2": 1080, "y2": 320})
        roi_tuple = (title_roi["x1"], title_roi["y1"], title_roi["x2"], title_roi["y2"])
        ignored_tags = (
            "loai", "lual", "loal", "lua", "nang", "gia", "da khoa", "dakhoa", "khoa",
            "ban tiem", "giao dich", "vut bo", "anh huong", "anhhuong", "suc khoe", "suckhoe",
            "choi qua", "180", "phut", "ngay", "thong bao", "he thong", "chu y", "luu y",
            "hanh trang", "sap xep", "thong tin", "tach", "bay ban", "x"
        )

        # 2.0 Kiểm tra xem vật phẩm có trong danh sách 17 vị trí cố định của Thương nhân Tây Vực hay không
        western_merchant_items = shop_cfg.get("western_merchant_items", {})
        fixed_slot_pos = None
        matched_fixed_item = None

        if western_merchant_items:
            for item_key, item_coords in western_merchant_items.items():
                is_m, sc = self._is_matching_shop_item(item_key, requested_item_cleaned)
                # Fixed coordinates are safe only for an unambiguous item
                # name match. A high fuzzy score alone matched Ám Yên to Ám
                # Phách in live use and selected the wrong slot.
                if is_m:
                    fixed_slot_pos = (int(item_coords["x"]), int(item_coords["y"]))
                    matched_fixed_item = item_key
                    break

        if fixed_slot_pos:
            fx, fy = fixed_slot_pos
            logger.info(
                f"[{ctx.account_id}] 🎯 Mua theo vị trí đã lưu: '{matched_fixed_item}' "
                f"tại ({fx}, {fy}). Bấm thẳng, không dò từng ô."
            )
            self._input.tap(fx, fy)
            time.sleep(0.3)
            found_target_slot = True
            target_slot_pos = (fx, fy)
        else:
            logger.error(
                f"[{ctx.account_id}] Chưa cấu hình tọa độ cho món '{requested_item_cleaned}'. "
                "Dừng mua, không dò từng ô và không mua đoán."
            )
            activity._write("⚠️", f"[{ctx.account_id}] Chua co toa do item {requested_item_cleaned}!")
            ctx.extra.pop("requested_shop_item", None)
            ctx.extra.pop("yellow_item", None)
            self._ensure_shop_and_inventory_closed(ctx)
            return BotState.CLASSIFY_QUEST

        if not found_target_slot:
            logger.warning(
                f"[{ctx.account_id}] Không xác nhận được món '{requested_item_cleaned}' tại tọa độ cố định; "
                "chỉ lúc này mới dò các ô để tránh mua nhầm."
            )
            for slot_idx, slot in enumerate(grid_slots):
                sx, sy = int(slot["x"]), int(slot["y"])
                logger.info(f"[{ctx.account_id}] Kiểm tra ô #{slot_idx + 1} tại ({sx}, {sy})...")
                self._input.tap(sx, sy)
                time.sleep(0.3)

                frame_item = self._screen.capture()
                boxes = self._text.read(frame_item, roi_tuple)
                boxes = sorted(boxes, key=lambda b: b.center_y)

                # Quét tất cả các text box trong tooltip tìm món đồ cần mua
                matched_box = None
                slot_best_score = -1.0
                slot_best_text = ""

                for b in boxes:
                    b_norm = normalize_game_text(b.text)
                    if any(tag in b_norm for tag in ignored_tags) or b_norm in ignored_tags or b_norm.isdigit():
                        continue
                    if len(b_norm) < 2:
                        continue

                    is_match, score = self._is_matching_shop_item(b.text, requested_item_cleaned)
                    if score > slot_best_score:
                        slot_best_score = score
                        slot_best_text = b.text
                    if is_match:
                        matched_box = b
                        break

                if slot_best_score > best_score:
                    best_score = slot_best_score
                    best_slot = (sx, sy)
                    best_item_name = slot_best_text

                if matched_box:
                    logger.info(f"[{ctx.account_id}] Ô #{slot_idx + 1}: ✅ ĐÚNG TÊN VẬT PHẨM: '{matched_box.text}' (độ khớp: {slot_best_score:.2f} với '{requested_item_cleaned}')! Tiến hành bấm Mua...")
                    found_target_slot = True
                    target_slot_pos = (sx, sy)
                    break
                elif slot_best_text:
                    logger.info(f"[{ctx.account_id}] Ô #{slot_idx + 1}: Tên hiển thị = '{slot_best_text}' (độ khớp: {slot_best_score:.2f} với '{requested_item_cleaned}')")
                    logger.info(f"[{ctx.account_id}] ❌ Chưa đúng món '{requested_item_cleaned}', đóng tooltip & nút Mua để tránh che ô tiếp theo...")
                    self._dismiss_item_tooltip(shop_cfg)
                else:
                    logger.debug(f"[{ctx.account_id}] Ô #{slot_idx + 1}: Không có tooltip hoặc ô trống, thử tiếp...")
                    self._dismiss_item_tooltip(shop_cfg)

        # Nếu chưa tìm thấy sau lượt quét đầu, thử cuộn nhẹ danh sách shop xuống để quét các ô phía dưới
        if not found_target_slot and len(grid_slots) > 1:
            logger.info(f"[{ctx.account_id}] Chưa thấy '{requested_item_cleaned}', cuộn danh sách Tạp hóa xuống để tìm tiếp...")
            self._input.swipe(375, 650, 375, 280, duration=300)
            time.sleep(0.3)
            bottom_slots = [s for s in grid_slots if s.get("y", 0) >= 500]
            for slot_idx, slot in enumerate(bottom_slots):
                sx, sy = int(slot["x"]), int(slot["y"])
                self._input.tap(sx, sy)
                time.sleep(0.3)
                frame_item = self._screen.capture()
                boxes = self._text.read(frame_item, roi_tuple)
                boxes = sorted(boxes, key=lambda b: b.center_y)

                matched_box = None
                slot_best_score = -1.0
                slot_best_text = ""

                for b in boxes:
                    b_norm = normalize_game_text(b.text)
                    if any(tag in b_norm for tag in ignored_tags) or b_norm in ignored_tags or b_norm.isdigit():
                        continue
                    if len(b_norm) < 2:
                        continue

                    is_match, score = self._is_matching_shop_item(b.text, requested_item_cleaned)
                    if score > slot_best_score:
                        slot_best_score = score
                        slot_best_text = b.text
                    if is_match:
                        matched_box = b
                        break

                if slot_best_score > best_score:
                    best_score = slot_best_score
                    best_slot = (sx, sy)
                    best_item_name = slot_best_text

                if matched_box:
                    logger.info(f"[{ctx.account_id}] Ô cuộn #{slot_idx + 1}: ✅ ĐÚNG TÊN VẬT PHẨM sau khi cuộn: '{matched_box.text}' (khớp: {slot_best_score:.2f})!")
                    found_target_slot = True
                    target_slot_pos = (sx, sy)
                    break
                elif slot_best_text:
                    logger.info(f"[{ctx.account_id}] Ô cuộn #{slot_idx + 1}: '{slot_best_text}' (khớp: {slot_best_score:.2f} với '{requested_item_cleaned}')")
                    self._dismiss_item_tooltip(shop_cfg)
                else:
                    self._dismiss_item_tooltip(shop_cfg)

        if not found_target_slot:
            is_explicit_single_slot = ("item_slot_linh_duoc" in shop_cfg and len(grid_slots) == 1)
            if is_explicit_single_slot or best_score >= 0.85:
                logger.warning(f"[{ctx.account_id}] Không tìm thấy ô khớp hoàn hảo, chọn ô ({best_slot[0]}, {best_slot[1]}) '{best_item_name}' với score={best_score:.2f}...")
                self._input.tap(best_slot[0], best_slot[1])
                target_slot_pos = best_slot
                time.sleep(0.18)
            else:
                logger.warning(f"[{ctx.account_id}] ⚠️ Không tìm thấy vật phẩm '{requested_item_cleaned}' trong danh sách ô đã duyệt (best score={best_score:.2f} '{best_item_name}')! Bỏ qua để tránh mua nhầm.")
                target_slot_pos = None

        # Bước 3: Bấm nút "Mua" (nút màu cam nổi lên theo vị trí item được chọn)
        buy_succeeded = False
        if target_slot_pos:
            frame_buy = self._screen.capture()
            shop_boxes = self._text.read(frame_buy) if hasattr(self, "_text") and self._text else []
            # Quét nút Mua trong khu vực shop (tránh tab Mua nhanh ở bên trái x <= 200)
            mua_boxes = [
                b for b in shop_boxes
                if (normalize_game_text(b.text).strip() == "mua" or "mua" in normalize_game_text(b.text).split())
                and 220 <= b.center_x <= 600 and 180 <= b.center_y <= 850
                and "nhanh" not in normalize_game_text(b.text)
            ]
            if mua_boxes:
                logger.info(f"[{ctx.account_id}] Tìm thấy nút Mua nổi tại ({mua_boxes[0].center_x}, {mua_boxes[0].center_y}) bằng OCR, bấm Mua...")
                self._input.tap(mua_boxes[0].center_x, mua_boxes[0].center_y)
            elif "buy_button" in shop_cfg:
                buy_pos = shop_cfg["buy_button"]
                logger.info(f"[{ctx.account_id}] Bấm nút Mua tại tọa độ cấu hình ({buy_pos['x']}, {buy_pos['y']})...")
                self._input.tap(buy_pos["x"], buy_pos["y"])
            else:
                self._input.tap(407, 535)

            # Bước 4: Chờ popup xác nhận số lượng mua ("Số lượng: - 1 +") -> bấm nút "Đồng ý"
            popup_detected = False
            for conf_attempt in range(3):
                time.sleep(0.35)
                frame_confirm = self._screen.capture()
                # The quantity dialog is small and sits over both shop panels.
                # OCR over the full game frame often misses its small labels;
                # read the dialog area separately to improve recognition.
                confirm_boxes = (
                    self._text.read(frame_confirm, (250, 520, 1350, 900))
                    if frame_confirm is not None and hasattr(self, "_text") and self._text
                    else []
                )
                norm_texts = [normalize_game_text(b.text).replace(" ", "") for b in confirm_boxes]

                # Tìm nút 'Đồng ý' / 'Xác nhận' trong popup
                target_confirm = None
                for b in confirm_boxes:
                    norm_b = normalize_game_text(b.text)
                    if (
                        any(w in norm_b for w in ("dong y", "xac nhan", "dongy", "dong", "dings"))
                        or fuzzy_score(norm_b, "dong y") >= 0.6
                    ) and 240 <= b.center_x <= 750 and 650 <= b.center_y <= 900:
                        target_confirm = b
                        break

                item_norm = normalize_game_text(requested_item_cleaned).replace(" ", "")
                item_title_seen = any(
                    b.center_x >= 280
                    and 500 <= b.center_y <= 780
                    and (
                        item_norm in normalize_game_text(b.text).replace(" ", "")
                        or (len(normalize_game_text(b.text).replace(" ", "")) >= 6 and normalize_game_text(b.text).replace(" ", "") in item_norm)
                        or fuzzy_score(item_norm, normalize_game_text(b.text).replace(" ", "")) >= 0.6
                    )
                    for b in confirm_boxes
                )
                has_quantity_fields = any(
                    any(k in t for k in ("soluong", "soluongmua", "dongia", "tongcong", "thanhtien", "luong", "ludng", "lugng"))
                    for t in norm_texts
                )
                has_modal_buttons = any(
                    any(k in t for k in ("dongy", "xacnhan", "roikhoi", "dings", "dong"))
                    for t in norm_texts
                )
                has_quantity_dialog = (
                    has_quantity_fields
                    or (item_title_seen and has_modal_buttons)
                    or any("roikhoi" in t for t in norm_texts)
                )

                if has_quantity_dialog and target_confirm:
                    popup_detected = True
                    logger.info(f"[{ctx.account_id}] Bấm nút xác nhận '{target_confirm.text}' tại ({target_confirm.center_x}, {target_confirm.center_y})...")
                    self._input.tap(target_confirm.center_x, target_confirm.center_y)
                    time.sleep(0.35)
                    break
                elif has_quantity_dialog:
                    popup_detected = True
                    conf_pos = shop_cfg.get("confirm_button", {"x": 314, "y": 816})
                    logger.info(f"[{ctx.account_id}] Bấm nút xác nhận cấu hình tại ({conf_pos['x']}, {conf_pos['y']})...")
                    self._input.tap(conf_pos["x"], conf_pos["y"])
                    time.sleep(0.35)
                    break
                else:
                    logger.debug(
                        f"[{ctx.account_id}] Chưa nhận diện popup mua (lượt {conf_attempt + 1}); "
                        f"OCR vùng popup: {[b.text for b in confirm_boxes]}"
                    )

            # Bước 4.1: Kiểm tra kết quả giao dịch sau khi bấm Đồng ý
            if popup_detected:
                time.sleep(0.3)
                post_buy_frame = self._screen.capture()
                post_boxes = (self._text.read(post_buy_frame) if hasattr(self, "_text") and self._text else []) if post_buy_frame is not None else []
                post_texts = [normalize_game_text(b.text).replace(" ", "") for b in post_boxes]

                # Kiểm tra thông báo thiếu xu / thiếu tiền
                has_insufficient_funds = any(any(k in t for k in ("khongduxu", "khongdutien", "khongdunganluong", "khongdubac")) for t in post_texts)

                if has_insufficient_funds:
                    logger.warning(f"[{ctx.account_id}] ⚠️ Game thông báo không đủ xu / tiền để mua '{requested_item_cleaned}': {post_texts}")
                    activity._write("⚠️", f"[{ctx.account_id}] Khong du xu de mua {requested_item_cleaned}!")
                    buy_succeeded = False
                else:
                    if self._is_inventory_full_or_overweight(post_buy_frame):
                        logger.warning(f"[{ctx.account_id}] ⚠️ [GHI CHÚ] Hành trang quá nặng hoặc đã đầy! Không thể nhận thêm vật phẩm.")
                        activity._write("⚠️", f"[{ctx.account_id}] Ghi chu: Hanh trang day/qua nang!")
                        self._save_progress(ctx, status_text="Hành trang đầy / quá nặng ⚠️")
                    logger.info(
                        f"[{ctx.account_id}] Đã bấm xác nhận trên popup số lượng của "
                        f"'{requested_item_cleaned}'; cần Chủ thương hội xác nhận nhiệm vụ."
                    )
                    buy_succeeded = True
            else:
                logger.warning(f"[{ctx.account_id}] ⚠️ Không mở được popup số lượng mua để xác nhận!")
                # Never leave a missed/poorly-recognized quantity dialog open:
                # it blocks the shop and the next quest state. This coordinate
                # is the dialog's safe "Rời khỏi" action, not the Buy button.
                cancel_pos = shop_cfg.get("cancel_button", {"x": 516, "y": 815})
                logger.info(
                    f"[{ctx.account_id}] Đóng popup mua chưa nhận diện tại "
                    f"({cancel_pos['x']}, {cancel_pos['y']}) để thoát trạng thái kẹt..."
                )
                self._input.tap(cancel_pos["x"], cancel_pos["y"])
                time.sleep(0.2)
                buy_succeeded = False

        # Bước 5: Đóng hoàn toàn cửa hàng Tạp hóa và Hành trang
        self._ensure_shop_and_inventory_closed(ctx)
        self._dismiss_unexpected_popups()

        # Bước 6: Nếu không tìm thấy hoặc chưa mua được vật phẩm -> không đánh dấu hoàn thành
        if not target_slot_pos or not buy_succeeded:
            logger.warning(f"[{ctx.account_id}] ⚠️ Không tìm thấy hoặc chưa mua được '{requested_item_cleaned}' (slot={target_slot_pos}, bought={buy_succeeded})! Không đánh dấu hoàn thành nhiệm vụ.")
            activity._write("⚠️", f"[{ctx.account_id}] Chua mua duoc {requested_item_cleaned}!")
            ctx.extra.pop("merchant_found_city", None)
            return BotState.CLASSIFY_QUEST

        # Bước 7: Đã xác nhận popup mua; chỉ Chủ thương hội xác nhận được
        # nhiệm vụ hoàn tất. Không ghi nhận giao dịch là thành công tại đây.
        logger.info(
            f"[{ctx.account_id}] Đã gửi xác nhận mua '{requested_item_cleaned}'. "
            "Quay về Chủ thương hội để xác minh nhiệm vụ."
        )
        activity._write(
            "🛍️",
            f"[{ctx.account_id}] Da gui xac nhan mua {requested_item_cleaned}; "
            "ve Chu thuong hoi xac minh.",
        )
        ctx.extra["returning_quest"] = True
        return BotState.RETURN_QUEST

    def _dismiss_item_tooltip(self, shop_cfg: Optional[dict] = None) -> None:
        """Đóng tooltip thông tin vật phẩm và nút Mua nổi để tránh che các ô vật phẩm khác."""
        try:
            if shop_cfg is None:
                coords = self._loader.load_ui_coords()
                shop_cfg = coords.get("shop", {})
            # 1. Bấm nút 'X' của tooltip tại (1073, 90) chuẩn 1600x900
            tooltip_x = shop_cfg.get("tooltip_close", {"x": 1073, "y": 90})
            self._input.tap(tooltip_x["x"], tooltip_x["y"])
            time.sleep(0.12)
            # 2. Bấm nhẹ điểm an toàn không có item (465, 750) để bỏ chọn nếu tooltip vẫn còn
            safe_pt = shop_cfg.get("safe_neutral_tap", {"x": 465, "y": 750})
            self._input.tap(safe_pt["x"], safe_pt["y"])
            time.sleep(0.1)
        except Exception as e:
            logger.debug(f"_dismiss_item_tooltip: {e}")

    @staticmethod
    def _is_matching_shop_item(seen_text: str, target_name: str) -> tuple[bool, float]:
        """Kiểm tra so khớp tên vật phẩm trong shop với tên yêu cầu của nhiệm vụ.
        Ngưỡng an toàn >= 0.80 (80%).
        Bắt buộc khớp các từ khóa đặc trưng (như 'kim', 'cang' hoặc 'linh', 'duoc').
        Tuyệt đối không khớp nếu có từ khóa thuộc tính xung đột (như 'Ám nguyên thạch' vs 'Kim cang nguyên thạch').
        """
        import re
        from .text_recognition import normalize_game_text, fuzzy_score
        raw_norm_seen = normalize_game_text(seen_text).strip()
        raw_norm_target = normalize_game_text(target_name).strip()
        if not raw_norm_seen or not raw_norm_target:
            return False, 0.0

        if raw_norm_seen == raw_norm_target:
            return True, 1.0

        # Chuẩn hóa các lỗi OCR dính chữ / sai font thường gặp trong shop game
        def _clean_shop_tokens(s: str) -> str:
            s = re.sub(r"\bdugc\b|\bduc\b", "duoc", s)
            s = re.sub(r"\bvyc\b|\bvtc\b|\bvy\b|\bvuyc\b", "vuc", s)
            s = re.sub(r"baotao|baotay", "bao tay ", s)
            s = re.sub(r"layvtc|layvuc", "tay vuc ", s)
            s = re.sub(r"linhdugc|linhduoc|linhduc", "linh duoc ", s)
            s = re.sub(r"tinhkim", "tinh kim ", s)
            s = re.sub(r"kimcang", "kim cang ", s)
            s = re.sub(r"nguyenthac|nguyenthach", " nguyen thach ", s)
            s = re.sub(r"\bthac\b", "thach", s)
            s = re.sub(r"\btring\b", "trung", s)
            s = re.sub(r"longdaibang", "long dai bang ", s)
            s = re.sub(r"trungdaibang", "trung dai bang ", s)
            s = re.sub(r"daibang", "dai bang ", s)
            return " ".join(s.split())

        norm_seen = _clean_shop_tokens(raw_norm_seen)
        norm_target = _clean_shop_tokens(raw_norm_target)

        # "Ám Yên" and "Âm/Ám Phách" become near-identical after accent
        # removal. The final qualifier distinguishes the actual shop slots.
        seen_tokens_exact = set(norm_seen.split())
        target_tokens_exact = set(norm_target.split())
        if "am" in seen_tokens_exact and "am" in target_tokens_exact:
            seen_am_item = "yen" if "yen" in seen_tokens_exact else (
                "phach" if "phach" in seen_tokens_exact else ""
            )
            target_am_item = "yen" if "yen" in target_tokens_exact else (
                "phach" if "phach" in target_tokens_exact else ""
            )
            if seen_am_item and target_am_item and seen_am_item != target_am_item:
                return False, 0.0

        score = max(fuzzy_score(raw_norm_seen, raw_norm_target), fuzzy_score(norm_seen, norm_target))
        seen_tokens = set(norm_seen.split())
        target_tokens = set(norm_target.split())

        generic_tokens = {'tay', 'vuc', 'vy', 'vtc', 'vuyc', 'phuong', 'cai', 'mon', 'vien'}
        distinctive_target = target_tokens - generic_tokens
        distinctive_seen = seen_tokens - generic_tokens

        # Các từ khóa phân loại thuộc tính đặc trưng bắt buộc không được mâu thuẫn
        attribute_qualifiers = {
            'kim', 'cang', 'am', 'quang', 'hoa', 'bang', 'loi', 'doc', 'moc', 'thuy', 'tho',
            'linh', 'duoc', 'bao', 'long', 'trung', 'thit', 'uyen', 'phach', 'tuyet', 'tinh',
            'thanh', 'luc', 'tru', 'yen', 'boi', 'dao', 'ho'
        }

        # Bổ ngữ thuộc tính trong target và seen: so khớp theo token nguyên vẹn hoặc fuzzy score cao,
        # tuyệt đối không dùng substring 'a in st' để tránh 'uyen' khớp nhầm trong 'nguyen'!
        target_attrs = {a for a in attribute_qualifiers if any(t == a or fuzzy_score(t, a) >= 0.85 for t in distinctive_target)}
        seen_attrs = {a for a in attribute_qualifiers if any(st == a or fuzzy_score(st, a) >= 0.85 for st in distinctive_seen)}

        # 1. Nếu seen chứa thuộc tính xung đột không có trong target (vd: seen có 'am' mà target không có 'am') -> LOẠI
        conflicting_seen = seen_attrs - target_attrs
        if conflicting_seen:
            return False, min(score * 0.3, 0.3)

        # 2. Nếu target có thuộc tính (vd: 'kim', 'cang') mà seen hoàn toàn không chứa -> LOẠI
        for ta in target_attrs:
            if not any(ta == st or fuzzy_score(ta, st) >= 0.80 for st in distinctive_seen):
                return False, min(score * 0.3, 0.3)

        # 3. Phải khớp ít nhất một từ khóa định danh chính (không tính hậu tố chung nguyen/thach nếu target có attribute)
        primary_target = target_attrs if target_attrs else distinctive_target
        matched_primary = False
        for pt in primary_target:
            if any(pt == st or fuzzy_score(pt, st) >= 0.75 for st in distinctive_seen):
                matched_primary = True
                break
        if not matched_primary:
            return False, score

        # Ngưỡng an toàn >= 0.80 (80%)
        is_match = (score >= 0.80) or (distinctive_target and len(distinctive_target & distinctive_seen) >= len(distinctive_target) - 1)
        return is_match, score

    def _ensure_shop_and_inventory_closed(self, ctx: Optional[StateContext] = None) -> bool:
        """
        Đảm bảo đóng hoàn toàn cả bảng Tạp hóa và bảng Hành trang (nếu đang mở):
        - Bấm nút 'X' của Tạp hóa tại (679, 110).
        - Bấm nút 'X' của Hành trang tại (1535, 109).
        - Đóng popup xác nhận sắp xếp tự động nếu bị mở nhầm.
        - Kiểm tra lại bằng OCR lặp lại tối đa 3 lần cho đến khi cả hai bảng đều đã đóng hoàn toàn.
        """
        coords = self._loader.load_ui_coords(reload=True)
        shop_cfg = coords.get("shop", {})
        close_shop_pos = shop_cfg.get("close_button", {"x": 679, "y": 110})
        close_inv_pos = shop_cfg.get("inventory_close", {"x": 1565, "y": 95})
        sort_cancel_pos = shop_cfg.get("sort_cancel_button", {"x": 1059, "y": 546})

        for attempt in range(3):
            frame = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
            if frame is not None and hasattr(self, "_text") and self._text:
                boxes = self._text.read(frame)
                norm_texts = [normalize_game_text(b.text) for b in boxes]
                if any(any(k in t for k in ("tu dong sap xep", "dung che do tu dong", "sap xep khong")) for t in norm_texts):
                    logger.info("Phát hiện popup 'Xác nhận tự động sắp xếp', bấm 'Hủy bỏ'...")
                    cancel_boxes = [b for b in boxes if any(w in normalize_game_text(b.text) for w in ("huy bo", "huy")) and b.center_y > 450]
                    if cancel_boxes:
                        self._input.tap(cancel_boxes[0].center_x, cancel_boxes[0].center_y)
                    else:
                        self._input.tap(sort_cancel_pos["x"], sort_cancel_pos["y"])
                    time.sleep(0.2)

            # 1. Bấm đóng Tạp hóa
            self._input.tap(close_shop_pos["x"], close_shop_pos["y"])
            time.sleep(0.15)
            # 2. Bấm đóng Hành trang
            self._input.tap(close_inv_pos["x"], close_inv_pos["y"])
            time.sleep(0.2)

            frame = self._screen.capture() if hasattr(self, "_screen") and self._screen else None
            if frame is None:
                break
            boxes = self._text.read(frame)
            norm_texts = [normalize_game_text(b.text) for b in boxes]

            def _is_shop_label(t):
                if any(k in t for k in ("hoan thanh", "doi thoai", "nhiem vu", "cung voi", "truy sat", "chu tap hoa", "chu tiem", "da tau")):
                    return False
                return any(k in t for k in ("tap hoa", "giam dinh", "sua chua", "mua nhanh", "ban nhanh", "cua hieu"))

            has_shop = any(_is_shop_label(t) for t in norm_texts)
            has_inv = any(any(k in t for k in ("hanh trang", "sap xep", "thong tin", "tach", "bay ban", "shop kim phieu", "shop ngan phieu")) for t in norm_texts)

            if not has_shop and not has_inv:
                logger.info("Đã đóng hoàn tất bảng Tạp hóa và Hành trang.")
                return True

            logger.info(f"Bảng Tạp hóa ({has_shop}) hoặc Hành trang ({has_inv}) vẫn chưa đóng (lần {attempt + 1}), thử lại...")

        return True

    def _return_to_nearest_city_commerce_chief(self, ctx: StateContext, current_city: Optional[str] = None) -> bool:
        """Quay về Chủ thương hội của thành hiện tại hoặc quay về Tuyền Châu."""
        # Chỉ xử lý quái chặn đường cho nhiệm vụ combat. Nhiệm vụ shop/NPC
        # không được phép suy đoán tên người chơi trên HUD là quái.
        if (
            ctx.current_quest_type == QuestType.COMBAT
            and not ctx.extra.get("combat_objective_complete")
            and not ctx.extra.get("returning_quest")
            and not ctx.extra.get("abandon_combat_quest")
            and self._attack_monster_blocking_travel(ctx)
        ):
            logger.info(f"[{ctx.account_id}] Đã ra lệnh diệt quái cản đường; chờ hết giao tranh rồi mới tìm đường về.")
            return False

        if ctx and (ctx.extra.get("combat_map") or ctx.current_quest_type == QuestType.COMBAT):
            if ctx.extra.get("combat_objective_complete"):
                logger.info(
                    f"[{ctx.account_id}] Đã xác nhận diệt đủ quái; di chuyển về Tuyền Châu gặp Chủ thương hội trả nhiệm vụ..."
                )
            else:
                logger.info(
                    f"[{ctx.account_id}] NV đánh quái chưa được xác nhận hoàn thành; "
                    "đang quay về Chủ thương hội để phục hồi điều hướng/kiểm tra nhiệm vụ..."
                )
        elif current_city is None and hasattr(self, "_is_near_commerce_chief") and self._is_near_commerce_chief():
            logger.info(f"[{ctx.account_id}] Đã đứng cạnh Chủ thương hội, không cần mở bản đồ.")
            return True

        cur_city = current_city
        if not cur_city:
            cur_city, _ = self._get_current_location_from_hud()
        if not cur_city:
            cur_city = ctx.extra.get("merchant_found_city", "")

        combat_map = ctx.extra.get("combat_map", "") if ctx else ""
        combat_incomplete = bool(
            ctx
            and ctx.current_quest_type == QuestType.COMBAT
            and not ctx.extra.get("combat_objective_complete")
            and not ctx.extra.get("abandon_combat_quest")
            and not ctx.extra.get("returning_quest")
        )
        if (
            combat_incomplete
            and cur_city
            and combat_map
            and self._is_same_combat_map(cur_city, combat_map, sect=ctx.extra.get("combat_sect"))
        ):
            logger.info(
                f"[{ctx.account_id}] Đã trở lại đúng map quái '{cur_city}' và nhiệm vụ chưa hoàn thành; "
                "hủy quy trình về Chủ thương hội để tiếp tục đánh quái."
            )
            return False

        logger.info(f"[{ctx.account_id}] Tìm đường quay về Chủ thương hội (thành hiện tại: '{cur_city}')...")
        time.sleep(0.05)

        from .text_recognition import normalize_game_text, fuzzy_score, is_same_city, KNOWN_CITIES
        norm_cur = normalize_game_text(cur_city) if cur_city else ""

        # 1. Thử tìm và trả nhiệm vụ tại Chủ thương hội ở thành phố hiện tại trước (vd: Biện Kinh, Thành Đô, Tuyền Châu, Dương Châu...)
        # Nếu có Chủ thương hội tại thành hiện tại -> di chuyển đến luôn để tối ưu thời gian!
        is_major_city = bool(cur_city and any(is_same_city(cur_city, mc) for mc in KNOWN_CITIES))
        if is_major_city and hasattr(self, "_ensure_map_open"):
            if self._ensure_map_open():
                time.sleep(0.15)
                coords = self._loader.load_ui_coords() if hasattr(self, "_loader") else {}
                nav_cfg = coords.get("navigation", {})
                find_npc_x = nav_cfg.get("map_tab_find_npc", {}).get("x", 376)
                find_npc_y = nav_cfg.get("map_tab_find_npc", {}).get("y", 161)
                search_x = nav_cfg.get("map_search_input", {}).get("x", 266)
                search_y = nav_cfg.get("map_search_input", {}).get("y", 222)

                self._input.tap(find_npc_x, find_npc_y)
                time.sleep(0.15)
                self._input.tap(search_x, search_y)
                time.sleep(0.1)
                try:
                    self._adb._run(["shell", "input keyevent 123; " + "; ".join(["input keyevent 67"] * 25)])
                    time.sleep(0.08)
                except Exception:
                    pass
                self._input_search_text("Chu thuong hoi", unicode_text="Chủ thương hội", ctx=ctx)
                time.sleep(0.2)

                if self._click_npc_search_result("Chu thuong hoi"):
                    real_cur_city, _ = self._get_current_location_from_hud()
                    effective_city = real_cur_city or cur_city
                    logger.info(f"[{ctx.account_id}] ⚡ [TỐI ƯU] Đã tìm thấy và di chuyển đến Chủ thương hội tại thành '{effective_city}'!")
                    activity._write("🏃", f"[{ctx.account_id}] Quay ve Chu thuong hoi ({effective_city})...")
                    ctx.extra["chief_city"] = effective_city
                    return True
                else:
                    logger.info(f"[{ctx.account_id}] Thành '{cur_city}' không có NPC Chủ thương hội trong danh sách; sẽ chuyển về Tuyền Châu.")
                    if hasattr(self, "_is_map_open") and self._is_map_open():
                        self._close_map()
                        time.sleep(0.15)

        # 2. Nếu thành hiện tại không có Chủ thương hội (hoặc tìm không ra):
        # Chuyển map quay về Chủ thương hội Tuyền Châu theo quy trình 8 bước chuẩn
        logger.info(f"[{ctx.account_id}] Thành hiện tại '{cur_city or 'chưa rõ'}' không có Chủ thương hội, chuyển map về Chủ thương hội Tuyền Châu (8 bước)...")
        ctx.extra["chief_city"] = "Tuyền Châu"
        return self._navigate_to_tuyen_chau_commerce_chief(ctx)

    def _handle_return_quest(self, ctx: StateContext) -> BotState:
        """Quay về Chủ Thương Hội ở thành lớn gần nhất để trả nhiệm vụ."""
        logger.info(f"[{ctx.account_id}] Đã hoàn thành mục tiêu nhiệm vụ, chuyển sang quay về Chủ Thương Hội...")
        ctx.extra["returning_quest"] = True
        ctx.last_action = "return_to_commerce_chief"
        self._checkpoint.save(ctx)

        # 0. Only accept the merchant location when the HUD confirms its city.
        cur_city_at_start, _ = self._get_current_location_from_hud()
        target_chief = ctx.extra.get("chief_city")
        is_chief_city = is_same_city(cur_city_at_start, target_chief) if target_chief else (
            is_same_city(cur_city_at_start, "Tuyền Châu") or (cur_city_at_start and any(is_same_city(cur_city_at_start, mc) for mc in KNOWN_CITIES))
        )
        if (
            cur_city_at_start
            and is_chief_city
            and ctx.current_quest_type != QuestType.NPC
            and self._is_near_commerce_chief()
        ):
            return BotState.MEET_NPC

        # 1. Chỉ kích hoạt tìm đường nếu trước đó chưa bắt đầu di chuyển:
        if not ctx.extra.get("chief_nav_started"):
            navigation_started = self._return_to_nearest_city_commerce_chief(ctx, current_city=cur_city_at_start)
            if navigation_started:
                ctx.extra["chief_nav_started"] = True

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
        # Dọn toàn bộ dữ liệu suy luận của nhiệm vụ vừa xong. Nếu giữ lại một
        # trường (đặc biệt requested_shop_item/combat_map), nó sẽ ghi đè kết quả
        # phân loại của nhiệm vụ kế tiếp.
        for key in (
            "quest_text", "yellow_tokens", "yellow_item", "requested_shop_item",
            "combat_map", "combat_monster", "combat_count", "combat_sect",
            "combat_objective_complete",
            "combat_avoided_directions", "combat_last_move_direction", "combat_reverse_vector",
            "quest_target_city", "quest_target_npc", "quest_detail_text",
            "multi_npc_route", "multi_npc_index", "multi_npc_completed",
            "multi_npc_needs_navigation",
            "contribution_option_selected", "quest_source", "combat_quest_start_time",
            "dismounted_for_combat", "returning_quest", "chief_nav_started", "chief_nav_failures",
            "chief_arrival_nudge_attempts",
            "teleport_to_combat_quest", "auto_walking", "merchant_search_retries",
            "out_of_money",
        ):
            ctx.extra.pop(key, None)
        return BotState.CHECK_LOCATION

    def _handle_cancel_quest(self, ctx: StateContext) -> BotState:
        logger.warning(f"[{ctx.account_id}] Cancelling quest (UNKNOWN type)")
        ctx.retry_count += 1
        ctx.extra.pop("out_of_money", None)
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

    def _is_account_completed_today(self, account_id: str, username: str = "") -> bool:
        """Kiểm tra xem tài khoản đã đạt đủ giới hạn nhiệm vụ (20/20) trong ngày hôm nay chưa."""
        try:
            today_str = datetime.now().strftime("%Y-%m-%d")
            runtime_dir = Path("runtime")
            # 1. Kiểm tra trực tiếp file theo account_id
            prog_file = runtime_dir / f"progress_{account_id}.json"
            if prog_file.exists():
                with open(prog_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    file_date = data.get("date")
                    if not file_date:
                        try:
                            file_date = datetime.fromtimestamp(prog_file.stat().st_mtime).strftime("%Y-%m-%d")
                        except Exception:
                            file_date = today_str
                    # Chỉ coi là đã hoàn thành nếu thuộc ngày hôm nay
                    if file_date == today_str:
                        total = data.get("total_quests_done", 0)
                        status = str(data.get("status_text", "")).lower()
                        if total >= 20 or "20/20" in status or "hết nhiệm vụ" in status or "hoàn thành" in status:
                            return True

            # 2. Kiểm tra theo username (phòng trường hợp người dùng đổi id tài khoản trong accounts.yaml)
            if username:
                for p in runtime_dir.glob("progress_*.json"):
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if data.get("username") == username:
                                file_date = data.get("date")
                                if not file_date:
                                    try:
                                        file_date = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d")
                                    except Exception:
                                        file_date = today_str
                                if file_date == today_str:
                                    total = data.get("total_quests_done", 0)
                                    status = str(data.get("status_text", "")).lower()
                                    if total >= 20 or "20/20" in status or "hết nhiệm vụ" in status or "hoàn thành" in status:
                                        return True
                    except Exception:
                        pass

            # 3. Kiểm tra DB xem hôm nay đã hoàn thành >= 20 nhiệm vụ chưa
            if hasattr(self, "_db") and self._db:
                today_completed = self._db.get_today_completed_quests(account_id)
                if today_completed >= 20:
                    return True
                if username and hasattr(self._db, "get_today_completed_quests_by_username"):
                    if self._db.get_today_completed_quests_by_username(username) >= 20:
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
        recovery_count = int(ctx.extra.get("recovery_count", 0)) + 1
        ctx.extra["recovery_count"] = recovery_count
        max_recovery = getattr(self._app_cfg.retry, "max_attempts", 3)

        logger.warning(
            f"[{ctx.account_id}] 🔄 [RECOVERY #{recovery_count}/{max_recovery}] "
            f"Gặp lỗi/mắc kẹt -> Tiến hành TẮT GAME và KHỞI ĐỘNG LẠI GAME..."
        )
        activity._write("♻️", f"[{ctx.account_id}] Loi ket noi/mac ket: Tat game va khoi dong lai game (#{recovery_count})...")

        # 1. Tắt game hoàn toàn
        try:
            logger.info(f"[{ctx.account_id}] Đang tắt game (force-stop)...")
            self._adb.stop_app()
        except Exception as e:
            logger.warning(f"[{ctx.account_id}] Lỗi khi tắt game: {e}")
        time.sleep(2.0)

        # 2. Khởi động lại game
        try:
            logger.info(f"[{ctx.account_id}] Đang khởi động lại game...")
            self._adb.start_app()
        except Exception as e:
            logger.warning(f"[{ctx.account_id}] Lỗi khi khởi động lại game: {e}")
        time.sleep(5.0)

        # 3. Dọn sạch các cờ trạng thái bị kẹt
        ctx.extra.pop("chief_nav_started", None)
        ctx.extra.pop("chief_nav_failures", None)
        ctx.extra.pop("chief_dialog_failures", None)
        ctx.extra.pop("nav_attempts", None)
        ctx.extra.pop("_adb_transient_retries", None)
        ctx.extra.pop("login_hud_collapsed", None)
        ctx.retry_count = 0  # Reset retry counter

        # 4. Kiểm tra số lần khôi phục tối đa
        if recovery_count > max_recovery:
            logger.error(
                f"[{ctx.account_id}] Đã tắt và khởi động lại game {recovery_count - 1} lần liên tiếp nhưng vẫn gặp lỗi. "
                f"Đã tắt game và khởi động lại sạch, chuyển sang tài khoản tiếp theo."
            )
            activity._write("⚠️", f"[{ctx.account_id}] Khoi dong lai {recovery_count - 1} lan van loi -> Chuyen tai khoan ke tiep")
            ctx.extra.pop("recovery_count", None)
            return BotState.NEXT_ACCOUNT

        logger.info(f"[{ctx.account_id}] Khởi động lại game thành công, chuyển sang START_GAME để vào lại game...")
        return BotState.START_GAME

    def _handle_error(self, ctx: StateContext) -> BotState:
        logger.error(
            f"[{ctx.account_id}] ERROR state: {ctx.error_code} – {ctx.error_message}"
        )
        try:
            logger.info(f"[{ctx.account_id}] Trạng thái ERROR: Tắt game và khởi động lại game...")
            self._adb.stop_app()
            time.sleep(2.0)
            self._adb.start_app()
            time.sleep(5.0)
        except Exception as e:
            logger.warning(f"[{ctx.account_id}] Lỗi khi tắt/khởi động lại app trong _handle_error: {e}")

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
            if self._is_account_completed_today(account.id, account.username):
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
                        try:
                            logger.info(f"[{account.id}] Gặp trạng thái ERROR ({err_msg}) -> Tắt game và khởi động lại game sạch...")
                            self._adb.stop_app()
                            time.sleep(2.0)
                            self._adb.start_app()
                            time.sleep(5.0)
                        except Exception as stop_err:
                            logger.warning(f"Không thể tắt/khởi động lại app sau lỗi: {stop_err}")
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
                    try:
                        self._adb.stop_app()
                        time.sleep(2.0)
                        self._adb.start_app()
                        time.sleep(5.0)
                    except Exception:
                        pass
                    self._db.end_run(run_id, "EXCEPTION", errors_count=1)
            else:
                logger.warning("StateMachine still running, skipping account start")

        logger.info("BotController: all accounts processed")

    def run_single_account(self, account_id: str) -> None:
        """Chạy một account cụ thể."""
        self._state_machine.run(account_id=account_id, initial_state=BotState.BOOT)
        if self._state_machine.context.current_state == BotState.ERROR:
            try:
                logger.info(f"[{account_id}] Gặp trạng thái ERROR -> Tắt game và khởi động lại game sạch...")
                self._adb.stop_app()
                time.sleep(2.0)
                self._adb.start_app()
                time.sleep(5.0)
            except Exception as stop_err:
                logger.warning(f"Không thể tắt/khởi động lại app sau lỗi: {stop_err}")

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
