"""
AutoQuestTool - Bot Controller
Orchestrator chính. Kết nối tất cả components.
GUI → BotController → StateMachine → Engine.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Callable

from loguru import logger

from .config_loader import ConfigLoader, AppConfig
from .adb_controller import ADBController
from .ldplayer_controller import LDPlayerController
from .screen_capture import ScreenCapture
from .template_manager import TemplateManager
from .vision_engine import VisionEngine
from .input_controller import InputController
from .state_machine import StateMachine, BotState, StateContext
from .quest_classifier import QuestClassifier
from .navigation_engine import NavigationEngine
from .checkpoint import Checkpoint
from .database import Database
from .watchdog import WatchdogService
from .error_codes import ErrorCode, BotError


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

        logger.info("BotController initialized")

    # ============================
    # Setup
    # ============================

    def _setup_logging(self) -> None:
        from loguru import logger
        import sys
        log_cfg = self._app_cfg.logging
        Path(log_cfg.log_dir).mkdir(parents=True, exist_ok=True)

        logger.remove()
        logger.add(
            sys.stdout,
            level=log_cfg.level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        )
        logger.add(
            f"{log_cfg.log_dir}/{{time:YYYY-MM-DD}}.log",
            rotation=log_cfg.rotation,
            retention=log_cfg.retention,
            level=log_cfg.level,
            encoding="utf-8",
        )

    def _create_adb(self) -> ADBController:
        """Tạo ADB hoặc MockADB tùy config."""
        if self._app_cfg.mode.mock_adb:
            from .mock.mock_adb import MockADBController
            logger.warning("Using MockADBController (mock_adb=true)")
            return MockADBController(self._app_cfg)
        return ADBController(self._app_cfg)

    # ============================
    # State Handlers Registration
    # ============================

    def _register_handlers(self) -> None:
        sm = self._state_machine
        sm.register(BotState.BOOT, self._handle_boot)
        sm.register(BotState.CONNECT_EMULATOR, self._handle_connect_emulator)
        sm.register(BotState.START_GAME, self._handle_start_game)
        sm.register(BotState.LOGIN, self._handle_login)
        sm.register(BotState.VERIFY_INGAME, self._handle_verify_ingame)
        sm.register(BotState.GET_QUEST, self._handle_get_quest)
        sm.register(BotState.CLASSIFY_QUEST, self._handle_classify_quest)
        sm.register(BotState.NAVIGATE, self._handle_navigate)
        sm.register(BotState.INTERACT_NPC, self._handle_interact_npc)
        sm.register(BotState.BUY_ITEM, self._handle_buy_item)
        sm.register(BotState.RETURN_QUEST, self._handle_return_quest)
        sm.register(BotState.COMPLETE_QUEST, self._handle_complete_quest)
        sm.register(BotState.CANCEL_QUEST, self._handle_cancel_quest)
        sm.register(BotState.TRANSFER_GOLD, self._handle_transfer_gold)
        sm.register(BotState.LOGOUT, self._handle_logout)
        sm.register(BotState.NEXT_ACCOUNT, self._handle_next_account)
        sm.register(BotState.RECOVERY, self._handle_recovery)
        sm.register(BotState.ERROR, self._handle_error)

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
            if ctx.retry_count >= self._app_cfg.retry.max_attempts:
                raise BotError(ErrorCode.ADB_002, "Cannot connect to ADB after max retries")
            time.sleep(self._app_cfg.retry.delay_between)
            return BotState.CONNECT_EMULATOR
        ctx.reset_retry()
        return BotState.START_GAME

    def _handle_start_game(self, ctx: StateContext) -> BotState:
        logger.info(f"[{ctx.account_id}] Starting game...")
        self._adb.start_app()
        return BotState.LOGIN

    def _handle_login(self, ctx: StateContext) -> BotState:
        """Login với account hiện tại."""
        accounts = self._loader.get_enabled_accounts()
        account = next((a for a in accounts if a.id == ctx.account_id), None)
        if not account:
            raise BotError(ErrorCode.LOGIN_001, f"Account not found: {ctx.account_id}")

        logger.info(f"[{ctx.account_id}] Logging in as {account.username}...")

        # Chờ màn hình login xuất hiện
        try:
            self._vision.wait_for_template(
                "login_screen",
                timeout=self._app_cfg.timing.login_timeout,
            )
        except BotError:
            raise BotError(ErrorCode.LOGIN_001, "Login screen not detected")

        # Nhập thông tin
        coords = self._loader.load_ui_coords()
        login_cfg = coords.get("login", {})

        user_x = login_cfg.get("username_field", {}).get("x", 640)
        user_y = login_cfg.get("username_field", {}).get("y", 350)
        pass_x = login_cfg.get("password_field", {}).get("x", 640)
        pass_y = login_cfg.get("password_field", {}).get("y", 420)
        btn_x  = login_cfg.get("login_button", {}).get("x", 640)
        btn_y  = login_cfg.get("login_button", {}).get("y", 490)

        self._input.clear_and_type(user_x, user_y, account.username)
        self._input.clear_and_type(pass_x, pass_y, account.password)
        self._input.tap(btn_x, btn_y)

        return BotState.VERIFY_INGAME

    def _handle_verify_ingame(self, ctx: StateContext) -> BotState:
        logger.info(f"[{ctx.account_id}] Verifying in-game...")
        try:
            self._vision.wait_for_template(
                "ingame_screen",
                timeout=self._app_cfg.timing.login_timeout,
            )
            ctx.reset_retry()
            logger.info(f"[{ctx.account_id}] Login verified OK")
            return BotState.GET_QUEST
        except BotError:
            ctx.retry_count += 1
            if ctx.retry_count >= self._app_cfg.retry.max_attempts:
                raise BotError(ErrorCode.LOGIN_003, "Not in-game after login max retries")
            return BotState.LOGIN

    def _handle_get_quest(self, ctx: StateContext) -> BotState:
        logger.info(f"[{ctx.account_id}] Getting quest...")
        # TODO: Tap nút nhận nhiệm vụ, chờ quest UI
        # Placeholder – sẽ implement sau khi có template assets thật
        ctx.last_action = "get_quest"
        self._checkpoint.save(ctx)
        return BotState.CLASSIFY_QUEST

    def _handle_classify_quest(self, ctx: StateContext) -> BotState:
        logger.info(f"[{ctx.account_id}] Classifying quest...")
        quest_type = self._classifier.classify()
        ctx.current_quest_type = quest_type
        ctx.last_action = f"classify:{quest_type}"
        self._checkpoint.save(ctx)

        if quest_type == ctx.current_quest_type.__class__.UNKNOWN:
            logger.warning(f"[{ctx.account_id}] Quest UNKNOWN – cancelling")
            return BotState.CANCEL_QUEST

        return BotState.NAVIGATE

    def _handle_navigate(self, ctx: StateContext) -> BotState:
        logger.info(f"[{ctx.account_id}] Navigating for quest type {ctx.current_quest_type}...")
        # TODO: Dựa vào quest type để navigate đến NPC/Shop phù hợp
        ctx.last_action = "navigate"
        self._checkpoint.save(ctx)
        return BotState.INTERACT_NPC

    def _handle_interact_npc(self, ctx: StateContext) -> BotState:
        logger.info(f"[{ctx.account_id}] Interacting with NPC...")
        # TODO: Talk NPC, xử lý dialog
        ctx.last_action = "interact_npc"
        self._checkpoint.save(ctx)
        return BotState.RETURN_QUEST

    def _handle_buy_item(self, ctx: StateContext) -> BotState:
        if self._app_cfg.mode.dry_run:
            logger.warning(f"[{ctx.account_id}] DRY_RUN: BUY_ITEM blocked")
            return BotState.RETURN_QUEST
        logger.info(f"[{ctx.account_id}] Buying item...")
        # TODO: implement shop buy
        ctx.last_action = "buy_item"
        return BotState.RETURN_QUEST

    def _handle_return_quest(self, ctx: StateContext) -> BotState:
        logger.info(f"[{ctx.account_id}] Returning to quest giver...")
        ctx.last_action = "return_quest"
        self._checkpoint.save(ctx)
        return BotState.COMPLETE_QUEST

    def _handle_complete_quest(self, ctx: StateContext) -> BotState:
        logger.info(f"[{ctx.account_id}] Completing quest...")
        ctx.total_quests_done += 1
        ctx.last_action = "complete_quest"
        self._db.log_quest(
            account_id=ctx.account_id,
            quest_type=ctx.current_quest_type.value,
            result="COMPLETED",
        )
        self._checkpoint.save(ctx)
        return BotState.GET_QUEST  # Vòng lặp quest tiếp theo

    def _handle_cancel_quest(self, ctx: StateContext) -> BotState:
        logger.warning(f"[{ctx.account_id}] Cancelling quest (UNKNOWN type)")
        self._db.log_quest(
            account_id=ctx.account_id,
            quest_type="UNKNOWN",
            result="CANCELLED",
        )
        return BotState.GET_QUEST

    def _handle_transfer_gold(self, ctx: StateContext) -> BotState:
        if self._app_cfg.mode.dry_run:
            logger.warning(f"[{ctx.account_id}] DRY_RUN: TRANSFER_GOLD blocked")
            return BotState.LOGOUT
        logger.info(f"[{ctx.account_id}] Transferring gold to master...")
        # TODO: Implement gold transfer
        return BotState.LOGOUT

    def _handle_logout(self, ctx: StateContext) -> BotState:
        if self._app_cfg.mode.dry_run:
            logger.warning(f"[{ctx.account_id}] DRY_RUN: LOGOUT blocked")
            return BotState.NEXT_ACCOUNT
        logger.info(f"[{ctx.account_id}] Logging out...")
        self._adb.stop_app()
        time.sleep(2)
        return BotState.NEXT_ACCOUNT

    def _handle_next_account(self, ctx: StateContext) -> BotState:
        logger.info(f"[{ctx.account_id}] Account done. Moving to next.")
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
        if self._on_status_update:
            self._on_status_update(ctx)

    # ============================
    # Public Run API
    # ============================

    def run_all_accounts(self) -> None:
        """Chạy tuần tự tất cả account đang enable."""
        accounts = self._loader.get_enabled_accounts()
        logger.info(f"BotController: running {len(accounts)} account(s)")

        for account in accounts:
            if not self._state_machine.is_running:
                # Đăng ký account vào DB
                self._db.upsert_account(
                    account.id, account.username, account.server
                )
                run_id = self._db.start_run(account.id)
                try:
                    self._state_machine.run(
                        account_id=account.id,
                        initial_state=BotState.BOOT,
                    )
                    self._db.end_run(
                        run_id,
                        final_state=self._state_machine.context.current_state.value,
                        quests_done=self._state_machine.context.total_quests_done,
                    )
                except Exception as e:
                    logger.exception(f"Unhandled error for {account.id}: {e}")
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
