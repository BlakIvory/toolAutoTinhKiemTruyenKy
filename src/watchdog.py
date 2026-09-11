"""
AutoQuestTool - Watchdog Service
Phát hiện crash, freeze, timeout và trigger recovery.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Callable

from loguru import logger

from .config_loader import AppConfig
from .adb_controller import ADBController
from .vision_engine import VisionEngine
from .error_codes import ErrorCode, BotError
from .state_machine import BotState, StateContext


class WatchdogEvent:
    GAME_CRASH    = "GAME_CRASH"
    GAME_FREEZE   = "GAME_FREEZE"
    LOADING_TIMEOUT = "LOADING_TIMEOUT"
    UNKNOWN_SCREEN  = "UNKNOWN_SCREEN"
    ADB_DISCONNECTED = "ADB_DISCONNECTED"
    STATE_TIMEOUT   = "STATE_TIMEOUT"


class WatchdogService:
    """
    Monitor bot trong background thread.
    Phát hiện: crash / freeze / disconnect / timeout.
    Recovery: tùy error policy trong config.
    """

    CHECK_INTERVAL = 5.0  # giây kiểm tra mỗi lần

    def __init__(
        self,
        config: AppConfig,
        adb: ADBController,
        vision: VisionEngine,
        on_event: Optional[Callable[[str, StateContext], None]] = None,
    ):
        self.config = config
        self._adb = adb
        self._vision = vision
        self._on_event = on_event
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._context: Optional[StateContext] = None
        self._last_state: Optional[BotState] = None
        self._state_entered_at: float = 0

    # ============================
    # Lifecycle
    # ============================

    def start(self, context: StateContext) -> None:
        """Khởi động watchdog thread."""
        self._context = context
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._watch_loop,
            name="WatchdogThread",
            daemon=True,
        )
        self._thread.start()
        logger.info("WatchdogService: started")

    def stop(self) -> None:
        self._stop_flag.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        logger.info("WatchdogService: stopped")

    def update_context(self, context: StateContext) -> None:
        """Cập nhật context từ state machine."""
        self._context = context

    # ============================
    # Watch Loop
    # ============================

    def _watch_loop(self) -> None:
        while not self._stop_flag.is_set():
            try:
                self._check()
            except Exception as e:
                logger.error(f"Watchdog check error: {e}")
            time.sleep(self.CHECK_INTERVAL)

    def _check(self) -> None:
        ctx = self._context
        if ctx is None:
            return

        # 1. ADB connectivity
        try:
            if not self._adb.is_connected():
                self._trigger(WatchdogEvent.ADB_DISCONNECTED, ctx)
                return
        except Exception:
            self._trigger(WatchdogEvent.ADB_DISCONNECTED, ctx)
            return

        # 2. Game process still running
        if not self._adb.is_app_running():
            logger.warning("Watchdog: game process not detected → GAME_CRASH")
            self._trigger(WatchdogEvent.GAME_CRASH, ctx)
            return

        # 3. State timeout
        elapsed = time.time() - ctx.state_entered_at
        state_timeout = self.config.timing.action_timeout * 5  # Watchdog timeout = 5x action timeout
        if elapsed > state_timeout:
            logger.warning(
                f"Watchdog: state {ctx.current_state} running {elapsed:.0f}s "
                f"(limit {state_timeout:.0f}s) → STATE_TIMEOUT"
            )
            self._trigger(WatchdogEvent.STATE_TIMEOUT, ctx)

    def _trigger(self, event: str, ctx: StateContext) -> None:
        logger.error(f"Watchdog EVENT: {event} | account={ctx.account_id} | state={ctx.current_state}")
        if self._on_event:
            try:
                self._on_event(event, ctx)
            except Exception as e:
                logger.error(f"Watchdog on_event callback error: {e}")
