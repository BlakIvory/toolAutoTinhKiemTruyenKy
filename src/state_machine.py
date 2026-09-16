"""
AutoQuestTool - State Machine
BotState enum + StateMachine quản lý workflow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Callable, Dict, Optional, Set

from loguru import logger

from .error_codes import ErrorCode, BotError
from .config_loader import AppConfig


# ============================
# Bot States
# ============================

class BotState(str, Enum):
    """Tất cả trạng thái của bot."""
    IDLE              = "IDLE"
    BOOT              = "BOOT"
    CONNECT_EMULATOR  = "CONNECT_EMULATOR"
    START_GAME        = "START_GAME"
    LOGIN             = "LOGIN"
    VERIFY_INGAME     = "VERIFY_INGAME"
    CHECK_LOCATION    = "CHECK_LOCATION"
    MOVE_TO_CITY      = "MOVE_TO_CITY"
    FIND_COMMERCE_CHIEF = "FIND_COMMERCE_CHIEF"
    MEET_NPC          = "MEET_NPC"
    GET_QUEST         = "GET_QUEST"
    CLASSIFY_QUEST    = "CLASSIFY_QUEST"
    NAVIGATE          = "NAVIGATE"
    INTERACT_NPC      = "INTERACT_NPC"
    BUY_ITEM          = "BUY_ITEM"
    CANCEL_QUEST      = "CANCEL_QUEST"
    RETURN_QUEST      = "RETURN_QUEST"
    COMPLETE_QUEST    = "COMPLETE_QUEST"
    TRANSFER_GOLD     = "TRANSFER_GOLD"
    LOGOUT            = "LOGOUT"
    NEXT_ACCOUNT      = "NEXT_ACCOUNT"
    FINISHED          = "FINISHED"
    RECOVERY          = "RECOVERY"
    ERROR             = "ERROR"
    PAUSED            = "PAUSED"


class QuestType(str, Enum):
    """Loại nhiệm vụ."""
    NPC              = "NPC"
    SPECIAL_NPC_SHOP = "SPECIAL_NPC_SHOP"
    COMBAT           = "COMBAT"
    UNKNOWN          = "UNKNOWN"


# ============================
# State Context
# ============================

@dataclass
class StateContext:
    """Dữ liệu runtime chia sẻ giữa các state."""
    account_id: str = ""
    current_state: BotState = BotState.IDLE
    previous_state: BotState = BotState.IDLE
    current_quest_type: QuestType = QuestType.UNKNOWN
    current_map: str = ""
    current_npc: str = ""
    current_shop: str = ""
    retry_count: int = 0
    error_code: str = ""
    error_message: str = ""
    last_action: str = ""
    state_entered_at: float = field(default_factory=time.time)
    total_quests_done: int = 0
    extra: dict = field(default_factory=dict)

    def reset_retry(self):
        self.retry_count = 0

    def increment_retry(self):
        self.retry_count += 1


# ============================
# State Machine
# ============================

# Type cho handler function: (context) → BotState (state tiếp theo)
StateHandler = Callable[[StateContext], BotState]


class StateMachine:
    """
    Quản lý workflow của bot theo state machine.
    Không dùng chuỗi if/else khổng lồ.
    Mỗi state có một handler function.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._handlers: Dict[BotState, StateHandler] = {}
        self._context = StateContext()
        self._running = False
        self._paused = False
        self._on_state_change: Optional[Callable] = None  # Callback cho GUI

        # Valid transitions (nếu muốn enforce)
        self._valid_transitions: Dict[BotState, Set[BotState]] = {}

    # ============================
    # Registration
    # ============================

    def register(self, state: BotState, handler: StateHandler) -> None:
        """Đăng ký handler cho một state."""
        self._handlers[state] = handler
        logger.debug(f"StateMachine: registered handler for {state}")

    def set_state_change_callback(self, callback: Callable) -> None:
        """Callback gọi mỗi khi state thay đổi (cho GUI)."""
        self._on_state_change = callback

    # ============================
    # Transitions
    # ============================

    def _transition(self, new_state: BotState) -> None:
        """Chuyển sang state mới."""
        old_state = self._context.current_state
        if old_state == new_state:
            return
        logger.info(
            f"[{self._context.account_id}] "
            f"STATE: {old_state} → {new_state}"
        )
        self._context.previous_state = old_state
        self._context.current_state = new_state
        self._context.state_entered_at = time.time()

        if self._on_state_change:
            try:
                self._on_state_change(old_state, new_state, self._context)
            except Exception:
                pass

    def _check_state_timeout(self) -> bool:
        """Kiểm tra state hiện tại có vượt timeout không."""
        elapsed = time.time() - self._context.state_entered_at
        timeout = self.config.timing.action_timeout
        if elapsed > timeout:
            logger.warning(
                f"State timeout: {self._context.current_state} "
                f"elapsed={elapsed:.1f}s limit={timeout}s"
            )
            return True
        return False

    # ============================
    # Run
    # ============================

    def run(
        self,
        account_id: str,
        initial_state: BotState = BotState.BOOT,
    ) -> None:
        """
        Chạy state machine cho một account.
        Block cho đến khi đến FINISHED hoặc ERROR.
        """
        self._context = StateContext(
            account_id=account_id,
            current_state=initial_state,
        )
        self._running = True
        logger.info(f"[{account_id}] StateMachine starting from {initial_state}")

        terminal_states = {BotState.FINISHED, BotState.ERROR}

        while self._running and self._context.current_state not in terminal_states:
            # Pause support
            if self._paused:
                time.sleep(0.5)
                continue

            state = self._context.current_state
            handler = self._handlers.get(state)

            if handler is None:
                logger.error(f"No handler for state: {state}")
                self._transition(BotState.ERROR)
                break

            try:
                next_state = handler(self._context)
                if next_state:
                    self._transition(next_state)
                else:
                    logger.warning(f"Handler for {state} returned None")
                    self._transition(BotState.ERROR)

            except BotError as e:
                logger.error(
                    f"[{account_id}] BotError in {state}: {e.code} – {e.message}"
                )
                self._context.error_code = e.code
                self._context.error_message = e.message
                self._transition(BotState.ERROR)

            except Exception as e:
                logger.exception(f"[{account_id}] Unexpected error in {state}: {e}")
                self._context.error_message = str(e)
                self._transition(BotState.ERROR)

        logger.info(
            f"[{account_id}] StateMachine ended: {self._context.current_state}"
        )
        self._running = False

    def stop(self) -> None:
        logger.info("StateMachine: STOP requested")
        self._running = False

    def pause(self) -> None:
        logger.info("StateMachine: PAUSE")
        self._paused = True
        self._context.current_state = BotState.PAUSED

    def resume(self) -> None:
        logger.info("StateMachine: RESUME")
        self._paused = False
        if self._context.current_state == BotState.PAUSED:
            self._transition(self._context.previous_state)

    @property
    def context(self) -> StateContext:
        return self._context

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused
