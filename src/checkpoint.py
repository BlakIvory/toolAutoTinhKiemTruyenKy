"""
AutoQuestTool - Checkpoint
Lưu và đọc runtime state từ state.json.
Dùng cho recovery khi tool bị crash.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict

from loguru import logger

from .state_machine import StateContext, BotState, QuestType


STATE_FILE = Path("runtime/state.json")


class Checkpoint:
    """
    Lưu trạng thái runtime vào state.json.
    Đọc lại để recovery khi restart.
    """

    def __init__(self, state_file: Path = STATE_FILE):
        self.state_file = state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    def save(self, context: StateContext) -> None:
        """Lưu context → state.json."""
        data = {
            "account_id": context.account_id,
            "current_state": context.current_state.value,
            "previous_state": context.previous_state.value,
            "current_quest_type": context.current_quest_type.value,
            "current_map": context.current_map,
            "current_npc": context.current_npc,
            "current_shop": context.current_shop,
            "retry_count": context.retry_count,
            "error_code": context.error_code,
            "error_message": context.error_message,
            "last_action": context.last_action,
            "total_quests_done": context.total_quests_done,
            "timestamp": datetime.utcnow().isoformat(),
        }
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Checkpoint saved: state={data['current_state']}")
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")

    def load(self) -> Optional[Dict[str, Any]]:
        """Đọc state.json. Trả về dict hoặc None nếu không có."""
        if not self.state_file.exists():
            return None
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Checkpoint loaded: {data}")
            return data
        except Exception as e:
            logger.error(f"Checkpoint load failed: {e}")
            return None

    def restore_context(self, data: Dict[str, Any]) -> StateContext:
        """Convert dict → StateContext."""
        ctx = StateContext()
        ctx.account_id = data.get("account_id", "")
        ctx.current_state = BotState(data.get("current_state", BotState.BOOT))
        ctx.previous_state = BotState(data.get("previous_state", BotState.IDLE))
        ctx.current_quest_type = QuestType(
            data.get("current_quest_type", QuestType.UNKNOWN)
        )
        ctx.current_map = data.get("current_map", "")
        ctx.current_npc = data.get("current_npc", "")
        ctx.current_shop = data.get("current_shop", "")
        ctx.retry_count = data.get("retry_count", 0)
        ctx.error_code = data.get("error_code", "")
        ctx.error_message = data.get("error_message", "")
        ctx.last_action = data.get("last_action", "")
        ctx.total_quests_done = data.get("total_quests_done", 0)
        return ctx

    def clear(self) -> None:
        """Xóa state.json sau khi hoàn thành."""
        if self.state_file.exists():
            self.state_file.unlink()
            logger.debug("Checkpoint cleared")

    def has_checkpoint(self) -> bool:
        return self.state_file.exists()
