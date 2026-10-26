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
            "current_state": context.current_state.value if getattr(context, "current_state", None) else "IDLE",
            "previous_state": context.previous_state.value if getattr(context, "previous_state", None) else None,
            "current_quest_type": context.current_quest_type.value if getattr(context, "current_quest_type", None) else None,
            "current_map": context.current_map,
            "current_npc": context.current_npc,
            "current_shop": context.current_shop,
            "retry_count": context.retry_count,
            "error_code": context.error_code,
            "error_message": context.error_message,
            "last_action": context.last_action,
            "total_quests_done": context.total_quests_done,
            "extra": context.extra,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": datetime.utcnow().isoformat(),
        }
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Checkpoint saved: state={data['current_state']}")
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")

    def is_expired(self) -> bool:
        """Kiểm tra state.json có phải từ ngày cũ không (nhiệm vụ Thương hội reset mỗi ngày)."""
        if not self.state_file.exists():
            return False
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            today_str = datetime.now().strftime("%Y-%m-%d")
            file_date = data.get("date")
            if not file_date:
                ts = data.get("timestamp", "")
                if ts and "T" in ts:
                    file_date = ts.split("T")[0]
            if not file_date:
                try:
                    file_date = datetime.fromtimestamp(self.state_file.stat().st_mtime).strftime("%Y-%m-%d")
                except Exception:
                    file_date = today_str
            return file_date != today_str
        except Exception:
            return False

    def load(self, allow_expired: bool = False) -> Optional[Dict[str, Any]]:
        """Đọc state.json. Trả về dict hoặc None nếu không có hoặc đã qua ngày mới."""
        if not self.state_file.exists():
            return None
        if not allow_expired and self.is_expired():
            logger.info("Checkpoint state.json đã qua ngày mới, tự động reset.")
            self.clear()
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
        ctx.extra = data.get("extra", {})
        return ctx

    def clear(self) -> None:
        """Xóa state.json sau khi hoàn thành."""
        if self.state_file.exists():
            self.state_file.unlink()
            logger.debug("Checkpoint cleared")

    def has_checkpoint(self) -> bool:
        return self.state_file.exists()
