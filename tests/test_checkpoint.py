"""
Tests: Checkpoint
"""

import pytest
import json
from pathlib import Path

from src.checkpoint import Checkpoint
from src.state_machine import StateContext, BotState, QuestType


@pytest.fixture
def checkpoint(tmp_path):
    return Checkpoint(state_file=tmp_path / "state.json")


def test_save_and_load(checkpoint):
    """Test save và load checkpoint."""
    ctx = StateContext()
    ctx.account_id = "ACC_01"
    ctx.current_state = BotState.NAVIGATE
    ctx.current_map = "main_city"
    ctx.total_quests_done = 5

    checkpoint.save(ctx)
    assert checkpoint.has_checkpoint()

    data = checkpoint.load()
    assert data is not None
    assert data["account_id"] == "ACC_01"
    assert data["current_state"] == "NAVIGATE"
    assert data["current_map"] == "main_city"
    assert data["total_quests_done"] == 5


def test_restore_context(checkpoint):
    """Test restore context từ saved data."""
    ctx = StateContext()
    ctx.account_id = "ACC_02"
    ctx.current_state = BotState.LOGIN
    ctx.current_quest_type = QuestType.NPC
    checkpoint.save(ctx)

    data = checkpoint.load()
    restored = checkpoint.restore_context(data)

    assert restored.account_id == "ACC_02"
    assert restored.current_state == BotState.LOGIN
    assert restored.current_quest_type == QuestType.NPC


def test_clear(checkpoint):
    """Test xóa checkpoint."""
    ctx = StateContext()
    checkpoint.save(ctx)
    assert checkpoint.has_checkpoint()
    checkpoint.clear()
    assert not checkpoint.has_checkpoint()


def test_no_checkpoint(checkpoint):
    """Test khi chưa có checkpoint."""
    assert not checkpoint.has_checkpoint()
    assert checkpoint.load() is None


def test_expired_checkpoint_resets_on_new_day(tmp_path):
    """Test checkpoint từ ngày cũ sẽ được tự động xóa khi load."""
    state_file = tmp_path / "state.json"
    old_data = {
        "account_id": "ACC_01",
        "current_state": "NAVIGATE",
        "total_quests_done": 15,
        "date": "2020-01-01",
        "timestamp": "2020-01-01T08:00:00",
    }
    state_file.write_text(json.dumps(old_data), encoding="utf-8")
    cp = Checkpoint(state_file=state_file)
    assert cp.is_expired() is True
    loaded = cp.load()
    assert loaded is None
    assert not state_file.exists()


def test_save_with_none_quest_type(checkpoint):
    """Test checkpoint save khi current_quest_type là None (không bị lỗi 'NoneType' object has no attribute 'value')."""
    ctx = StateContext()
    ctx.account_id = "ACC_03"
    ctx.current_state = BotState.GET_QUEST
    ctx.current_quest_type = None

    checkpoint.save(ctx)
    data = checkpoint.load()
    assert data["current_quest_type"] is None


