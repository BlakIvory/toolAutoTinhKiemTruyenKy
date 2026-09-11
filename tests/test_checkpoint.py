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
