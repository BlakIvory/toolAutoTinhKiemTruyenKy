import time
from unittest.mock import MagicMock, patch
import pytest

from src.bot_controller import BotController
from src.state_machine import BotState, StateContext
from src.quest_classifier import QuestType


def test_should_abandon_combat_immediately_true():
    bot = BotController.__new__(BotController)
    cfg_mock = MagicMock()
    cfg_mock.commerce_optimization.abandon_combat_immediately = True
    bot._cfg = cfg_mock

    assert bot._should_abandon_combat_immediately() is True


def test_quest_timeout_triggers_and_returns_commerce_chief():
    bot = BotController.__new__(BotController)
    cfg_mock = MagicMock()
    cfg_mock.commerce_optimization.quest_timeout_minutes = 15
    bot._cfg = cfg_mock
    bot._checkpoint = MagicMock()

    ctx = StateContext(account_id="ACC_01")
    # Set quest started 16 minutes ago
    ctx.extra["current_quest_start_time"] = time.time() - (16 * 60)

    assert bot._check_quest_timeout(ctx) is True
    assert ctx.extra.get("abandon_combat_quest") is True
    assert ctx.extra.get("returning_quest") is True
    assert "15 phút" in ctx.extra.get("abandon_combat_quest_reason", "")


def test_quest_timeout_not_triggered_under_15_mins():
    bot = BotController.__new__(BotController)
    cfg_mock = MagicMock()
    cfg_mock.commerce_optimization.quest_timeout_minutes = 15
    bot._cfg = cfg_mock

    ctx = StateContext(account_id="ACC_01")
    # Set quest started 5 minutes ago
    ctx.extra["current_quest_start_time"] = time.time() - (5 * 60)

    assert bot._check_quest_timeout(ctx) is False
    assert ctx.extra.get("abandon_combat_quest") is None
