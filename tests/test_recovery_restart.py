"""
Test recovery and error restart behavior.
"""
import pytest
from unittest.mock import MagicMock, patch
from src.bot_controller import BotController
from src.state_machine import BotState, StateContext
from src.config_loader import AppConfig, RetryConfig, GameConfig, EmulatorConfig, ADBConfig


def test_recovery_stops_and_restarts_game():
    bot = BotController.__new__(BotController)
    bot._adb = MagicMock()
    bot._app_cfg = MagicMock()
    bot._app_cfg.retry.max_attempts = 3
    
    ctx = StateContext(account_id="ACC_04")
    ctx.extra["chief_nav_started"] = True
    ctx.extra["chief_nav_failures"] = 3
    ctx.extra["chief_dialog_failures"] = 2
    ctx.retry_count = 2

    with patch("src.bot_controller.time.sleep"):
        next_state = bot._handle_recovery(ctx)

    # Đảm bảo đã gọi stop_app và start_app
    bot._adb.stop_app.assert_called_once()
    bot._adb.start_app.assert_called_once()

    # Phải chuyển về START_GAME để chờ game load và login đàng hoàng
    assert next_state == BotState.START_GAME
    # Các cờ trạng thái bị kẹt phải được xóa sạch
    assert "chief_nav_started" not in ctx.extra
    assert "chief_nav_failures" not in ctx.extra
    assert "chief_dialog_failures" not in ctx.extra
    assert ctx.retry_count == 0
    assert ctx.extra.get("recovery_count") == 1


def test_recovery_exceeded_switches_next_account_after_clean_restart():
    bot = BotController.__new__(BotController)
    bot._adb = MagicMock()
    bot._app_cfg = MagicMock()
    bot._app_cfg.retry.max_attempts = 3
    
    ctx = StateContext(account_id="ACC_04")
    ctx.extra["recovery_count"] = 3  # Đã thử 3 lần trước đó

    with patch("src.bot_controller.time.sleep"):
        next_state = bot._handle_recovery(ctx)

    # Vẫn phải tắt và khởi động lại game sạch
    bot._adb.stop_app.assert_called_once()
    bot._adb.start_app.assert_called_once()

    # Sau khi vượt quá số lần recovery, chuyển sang NEXT_ACCOUNT thay vì crash
    assert next_state == BotState.NEXT_ACCOUNT
