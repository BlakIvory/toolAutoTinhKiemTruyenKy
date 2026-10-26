import pytest
from unittest.mock import MagicMock
import numpy as np
from src.state_machine import BotState, StateContext
from src.text_recognition import TextBox
from src.bot_controller import BotController


def test_handle_meet_npc_fast_reopen_does_not_call_dismiss_popups():
    """Kiểm tra: sau khi hoàn thành nhiệm vụ và đóng dialog, bot không gọi _dismiss_unexpected_popups gây nghẽn OCR."""
    bot = BotController.__new__(BotController)
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    bot._check_active_commerce_quest = MagicMock(return_value=None)
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bot._input = MagicMock()
    bot._checkpoint = MagicMock()
    bot._save_progress = MagicMock()
    bot._ensure_npc_dialog_closed = MagicMock()
    bot._dismiss_unexpected_popups = MagicMock()
    bot._select_contribute_to_receive_quest = MagicMock(return_value=BotState.GET_QUEST)

    complete_box = TextBox("Nhiem vudahoan thanh!", 0.95, 361, 693)
    bot._text = MagicMock()
    bot._text.read.side_effect = [
        [complete_box],
        [],
    ]

    ctx = StateContext(account_id="ACC_TEST")
    ctx.extra = {"returning_quest": True}

    res = bot._handle_meet_npc(ctx)

    assert res == BotState.GET_QUEST
    # Không được gọi _dismiss_unexpected_popups khi mở lại đối thoại
    bot._dismiss_unexpected_popups.assert_not_called()
