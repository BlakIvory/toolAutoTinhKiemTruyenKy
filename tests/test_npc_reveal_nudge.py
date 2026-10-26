from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.bot_controller import BotController
from src.state_machine import StateContext, BotState
from src.text_recognition import TextBox


def test_nudge_left_right_swipes_joystick_left_then_right():
    bot = BotController.__new__(BotController)
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "combat": {
            "joystick_center": {"x": 220, "y": 680}
        }
    }
    ctx = StateContext(account_id="ACC_03")

    with patch("time.sleep", return_value=None):
        res = bot._nudge_left_right_to_reveal_npc(ctx)

    assert res is True
    assert bot._input.swipe.call_count == 2
    # Call 1: left (220, 680) -> (120, 680)
    bot._input.swipe.assert_any_call(220, 680, 120, 680, duration=350)
    # Call 2: right (220, 680) -> (320, 680)
    bot._input.swipe.assert_any_call(220, 680, 320, 680, duration=350)


def test_navigate_nudges_when_npc_unseen_after_arriving():
    bot = BotController.__new__(BotController)
    bot._checkpoint = MagicMock()
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {"combat": {"joystick_center": {"x": 220, "y": 680}}}
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bot._dismiss_unexpected_popups = MagicMock(return_value=False)
    bot._ensure_npc_dialog_closed = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._is_npc_interact_bubble_present = MagicMock(return_value=True)
    bot._click_npc_interact_bubble = MagicMock()
    bot._nudge_left_right_to_reveal_npc = MagicMock(return_value=True)

    # Đứng yên tại tọa độ đích (200, 200) của Dương Châu
    bot._get_current_location_from_hud = MagicMock(return_value=("Dương Châu", (200, 200)))

    # Frame ban đầu: chưa thấy tên NPC "Chủ Tạp hóa" do culling chống hack
    # Sau khi nudge: frame hiển thị "Chủ Tạp hóa"
    bot._text = MagicMock()
    dialog_open = False
    bot._is_npc_dialog_open = MagicMock(side_effect=lambda *args: dialog_open)
    bot._find_dialog_buttons = MagicMock(return_value=[])

    def mock_read(frame, region=None):
        if bot._nudge_left_right_to_reveal_npc.called:
            return [TextBox("Chủ Tạp hóa", 0.98, 750, 400)]
        return []
    bot._text.read = MagicMock(side_effect=mock_read)

    def on_click_bubble(frame=None):
        nonlocal dialog_open
        if bot._nudge_left_right_to_reveal_npc.called:
            dialog_open = True
    bot._click_npc_interact_bubble.side_effect = on_click_bubble

    ctx = StateContext(account_id="ACC_03")
    ctx.extra = {
        "quest_target_city": "Dương Châu",
        "quest_target_npc": "Chủ Tạp hóa",
    }

    # Giả lập thời gian tăng dần để stable_for >= 0.8s
    times = [100.0 + 0.5 * idx for idx in range(30)]
    t_idx = 0
    def mock_time():
        nonlocal t_idx
        v = times[min(t_idx, len(times) - 1)]
        t_idx += 1
        return v

    with patch("time.sleep", return_value=None), patch("time.time", side_effect=mock_time):
        res = bot._handle_navigate(ctx)

    assert bot._nudge_left_right_to_reveal_npc.call_count >= 1
    assert bot._click_npc_interact_bubble.call_count >= 1
    assert res == BotState.INTERACT_NPC


def test_interact_npc_nudges_if_dialog_not_opened():
    bot = BotController.__new__(BotController)
    bot._checkpoint = MagicMock()
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {"combat": {"joystick_center": {"x": 220, "y": 680}}}
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bot._get_current_location_from_hud = MagicMock(return_value=("Dương Châu", (200, 200)))
    bot._is_npc_interact_bubble_present = MagicMock(return_value=True)
    bot._click_npc_interact_bubble = MagicMock()
    bot._nudge_left_right_to_reveal_npc = MagicMock(return_value=True)
    bot._ensure_npc_dialog_closed = MagicMock()
    def mock_finish(*args):
        return (800, 750) if dialog_state else None
    bot._find_finish_dialogue_button = MagicMock(side_effect=mock_finish)

    dialog_state = False
    def mock_dialog_open(*args):
        return dialog_state
    bot._is_npc_dialog_open = MagicMock(side_effect=mock_dialog_open)

    def mock_find_buttons(*args):
        return [(300, 400)] if dialog_state else []
    bot._find_dialog_buttons = MagicMock(side_effect=mock_find_buttons)

    def on_click(frame=None):
        nonlocal dialog_state
        if bot._nudge_left_right_to_reveal_npc.called:
            dialog_state = True
    bot._click_npc_interact_bubble.side_effect = on_click

    def on_tap(x, y):
        nonlocal dialog_state
        if (x, y) == (800, 750):
            dialog_state = False
    bot._input.tap.side_effect = on_tap

    ctx = StateContext(account_id="ACC_03")
    ctx.extra = {
        "quest_target_city": "Dương Châu",
        "quest_target_npc": "Chủ Tạp hóa",
    }

    with patch("time.sleep", return_value=None):
        res = bot._handle_interact_npc(ctx)

    assert bot._nudge_left_right_to_reveal_npc.call_count >= 1
    assert bot._click_npc_interact_bubble.call_count >= 2
    assert res == BotState.RETURN_QUEST
