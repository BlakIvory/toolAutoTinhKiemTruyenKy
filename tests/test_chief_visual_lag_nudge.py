import pytest
from unittest.mock import MagicMock
from src.state_machine import BotState, StateContext
from src.text_recognition import TextBox
from src.bot_controller import BotController


def test_is_near_commerce_chief_matches_without_space():
    bot = BotController.__new__(BotController)
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._text = MagicMock()
    # RapidOCR reads 'Chu thuonghoi' without space
    bot._text.read.return_value = [
        TextBox(text="Chu thuonghoi", confidence=0.9, center_x=807, center_y=352)
    ]
    res = bot._is_near_commerce_chief("dummy_scr")
    assert res is True


def test_nudge_left_right_alternates_directions():
    bot = BotController.__new__(BotController)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "combat": {"joystick_center": {"x": 220, "y": 680}}
    }
    bot._input = MagicMock()

    ctx = StateContext(account_id="ACC_01")

    # 1st call: horizontal swipe (left then right)
    bot._nudge_left_right_to_reveal_npc(ctx)
    assert bot._input.swipe.call_count == 2
    args1 = bot._input.swipe.call_args_list[0][0]
    args2 = bot._input.swipe.call_args_list[1][0]
    # start_x=220, end_x=120, y unchanged=680
    assert args1[0] == 220 and args1[2] == 120 and args1[1] == args1[3] == 680
    assert args2[0] == 220 and args2[2] == 320 and args2[1] == args2[3] == 680

    bot._input.swipe.reset_mock()

    # 2nd call: vertical swipe (up then down)
    bot._nudge_left_right_to_reveal_npc(ctx)
    assert bot._input.swipe.call_count == 2
    args3 = bot._input.swipe.call_args_list[0][0]
    args4 = bot._input.swipe.call_args_list[1][0]
    # start_y=680, end_y=580, x unchanged=220
    assert args3[1] == 680 and args3[3] == 580 and args3[0] == args3[2] == 220
    assert args4[1] == 680 and args4[3] == 780 and args4[0] == args4[2] == 220


def test_find_commerce_chief_nudges_when_dialog_unconfirmed_and_remaps_on_3_failures():
    bot = BotController.__new__(BotController)
    bot._attack_monster_blocking_travel = MagicMock(return_value=False)
    bot._get_current_location_from_hud = MagicMock(return_value=("Tuyền Châu", (192, 183)))
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._open_commerce_chief_dialog = MagicMock(return_value=False)
    bot._verify_commerce_chief_dialog = MagicMock(return_value=False)
    bot._ensure_npc_dialog_closed = MagicMock()
    bot._nudge_left_right_to_reveal_npc = MagicMock(return_value=True)
    bot._navigate_to_tuyen_chau_commerce_chief = MagicMock(return_value=True)

    ctx = StateContext(account_id="ACC_01")

    # 1st attempt: fails, nudges
    state1 = bot._handle_find_commerce_chief(ctx)
    assert state1 == BotState.FIND_COMMERCE_CHIEF
    assert ctx.extra["chief_dialog_failures"] == 1
    assert bot._nudge_left_right_to_reveal_npc.call_count == 1
    bot._navigate_to_tuyen_chau_commerce_chief.assert_not_called()

    # 2nd attempt: fails, nudges again
    state2 = bot._handle_find_commerce_chief(ctx)
    assert state2 == BotState.FIND_COMMERCE_CHIEF
    assert ctx.extra["chief_dialog_failures"] == 2
    assert bot._nudge_left_right_to_reveal_npc.call_count == 2
    bot._navigate_to_tuyen_chau_commerce_chief.assert_not_called()

    # 3rd attempt: triggers re-navigation via Minimap
    state3 = bot._handle_find_commerce_chief(ctx)
    assert state3 == BotState.FIND_COMMERCE_CHIEF
    assert "chief_dialog_failures" not in ctx.extra
    bot._navigate_to_tuyen_chau_commerce_chief.assert_called_once()
