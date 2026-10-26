from unittest.mock import MagicMock, patch
from src.text_recognition import TextBox
from src.bot_controller import BotController
from src.state_machine import StateContext, BotState


def test_meet_npc_returns_logout_when_20_quests_done():
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._screen = MagicMock()
    bot._input = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._checkpoint = MagicMock()
    bot._save_progress = MagicMock()
    bot._vision = MagicMock()
    bot._find_dialog_buttons = MagicMock(return_value=[(400, 600)])
    bot._get_username = MagicMock(return_value='hk0001')
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    # Dialog is open before turn-in and closes automatically on success.
    bot._is_npc_dialog_open = MagicMock(side_effect=[True, False])

    ctx = StateContext(account_id='ACC_01')
    ctx.total_quests_done = 19
    ctx.extra = {'returning_quest': True}

    bot._text.read.side_effect = [
        [TextBox(text='Hoàn thành nhiệm vụ', confidence=0.9, center_x=400, center_y=600)],
        [TextBox(text='Chúc mừng bạn đã hoàn thành nhiệm vụ!', confidence=0.9, center_x=500, center_y=400)],
    ]

    next_state = bot._handle_meet_npc(ctx)
    assert ctx.total_quests_done == 20
    assert next_state == BotState.LOGOUT


def test_get_quest_returns_logout_when_npc_says_exhausted():
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._screen = MagicMock()
    bot._input = MagicMock()
    bot._save_progress = MagicMock()
    bot._checkpoint = MagicMock()

    ctx = StateContext(account_id='ACC_01')

    bot._text.read.return_value = [
        TextBox(text='Hôm nay ngươi đã hoàn thành hết nhiệm vụ, ngày mai hãy đến!', confidence=0.9, center_x=500, center_y=400)
    ]

    next_state = bot._handle_get_quest(ctx)
    assert next_state == BotState.LOGOUT


def test_start_game_logs_out_old_account_if_ingame_and_not_skip_login():
    bot = BotController.__new__(BotController)
    bot._adb = MagicMock()
    bot._adb.is_connected.return_value = True
    bot._screen = MagicMock()
    bot._is_ingame = MagicMock(return_value=True)
    bot._logout_current_account = MagicMock(return_value=True)
    bot._skip_login = False

    ctx = StateContext(account_id='ACC_02')

    next_state = bot._handle_start_game(ctx)
    bot._logout_current_account.assert_called_once_with(ctx)
    assert next_state == BotState.LOGIN


def test_is_account_completed_today():
    bot = BotController.__new__(BotController)
    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", MagicMock()):
            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")
            with patch("json.load", return_value={"total_quests_done": 20, "status_text": "Đã hoàn thành 20/20", "date": today}):
                assert bot._is_account_completed_today("ACC_01") is True


def test_is_account_completed_today_returns_false_when_recorded_yesterday():
    bot = BotController.__new__(BotController)
    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", MagicMock()):
            with patch("json.load", return_value={"total_quests_done": 20, "status_text": "Đã hoàn thành 20/20", "date": "2020-01-01"}):
                assert bot._is_account_completed_today("ACC_01") is False


def test_verify_ingame_switches_to_next_account_on_account_in_use_popup():
    import numpy as np
    bot = BotController.__new__(BotController)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._input = MagicMock()
    bot._save_progress = MagicMock()
    bot._get_username = MagicMock(return_value="hkcbq01")
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {}
    bot._app_cfg = MagicMock()
    bot._app_cfg.timing.login_timeout = 5
    bot._vision = MagicMock()
    bot._vision.find_template.return_value = MagicMock(found=False, confidence=0.0)

    # Mock _check_and_dismiss_login_error returning True with "Tài khoản hiện đang sử dụng!"
    bot._check_and_dismiss_login_error = MagicMock(return_value=(True, "Tài khoản hiện đang sử dụng!"))

    ctx = StateContext(account_id="ACC_01")
    next_state = bot._handle_verify_ingame(ctx)

    assert next_state == BotState.NEXT_ACCOUNT
    bot._check_and_dismiss_login_error.assert_called()


def test_check_and_dismiss_login_error_detects_and_taps_tro_lai():
    import numpy as np
    bot = BotController.__new__(BotController)
    bot._screen = MagicMock()
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {"login": {"popup_dismiss_button": {"x": 771, "y": 395}}}
    bot._vision = MagicMock()
    bot._vision.find_template.return_value = MagicMock(found=False, confidence=0.0)

    dummy_frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox(text="Tai khoan hien dang su dung!", confidence=0.95, center_x=800, center_y=350),
        TextBox(text="Tro lai", confidence=0.92, center_x=771, center_y=410),
    ]

    has_error, error_msg = bot._check_and_dismiss_login_error(dummy_frame)
    assert has_error is True
    assert "đang sử dụng" in error_msg
    bot._input.tap.assert_called_with(771, 410)

