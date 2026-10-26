import pytest
from unittest.mock import MagicMock
import numpy as np
from src.state_machine import BotState, StateContext, QuestType
from src.bot_controller import BotController


def test_fast_combat_start_immediately_moves_joystick_without_heavy_ocr():
    """Kiểm tra: khi vừa tới map quái, bot không chạy OCR toàn màn hình popup hay banner check,
    mà lập tức kiểm tra quái hoặc vuốt joystick tuần tra ngay trong vòng đầu tiên."""
    bot = BotController.__new__(BotController)
    bot._screen = MagicMock()
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._screen.capture.return_value = frame

    bot._text = MagicMock()
    bot._text.read.return_value = []

    bot._checkpoint = MagicMock()
    bot._check_quest_timeout = MagicMock(return_value=False)
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "combat": {
            "joystick_center": {"x": 220, "y": 680},
            "patrol_minimap_points": [
                {"x": 1150, "y": 550, "name": "Đông - Sâu trong hang", "dx": 130, "dy": 40}
            ]
        }
    }
    bot._dismiss_unexpected_popups = MagicMock(return_value=False)
    bot._ensure_combat_skills_visible = MagicMock(return_value=True)
    bot._collapse_chat_bar = MagicMock()
    bot._ensure_dismounted_from_horse = MagicMock(return_value=True)
    bot._get_current_location_from_hud = MagicMock(return_value=("Dương Trung động", (200, 200)))
    bot._is_same_combat_map = MagicMock(return_value=True)
    bot._is_map_open = MagicMock(return_value=False)
    bot._find_monster_on_screen = MagicMock(return_value=None)
    bot.clamp_to_safe_combat_ground = MagicMock(return_value=(1150, 550))
    bot._move_combat_patrol = MagicMock()

    ctx = StateContext(account_id="ACC_TEST")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.extra = {
        "combat_map": "Dương Trung động",
        "combat_monster": "Quan sự",
        "combat_count": 10,
    }

    # Giới hạn vòng lặp sau 1 chu kỳ để kiểm tra hành vi đầu tiên
    def mock_move(pt, w, h):
        raise KeyboardInterrupt("Stop test after first patrol")

    bot._move_combat_patrol.side_effect = mock_move

    try:
        bot._handle_navigate(ctx)
    except KeyboardInterrupt:
        pass

    # 1. Popup dismiss toàn màn hình (gây nghẽn 15s) KHÔNG ĐƯỢC CHẠY ở chu kỳ đầu
    bot._dismiss_unexpected_popups.assert_not_called()

    # 2. Joystick tuần tra ĐÃ ĐƯỢC GỌI NGAY LẬP TỨC ở chu kỳ đầu tiên
    assert bot._move_combat_patrol.call_count == 1
