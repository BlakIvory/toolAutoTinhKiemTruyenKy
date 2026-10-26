import pytest
from unittest.mock import MagicMock, patch, call
import numpy as np

from src.bot_controller import BotController
from src.state_machine import StateContext, BotState, QuestType
from src.text_recognition import TextBox


def test_abandon_shop_quest_with_4_buttons_uses_shop_coordinates():
    """Kiểm tra khi hủy NV SPECIAL_NPC_SHOP hoặc hộp thoại có 4 nút,
    khi OCR không bắt được nút hủy, bot sẽ fallback đúng vào tọa độ hàng 2 cột 1 (380, 700)
    chứ không bấm nhầm vào (1167, 580) của NV đánh quái."""
    bot = BotController.__new__(BotController)
    bot._checkpoint = MagicMock()
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "abandon_quest_option": {"x": 1167, "y": 580},
            "abandon_quest_option_shop": {"x": 380, "y": 700},
            "abandon_quest_confirm": {"x": 303, "y": 571},
            "abandon_quest_finish": {"x": 309, "y": 543},
        }
    }
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._is_npc_dialog_open = MagicMock(return_value=True)
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    bot._sync_quest_progress_from_text = MagicMock()
    bot._save_progress = MagicMock()
    bot._get_username = MagicMock(return_value="ACC_02")
    bot._select_contribute_to_receive_quest = MagicMock(return_value=BotState.GET_QUEST)

    # OCR không tìm thấy text "Ta muốn hủy bỏ...", chỉ thấy các nút khác
    btn_done = TextBox("Nhiệm vụ đã hoàn thành!", 0.95, 340, 590)
    btn_view = TextBox("Xem tình hình nhiệm vụ thương hội đã hoàn thành", 0.95, 780, 590)
    btn_not_done = TextBox("Nhiệm vụ vẫn chưa hoàn thành", 0.95, 1170, 590)
    confirm_popup_btn = TextBox("Hủy bỏ nhiệm vụ", 0.95, 303, 571)

    bot._text = MagicMock()
    bot._text.read.side_effect = [
        [btn_done, btn_view, btn_not_done],  # Initial dialog boxes
        [],                                  # Fresh capture step 1 (still no text found)
        [confirm_popup_btn],                 # Step 2 confirm popup appears
        [],                                  # Step 3 finish
    ]

    ctx = StateContext(account_id="ACC_02")
    ctx.current_quest_type = QuestType.SPECIAL_NPC_SHOP
    ctx.total_quests_done = 8
    ctx.extra = {
        "abandon_combat_quest": True,
        "abandon_combat_quest_reason": "Quá 15 phút chưa hoàn thành (17 phút)",
        "returning_quest": True,
    }

    with patch("time.sleep", return_value=None):
        with patch.object(bot, "_ensure_npc_dialog_closed", return_value=True):
            next_state = bot._handle_meet_npc(ctx)

    # Đã bấm đúng tọa độ (380, 700) cho shop quest thay vì (1167, 580)
    tap_calls = [call.args for call in bot._input.tap.call_args_list]
    assert (380, 700) in tap_calls
    assert (1167, 580) not in tap_calls

    # Xác nhận tăng tiến độ vì hủy thành công
    assert ctx.total_quests_done == 9
    assert next_state == BotState.GET_QUEST


def test_abandon_fails_verification_when_popup_never_appears():
    """Kiểm tra: nếu Bước 1 không tìm thấy text và popup xác nhận không bao giờ xuất hiện,
    bot xác minh thất bại, KHÔNG tăng tiến độ total_quests_done và giữ nguyên cờ để thử lại."""
    bot = BotController.__new__(BotController)
    bot._checkpoint = MagicMock()
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "abandon_quest_option": {"x": 1167, "y": 580},
            "abandon_quest_option_shop": {"x": 380, "y": 700},
            "abandon_quest_confirm": {"x": 303, "y": 571},
            "abandon_quest_finish": {"x": 309, "y": 543},
        }
    }
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._is_npc_dialog_open = MagicMock(return_value=True)
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    bot._sync_quest_progress_from_text = MagicMock()
    bot._save_progress = MagicMock()
    bot._get_username = MagicMock(return_value="ACC_02")
    bot._ensure_npc_dialog_closed = MagicMock()

    bot._text = MagicMock()
    # Mọi lần quét OCR đều không thấy nút hủy và không thấy popup
    bot._text.read.return_value = []

    ctx = StateContext(account_id="ACC_02")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.total_quests_done = 5
    ctx.extra = {
        "abandon_combat_quest": True,
        "abandon_combat_quest_reason": "Quá 10 phút",
    }

    with patch("time.sleep", return_value=None):
        next_state = bot._handle_meet_npc(ctx)

    # KHÔNG được tăng số lượng nhiệm vụ đã làm!
    assert ctx.total_quests_done == 5
    # Cờ hủy vẫn còn nguyên để thử lại
    assert ctx.extra.get("abandon_combat_quest") is True
    # Chuyển về FIND_COMMERCE_CHIEF để thử lại
    assert next_state == BotState.FIND_COMMERCE_CHIEF
    bot._ensure_npc_dialog_closed.assert_called_once()
