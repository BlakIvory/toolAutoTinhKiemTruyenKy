import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.bot_controller import BotController, BotState, StateContext
from src.quest_classifier import (
    extract_dialogue_target,
    is_auto_abandon_dialogue_quest,
    parse_yellow_entities,
    QuestType,
)
from src.text_recognition import TextBox

def test_extract_dialogue_target_au_dong():
    text = (
        "Gần phía Bắc Nhạn Môn quan gần đây xuất hiện 10 vạn quân Liêu, xem ra sắp có chiến sự lớn. "
        "Vì Quốc gia đại sự, Chủ thương hội trong thành đang thu thập các loại vật tư. "
        "Làm con dân Đại Tống nên dốc chút sức lực trước vận nước lâm nguy. "
        "Giúp chủ thương hội đến Tuyền Châu thông báo cho ấu đồng. "
        "Cùng với Xa phu đối thoại, hoàn thành nhiệm vụ"
    )
    yellow_tokens = ["Tuyền Châu", "ấu đồng"]

    # 1. Có yellow tokens
    city, npc = extract_dialogue_target(text, with_accents=True, yellow_tokens=yellow_tokens)
    assert city == "Tuyền Châu"
    assert npc == "Ấu đồng"

    # 2. Không có yellow tokens (full text OCR)
    city_raw, npc_raw = extract_dialogue_target(text, with_accents=True, yellow_tokens=None)
    assert city_raw == "Tuyền Châu"
    assert npc_raw == "Ấu đồng"


def test_is_auto_abandon_dialogue_quest():
    assert is_auto_abandon_dialogue_quest(npc="Ấu đồng") is True
    assert is_auto_abandon_dialogue_quest(npc="au dong") is True
    assert is_auto_abandon_dialogue_quest(text="thong bao cho au dong tai Tuyen Chau") is True
    assert is_auto_abandon_dialogue_quest(npc="Khấu Chuẩn") is False
    assert is_auto_abandon_dialogue_quest(npc="Xa phu") is False


def test_classify_quest_abandons_au_dong_immediately():
    bot = BotController.__new__(BotController)
    bot._checkpoint = MagicMock()
    bot._classifier = MagicMock()
    bot._classifier.classify.return_value = QuestType.NPC
    bot._return_to_nearest_city_commerce_chief = MagicMock(return_value=True)
    bot._save_progress = MagicMock()
    bot._is_four_refugee_quest = MagicMock(return_value=False)

    ctx = StateContext(account_id="ACC_02")
    ctx.extra = {
        "quest_text": "Giúp chủ thương hội đến Tuyền Châu thông báo cho ấu đồng.",
        "quest_target_city": "Tuyền Châu",
        "quest_target_npc": "Ấu đồng",
    }

    res = bot._handle_classify_quest(ctx)

    assert res == BotState.RETURN_QUEST
    assert ctx.extra.get("abandon_dialogue_quest") is True
    assert ctx.extra.get("abandon_combat_quest") is True
    assert "Ấu đồng" in ctx.extra.get("abandon_combat_quest_reason", "")
    bot._return_to_nearest_city_commerce_chief.assert_called_once_with(ctx)


def test_return_quest_executes_abandon_for_dialogue_quest():
    bot = BotController.__new__(BotController)
    bot._checkpoint = MagicMock()
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "abandon_quest_option": {"x": 1167, "y": 576},
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
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox("Ta muốn hủy bỏ nhiệm vụ đang làm", 0.95, 1167, 576),
    ]

    ctx = StateContext(account_id="ACC_02")
    ctx.extra = {
        "quest_text": "Giúp chủ thương hội đến Tuyền Châu thông báo cho ấu đồng.",
        "quest_target_city": "Tuyền Châu",
        "quest_target_npc": "Ấu đồng",
        "abandon_dialogue_quest": True,
        "abandon_combat_quest": True,
        "abandon_combat_quest_reason": "Nhiệm vụ gặp NPC Ấu đồng (tự động hủy)",
        "returning_quest": True,
    }

    with patch("time.sleep", return_value=None):
        # Trả về FIND_COMMERCE_CHIEF hoặc tiếp tục vòng lặp sau khi hủy
        with patch.object(bot, "_ensure_npc_dialog_closed", return_value=True):
            res = bot._handle_meet_npc(ctx)

    # Đã gọi bấm nút hủy
    bot._input.tap.assert_any_call(1167, 576)
    # Các cờ hủy được dọn dẹp sạch sau khi hủy thành công
    assert "abandon_dialogue_quest" not in ctx.extra
    assert ctx.total_quests_done == 1
