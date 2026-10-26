from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.bot_controller import BotController
from src.state_machine import StateContext, QuestType, BotState
from src.text_recognition import TextBox, is_western_merchant_name


def test_is_western_merchant_name_strictly_filters_dialogue():
    # Valid NPC name tags: short and concise
    assert is_western_merchant_name("Thương nhân Tây Vực")
    assert is_western_merchant_name("Thuong nhan Tay Ho")
    assert is_western_merchant_name("Thương nhân phương tây")
    
    # Invalid: Long sentences from dialogue / quest text must NOT be recognized as merchant
    assert not is_western_merchant_name("Ta da dat 1 bao dao 0 cho Thuong nhan Tay Vuc, xin giup ta lay ve! Co the den Bien Ki")
    assert not is_western_merchant_name("Xin giup ta den gap Thuong nhan Tay Vuc de mua do")
    assert not is_western_merchant_name("Long Tuyen Thon")


def test_handle_get_quest_does_not_revive_ghost_quest_on_unknown_dialog():
    bot = BotController.__new__(BotController)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._text = MagicMock()
    bot._classifier = MagicMock()
    bot._sync_quest_progress_from_text = MagicMock()
    bot._checkpoint = MagicMock()

    # OCR returns background text instead of valid quest
    bg_box = TextBox("Tan Thu Bo Auto huyen thu amTrongY Chu thuonghoi", 0.9, 500, 400)
    bot._text.read.return_value = [bg_box]
    bot._classifier.classify.return_value = QuestType.UNKNOWN

    ctx = StateContext(account_id="ACC_01")
    # Simulate stale leftover fields from a previous combat quest
    ctx.extra = {
        "combat_map": "Vu Lang son",
        "combat_monster": "Xí Hỏa Hắc giáo đồ",
        "combat_count": 10,
    }

    next_state = bot._handle_get_quest(ctx)

    # Must NOT become BotState.CLASSIFY_QUEST with COMBAT type
    assert next_state in (BotState.GET_QUEST, BotState.MEET_NPC)
    assert ctx.extra.get("combat_map") is None
    assert ctx.extra.get("combat_monster") is None


def test_check_quest_panel_returns_none_when_no_commerce_quest():
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._screen = MagicMock()
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "quest_button": {"x": 1460, "y": 765},
            "quest_panel_close": {"x": 1466, "y": 125},
            "quest_panel_merchant_row": {"x": 482, "y": 445},
        }
    }
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._is_quest_panel_open = MagicMock(return_value=True)
    bot._ensure_quest_panel_closed = MagicMock()
    bot._sync_quest_progress_from_text = MagicMock()

    # The quest panel list only has Tân Thủ / Dã Tẩu, no "Thương hội"
    other_quest_box = TextBox("Nhiệm vụ hướng dẫn Tân Thủ", 0.95, 300, 300)
    bot._text.read.side_effect = [
        [TextBox("Nhiệm vụ", 0.95, 1460, 765)],  # icon check
        [other_quest_box],                      # quest list check
    ]

    ctx = StateContext(account_id="ACC_01")
    result = bot._check_quest_panel(ctx)

    # Must return None and not blindly click row 482/445
    assert result is None
    bot._ensure_quest_panel_closed.assert_called_once()


def test_combat_auto_button_activated_when_monster_spotted():
    bot = BotController.__new__(BotController)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "combat": {
            "auto_button": {"x": 1308, "y": 241},
            "horse_button": {"x": 1553, "y": 370},
        }
    }
    bot._input = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._ensure_quest_panel_closed = MagicMock()
    bot._ensure_combat_skills_visible = MagicMock()
    bot._ensure_dismounted_from_horse = MagicMock()
    bot._collapse_chat_bar = MagicMock()
    bot._get_current_location_from_hud = MagicMock(return_value=("Ngũ Độc Mê Cung", (100, 100)))
    bot._is_same_combat_map = MagicMock(return_value=True)
    bot._checkpoint = MagicMock()
    bot._clear_monsters_around_character = MagicMock()
    bot._return_to_nearest_city_commerce_chief = MagicMock(return_value=True)

    # Monster spotted at (833, 494)
    cycle = [0]
    def mock_find(frame, monster, blocked_positions=None):
        cycle[0] += 1
        if cycle[0] == 1:
            return (833, 494)
        return None
    bot._find_monster_on_screen = MagicMock(side_effect=mock_find)

    # After cycle 1, quest finishes
    read_c = [0]
    def mock_read(frame, roi=None):
        read_c[0] += 1
        if read_c[0] >= 5:
            return [TextBox("Tiêu diệt Thực nhân thảo (10/10)", 0.95, 100, 300)]
        return []
    bot._text = MagicMock()
    bot._text.read.side_effect = mock_read

    ctx = StateContext(account_id="ACC_02")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.extra = {
        "combat_map": "Ngũ Độc Mê Cung",
        "combat_monster": "Thực nhân thảo",
        "combat_count": 10,
    }

    bot._handle_navigate(ctx)

    # Verify that auto_button at (1308, 241) was tapped to engage auto attack
    tap_calls = [call[0] for call in bot._input.tap.call_args_list]
    assert (1308, 241) in tap_calls


def test_meet_npc_no_quest_dialog_taps_contribute_and_never_checks_quest_icon():
    bot = BotController.__new__(BotController)
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._sync_quest_progress_from_text = MagicMock()
    bot._save_progress = MagicMock()
    bot._input = MagicMock()
    bot._check_active_commerce_quest = MagicMock()

    # Boxes representing the exact screenshot: "Vì triều đình cống hiến sức lực"
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox("Hoàng thượng vừa ban chiếu: Nhân sĩ nào giúp sức Triều đình sẽ được trọng thưởng! Mỗi ngày có thể nhận tối đa 20 lần nhiệm vụ!", 0.95, 500, 250),
        TextBox("Vì triều đình cống hiến sức lực", 0.95, 326, 578),
        TextBox("Xem tình hình nhiệm vụ thương hội đã hoàn thành", 0.95, 522, 369),
        TextBox("Giới thiệu tỉ mỉ nhiệm vụ thương hội", 0.95, 762, 376),
    ]

    ctx = StateContext(account_id="ACC_01")
    ctx.extra = {
        "combat_monster": "Sói điên",
        "has_active_quest": True,
    }

    next_state = bot._handle_meet_npc(ctx)

    assert next_state == BotState.GET_QUEST
    bot._input.tap.assert_called_with(326, 578)
    bot._check_active_commerce_quest.assert_not_called()
    assert ctx.extra.get("combat_monster") is None
    assert ctx.extra.get("has_active_quest") is None


def test_get_quest_with_imperial_edict_re_taps_contribute_never_opens_quest_icon():
    bot = BotController.__new__(BotController)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._sync_quest_progress_from_text = MagicMock()
    bot._input = MagicMock()
    bot._check_active_commerce_quest = MagicMock()
    bot._classifier = MagicMock()

    # The dialog still shows the imperial edict
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox("Hoàng thượng vừa ban chiếu: Nhân sĩ nào giúp sức Triều đình sẽ được trọng thưởng! Mỗi ngày có thể nhận tối đa 20 lần nhiệm vụ!", 0.95, 500, 250),
        TextBox("Vì triều đình cống hiến sức lực", 0.95, 326, 578),
        TextBox("Xem tình hình nhiệm vụ thương hội đã hoàn thành", 0.95, 522, 369),
    ]

    ctx = StateContext(account_id="ACC_01")

    # Mock extract_yellow_text_boxes returning background stall and daily 20 lan
    with patch("src.quest_classifier.extract_yellow_text_boxes") as mock_y:
        mock_y.return_value = [
            TextBox("tral cay le", 0.9, 301, 143),
            TextBox("20 lan", 0.9, 500, 250),
        ]
        next_state = bot._handle_get_quest(ctx)

    assert next_state == BotState.GET_QUEST
    bot._input.tap.assert_called_with(326, 578)
    bot._check_active_commerce_quest.assert_not_called()
    assert ctx.extra.get("combat_monster") is None

