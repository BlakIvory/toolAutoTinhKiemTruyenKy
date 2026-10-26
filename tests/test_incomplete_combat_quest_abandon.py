from unittest.mock import MagicMock

import pytest

from src.bot_controller import BotController
from src.state_machine import BotState, QuestType, StateContext
from src.quest_classifier import extract_combat_target


def test_extract_combat_target_from_commerce_chief_dialogue():
    text = (
        "Gần đây có rất nhiều Nữ Kiếm Khách bị trục xuất khỏi sư môn tụ tập "
        "ở Nga My Phái gây rối, hãy đi trừng trị 10 ả. "
        "Có cần ta đưa đến nơi làm nhiệm vụ không?"
    )

    assert extract_combat_target(text) == ("Nga My", "Nữ Kiếm khách", 10)


def test_extract_combat_target_from_quest_panel_wording():
    text = (
        "Gần đây có rất nhiều Nữ Kiếm Khách bị trục xuất khỏi sư môn tụ tập "
        "ở Nga My Phái, thậm chí đã đả thương không ít thương khách, "
        "hãy giúp đi trừng trị 10 tên!"
    )

    assert extract_combat_target(text, ["Nt Kiem khach", "10", "!!"]) == (
        "Nga My", "Nữ Kiếm khách", 10
    )


def test_yellow_monster_and_count_cannot_be_misclassified_as_npc():
    from src.quest_classifier import QuestClassifier

    classifier = QuestClassifier.__new__(QuestClassifier)
    classifier._vision = MagicMock()

    result = classifier.classify(
        text="Nhiệm vụ Thương hội chuyển thư",
        yellow_tokens=["Nt Kiem khach", "10", "!!"],
    )

    assert result == QuestType.COMBAT


def test_handle_classify_prioritizes_current_yellow_combat_evidence():
    bot = BotController.__new__(BotController)
    bot._classifier = MagicMock()
    bot._classifier.classify.return_value = QuestType.NPC
    bot._checkpoint = MagicMock()
    bot._save_progress = MagicMock()

    ctx = StateContext(account_id="ACC_03")
    ctx.extra = {
        "quest_text": "Nhiệm vụ Thương hội chuyển thư",
        "yellow_tokens": ["Nt Kiem khach", "10", "!!"],
    }

    next_state = bot._handle_classify_quest(ctx)

    assert ctx.current_quest_type == QuestType.COMBAT
    assert ctx.extra["combat_monster"] == "Nữ Kiếm khách"
    assert ctx.extra["combat_count"] == 10
    assert ctx.extra["abandon_combat_quest"] is True
    assert "địa điểm" in ctx.extra["abandon_combat_quest_reason"]
    assert next_state == BotState.FIND_COMMERCE_CHIEF


@pytest.mark.parametrize(
    "combat_data,missing_label",
    [
        ({"combat_monster": "Nữ Kiếm khách", "combat_count": 10}, "địa điểm/map đánh quái"),
        ({"combat_map": "Nga My", "combat_count": 10}, "tên quái"),
        ({"combat_map": "Nga My", "combat_monster": "Nữ Kiếm khách"}, "số lượng quái"),
    ],
)
def test_incomplete_combat_metadata_is_abandoned(combat_data, missing_label):
    bot = BotController.__new__(BotController)
    bot._classifier = MagicMock(return_value=QuestType.COMBAT)
    bot._checkpoint = MagicMock()
    bot._save_progress = MagicMock()

    ctx = StateContext(account_id="ACC_TEST")
    ctx.extra = {"quest_text": "Hãy đi trừng trị quân gây rối", **combat_data}

    next_state = bot._handle_classify_quest(ctx)

    assert next_state == BotState.FIND_COMMERCE_CHIEF
    assert ctx.extra["abandon_combat_quest"] is True
    assert ctx.extra["returning_quest"] is True
    assert missing_label in ctx.extra["abandon_combat_quest_reason"]
    bot._checkpoint.save.assert_called_with(ctx)


def test_explicit_dialogue_clears_stale_combat_cache_instead_of_abandoning():
    bot = BotController.__new__(BotController)
    bot._classifier = MagicMock(return_value=QuestType.NPC)
    bot._checkpoint = MagicMock()
    bot._save_progress = MagicMock()
    bot._navigate_to_dialogue_npc = MagicMock(return_value=True)

    ctx = StateContext(account_id="ACC_03")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.extra = {
        "quest_text": (
            "Giup Thuong hoi den Tuong Duong phu noi voi Xa phu: hang da den!"
        ),
        "quest_target_city": "Tương Dương",
        "quest_target_npc": "Xa phu",
        # Dữ liệu sai/cũ từ nhiệm vụ đánh Nhện.
        "combat_map": "Tường Vân động",
        "combat_monster": "Nhện độc",
        "combat_count": 10,
        "abandon_combat_quest": True,
        "abandon_combat_quest_reason": "map",
    }

    next_state = bot._handle_classify_quest(ctx)

    assert next_state == BotState.NAVIGATE
    assert ctx.current_quest_type == QuestType.NPC
    assert "combat_map" not in ctx.extra
    assert "combat_monster" not in ctx.extra
    assert "combat_count" not in ctx.extra
    assert "abandon_combat_quest" not in ctx.extra
    bot._navigate_to_dialogue_npc.assert_called_once_with(
        ctx, "Tương Dương", "Xa phu"
    )


def test_abandon_quest_increments_total_quests_done():
    from src.text_recognition import TextBox
    import numpy as np

    bot = BotController.__new__(BotController)
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    bot._ensure_npc_dialog_closed = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bot._input = MagicMock()
    bot._checkpoint = MagicMock()
    bot._save_progress = MagicMock()
    bot._get_username = MagicMock(return_value="test_user")
    bot._select_contribute_to_receive_quest = MagicMock(return_value=BotState.GET_QUEST)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {}

    cancel_btn = TextBox("Ta muon huy bo nhiem vu dang lam", 0.95, 1167, 576)
    bot._text = MagicMock()
    bot._text.read.side_effect = [
        [cancel_btn],   # Initial meet dialog
        [],             # Step 2 confirm
        [],             # Step 3 finish
    ]

    ctx = StateContext(account_id="ACC_04")
    ctx.total_quests_done = 2
    ctx.extra = {
        "abandon_combat_quest": True,
        "abandon_combat_quest_reason": "Quá 10 phút chưa hoàn thành",
    }

    next_state = bot._handle_meet_npc(ctx)

    assert ctx.total_quests_done == 3
    assert "abandon_combat_quest" not in ctx.extra
    assert any(h["num"] == 3 and "Hủy NV" in h["detail"] for h in ctx.extra.get("quest_history_list", []))
    assert next_state == BotState.GET_QUEST


def test_combat_timeout_abandon_with_stale_dialogue_keys_abandons_successfully():
    """Kiểm tra: khi bị timeout quá 10 phút, dù ctx.extra còn sót lại quest_target_city/npc
    từ nhiệm vụ đối thoại cũ, bot vẫn phải tiến hành hủy nhiệm vụ bình thường mà không bị xóa cờ."""
    from src.text_recognition import TextBox
    import numpy as np

    bot = BotController.__new__(BotController)
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    bot._ensure_npc_dialog_closed = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bot._input = MagicMock()
    bot._checkpoint = MagicMock()
    bot._save_progress = MagicMock()
    bot._get_username = MagicMock(return_value="test_user")
    bot._select_contribute_to_receive_quest = MagicMock(return_value=BotState.GET_QUEST)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {}

    cancel_btn = TextBox("Ta muon huy bo nhiem vu dang lam", 0.95, 1167, 576)
    bot._text = MagicMock()
    bot._text.read.side_effect = [
        [cancel_btn],   # Initial meet dialog
        [],             # Step 2 confirm
        [],             # Step 3 finish
    ]

    ctx = StateContext(account_id="ACC_02")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.total_quests_done = 8
    ctx.extra = {
        "abandon_combat_quest": True,
        "abandon_combat_quest_reason": "Quá 10 phút chưa hoàn thành (10 phút)",
        "combat_map": "Tầng 1 Dương Trung động",
        "combat_monster": "Quân Liêu",
        # Cờ đối thoại cũ còn sót lại từ nhiệm vụ trước
        "quest_target_city": "Biện Kinh",
        "quest_target_npc": "Xa phu",
    }

    next_state = bot._handle_meet_npc(ctx)

    assert ctx.total_quests_done == 9
    assert "abandon_combat_quest" not in ctx.extra
    # Cờ đối thoại cũ phải được dọn dẹp
    assert "quest_target_city" not in ctx.extra
    assert "quest_target_npc" not in ctx.extra
    assert any(h["num"] == 9 and "Hủy NV" in h["detail"] for h in ctx.extra.get("quest_history_list", []))
    assert next_state == BotState.GET_QUEST


