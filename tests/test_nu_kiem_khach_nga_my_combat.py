"""
Test classification and target extraction for Nu Kiem Khach Nga My combat quest,
and ensure near commerce chief coordinate checks don't produce false positives in Nga My.
"""
import pytest
from unittest.mock import MagicMock
from src.quest_classifier import extract_combat_target, QuestClassifier, QuestType
from src.bot_controller import BotController
from src.state_machine import StateContext


def test_extract_combat_target_nu_kiem_khach_nga_my():
    quest_raw = (
        "y huong dan Tan Thu Auto ie Chu thuong hoi:Gan day co rat nhieu Nt Kiem Khach "
        "bi tryc xuat khoi su mon ty tap 0 Nga My Phai gay roi, hay di truing tri 10 a. "
        "am Co can ta dua den noi lam nhiem vukhong? hie Vi o Dong Bac Tuye am Trong Yem "
        "Phien dua ta di mot chuyen Khong can dau Cung voi Bach Hieu Sinh doi thoai,hoan thanh nhiem vu Monphai"
    )
    yellow_tokens = ['Ngudi Phat Thud', 'y huong dan Tan Thu', "uep uer'", 'Nt Kiem Khach', '10', 'am']

    c_map, c_monster, c_count = extract_combat_target(quest_raw, yellow_tokens=yellow_tokens)

    assert c_map == "Nga My"
    assert c_monster == "Nữ Kiếm khách"
    assert c_count == 10


def test_classify_nu_kiem_khach_not_overridden_by_monphai_quest():
    quest_raw = (
        "Chu thuong hoi:Gan day co rat nhieu Nt Kiem Khach "
        "bi tryc xuat khoi su mon ty tap 0 Nga My Phai gay roi, hay di truing tri 10 a. "
        "Co can ta dua den noi lam nhiem vukhong? Phien dua ta di mot chuyen Khong can dau "
        "Cung voi Bach Hieu Sinh doi thoai,hoan thanh nhiem vu Monphai"
    )
    yellow_tokens = ['Nt Kiem Khach', '10']
    classifier = QuestClassifier.__new__(QuestClassifier)

    q_type = classifier.classify(text=quest_raw, yellow_tokens=yellow_tokens)
    assert q_type == QuestType.COMBAT


def test_is_near_commerce_chief_rejects_nga_my_at_192_183():
    bot = BotController.__new__(BotController)
    bot._is_map_open = MagicMock(return_value=False)
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._text = MagicMock()
    bot._text.read.return_value = []
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "major_cities": [
                {"name": "Tuyền Châu", "x": 192, "y": 182}
            ]
        }
    }
    # Đang ở Nga My, tọa độ (192, 183)
    bot._get_current_location_from_hud = MagicMock(return_value=("Nga My", (192, 183)))

    assert bot._is_near_commerce_chief() is False
