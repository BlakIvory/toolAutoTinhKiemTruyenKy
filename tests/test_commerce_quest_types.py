from unittest.mock import Mock
from src.quest_classifier import (
    QuestClassifier,
    QuestType,
    extract_dialogue_target,
    extract_shop_item,
    extract_combat_target,
)


def make_classifier():
    vision = Mock()
    vision._screen.capture.return_value = None
    vision._templates.has.return_value = False
    return QuestClassifier(vision, Mock())


def test_type_1_npc_dialogue():
    clf = make_classifier()
    text = "Hãy đến Thành Đô gặp Nạn dân Đinh nói hàng đến rồi, có rảnh đến lấy."
    assert clf.classify(text=text) == QuestType.NPC
    city, npc = extract_dialogue_target(text)
    from src.text_recognition import normalize_game_text
    assert "thanh do" in normalize_game_text(city)
    assert npc == "nan dan dinh"


def test_type_2_western_merchant():
    clf = make_classifier()
    cases = [
        ("Hãy mua Mật Lị Hương Nang ở chỗ thương nhân phương tây", "mat li huong nang"),
        ("Đến gặp Thương nhân Tây Vực mua 1 Ngọc Bội", "ngoc boi"),
        ("Hãy mua Vũ Khí từ thương nhân phương tây", "vu khi"),
    ]
    for text, expected_item in cases:
        assert clf.classify(text=text) == QuestType.SPECIAL_NPC_SHOP
        assert extract_shop_item(text) == expected_item


def test_type_3_combat():
    clf = make_classifier()
    text1 = "Hãy đến Tường Vân động tiêu diệt 20 Sói điên"
    assert clf.classify(text=text1) == QuestType.COMBAT
    map_name, monster, count = extract_combat_target(text1)
    assert map_name == "tuong van dong"
    assert monster == "soi dien"
    assert count == 20

    text2 = "Đến Phục Ngưu sơn đánh bại 15 Khỉ đột"
    assert clf.classify(text=text2) == QuestType.COMBAT
    map_name2, monster2, count2 = extract_combat_target(text2)
    assert map_name2 == "phuc nguu son"
    assert monster2 == "khi dot"
    assert count2 == 15
