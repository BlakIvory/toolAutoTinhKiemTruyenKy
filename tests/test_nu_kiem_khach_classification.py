from src.quest_classifier import (
    QuestClassifier,
    match_western_merchant_item,
    parse_yellow_entities,
)
from src.state_machine import QuestType


def test_ocr_nt_kiem_khach_is_monster_not_shop_item():
    entities = parse_yellow_entities(["Nt Kiem Khach", "10"])

    assert entities["monster"] == "Nữ Kiếm khách"
    assert entities["count"] == 10
    assert entities["item"] == ""


def test_explicit_trung_tri_overrides_bad_yellow_item_inference():
    classifier = QuestClassifier.__new__(QuestClassifier)
    text = (
        "Gần đây có rất nhiều Nữ Kiếm Khách bị trục xuất khỏi sư môn "
        "tụ tập ở Nga My Phái gây rối, hãy đi trừng trị 10 tên!"
    )

    result = classifier.classify(text=text, yellow_tokens=["Nt Kiem Khach", "10"])

    assert result == QuestType.COMBAT


def test_only_whitelisted_names_are_merchant_items():
    assert match_western_merchant_item("Kim Cang Nguyên Thạch") == "Kim Cang Nguyên Thạch"
    assert match_western_merchant_item("Nt Kiem Khach") == ""
    assert match_western_merchant_item("Ngọc Bội") == ""
    assert match_western_merchant_item("Vũ Khí") == ""
