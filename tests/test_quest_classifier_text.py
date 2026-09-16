from unittest.mock import Mock

from src.quest_classifier import QuestClassifier, extract_dialogue_target, extract_shop_item
from src.state_machine import QuestType


def classifier():
    vision = Mock()
    vision._screen.capture.return_value = None
    vision._templates.has.return_value = False
    return QuestClassifier(vision, Mock())


def test_classifies_shop_text_without_accents():
    assert classifier().classify(text="Tim thuong nhan phuong tay de mua vat pham") == QuestType.SPECIAL_NPC_SHOP


def test_classifies_combat_text_with_surrounding_words():
    assert classifier().classify(text="Den ngoai thanh danh quai tieu diet 10 ten") == QuestType.COMBAT


def test_classifies_npc_conversation():
    assert classifier().classify(text="Di noi chuyen va doi thoai voi NPC") == QuestType.NPC


def test_extracts_dialogue_city_and_npc_from_sample():
    city, npc = extract_dialogue_target(
        "Giúp Thương hội đến Thành Đô phủ nói với Chu Thiên Nhân: hàng đã đến!"
    )
    assert city == "Thành Đô"
    assert npc == "chu thien nhan"


def test_extracts_western_merchant_item_from_sample():
    text = "Ta đã đặt một quả trứng đại bàng ở chỗ Thương nhân Tây Vực, xin giúp ta lấy về"
    assert extract_shop_item(text) == "qua trung dai bang"


def test_classifies_western_merchant_sample_as_shop():
    text = "Ta đã đặt một quả trứng đại bàng ở chỗ Thương nhân Tây Vực, xin giúp ta lấy về"
    assert classifier().classify(text=text) == QuestType.SPECIAL_NPC_SHOP


def test_extracts_unseen_map_dynamically():
    city, npc = extract_dialogue_target(
        "Hãy đến Bắc Nhạn Môn Quan nói với Lý Tướng Quân để báo tin"
    )
    assert city == "bac nhan mon quan"
    assert npc == "ly tuong quan"


def test_each_quest_gets_its_own_target():
    assert extract_dialogue_target("Đến Đại Lý nói với Đoàn Công Tử để hỏi thăm") == (
        "Đại Lý", "doan cong tu"
    )


def test_rejects_screen_noise_and_does_not_classify_as_shop():
    clf = classifier()
    # Random player names / screen words must NOT match Western Merchant
    assert clf.classify(text="Truong Dai Hiu") == QuestType.UNKNOWN
    assert clf.classify(text="Pham Trong Y HKUOOUZ Nhat DO HK00001") == QuestType.UNKNOWN
    # Security/account popups must NOT match
    assert clf.classify(text="2. NPH se khong yeu cau nguoi choi phai cung cap thong tin tai khoan") == QuestType.UNKNOWN
    assert clf.classify(text="Thuong xuyen thay doi mat khau de tranh bi mat tai khoan") == QuestType.UNKNOWN


def test_shop_quest_without_item_is_not_classified_as_shop():
    clf = classifier()
    # Text mentioning merchant or trade in general without item or western merchant
    assert clf.classify(text="Gap thuong nhan de hoi tham") == QuestType.NPC


def test_extract_dialogue_target_with_proper_accents():
    from src.quest_classifier import extract_dialogue_target, get_canonical_npc_name

    assert get_canonical_npc_name("bach hieu sinh") == "Bách Hiểu Sinh"
    assert get_canonical_npc_name("nan dan dinh") == "Nạn dân Đinh"
    assert get_canonical_npc_name("van du dao nhan") == "Vân Du đạo nhân"

    c1, n1 = extract_dialogue_target(
        "Giúp Thương hội đến Thành Đô phủ nói với Bách Hiểu Sinh: hàng đã đến!",
        with_accents=True
    )
    assert c1 == "Thành Đô"
    assert n1 == "Bách Hiểu Sinh"

    c2, n2 = extract_dialogue_target(
        "Hãy đến Thành Đô gặp Nạn dân Đinh nói hàng đến rồi, có rảnh đến lấy.",
        with_accents=True
    )
    assert c2 == "Thành Đô"
    assert n2 == "Nạn dân Đinh"

    c3, n3 = extract_dialogue_target(
        "Giúp Thương hội đến Biện Kinh nói với Vân Du đạo nhân: hàng đã đến!",
        with_accents=True
    )
def test_extract_and_classify_linh_duoc_tay_vuc():
    quest_text = (
        "Ta đã đặt một bình linh dược Tây Vực ở chỗ Thương nhân Tây Vực, xin giúp ta lấy về! "
        "Có thể đến Biện Kinh, Thành Đô, Tuyền Châu, Tương Dương, Dương Châu, Đại Lý để tìm ông ta."
    )
    assert classifier().classify(text=quest_text) == QuestType.SPECIAL_NPC_SHOP
    assert extract_shop_item(quest_text) == "linh duoc tay vuc"



