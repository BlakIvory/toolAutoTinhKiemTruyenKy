from unittest.mock import MagicMock
import pytest
from src.quest_classifier import QuestClassifier, QuestType, extract_shop_item, match_western_merchant_item, parse_yellow_entities


def test_classify_thit_lac_da_with_yellow_tokens():
    """Nhiệm vụ mua thịt lạc đà có chữ vàng phải nhận dạng thành SPECIAL_NPC_SHOP."""
    vision = MagicMock()
    vision.find_template.return_value.found = False
    qc = QuestClassifier(vision, {})

    text = (
        "Gần phía Bắc Nhạn Môn quan gần đây xuất hiện 10 vạn quân Liêu, xem ra sắp có chiến sự lớn. "
        "Vì Quốc gia đại sự, Chủ thương hội trong thành đang thu thập các loại vật tư. "
        "Làm con dân Đại Tống nên dốc chút sức lực trước vận nước lâm nguy. "
        "Ta đã đặt một miếng thịt lạc đà ở Thương nhân Tây Vực, xin giúp ta lấy về! "
        "Có thể đến Biện Kinh, Thành Đô, Tuyền Châu, Tương Dương, Dương Châu, Đại Lý để tìm ông ta."
    )
    # 1. Với chữ vàng "thịt lạc đà"
    assert qc.classify(text=text, yellow_tokens=["thịt lạc đà"]) == QuestType.SPECIAL_NPC_SHOP

    # 2. Với chữ vàng OCR có dấu lệch nhỏ "thrt Iac da"
    entities = parse_yellow_entities(["thrt Iac da"])
    assert entities.get("item") == "Thịt Lạc Đà"
    assert qc.classify(text=text, yellow_tokens=["thrt Iac da"]) == QuestType.SPECIAL_NPC_SHOP

    # 3. Kể cả khi yellow_tokens rỗng, full text vẫn phải nhận dạng đúng SPECIAL_NPC_SHOP
    assert qc.classify(text=text, yellow_tokens=[]) == QuestType.SPECIAL_NPC_SHOP


def test_extract_shop_item_thit_lac_da():
    """Hàm extract_shop_item trích xuất chính xác 'Thịt Lạc Đà'."""
    text = "Ta đã đặt một miếng thịt lạc đà ở Thương nhân Tây Vực, xin giúp ta lấy về!"
    assert extract_shop_item(text) == "Thịt Lạc Đà"
    assert match_western_merchant_item("thịt lạc đà") == "Thịt Lạc Đà"
    assert match_western_merchant_item("miếng thịt lạc đà") == "Thịt Lạc Đà"


def test_empty_quest_text_never_falls_back_to_npc():
    """Khi quest_text và yellow_tokens rỗng, tuyệt đối không đoán mò thành QuestType.NPC qua template match."""
    vision = MagicMock()
    vision.find_template.return_value.found = True  # Giả lập trên màn hình có template quest_icon_npc của Tân thủ
    qc = QuestClassifier(vision, {})

    # Khi không có text và không có yellow_tokens, classify không được nhận bừa NPC từ HUD
    assert qc.classify(text="", yellow_tokens=[]) == QuestType.UNKNOWN
