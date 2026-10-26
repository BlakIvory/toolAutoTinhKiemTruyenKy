from unittest.mock import Mock
from src.quest_classifier import (
    QuestClassifier,
    QuestType,
    extract_dialogue_target,
    extract_shop_item,
    extract_combat_target,
    match_western_merchant_item,
)
from src.text_recognition import normalize_game_text


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
        ("Hãy mua Bảo Đao Tây Vực ở chỗ thương nhân phương tây", "Bảo Đao Tây Vực"),
        ("Đến gặp Thương nhân Tây Vực mua 1 Dạ Quang Bôi", "Dạ Quang Bôi"),
        ("Hãy mua Kim Cang Nguyên Thạch từ thương nhân phương tây", "Kim Cang Nguyên Thạch"),
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


def test_linh_duoc_tay_vuc_user_screenshot_quest():
    clf = make_classifier()
    full_quest = (
        "Gần phía Bắc Nhạn Môn quan gần đây xuất hiện 10 vạn quân Liêu, xem ra sắp có chiến sự lớn. "
        "Vì Quốc gia đại sự, Chủ thương hội trong thành đang thu thập các loại vật tư. "
        "Làm con dân Đại Tống nên dốc chút sức lực trước vận nước lâm nguy. "
        "Ta đã đặt một bình linh dược Tây Vực ở chỗ Thương nhân Tây Vực, xin giúp ta lấy về! "
        "Có thể đến Biện Kinh, Thành Đô, Tuyền Châu, Tương Dương, Dương Châu, Đại Lý để tìm ông ta."
    )
    # 1. Phân loại quest type
    assert clf.classify(text=full_quest) == QuestType.SPECIAL_NPC_SHOP

    item = extract_shop_item(full_quest)
    assert "linh duoc" in normalize_game_text(item)

    # 3. Không được nhầm lẫn thành đối thoại NPC
    city, npc = extract_dialogue_target(full_quest, with_accents=True)
    assert city == ""
    assert npc == ""

    # 4. Kiểm tra với chữ vàng được trích xuất
    yellow_tokens = ["linh dược Tây Vực"]
    assert clf.classify(text="", yellow_tokens=yellow_tokens) == QuestType.SPECIAL_NPC_SHOP
    item_from_yellow = extract_shop_item("", yellow_tokens=yellow_tokens)
    assert item_from_yellow == "Linh Dược Tây Vực"

    # 5. Kiểm tra đoạn đuôi nếu bị cắt dòng trên trong OCR
    tail_text = "Có thể đến Biện Kinh, Thành Đô, Tuyền Châu, Tương Dương, Dương Châu, Đại Lý để tìm ông ta. Kết thúc đối thoại"
    assert clf.classify(text=tail_text) == QuestType.UNKNOWN
    c_tail, n_tail = extract_dialogue_target(tail_text, with_accents=True)
    assert c_tail == ""
    assert n_tail == ""


def test_reordered_amber_stone_name_is_shop_quest():
    clf = make_classifier()
    text = (
        "Gan phia Bac Nhan Mon quan xuat hien 10 van quan Lieu. "
        "Ta da dat lam 1 Nguyen Thach Ho Phach o cho Thuong nhan Tay Vuc, "
        "xin giup ta lay ve."
    )

    assert match_western_merchant_item("Nguyen Thach Ho Phach") == "Hổ Phách Nguyên Thạch"
    assert clf.classify(
        text=text,
        yellow_tokens=["Nguyen Thach Ho Phach"],
    ) == QuestType.SPECIAL_NPC_SHOP


def test_cam_nuong_thanh_do_dialogue_quest():
    clf = make_classifier()
    text = (
        "Gần phía Bắc Nhạn Môn quan gần đây xuất hiện 10 vạn quân Liêu, xem ra sắp có chiến sự lớn. "
        "Vì Quốc gia đại sự, Chủ thương hội trong thành đang thu thập các loại vật tư. "
        "Làm con dân Đại Tống nên dốc chút sức lực trước vận nước lâm nguy. "
        "Giúp Thương hội đến Thành Đô phủ nói với Cẩm Nương: hàng đã đến!"
    )
    yellow_tokens = ["n", "Thanh D8 phi", "Cam Nuong"]

    # 1. Phân loại quest type
    assert clf.classify(text=text, yellow_tokens=yellow_tokens) == QuestType.NPC

    # 2. Trích xuất thành phố và NPC
    city, npc = extract_dialogue_target(text, with_accents=True, yellow_tokens=yellow_tokens)
    assert city == "Thành Đô"
    assert npc == "Cẩm Nương"

    # 3. Không được nhầm lẫn thành đánh quái dù có đoạn lore '10 vạn quân Liêu'
    map_name, monster, count = extract_combat_target(text, yellow_tokens=yellow_tokens)
    assert map_name == ""
    assert monster == ""
    assert count == 0


def test_ho_uyen_long_lac_da_ocr_distortions():
    from src.quest_classifier import parse_yellow_entities

    clf = make_classifier()

    # 1. Test trích xuất từ chữ vàng trên Bảng nhiệm vụ (như ảnh user gửi: "Hj Uyen long lac da")
    entities = parse_yellow_entities(["Hj Uyen long lac da"])
    assert entities["item"] == "Hộ Uyển Lông Lạc Đà"

    # 2. Test trích xuất từ chữ vàng bị méo font hộp thoại ("Da Nhung Tha Thao" / "Da Nhung Thi Thao")
    entities_npc = parse_yellow_entities(["Da Nhung Tha Thao"])
    assert entities_npc["item"] == "Hộ Uyển Lông Lạc Đà"

    # 3. Test với nội dung nhiệm vụ đầy đủ
    quest_text = (
        "Gần phía Bắc Nhạn Môn quan gần đây xuất hiện 10 vạn quân Liêu, xem ra sắp có chiến sự lớn. "
        "Vì Quốc gia đại sự, Chủ thương hội trong thành đang thu thập các loại vật tư. "
        "Làm con dân Đại Tống nên dốc chút sức lực trước vận nước lâm nguy. "
        "Ta đã đặt 1 Hộ Uyển lông lạc đà ở chỗ Thương nhân Tây Vực, xin giúp ta lấy về! "
        "Có thể đến Biện Kinh, Thành Đô, Tuyền Châu, Tương Dương, Dương Châu, Đại Lý để tìm ông ta."
    )
    assert clf.classify(text=quest_text) == QuestType.SPECIAL_NPC_SHOP
    assert extract_shop_item(quest_text) == "Hộ Uyển Lông Lạc Đà"

    # 4. Match direct aliases
    assert match_western_merchant_item("da nhung thi thao") == "Hộ Uyển Lông Lạc Đà"
    assert match_western_merchant_item("Hj Uyen long lac da") == "Hộ Uyển Lông Lạc Đà"


def test_long_lac_da_without_ho_uyen():
    from src.quest_classifier import parse_yellow_entities, extract_shop_item, match_western_merchant_item

    # 1. Chữ vàng chỉ có 'lông lạc đà' -> phải ra 'Lông Lạc Đà', tuyệt đối không nhầm sang 'Hộ Uyển Lông Lạc Đà'
    entities = parse_yellow_entities(["lông lạc đà"])
    assert entities["item"] == "Lông Lạc Đà"

    entities_norm = parse_yellow_entities(["long lac da"])
    assert entities_norm["item"] == "Lông Lạc Đà"

    # 2. Nhiệm vụ đầy đủ chỉ yêu cầu 'lông lạc đà'
    quest_text = (
        "Gần phía Bắc Nhạn Môn quan gần đây xuất hiện 10 vạn quân Liêu, xem ra sắp có chiến sự lớn. "
        "Vì Quốc gia đại sự, Chủ thương hội trong thành đang thu thập các loại vật tư. "
        "Làm con dân Đại Tống nên dốc chút sức lực trước vận nước lâm nguy. "
        "Ta đã đặt một sợi lông lạc đà ở chỗ Thương nhân Tây Vực, xin giúp ta lấy về! "
        "Có thể đến Biện Kinh, Thành Đô, Tuyền Châu, Tương Dương, Dương Châu, Đại Lý để tìm ông ta."
    )
    assert extract_shop_item(quest_text) == "Lông Lạc Đà"

    # 3. Direct matching
    assert match_western_merchant_item("lông lạc đà") == "Lông Lạc Đà"
    assert match_western_merchant_item("một sợi lông lạc đà") == "Lông Lạc Đà"
    assert match_western_merchant_item("1 sợi lông lạc đà") == "Lông Lạc Đà"

def test_bao_dao_tay_vuc_user_screenshot_quest():
    from src.quest_classifier import (
        QuestClassifier,
        parse_yellow_entities,
        extract_shop_item,
        extract_dialogue_target,
        QuestType,
    )

    clf = QuestClassifier(None, {})
    # Quest text matches user's screenshot exactly, including residual bottom banner
    full_quest = (
        "Gần phía Bắc Nhạn Môn quan gần đây xuất hiện 10 vạn quân Liêu, xem ra sắp có chiến sự lớn. "
        "Vì Quốc gia đại sự, Chủ thương hội trong thành đang thu thập các loại vật tư. "
        "Làm con dân Đại Tống nên dốc chút sức lực trước vận nước lâm nguy. "
        "Ta đã đặt 1 bảo đao ở chỗ Thương nhân Tây Vực, xin giúp ta lấy về! "
        "Có thể đến Biện Kinh, Thành Đô, Tuyền Châu, Tương Dương, Dương Châu, Đại Lý để tìm ông ta. "
        "Cùng với Chủ Tiệm Nữ phục đối thoại, hoàn thành nhiệm vụ"
    )
    yellow_tokens = ["bao dao"]

    # 1. parse_yellow_entities must extract item as 'Bảo Đao Tây Vực', NOT NPC 'Bộ đầu'
    entities = parse_yellow_entities(yellow_tokens)
    assert entities.get("item") == "Bảo Đao Tây Vực"
    assert entities.get("npc") == ""

    # 2. extract_dialogue_target must NOT match 'Bộ đầu'
    city, npc = extract_dialogue_target(full_quest, yellow_tokens=yellow_tokens, with_accents=True)
    assert npc != "Bộ đầu"

    # 3. classify must classify as SPECIAL_NPC_SHOP, NOT NPC
    quest_type = clf.classify(text=full_quest, yellow_tokens=yellow_tokens)
    assert quest_type == QuestType.SPECIAL_NPC_SHOP

    # 4. extract_shop_item extracts 'Bảo Đao Tây Vực'
    item = extract_shop_item(full_quest, yellow_tokens=yellow_tokens)
    assert item == "Bảo Đao Tây Vực"

