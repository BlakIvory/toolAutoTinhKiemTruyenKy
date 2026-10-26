from unittest.mock import Mock

from src.quest_classifier import QuestClassifier, extract_dialogue_target, extract_shop_item
from src.state_machine import QuestType


def classifier():
    vision = Mock()
    vision._screen.capture.return_value = None
    vision._templates.has.return_value = False
    return QuestClassifier(vision, Mock())


def test_world_hud_noise_is_not_classified_as_combat_quest():
    from src.quest_classifier import parse_yellow_entities

    yellow_tokens = [
        "y huong dan Tan Thu",
        "yq dong gia re day",
        "g duong chu",
        "Tong hop",
    ]

    entities = parse_yellow_entities(yellow_tokens)
    assert entities["monster"] == ""
    assert classifier().classify(
        text="Pham Trong Y Chu thuong hoi Cai Bang duong chu Tong hop Mat He thong",
        yellow_tokens=yellow_tokens,
    ) == QuestType.UNKNOWN


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
    assert extract_shop_item(text) == "Trứng Đại Bàng"


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


def test_dialogue_quest_with_two_cities_is_not_misclassified_as_shop():
    text = "Hay den Bien Kinh gap Ly Phong Thu, sau do tro ve Tuyen Chau bao tin"
    assert classifier().classify(text=text) == QuestType.NPC


def test_yellow_npc_entity_classifies_dialogue_quest():
    assert classifier().classify(yellow_tokens=["Tuyen Chau", "Ly Phong Thu"]) == QuestType.NPC


def test_extracts_quan_vien_thoai_nhiem_from_yellow_tokens_and_full_text():
    from src.quest_classifier import parse_yellow_entities

    tokens = ["Thanh Do phi", "Quan vien thoai nhiem", "!"]
    entities = parse_yellow_entities(tokens)
    assert entities["city"] == "Thành Đô"
    assert entities["npc"] == "Quan viên thoái nhiệm"

    text = (
        "Giup Thuong hoi den Thanh Do phu noi voi Quan vien thoai nhiem: "
        "hang da den!"
    )
    assert extract_dialogue_target(
        text, with_accents=True, yellow_tokens=tokens
    ) == ("Thành Đô", "Quan viên thoái nhiệm")


def test_dialogue_target_falls_back_to_full_text_when_yellow_has_only_city():
    text = (
        "Giup Thuong hoi den Thanh Do phu noi voi Quan vien thoai nhiem: "
        "hang da den!"
    )
    assert extract_dialogue_target(
        text, with_accents=True, yellow_tokens=["Thanh Do phi"]
    ) == ("Thành Đô", "Quan viên thoái nhiệm")


def test_unknown_yellow_npc_is_not_guessed_as_shop_item():
    from src.quest_classifier import parse_yellow_entities

    entities = parse_yellow_entities(["Nguyen Van A", "Bien Kinh"])
    assert entities["item"] == ""


def test_combat_description_with_monster_before_count_keeps_monster():
    from src.quest_classifier import extract_combat_target

    result = extract_combat_target("Co rat nhieu soi hoang gay hai, hay di tieu diet 10 con.")
    assert result == ("", "Sói hoang", 10)


def test_combat_count_is_unknown_instead_of_inventing_twenty():
    from src.quest_classifier import extract_combat_target

    assert extract_combat_target("Hay di tieu diet soi hoang") == ("", "Sói hoang", 0)


def test_extract_dialogue_target_with_proper_accents():
    from src.quest_classifier import extract_dialogue_target, get_canonical_npc_name

    assert get_canonical_npc_name("bach hieu sinh") == "Bách Hiểu Sinh"
    assert get_canonical_npc_name("nan dan dinh") == "Nạn dân Đinh"
    assert get_canonical_npc_name("nan dan giap") == "Nạn dân Giáp"
    assert get_canonical_npc_name("nan dan at") == "Nạn dân Ất"
    assert get_canonical_npc_name("nan dan binh") == "Nạn dân Bính"
    assert get_canonical_npc_name("van du dao nhan") == "Vân Du đạo nhân"
    assert get_canonical_npc_name("Tiêu sư") == "Tiêu sư"
    assert get_canonical_npc_name("Tiêu sứ") == "Tiêu sư"

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

    c_tieu_su, n_tieu_su = extract_dialogue_target(
        "Giúp Thương hội đến Biện Kinh nói với Tiêu sư: hàng đã đến!",
        with_accents=True,
    )
    assert c_tieu_su == "Biện Kinh"
    assert n_tieu_su == "Tiêu sư"

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
    assert extract_shop_item(quest_text) == "Linh Dược Tây Vực"


def test_clean_shop_item_name():
    from src.quest_classifier import clean_shop_item_name
    assert clean_shop_item_name("Kim Cang Nguyen Thach y Thuong hoi") == "Kim Cang Nguyen Thach"
    assert clean_shop_item_name("1 Kim Cang Nguyen Thach y Thuong hoi") == "Kim Cang Nguyen Thach"
    assert clean_shop_item_name("1 viên Kim Cang Nguyên Thạch ý Thương hội") == "Kim Cang Nguyên Thạch"
    assert clean_shop_item_name("Linh Dược Tây Vực") == "Linh Dược Tây Vực"


def test_extract_kim_cang_nguyen_thach_from_user_sample():
    text = (
        "Ta da dat lam 1 Kim Cang Nguyen Thach 0 cho Thuong nhan Tay Vuc, xin giup ta "
        "g duong chi en Bien Kinh, Thanh Do, Tuyen Chau, Tuong Duong, Duong Chau, Dai Ly de tim"
    )
    assert classifier().classify(text=text) == QuestType.SPECIAL_NPC_SHOP
    assert extract_shop_item(text) == "Kim Cang Nguyên Thạch"


def test_extract_long_dai_bang_and_parse_yellow_entities():
    from src.quest_classifier import parse_yellow_entities

    # Case user provided in screenshot:
    # "Ta đã đặt 1 lông đại bàng ở chỗ Thương nhân Tây Vực, xin giúp ta lấy về! Có thể đến Biện Kinh..."
    # Yellow text highlighted is "lông đại bàng"
    res = parse_yellow_entities(["lông đại bàng"])
    assert res["item"] == "Lông Đại Bàng"
    assert res["monster"] == ""

    # Test that full quest text correctly extracts item as well
    quest_text = (
        "Gần phía Bắc Nhạn Môn quan gần đây xuất hiện 10 vạn quân Liêu, xem ra sắp có chiến sự lớn. "
        "Vì Quốc gia đại sự, Chủ thương hội trong thành đang thu thập các loại vật tư. "
        "Làm con dân Đại Tống nên dốc chút sức lực trước vận nước lâm nguy. "
        "Ta đã đặt 1 lông đại bàng ở chỗ Thương nhân Tây Vực, xin giúp ta lấy về! "
        "Có thể đến Biện Kinh, Thành Đô, Tuyền Châu, Tương Dương, Dương Châu, Đại Lý để tìm ông ta."
    )
    assert extract_shop_item(quest_text) in ("long dai bang", "Lông Đại Bàng")


def test_is_matching_shop_item_discrimination():
    from src.bot_controller import BotController

    bot = BotController.__new__(BotController)

    # 1. Matching 'lông đại bàng'
    is_match, score = bot._is_matching_shop_item("Lông đại bàng", "lông đại bàng")
    assert is_match is True
    assert score >= 0.80

    # 2. 'Trứng đại bàng' vs 'lông đại bàng' MUST NOT match
    is_match, score = bot._is_matching_shop_item("Trứng đại bàng", "lông đại bàng")
    assert is_match is False
    assert score < 0.80

    # 3. 'Tring Dai Bang' (OCR typo for trứng) vs 'lông đại bàng' MUST NOT match
    is_match, score = bot._is_matching_shop_item("Tring Đại Bàng", "lông đại bàng")
    assert is_match is False

    # 4. 'Tinh kim Tây Vực' vs 'Linh dược Tây Vực' MUST NOT match
    is_match, score = bot._is_matching_shop_item("Tinh kim Tây Vực", "Linh dược Tây Vực")
    assert is_match is False
    assert score < 0.80

    # 5. Banner noise like 'anhhuongxau' MUST NOT match 'long dai bang' or any shop item
    is_match, score = bot._is_matching_shop_item("anhhuongxau", "lông đại bàng")
    assert is_match is False
    assert score < 0.50


def test_extract_ngu_doc_me_cung_and_thuc_nhan_thao():
    from src.quest_classifier import parse_yellow_entities, parse_sect_and_combat_map, extract_combat_target

    tokens = ["Ngi Doc giao me cung tang 2", "Thyc nhan thao", "10 cay", "g1"]
    entities = parse_yellow_entities(tokens)
    assert entities["map_name"] == "Ngũ Độc Mê Cung tầng 2"
    assert entities["sect"] == "Ngũ Độc"
    assert entities["monster"] in ("Thực nhân thảo", "Thyc nhan thao")
    assert entities["count"] == 10
    assert entities["item"] == ""

    # Test parse_sect_and_combat_map directly
    sect, c_map = parse_sect_and_combat_map("Ngi Doc giao me cung tang 2", with_accents=True)
    assert sect == "Ngũ Độc"
    assert c_map == "Ngũ Độc Mê Cung tầng 2"

    # Test extract_combat_target with text
    quest_text = "Chu thuong hoi: Ngi Doc giao me cung tang 2 co loai Thyc nhan thao rat nguy hiem, hay di diet tri 10 cay."
    map_name, monster, count = extract_combat_target(quest_text)
    assert "me cung" in map_name
    assert "2" in map_name
    assert monster in ("thuc nhan thao", "thyc nhan thao", "Thực nhân thảo")
    assert count == 10

    # Test extract_combat_target with yellow tokens
    map_name, monster, count = extract_combat_target(quest_text, yellow_tokens=tokens)
    assert map_name == "Ngũ Độc Mê Cung tầng 2"
    assert monster in ("Thực nhân thảo", "Thyc nhan thao")
    assert count == 10


def test_hud_map_recognition_duong_trung_dong_and_towns():
    from src.bot_controller import BotController
    from src.text_recognition import TextBox

    bot = BotController.__new__(BotController)
    mock_text = Mock()
    bot._text = mock_text

    # Giả lập các box OCR khi ở Dương Trung động (kênh 1, tọa độ 167/186)
    # Font game thường bị OCR đọc nhầm 'Duong' thành 'Dugng'
    mock_boxes = [
        TextBox(text="Dugng", confidence=0.85, center_x=1476, center_y=12),
        TextBox(text="Trung", confidence=0.99, center_x=1531, center_y=13),
        TextBox(text="dong-1", confidence=0.64, center_x=1578, center_y=13),
        TextBox(text="167/186", confidence=1.00, center_x=1538, center_y=57),
    ]
    mock_text.read.return_value = mock_boxes

    import numpy as np
    dummy_frame = np.zeros((900, 1600, 3), dtype=np.uint8)

    city, coords = bot._get_current_location_from_hud(frame=dummy_frame)
    assert city == "Dương Trung động"
    assert coords == (167, 186)

    # Giả lập khi ở thành thị lớn (Biện Kinh-1)
    mock_text.read.return_value = [
        TextBox(text="Bien", confidence=0.95, center_x=1480, center_y=12),
        TextBox(text="Kinh-1", confidence=0.95, center_x=1540, center_y=12),
        TextBox(text="200/200", confidence=1.00, center_x=1538, center_y=57),
    ]
    city, coords = bot._get_current_location_from_hud(frame=dummy_frame)
    assert city == "Biện Kinh"
    assert coords == (200, 200)

    # Giả lập khi ở thôn trấn (Ba Lăng huyện-1)
    mock_text.read.return_value = [
        TextBox(text="Ba Lang Huyen-1", confidence=0.90, center_x=1500, center_y=12),
        TextBox(text="190/210", confidence=1.00, center_x=1538, center_y=57),
    ]
    city, coords = bot._get_current_location_from_hud(frame=dummy_frame)
    assert city == "Ba Lăng huyện"
    assert coords == (190, 210)


def test_extract_bat_phu_ha_dong_su_combat_quest():
    from src.quest_classifier import parse_yellow_entities, extract_combat_target

    tokens = ["TieuLongNu", "Tang 2 Duong Trung dong", "10", "Bat phy Ha Dor"]
    entities = parse_yellow_entities(tokens)
    assert entities["map_name"] == "Tầng 2 Dương Trung động"
    assert entities["count"] == 10
    assert entities["monster"] == "Bát phụ Hà Đông Sư"
    assert entities["item"] == ""

    quest_text = (
        "Gần phía Bắc Nhạn Môn quan gần đây xuất hiện 10 vạn quân Liêu, xem ra sắp có chiến sự lớn. "
        "Vì Quốc gia đại sự, Chủ thương hội trong thành đang thu thập các loại vật tư. "
        "Làm con dân Đại Tống nên dốc chút sức lực trước vận nước lâm nguy. "
        "Tầng 2 Dương Trung động có rất nhiều kẻ tụ tập tạo phản, hãy đi diệt 10 tên Bát phụ Hà Đông Sư"
    )
    c_map, c_monster, c_count = extract_combat_target(quest_text, yellow_tokens=tokens)
    assert "Dương Trung động" in c_map
    assert c_monster == "Bát phụ Hà Đông Sư"
    assert c_count == 10

    # Test without yellow tokens (only description text)
    c_map2, c_monster2, c_count2 = extract_combat_target(quest_text)
    from src.text_recognition import normalize_game_text
    assert "duong trung dong" in normalize_game_text(c_map2)
    assert "bat phu ha dong su" in normalize_game_text(c_monster2)
    assert c_count2 == 10


def test_extract_van_mong_trach_coc_doc_combat_quest():
    from src.quest_classifier import parse_yellow_entities, extract_combat_target
    from src.text_recognition import normalize_game_text

    # 1. Test trích xuất từ chữ màu vàng (bao gồm cả trường hợp OCR đọc 'Goc dpc')
    tokens_ocr = ["Van Mong trach", "Goc dpc", "10 con"]
    entities_ocr = parse_yellow_entities(tokens_ocr)
    assert entities_ocr["map_name"] == "Vân Mộng trạch"
    assert entities_ocr["count"] == 10
    assert entities_ocr["monster"] == "Cóc độc"
    assert entities_ocr["item"] == ""

    # 2. Test trích xuất chữ chuẩn "Coc doc"
    tokens_std = ["Van Mong trach", "Coc doc", "10 con"]
    entities_std = parse_yellow_entities(tokens_std)
    assert entities_std["map_name"] == "Vân Mộng trạch"
    assert entities_std["count"] == 10
    assert entities_std["monster"] == "Cóc độc"
    assert entities_std["item"] == ""

    # 3. Test với nội dung nhiệm vụ đầy đủ
    quest_text = (
        "Gần phía Bắc Nhạn Môn quan gần đây xuất hiện 10 vạn quân Liêu, xem ra sắp có chiến sự lớn. "
        "Vì Quốc gia đại sự, Chủ thương hội trong thành đang thu thập các loại vật tư. "
        "Làm con dân Đại Tống nên dốc chút sức lực trước vận nước lâm nguy. "
        "Xung quanh Vân Mộng trạch có nhiều Cóc độc gây hại, hãy đi diệt 10 con ."
    )
    # Phân loại quest type
    assert classifier().classify(text=quest_text) == QuestType.COMBAT

    # Trích xuất mục tiêu đánh quái có tokens chữ vàng
    c_map, c_monster, c_count = extract_combat_target(quest_text, yellow_tokens=tokens_ocr)
    assert "Vân Mộng trạch" in c_map
    assert c_monster == "Cóc độc"
    assert c_count == 10

    # Trích xuất mục tiêu đánh quái chỉ từ text mô tả
    c_map2, c_monster2, c_count2 = extract_combat_target(quest_text)
    assert "van mong trach" in normalize_game_text(c_map2)
    assert "coc doc" in normalize_game_text(c_monster2)
    assert c_count2 == 10


def test_extract_van_mong_trach_ca_sau_keeps_canonical_accents():
    from src.quest_classifier import parse_yellow_entities, extract_combat_target

    tokens = ["Ba Lang huyen", "Van Mong trach", "Ca sau", "10 con"]
    entities = parse_yellow_entities(tokens)
    assert entities["map_name"] == "Vân Mộng trạch"
    assert entities["monster"] == "Cá sấu"
    assert entities["count"] == 10

    quest_text = (
        "Xung quanh Vân Mộng trạch có nhiều Cá sấu gây hại, "
        "hãy đi diệt 10 con."
    )
    c_map, c_monster, c_count = extract_combat_target(
        quest_text, yellow_tokens=tokens
    )
    assert c_map == "Vân Mộng trạch"
    assert c_monster == "Cá sấu"
    assert c_count == 10


def test_cung_voi_chu_tuu_lau_classified_as_npc_not_combat():
    quest_text = (
        "Gần phía Bắc Nhạn Môn quan gần đây xuất hiện 10 vạn quân Liêu, xem ra sắp có chiến sự lớn. "
        "Vì Quốc gia đại sự, Chủ thương hội trong thành đang thu thập các loại vật tư. "
        "Làm con dân Đại Tống nên dốc chút sức lực trước vận nước lâm nguy. "
        "tầng 1 Dương Trung động có nhiều Quan sứ đả thương người dân vô tội, hãy đi diệt 10 tên. "
        "Cùng với Chủ tửu lầu đối thoại, hoàn thành nhiệm vụ"
    )
    yellow_tokens = ["Quan sứ", "10"]
    # 1. Phân loại phải là NPC vì mục tiêu khả thi duy nhất là "Cùng với Chủ tửu lầu đối thoại"
    assert classifier().classify(text=quest_text, yellow_tokens=yellow_tokens) == QuestType.NPC

    # 2. Trích xuất NPC phải ra "Chủ tửu lầu" và city không bị gán chuỗi rác
    city, npc = extract_dialogue_target(quest_text, with_accents=True)
    assert npc == "Chủ tửu lầu"
    assert city == ""  # Thành phố sẽ lấy từ HUD (thành phố hiện tại)


def test_chu_ban_cu_dialogue_target_and_classification():
    quest_text = (
        "Gần phía Bắc Nhạn Môn quan gần đây xuất hiện 10 vạn quân Liêu, xem ra sắp có chiến sự lớn. "
        "Vì Quốc gia đại sự, Chủ thương hội trong thành đang thu thập các loại vật tư. "
        "Làm con dân Đại Tống nên dốc chút sức lực trước vận nước lâm nguy. "
        "Giúp chủ thương hội đến Tuyền Châu thông báo cho Chu Bân Cử"
    )
    yellow_tokens = ["Tuyền Châu", "Chu Bân Cử"]
    assert classifier().classify(text=quest_text, yellow_tokens=yellow_tokens) == QuestType.NPC

    city, npc = extract_dialogue_target(quest_text, with_accents=True)
    assert city == "Tuyền Châu"
    assert npc == "Chu Bân Cử"

    from src.quest_classifier import get_canonical_npc_name
    assert get_canonical_npc_name("Chu Bân Cử") == "Chu Bân Cử"
    assert get_canonical_npc_name("chu ban cu") == "Chu Bân Cử"
    assert get_canonical_npc_name("Chủ Bàn Cử") == "Chu Bân Cử"


def test_nha_dich_dialogue_target_and_classification():
    quest_text = (
        "Gần phía Bắc Nhạn Môn quan gần đây xuất hiện 10 vạn quân Liêu, xem ra sắp có chiến sự lớn. "
        "Vì Quốc gia đại sự, Chủ thương hội trong thành đang thu thập các loại vật tư. "
        "Làm con dân Đại Tống nên dốc chút sức lực trước vận nước lâm nguy. "
        "Giúp chủ thương hội đến Tuyền Châu thông báo cho nha dịch\n"
        "Cùng với Nha dịch đối thoại, hoàn thành nhiệm vụ"
    )
    yellow_tokens = ["Tuyền Châu", "nha dịch"]
    assert classifier().classify(text=quest_text, yellow_tokens=yellow_tokens) == QuestType.NPC

    from src.quest_classifier import parse_yellow_entities
    entities = parse_yellow_entities(yellow_tokens)
    assert entities["city"] == "Tuyền Châu"
    assert entities["npc"] == "Nha dịch"

    city, npc = extract_dialogue_target(quest_text, with_accents=True, yellow_tokens=yellow_tokens)
    assert city == "Tuyền Châu"
    assert npc == "Nha dịch"


def test_nan_dan_dialogue_target_and_discrimination():
    quest_text = (
        "Gần phía Bắc Nhạn Môn quan gần đây xuất hiện 10 vạn quân Liêu, xem ra sắp có chiến sự lớn. "
        "Vì Quốc gia đại sự, Chủ thương hội trong thành đang thu thập các loại vật tư. "
        "Làm con dân Đại Tống nên dốc chút sức lực trước vận nước lâm nguy. "
        "Giúp Thương hội đến Thành Đô phủ nói với Nạn dân: hàng đã đến!"
    )
    yellow_tokens = ["Thành Đô phủ", "Nạn dân"]
    assert classifier().classify(text=quest_text, yellow_tokens=yellow_tokens) == QuestType.NPC

    from src.quest_classifier import parse_yellow_entities, get_canonical_npc_name
    entities = parse_yellow_entities(yellow_tokens)
    assert entities["city"] == "Thành Đô"
    assert entities["npc"] == "Nạn dân"

    # Must extract exact "Nạn dân", NEVER "Nạn dân Giáp"
    city, npc = extract_dialogue_target(quest_text, with_accents=True, yellow_tokens=yellow_tokens)
    assert city == "Thành Đô"
    assert npc == "Nạn dân"

    city_no_yellow, npc_no_yellow = extract_dialogue_target(quest_text, with_accents=True)
    assert city_no_yellow == "Thành Đô"
    assert npc_no_yellow == "Nạn dân"

    # Canonical naming tests
    assert get_canonical_npc_name("Nạn dân") == "Nạn dân"
    assert get_canonical_npc_name("nan dan") == "Nạn dân"
    assert get_canonical_npc_name("nan dan giap") == "Nạn dân Giáp"
    assert get_canonical_npc_name("Nạn dân Giáp") == "Nạn dân Giáp"
    assert get_canonical_npc_name("nan dan dinh") == "Nạn dân Đinh"


def test_chu_tiem_vu_khi_dialogue_target_and_discrimination():
    quest_text = (
        "Gần phía Bắc Nhạn Môn quan gần đây xuất hiện 10 vạn quân Liêu, xem ra sắp có chiến sự lớn. "
        "Vì Quốc gia đại sự, Chủ thương hội trong thành đang thu thập các loại vật tư. "
        "Làm con dân Đại Tống nên dốc chút sức lực trước vận nước lâm nguy. "
        "Giúp Thương hội đến Thành Đô phủ nói với Chủ tiệm vũ khí: hàng đã đến!"
    )
    yellow_tokens = ["Thành Đô phủ", "Chủ tiệm vũ khí"]
    assert classifier().classify(text=quest_text, yellow_tokens=yellow_tokens) == QuestType.NPC

    from src.quest_classifier import parse_yellow_entities, get_canonical_npc_name
    entities = parse_yellow_entities(yellow_tokens)
    assert entities["city"] == "Thành Đô"
    assert entities["npc"] == "Chủ Tiệm Vũ khí"

    # Must extract exact "Chủ Tiệm Vũ khí", NEVER "Chủ Tiệm Nữ phục"
    city, npc = extract_dialogue_target(quest_text, with_accents=True, yellow_tokens=yellow_tokens)
    assert city == "Thành Đô"
    assert npc == "Chủ Tiệm Vũ khí"

    city_no_yellow, npc_no_yellow = extract_dialogue_target(quest_text, with_accents=True)
    assert city_no_yellow == "Thành Đô"
    assert npc_no_yellow == "Chủ Tiệm Vũ khí"

    # Canonical naming tests
    assert get_canonical_npc_name("Chủ tiệm vũ khí") == "Chủ Tiệm Vũ khí"
    assert get_canonical_npc_name("chu tiem vu khi") == "Chủ Tiệm Vũ khí"
    assert get_canonical_npc_name("tiem vu khi") == "Chủ Tiệm Vũ khí"
    assert get_canonical_npc_name("Tiệm Vũ khí") == "Chủ Tiệm Vũ khí"


def test_coc_doc_quest_classification_not_cam_quan():
    """Nhiệm vụ Cóc độc tại Vân Mộng trạch phải là COMBAT, không được nhầm 'Clan dan' thành 'Cấm quân'."""
    quest_text = (
        "Gần phía Bắc Nhạn Môn quan gần đây xuất hiện 10 vạn quân Liêu, xem ra sắp có chiến sự lớn. "
        "Vì Quốc gia đại sự, Chủ thương hội trong thành đang thu thập các loại vật tư. "
        "Làm con dân Đại Tống nên dốc chút sức lực trước vận nước lâm nguy. "
        "Xung quanh Vân Mộng trạch có nhiều Cóc độc gây hại, hãy đi diệt 10 con."
    )
    yellow_tokens = ["Van Mong trach", "Coc doc", "10 con"]
    assert classifier().classify(text=quest_text, yellow_tokens=yellow_tokens) == QuestType.COMBAT

    from src.quest_classifier import parse_yellow_entities
    entities = parse_yellow_entities(yellow_tokens)
    assert entities.get("monster") == "Cóc độc"
    assert entities.get("count") == 10
    assert entities.get("map_name") == "Vân Mộng trạch"
    assert entities.get("npc") == ""

    # Test với các token vàng đọc từ OCR có dính 'Clan dan' ('con dân')
    get_quest_tokens = ["Ngudi Phat Thud", "Tan Thu", "Clan dan", "Van Mong trach", "coc doc", "10 con"]
    entities_gq = parse_yellow_entities(get_quest_tokens)
    assert entities_gq.get("npc") != "Cấm quân"
    assert entities_gq.get("monster") == "Cóc độc"
    assert entities_gq.get("count") == 10





