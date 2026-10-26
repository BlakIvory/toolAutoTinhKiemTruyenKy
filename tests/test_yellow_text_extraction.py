from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.quest_classifier import (
    extract_yellow_text_boxes,
    parse_yellow_entities,
    extract_combat_target,
    extract_shop_item,
    extract_dialogue_target,
    QuestClassifier,
)
from src.state_machine import QuestType, BotState, StateContext
from src.bot_controller import BotController


def test_yellow_text_extraction_from_real_screenshot():
    # Test on the real screenshot if present
    img_path = Path(r"C:\Users\nguye\.gemini\antigravity-ide\brain\7800f102-5e7f-45b2-9d38-169733f8b3a1\.user_uploaded\media_1789883105222.png")
    if not img_path.exists():
        pytest.skip("Test image not available in this environment")

    import cv2
    img = cv2.imread(str(img_path))
    assert img is not None
    # Crop right panel where quest content is displayed
    yellow_boxes = extract_yellow_text_boxes(img, region=(300, 50, 1000, 550))
    tokens = [b.text for b in yellow_boxes]
    assert len(tokens) >= 3

    entities = parse_yellow_entities(tokens)
    assert "bang tam dong" in entities["map_name"].lower() or "băng tâm động" in entities["map_name"].lower()
    from src.text_recognition import normalize_game_text
    assert normalize_game_text(entities["monster"]) == "tuyet quai"
    assert entities["count"] == 10


def test_parse_yellow_entities_combat():
    tokens = ["Băng Tâm động", "tuyết quái", "10"]
    entities = parse_yellow_entities(tokens)
    assert entities["map_name"] == "Băng Tâm động"
    assert entities["monster"] in ("tuyết quái", "Tuyết quái")
    assert entities["count"] == 10


def test_yellow_noise_does_not_overwrite_recognized_monster():
    tokens = ["Tang 2 Duong Trung dong", "10", "Bat phy Ha Dong", "oi"]
    entities = parse_yellow_entities(tokens)

    from src.text_recognition import normalize_game_text
    assert normalize_game_text(entities["monster"]) == "bat phu ha dong su"


def test_yellow_noise_before_monster_is_ignored():
    entities = parse_yellow_entities(["oi", "Bat phy Ha Dong", "10"])

    from src.text_recognition import normalize_game_text
    assert normalize_game_text(entities["monster"]) == "bat phu ha dong su"


def test_parse_yellow_entities_shop():
    tokens = ["Bảo Đao Tây Vực", "1"]
    entities = parse_yellow_entities(tokens)
    assert entities["item"] == "Bảo Đao Tây Vực"
    assert entities["count"] == 1


def test_parse_yellow_entities_dialogue():
    tokens = ["Khấu Chuẩn", "Biện Kinh"]
    entities = parse_yellow_entities(tokens)
    assert entities["npc"] in ("Khấu Chuẩn", "Khẩu Chuẩn")
    assert entities["city"] == "Biện Kinh"


def test_parse_yellow_entities_chu_tiem_nu_phuc():
    tokens = ["Dương Châu", "Chủ Tiệm Nữ phục"]
    entities = parse_yellow_entities(tokens)
    assert entities["city"] == "Dương Châu"
    assert entities["npc"] == "Chủ Tiệm Nữ phục"
    assert entities["item"] == ""


def test_parse_yellow_entities_chu_tiem_nam_phuc():
    tokens = ["Duong Chau", "Chi Tiem Nam phyc"]
    entities = parse_yellow_entities(tokens)
    assert entities["city"] == "Dương Châu"
    assert entities["npc"] == "Chủ Tiệm Nam phục"
    assert entities["item"] == ""


def test_parse_yellow_entities_dialogue_ocr_alias():
    tokens = ["Duong Chau", "Chu tiem trang phyc nt"]
    entities = parse_yellow_entities(tokens)
    assert entities["city"] == "Dương Châu"
    assert entities["npc"] == "Chủ Tiệm Nữ phục"
    assert entities["item"] == ""


def test_tutorial_hud_yellow_noise_is_not_a_monster():
    entities = parse_yellow_entities(["ry huong dan Tan Thu", "19", "g1"])

    assert entities["monster"] == ""
    assert entities["map_name"] == ""


def test_xa_phu_dialogue_overrides_background_army_wording_and_yellow_noise():
    quest_text = (
        "Gan phia Bac Nhan Mon quan gan day xuat hien 10 van quan Lieu. "
        "Giup Thuong hoi den Tuong Duong phu noi voi Xa phu: hang da den!"
    )
    classifier = QuestClassifier(MagicMock(), MagicMock())

    assert classifier.classify(
        text=quest_text,
        yellow_tokens=["ry huong dan Tan Thu", "19", "g1"],
    ) == QuestType.NPC

    city, npc = extract_dialogue_target(quest_text, with_accents=True)
    assert city == "Tương Dương"
    assert npc == "Xa phu"


def test_extract_dialogue_target_thong_bao_cho():
    quest_text = (
        "Gần phía Bắc Nhạn Môn quan gần đây xuất hiện 10 vạn quân Liêu, xem ra sắp có chiến sự lớn. "
        "Vì Quốc gia đại sự, Chủ thương hội trong thành đang thu thập các loại vật tư. "
        "Làm con dân Đại Tống nên dốc chút sức lực trước vận nước lâm nguy. "
        "Giúp chủ thương hội đến Dương Châu thông báo cho Chủ Tiệm Nữ phục"
    )
    city, npc = extract_dialogue_target(quest_text, with_accents=True)
    assert city == "Dương Châu"
    assert npc == "Chủ Tiệm Nữ phục"


def test_extract_dialogue_target_from_chief_dialog_ocr():
    quest_text = (
        "Biel Chi thuong hoi:Xin giup ta den Duong Chau noi voi Chi tiem trang phyc ni den nhan hang! "
        "am Thanh Do phu va Vu nie Vi o Dong Bac Tuye g dudng chu"
    )
    city, npc = extract_dialogue_target(quest_text, with_accents=True)
    assert city == "Dương Châu"
    assert npc == "Chủ Tiệm Nữ phục"


def test_extract_combat_target_with_user_sample():
    quest_text = (
        "Gần phía Bắc Nhạn Môn quan gần đây xuất hiện 10 vạn quân Liêu, xem ra sắp có chiến sự lớn. "
        "Vì Quốc gia đại sự, Chủ thương hội trong thành đang thu thập các loại vật tư. "
        "Làm con dân Đại Tống nên dốc chút sức lực trước vận nước lâm nguy. "
        "Sau núi Nga My ở Băng Tâm động có rất nhiều tuyết quái, khiến cho đệ tử Nga My cảm thấy bất an, hãy đi diệt 10 con!"
    )
    # Test without yellow tokens (pure text regex)
    map_name, monster, count = extract_combat_target(quest_text)
    assert map_name == "bang tam dong"
    assert monster == "tuyet quai"
    assert count == 10

    # Test with yellow tokens (priority)
    yellow_tokens = ["Băng Tâm động", "tuyết quái", "10"]
    map_name_y, monster_y, count_y = extract_combat_target(quest_text, yellow_tokens)
    assert map_name_y == "Băng Tâm động"
    assert monster_y in ("tuyết quái", "Tuyết quái")
    assert count_y == 10


def test_classifier_recognizes_user_sample_as_combat():
    classifier = QuestClassifier(MagicMock(), MagicMock())
    quest_text = (
        "Sau núi Nga My ở Băng Tâm động có rất nhiều tuyết quái, khiến cho đệ tử Nga My cảm thấy bất an, hãy đi diệt 10 con!"
    )
    q_type = classifier.classify(text=quest_text)
    assert q_type == QuestType.COMBAT


def test_cai_bang_tuong_van_dong_combat_quest():
    quest_text = (
        "Gần phía Bắc Nhạn Môn quan gần đây xuất hiện 10 vạn quân Liêu, xem ra sắp có chiến sự lớn. "
        "Vì Quốc gia đại sự, Chủ thương hội trong thành đang thu thập các loại vật tư. "
        "Làm con dân Đại Tống nên dốc chút sức lực trước vận nước lâm nguy.\n"
        "Cái Bang Tường Vân động có rất nhiều nhện độc, hãy đi diệt 10 con!"
    )

    # 1. Phân tích token chữ vàng: tách địa bàn 'Cái Bang' và map 'Tường Vân động'
    yellow_tokens = ["Cái Bang Tường Vân động", "nhện độc", "10"]
    entities = parse_yellow_entities(yellow_tokens)
    assert entities["map_name"] == "Tường Vân động"
    assert entities["sect"] == "Cái Bang"
    assert entities["monster"] in ("nhện độc", "Nhện độc")
    assert entities["count"] == 10
    assert entities["city"] == ""

    # 2. Trích xuất mục tiêu với chữ vàng
    map_y, monster_y, count_y = extract_combat_target(quest_text, yellow_tokens)
    assert map_y == "Tường Vân động"
    assert monster_y in ("nhện độc", "Nhện độc")
    assert count_y == 10

    # 3. Trích xuất thuần regex (loại bỏ 'trong thành', 'Nhạn Môn quan', tách Cái Bang -> Tường Vân động)
    map_raw, monster_raw, count_raw = extract_combat_target(quest_text)
    assert map_raw == "tuong van dong"
    assert monster_raw == "nhen doc"
    assert count_raw == 10

    # 4. Phân loại quest type
    classifier = QuestClassifier(MagicMock(), MagicMock())
    assert classifier.classify(text=quest_text) == QuestType.COMBAT


def test_combat_patrol_loop_and_return_to_commerce_chief():
    from src.text_recognition import TextBox

    bot = BotController.__new__(BotController)
    bot._app_cfg = MagicMock()
    bot._app_cfg.mode.dry_run = False
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "combat": {
            "auto_button": {"x": 1306, "y": 240},
            "patrol_minimap_points": [{"x": 1440, "y": 150}, {"x": 1530, "y": 190}],
            "target_hud_region": {"x1": 450, "y1": 60, "x2": 1100, "y2": 130},
            "left_tracker_region": {"x1": 50, "y1": 250, "x2": 350, "y2": 500},
        }
    }
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    call_count = [0]
    def mock_read(frame, roi=None):
        call_count[0] += 1
        # On 2nd frame, report completion in tracker
        if call_count[0] >= 3:
            return [TextBox("Tiêu diệt Tuyết quái (10/10)", 0.95, 100, 300)]
        return []

    bot._text = MagicMock()
    bot._text.read.side_effect = mock_read
    bot._input = MagicMock()
    bot._checkpoint = MagicMock()
    bot._return_to_nearest_city_commerce_chief = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._get_current_location_from_hud = MagicMock(return_value=("Băng Tâm động", (100, 100)))

    ctx = StateContext(account_id="ACC_01")
    ctx.current_quest_type = QuestType.COMBAT
    ctx.extra["combat_map"] = "Băng Tâm động"
    ctx.extra["combat_monster"] = "tuyết quái"
    ctx.extra["combat_count"] = 10

    next_state = bot._handle_navigate(ctx)

    # Bot không được chủ động bật Auto; chuột phải vào quái là đủ.
    assert (1306, 240) not in [call.args for call in bot._input.tap.call_args_list]
    # Verify return to commerce chief was called
    bot._return_to_nearest_city_commerce_chief.assert_called_once_with(ctx)
    assert ctx.extra.get("returning_quest") is True
    assert next_state == BotState.RETURN_QUEST


def test_meet_npc_teleports_to_combat_quest_when_unfinished_active():
    from src.text_recognition import TextBox

    bot = BotController.__new__(BotController)
    bot._input = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    bot._check_active_commerce_quest = MagicMock(return_value=(
        "Sau núi Nga My ở Băng Tâm động có rất nhiều tuyết quái, hãy đi diệt 10 con!",
        None
    ))
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox("Hãy đưa ta tới nơi nhiệm vụ", 0.95, 330, 575),
        TextBox("Nhiệm vụ vẫn chưa hoàn thành", 0.90, 700, 575),
    ]

    ctx = StateContext(account_id="ACC_01")
    ctx.extra = {}

    next_state = bot._handle_meet_npc(ctx)

    assert next_state == BotState.NAVIGATE
    assert ctx.current_quest_type == QuestType.COMBAT
    # Always try to turn in first; only after the NPC confirms unfinished may
    # the bot use the teleport option to continue the current quest.
    assert bot._input.tap.call_args_list[0].args == (360, 693)
    assert bot._input.tap.call_args_list[1].args == (330, 575)


def test_extract_combat_target_ocr_0_quirk():
    """Kiểm tra trường hợp chữ 'ở' bị OCR nhận diện thành số '0'."""
    raw_ocr = (
        "Sau nui Nga My 0 Bang Tam dong co rat nhieu tuyet quai, khien cho de ti Nga My cam thay an, hay di diet 10 con!"
    )
    map_name, monster, count = extract_combat_target(raw_ocr)
    assert "bang tam dong" in map_name
    assert monster == "tuyet quai"
    assert count == 10


def test_handle_navigate_missing_combat_fields_abandons_without_opening_panel():
    """Thiếu bất kỳ dữ liệu bắt buộc nào thì hủy, không suy đoán giữa bãi quái."""
    from src.text_recognition import TextBox

    bot = BotController.__new__(BotController)
    bot._app_cfg = MagicMock()
    bot._app_cfg.mode.dry_run = False
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "combat": {
            "auto_button": {"x": 1306, "y": 240},
            "patrol_minimap_points": [{"x": 1440, "y": 150}],
            "target_hud_region": {"x1": 450, "y1": 60, "x2": 1100, "y2": 130},
            "left_tracker_region": {"x1": 50, "y1": 250, "x2": 350, "y2": 500},
        }
    }
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    # Giả lập khi gọi _check_quest_panel thì điền thông tin vào extra
    def mock_check_quest_panel(ctx):
        ctx.extra["combat_monster"] = "tuyết quái"
        ctx.extra["combat_map"] = "Băng Tâm động"
        ctx.extra["combat_count"] = 10
        return "Nhiệm vụ Thương hội", None

    bot._check_quest_panel = MagicMock(side_effect=mock_check_quest_panel)
    bot._text = MagicMock()
    # Báo hoàn thành ngay lập tức
    bot._text.read.return_value = [TextBox("Tiêu diệt Tuyết quái (10/10)", 0.95, 100, 300)]
    bot._input = MagicMock()
    bot._checkpoint = MagicMock()
    bot._return_to_nearest_city_commerce_chief = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._get_current_location_from_hud = MagicMock(return_value=("Băng Tâm động", (100, 100)))

    ctx = StateContext(account_id="ACC_02")
    ctx.current_quest_type = QuestType.COMBAT
    # Cố tình để trống quái và map như trường hợp user gặp phải
    ctx.extra = {}

    next_state = bot._handle_navigate(ctx)

    bot._check_quest_panel.assert_not_called()
    assert ctx.extra["abandon_combat_quest"] is True
    assert ctx.extra["returning_quest"] is True
    assert "tên quái" in ctx.extra["abandon_combat_quest_reason"]
    assert "địa điểm" in ctx.extra["abandon_combat_quest_reason"]
    assert "số lượng" in ctx.extra["abandon_combat_quest_reason"]
    assert next_state == BotState.FIND_COMMERCE_CHIEF
