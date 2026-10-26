import pytest
from src.quest_classifier import QuestClassifier, QuestType, extract_combat_target, parse_sect_and_combat_map
from src.text_recognition import normalize_game_text

def test_extract_combat_target_trung_tri_bat_phu():
    text = "Chủ thương hội: Tầng 2 Dương Trung động ở Hạnh Hoa thôn có nhiều thôn dân tạo phản, hãy đi trừng trị 10 tên Bát Phụ Hà Đông sư. Có cần ta đưa đến nơi làm nhiệm vụ không?"
    map_name, monster, count = extract_combat_target(text)
    assert "tang 2 duong trung dong" in normalize_game_text(map_name) or "duong trung dong" in normalize_game_text(map_name)
    assert "bat phu ha dong su" in monster.lower()
    assert count == 10

def test_classify_trung_tri_combat_quest():
    classifier = QuestClassifier.__new__(QuestClassifier)
    classifier.TEXT_RULES = QuestClassifier.TEXT_RULES
    text = "Chủ thương hội: Tầng 2 Dương Trung động ở Hạnh Hoa thôn có nhiều thôn dân tạo phản, hãy đi trừng trị 10 tên Bát Phụ Hà Đông sư. Có cần ta đưa đến nơi làm nhiệm vụ không?"
    q_type = classifier.classify(frame=None, text=text)
    assert q_type == QuestType.COMBAT

def test_phien_dua_ta_teleport_button_keyword():
    btn_text = "Phiền đưa ta đi một chuyến"
    b_norm = normalize_game_text(btn_text)
    keywords = (
        "dua ta toi noi", "dua ta den", "toi noi nhiem vu", "den noi nhiem vu",
        "di den noi", "dua ta toi", "den noi lam", "lam nhiem vu", "noi lam nhiem vu",
        "phien dua ta", "dua ta di", "di mot chuyen", "phien dua ta di", "dua den noi"
    )
    assert any(kw in b_norm for kw in keywords)

def test_extract_combat_target_van_mong_trach_coc_doc():
    text = "Chủ thương hội: Gần Vân Mộng trạch có rất nhiều cóc độc, hãy đi bắt 10 con. Có cần ta đưa đến nơi làm nhiệm vụ không?"
    map_name, monster, count = extract_combat_target(text)
    assert "van mong trach" in normalize_game_text(map_name)
    assert "coc doc" in normalize_game_text(monster)
    assert count == 10

def test_classify_van_mong_trach_coc_doc():
    classifier = QuestClassifier.__new__(QuestClassifier)
    classifier.TEXT_RULES = QuestClassifier.TEXT_RULES
    text = "Chủ thương hội: Gần Vân Mộng trạch có rất nhiều cóc độc, hãy đi bắt 10 con. Có cần ta đưa đến nơi làm nhiệm vụ không?"
    q_type = classifier.classify(frame=None, text=text)
    assert q_type == QuestType.COMBAT

def test_is_npc_dialog_open_detects_commerce_chief_teleport_prompt():
    import numpy as np
    from unittest.mock import MagicMock
    from src.bot_controller import BotController
    from src.text_recognition import TextBox

    bot = BotController.__new__(BotController)
    bot._vision = MagicMock()
    bot._vision.find_template.return_value = MagicMock(found=False)
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox("Chủ thương hội: Gần Vân Mộng trạch có rất nhiều cóc độc, hãy đi bắt 10 con.", 0.95, 450, 250),
        TextBox("Có cần ta đưa đến nơi làm nhiệm vụ không?", 0.95, 450, 300),
        TextBox("Phiền đưa ta đi một chuyến", 0.95, 300, 380),
        TextBox("Không cần đâu", 0.95, 600, 380),
    ]
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)
    assert bot._is_npc_dialog_open(frame) is True

def test_get_quest_taps_teleport_button_when_present():
    from unittest.mock import MagicMock
    from src.bot_controller import BotController
    from src.state_machine import BotState, StateContext
    from src.text_recognition import TextBox

    bot = BotController.__new__(BotController)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = None
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox("Chủ thương hội: Gần Vân Mộng trạch có rất nhiều cóc độc, hãy đi bắt 10 con.", 0.95, 450, 250),
        TextBox("Có cần ta đưa đến nơi làm nhiệm vụ không?", 0.95, 450, 300),
        TextBox("Phiền đưa ta đi một chuyến", 0.95, 300, 380),
        TextBox("Không cần đâu", 0.95, 600, 380),
    ]
    classifier = QuestClassifier.__new__(QuestClassifier)
    classifier.TEXT_RULES = QuestClassifier.TEXT_RULES
    bot._classifier = classifier
    bot._checkpoint = MagicMock()
    bot._input = MagicMock()

    ctx = StateContext(account_id="ACC_03")
    next_state = bot._handle_get_quest(ctx)

    assert next_state == BotState.NAVIGATE
    assert ctx.current_quest_type == QuestType.COMBAT
    assert "van mong trach" in normalize_game_text(ctx.extra.get("combat_map", ""))
    assert "coc doc" in normalize_game_text(ctx.extra.get("combat_monster", ""))
    bot._input.tap.assert_called_once_with(300, 380)


def test_extract_combat_target_luong_son_bac_son_tac():
    """Trích xuất địa điểm Lương Sơn Bạc và quái Sơn tặc khi địa điểm đứng trước 'có rất nhiều'."""
    text = (
        "Chu thuong hoi:Luong Son Bac co rat nhieu Son tac lam phan, hay di truing tri 10 ten. "
        "Co can ta dua den noi lam nhiem vy khong? Phien duata dimotchuyen Khongcandau"
    )
    map_name, monster, count = extract_combat_target(text)
    assert "luong son bac" in normalize_game_text(map_name)
    assert "son tac" in normalize_game_text(monster)
    assert count == 10


def test_get_quest_teleports_when_text_has_spaceless_phien_duata():
    """Nhận diện đúng nút teleport khi OCR đọc dính chữ 'Phien duata dimotchuyen'."""
    from unittest.mock import MagicMock
    from src.bot_controller import BotController
    from src.state_machine import BotState, StateContext
    from src.text_recognition import TextBox

    bot = BotController.__new__(BotController)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = None
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox("Chu thuong hoi:Luong Son Bac co rat nhieu Son tac lam phan, hay di truing tri 10 ten.", 0.95, 450, 250),
        TextBox("Co can ta dua den noi lam nhiem vy khong?", 0.95, 450, 300),
        TextBox("Phien duata dimotchuyen", 0.95, 300, 380),
        TextBox("Khongcandau", 0.95, 600, 380),
    ]
    classifier = QuestClassifier.__new__(QuestClassifier)
    classifier.TEXT_RULES = QuestClassifier.TEXT_RULES
    classifier._vision = MagicMock()
    bot._classifier = classifier
    bot._checkpoint = MagicMock()
    bot._input = MagicMock()

    ctx = StateContext(account_id="ACC_03")
    next_state = bot._handle_get_quest(ctx)

    assert next_state == BotState.NAVIGATE
    assert ctx.current_quest_type == QuestType.COMBAT
    assert "luong son bac" in normalize_game_text(ctx.extra.get("combat_map", ""))
    assert "son tac" in normalize_game_text(ctx.extra.get("combat_monster", ""))
    assert ctx.extra.get("combat_count") == 10
    bot._input.tap.assert_called_once_with(300, 380)



