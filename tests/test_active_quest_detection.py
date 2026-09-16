from unittest.mock import MagicMock, call
from src.text_recognition import TextBox
from src.bot_controller import BotController
from src.state_machine import StateContext, BotState, QuestType


def test_method_1_left_tracker_direct():
    """Cách 1: Quét trực tiếp thanh nhiệm vụ bên trái thấy ngay Thương hội."""
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._screen = MagicMock()
    bot._input = MagicMock()

    bot._screen.capture.return_value = "dummy_frame"
    bot._text.read.return_value = [
        TextBox(text="Nhiệm vụ Thương hội", confidence=0.9, center_x=178, center_y=264),
        TextBox(text="Giúp Thương hội đến Biện Kinh nói với Vân Du đạo nhân: hàng đã đến!", confidence=0.9, center_x=199, center_y=314),
    ]

    result = bot._check_left_quest_tracker()
    assert result is not None
    quest_text, click_pos = result
    assert "Bien Kinh" in quest_text or "Biện Kinh" in quest_text
    assert click_pos == (178, 264)
    bot._input.swipe.assert_not_called()


def test_method_1_left_tracker_after_scroll():
    """Cách 1: Đầu danh sách chỉ có Dã Tẩu / Tân Thủ, sau khi cuộn tới cuối mới thấy Thương hội."""
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._screen = MagicMock()
    bot._input = MagicMock()

    bot._screen.capture.return_value = "dummy_frame"
    bot._text.read.side_effect = [
        [
            TextBox(text="Nhiệm vụ Dã Tẩu", confidence=0.9, center_x=180, center_y=250),
            TextBox(text="Đến gặp Dã Tẩu để nhận thưởng", confidence=0.9, center_x=180, center_y=290),
        ],
        [
            TextBox(text="Nhiệm vụ Thương hội", confidence=0.9, center_x=180, center_y=320),
            TextBox(text="Tìm Thương nhân Tây Vực mua Trân Châu", confidence=0.9, center_x=180, center_y=360),
        ]
    ]

    result = bot._check_left_quest_tracker()
    assert result is not None
    quest_text, click_pos = result
    assert "Tran Chau" in quest_text or "Trân Châu" in quest_text
    assert click_pos == (180, 320)
    bot._input.swipe.assert_called_with(200, 480, 200, 240, duration=250)


def test_method_2_quest_panel():
    """Cách 2: Thanh nhiệm vụ bên trái không có, bấm icon góc phải dưới để mở bảng nhiệm vụ."""
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._screen = MagicMock()
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._find_blue_npc_link = MagicMock(return_value=None)
    bot._loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "quest_button": {"x": 1404, "y": 755},
            "quest_panel_close": {"x": 1495, "y": 95},
        }
    }

    bot._screen.capture.return_value = "dummy_frame"
    bot._text.read.side_effect = [
        [TextBox(text="Nhiệm vụ Thương hội", confidence=0.9, center_x=250, center_y=300)],  # quest panel list
        [TextBox(text="Mua 1 Bình Định Quốc", confidence=0.9, center_x=600, center_y=300)],  # quest panel detail
    ]

    result = bot._check_active_commerce_quest(full_check=True)
    assert result is not None
    quest_text, click_pos = result
    assert "Bình Định Quốc" in quest_text
    bot._input.tap.assert_any_call(1404, 755)
    bot._input.tap.assert_any_call(1495, 95)


def test_method_3_chief_dialogue_unfinished():
    """Cách 3: Bấm Hoàn thành nhiệm vụ nhưng Chủ thương hội thông báo chưa hoàn thành -> Đi làm NV."""
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._screen = MagicMock()
    bot._input = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._find_dialog_buttons = MagicMock(return_value=[(400, 600)])
    bot._vision = MagicMock()
    bot._checkpoint = MagicMock()

    ctx = StateContext(account_id="ACC_01")
    ctx.total_quests_done = 5
    ctx.extra = {"returning_quest": True}

    bot._text.read.side_effect = [
        [TextBox(text="Hoàn thành nhiệm vụ", confidence=0.9, center_x=400, center_y=600)],
        [TextBox(text="Ngươi chưa hoàn thành nhiệm vụ Thương hội, hãy mau đi làm!", confidence=0.9, center_x=500, center_y=400)],
    ]
    bot._check_active_commerce_quest = MagicMock(return_value=(
        "Giúp Thương hội đến Biện Kinh gặp Vân Du đạo nhân",
        (180, 310)
    ))

    next_state = bot._handle_meet_npc(ctx)

    assert next_state == BotState.CLASSIFY_QUEST
    assert ctx.total_quests_done == 5
    assert "returning_quest" not in ctx.extra
    assert "Vân Du" in ctx.extra["quest_text"]
    bot._press_buff_speed.assert_called_once()


def test_find_commerce_chief_diverts_to_classify_when_active_quest_exists():
    bot = BotController.__new__(BotController)
    bot._input = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._check_active_commerce_quest = MagicMock(return_value=(
        "Giúp Thương hội đến Biện Kinh nói với Vân Du đạo nhân: hàng đã đến!",
        None
    ))

    ctx = StateContext(account_id="ACC_01")
    ctx.extra = {}

    next_state = bot._handle_find_commerce_chief(ctx)
    assert next_state == BotState.CLASSIFY_QUEST
    assert ctx.extra["quest_text"] == "Giúp Thương hội đến Biện Kinh nói với Vân Du đạo nhân: hàng đã đến!"
    bot._press_buff_speed.assert_called_once()


def test_find_commerce_chief_interacts_immediately_when_already_near_npc():
    """Nếu chưa có nhiệm vụ nhưng nhân vật đã đứng cạnh Chủ thương hội -> lập tức đối thoại nhận nhiệm vụ (MEET_NPC)."""
    bot = BotController.__new__(BotController)
    bot._input = MagicMock()
    bot._ensure_map_open = MagicMock()
    bot._check_active_commerce_quest = MagicMock(return_value=None)
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._find_dialog_buttons = MagicMock(return_value=[])
    bot._is_npc_dialog_open = MagicMock(side_effect=[False, True])

    ctx = StateContext(account_id="ACC_01")
    ctx.extra = {}

    next_state = bot._handle_find_commerce_chief(ctx)
    assert next_state == BotState.MEET_NPC
    # Tap NPC tại (780, 380) để mở đối thoại
    bot._input.tap.assert_called_with(780, 380)
    # Tuyệt đối không mở bản đồ lại
    bot._ensure_map_open.assert_not_called()


def test_get_quest_returns_meet_npc_when_no_valid_quest_received():
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._screen = MagicMock()
    bot._classifier = MagicMock()
    bot._classifier.classify.return_value = QuestType.UNKNOWN
    bot._check_active_commerce_quest = MagicMock(return_value=None)

    # OCR reads random noise (player names)
    bot._screen.capture.return_value = "dummy"
    bot._text.read.return_value = [
        TextBox(text="Truong Dai Hiu", confidence=0.8, center_x=500, center_y=600)
    ]

    ctx = StateContext(account_id="ACC_01")
    ctx.extra = {"no_quest_attempts": 2}

    next_state = bot._handle_get_quest(ctx)
    # After 3 attempts with no quest received, must return MEET_NPC to receive properly
    assert next_state == BotState.MEET_NPC
    assert ctx.extra["no_quest_attempts"] == 0


def test_classify_quest_rejects_empty_shop_item_and_returns_meet_npc():
    bot = BotController.__new__(BotController)
    bot._classifier = MagicMock()
    bot._classifier.classify.return_value = QuestType.SPECIAL_NPC_SHOP
    bot._check_active_commerce_quest = MagicMock(return_value=None)
    bot._checkpoint = MagicMock()

    ctx = StateContext(account_id="ACC_01")
    # quest_text is bogus, no item can be extracted
    ctx.extra = {"quest_text": "Truong Dai Hiu"}

    next_state = bot._handle_classify_quest(ctx)
    # Must NOT proceed to BUY_ITEM with empty item! Must return MEET_NPC
    assert next_state == BotState.MEET_NPC


def test_search_and_select_npc_rejects_chu_tap_hoa():
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._screen = MagicMock()
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "navigation": {
            "map_tab_find_npc": {"x": 376, "y": 161},
            "map_search_input": {"x": 266, "y": 222},
        }
    }
    bot._adb = MagicMock()
    bot._type_unicode_text = MagicMock()
    bot._dismiss_unexpected_popups = MagicMock()

    bot._screen.capture.return_value = "dummy"
    # OCR only sees 'Chu Tap hoa' and 'Chu Duoc diem', not 'Chu Thuong hoi'
    bot._text.read.return_value = [
        TextBox(text="Chu Tap hoa", confidence=0.9, center_x=137, center_y=474),
        TextBox(text="Chu Duoc diem", confidence=0.9, center_x=137, center_y=420),
    ]

    ctx = StateContext(account_id="ACC_01")
    found = bot._search_and_select_npc(ctx, "Chủ thương hội")
    # Must NOT match Chu Tap hoa!
    assert found is False
    # Must NOT tap (137, 474)
    for call_args in bot._input.tap.call_args_list:
        assert call_args[0] != (137, 474)


def test_find_commerce_chief_uses_hardcoded_city_coordinates():
    bot = BotController.__new__(BotController)
    bot._input = MagicMock()
    bot._ensure_map_open = MagicMock()
    bot._close_map = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._check_active_commerce_quest = MagicMock(return_value=None)
    bot._is_near_commerce_chief = MagicMock(return_value=False)
    bot._navigate_to_tuyen_chau_commerce_chief = MagicMock(return_value=True)
    bot._find_dialog_buttons = MagicMock(return_value=[(400, 600)])
    bot._vision = MagicMock()

    ctx = StateContext(account_id="ACC_01")
    ctx.extra = {}

    next_state = bot._handle_find_commerce_chief(ctx)
    assert next_state == BotState.MEET_NPC
    bot._navigate_to_tuyen_chau_commerce_chief.assert_called_once_with(ctx)



