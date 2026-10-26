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
    """ Thanh nhiệm vụ bên trái không có, bấm icon góc phải dưới để mở bảng nhiệm vụ."""
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._screen = MagicMock()
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._find_blue_npc_link = MagicMock(return_value=None)
    bot._is_quest_panel_open = MagicMock(side_effect=[False, True, True, True, True])
    bot._loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "quest_button": {"x": 1403, "y": 755},
            "quest_panel_close": {"x": 1495, "y": 95},
        }
    }

    bot._screen.capture.return_value = "dummy_frame"
    bot._text.read.side_effect = [
        [TextBox(text="Nhiệm vụ", confidence=0.9, center_x=1403, center_y=755)],  # check quest button
        [TextBox(text="Nhiệm vụ Thương hội", confidence=0.9, center_x=250, center_y=300)],  # quest panel list
        [TextBox(text="Mua 1 Bình Định Quốc", confidence=0.9, center_x=600, center_y=300)],  # quest panel detail
    ]

    result = bot._check_active_commerce_quest(full_check=True)
    assert result is not None
    quest_text, click_pos = result
    assert "Bình Định Quốc" in quest_text
    bot._input.tap.assert_any_call(1403, 755)
    bot._input.tap.assert_any_call(1495, 95)


def test_method_3_chief_dialogue_unfinished():
    """ Bấm Hoàn thành nhiệm vụ nhưng Chủ thương hội thông báo chưa hoàn thành -> Đi làm NV."""
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._screen = MagicMock()
    bot._input = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._find_dialog_buttons = MagicMock(return_value=[(400, 600)])
    bot._vision = MagicMock()
    bot._checkpoint = MagicMock()
    bot._is_near_commerce_chief = MagicMock(return_value=True)
    bot._ensure_npc_dialog_closed = MagicMock(return_value=True)

    ctx = StateContext(account_id="ACC_01")
    ctx.total_quests_done = 5
    ctx.extra = {"returning_quest": True}

    bot._text.read.side_effect = [
        [TextBox(text="Hoàn thành nhiệm vụ", confidence=0.9, center_x=400, center_y=600)],
        [TextBox(text="Ngươi chưa hoàn thành nhiệm vụ Thương hội, hãy mau đi làm!", confidence=0.9, center_x=500, center_y=400)],
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


def test_find_commerce_chief_navigates_to_npc_even_if_active_quest_exists():
    bot = BotController.__new__(BotController)
    bot._input = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._is_near_commerce_chief = MagicMock(side_effect=[False, True])
    bot._navigate_to_tuyen_chau_commerce_chief = MagicMock(return_value=True)
    bot._vision = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = "dummy_frame"
    bot._dismiss_unexpected_popups = MagicMock()
    bot._open_commerce_chief_dialog = MagicMock(return_value=True)
    bot._is_npc_dialog_open = MagicMock(return_value=True)
    bot._verify_commerce_chief_dialog = MagicMock(return_value=True)

    ctx = StateContext(account_id="ACC_01")
    ctx.extra = {}

    from unittest.mock import patch
    with patch("src.bot_controller.time.sleep"):
        next_state = bot._handle_find_commerce_chief(ctx)
    assert next_state == BotState.MEET_NPC
    bot._navigate_to_tuyen_chau_commerce_chief.assert_called_once_with(ctx)


def test_quest_completion_banner_is_not_npc_dialog():
    import numpy as np

    bot = BotController.__new__(BotController)
    bot._vision = MagicMock()
    bot._vision.find_template.return_value = MagicMock(found=False)
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox(
            "Cung voi Xa phu doi thoai, hoan thanh nhiem vu",
            0.95,
            815,
            674,
        )
    ]
    frame = np.zeros((900, 1600, 3), dtype=np.uint8)

    assert bot._is_npc_dialog_open(frame) is False


def test_close_dialog_never_clicks_quest_completion_banner(monkeypatch):
    import numpy as np

    bot = BotController.__new__(BotController)
    bot._is_map_open = MagicMock(return_value=False)
    bot._is_npc_dialog_open = MagicMock(side_effect=[True, False])
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox(
            "Cung voi Xa phu doi thoai, hoan thanh nhiem vu",
            0.95,
            815,
            674,
        )
    ]
    bot._input = MagicMock()
    monkeypatch.setattr("src.bot_controller.time.sleep", lambda _seconds: None)

    assert bot._ensure_npc_dialog_closed(max_attempts=1) is True
    bot._input.tap.assert_called_once_with(800, 200)
    assert call(815, 674) not in bot._input.tap.call_args_list


def test_close_dialog_accepts_compact_finish_text(monkeypatch):
    import numpy as np

    bot = BotController.__new__(BotController)
    bot._is_map_open = MagicMock(return_value=False)
    bot._is_npc_dialog_open = MagicMock(side_effect=[True, False])
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox("Nguoi chua hoan thanh nhiem vu ta giao", 0.95, 456, 338),
        TextBox("Ketthucdoi thoai", 0.95, 278, 560),
    ]
    bot._input = MagicMock()
    monkeypatch.setattr("src.bot_controller.time.sleep", lambda _seconds: None)

    assert bot._ensure_npc_dialog_closed(max_attempts=1) is True
    bot._input.tap.assert_called_once_with(278, 560)
    assert call(456, 338) not in bot._input.tap.call_args_list


def test_close_dialog_never_clicks_bottom_center_chat(monkeypatch):
    import numpy as np

    bot = BotController.__new__(BotController)
    bot._is_map_open = MagicMock(return_value=False)
    bot._is_npc_dialog_open = MagicMock(side_effect=[True, False])
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._text = MagicMock()
    # OCR chat có thể bị nhận nhầm thành nút kết thúc hội thoại.
    bot._text.read.return_value = [
        TextBox("Ket thuc doi thoai", 0.95, 800, 780),
    ]
    bot._input = MagicMock()
    monkeypatch.setattr("src.bot_controller.time.sleep", lambda _seconds: None)

    assert bot._ensure_npc_dialog_closed(max_attempts=1) is True
    assert call(800, 780) not in bot._input.tap.call_args_list
    bot._input.tap.assert_called_once_with(800, 200)


def test_close_dialog_never_clicks_bottom_left_chat_that_looks_like_xa_phu(monkeypatch):
    import numpy as np

    bot = BotController.__new__(BotController)
    bot._is_map_open = MagicMock(return_value=False)
    bot._is_npc_dialog_open = MagicMock(side_effect=[True, False])
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((900, 1600, 3), dtype=np.uint8)
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox("Không đi đâu cả", 0.95, 271, 780),
    ]
    bot._input = MagicMock()
    monkeypatch.setattr("src.bot_controller.time.sleep", lambda _seconds: None)

    assert bot._ensure_npc_dialog_closed(max_attempts=1) is True
    assert call(271, 780) not in bot._input.tap.call_args_list
    bot._input.tap.assert_called_once_with(800, 200)


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
    # Tap mở đối thoại (ưu tiên icon bong bóng đối thoại 1220, 299 hoặc 1268, 335)
    assert bot._input.tap.call_args[0] in [(1220, 299), (1268, 335), (800, 350)]
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


def test_dialogue_xa_phu_accepts_any_tuong_duong_gate():
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._screen = MagicMock()
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {"navigation": {}}
    bot._is_map_open = MagicMock(return_value=True)
    bot._ensure_map_open = MagicMock()
    bot._screen.capture.return_value = "dummy"
    bot._text.read.return_value = [
        TextBox(text="Xa phu Nam", confidence=0.95, center_x=150, center_y=385),
        TextBox(text="Xa phu Tay", confidence=0.95, center_x=150, center_y=570),
        TextBox(text="Xa phu Bac", confidence=0.95, center_x=150, center_y=755),
    ]

    ctx = StateContext(account_id="ACC_01")
    found = bot._search_and_select_npc(
        ctx,
        "Xa phu",
        allow_directional_xa_phu=True,
    )

    assert found is True
    # Kết quả hợp lệ đầu tiên đang hiện trên danh sách được chọn ngay.
    bot._input.tap.assert_any_call(150, 385)


def test_search_npc_falls_back_to_first_word_and_picks_closest_name(monkeypatch):
    bot = BotController.__new__(BotController)
    bot._text = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = "dummy"
    bot._input = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {"navigation": {}}
    bot._adb = MagicMock()
    bot._type_unicode_text = MagicMock()
    bot._is_map_open = MagicMock(return_value=True)
    bot._ensure_map_open = MagicMock(return_value=True)
    monkeypatch.setattr("src.bot_controller.time.sleep", lambda _seconds: None)

    # Trang đầu và kết quả tên đầy đủ đều rỗng. Khi tìm "Tiêu",
    # phải chọn "Tiêu sứ" (biến thể OCR gần Tiêu sư) thay vì Tiêu cục.
    bot._text.read.side_effect = [
        [],
        [],
        [
            TextBox("Tiêu cục", 0.95, 150, 410),
            TextBox("Tiêu sứ", 0.91, 150, 465),
        ],
    ]

    ctx = StateContext(account_id="ACC_01")
    assert bot._search_and_select_npc(ctx, "Tiêu sư") is True
    assert bot._type_unicode_text.call_args_list == [call(ctx, "Tiêu sư"), call(ctx, "Tiêu")]
    bot._input.tap.assert_any_call(150, 465)
    assert call(150, 410) not in bot._input.tap.call_args_list


def test_tuong_duong_dialogue_enables_directional_xa_phu_only_for_dialogue():
    bot = BotController.__new__(BotController)
    bot._dismiss_unexpected_popups = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {"navigation": {}}
    bot._input = MagicMock()
    bot._adb = MagicMock()
    bot._type_unicode_text = MagicMock()
    bot._click_map_search_result = MagicMock(return_value=True)
    bot._is_map_open = MagicMock(return_value=True)
    bot._ensure_map_open = MagicMock(return_value=True)
    bot._close_map = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._search_and_select_npc = MagicMock(return_value=True)

    ctx = StateContext(account_id="ACC_01")
    found = bot._navigate_to_dialogue_npc(ctx, "Tuong Duong", "Xa phu")

    assert found is True
    bot._type_unicode_text.assert_called_once_with(ctx, "Tuong Duong")
    bot._click_map_search_result.assert_called_once_with("Tuong Duong")
    bot._search_and_select_npc.assert_called_once_with(
        ctx,
        "Xa phu",
        allow_directional_xa_phu=True,
    )


def test_navigate_clicks_interact_bubble_when_arriving_at_dialogue_npc(monkeypatch):
    bot = BotController.__new__(BotController)
    bot._checkpoint = MagicMock()
    bot._ensure_npc_dialog_closed = MagicMock()
    bot._find_dialog_buttons = MagicMock(return_value=[])
    bot._is_npc_dialog_open = MagicMock(side_effect=[False, True])
    bot._is_npc_interact_bubble_present = MagicMock(return_value=True)
    bot._click_npc_interact_bubble = MagicMock(return_value=True)
    bot._press_buff_speed = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = "frame"
    bot._get_current_location_from_hud = MagicMock(return_value=("Tuong Duong", (180, 180)))
    monkeypatch.setattr("src.bot_controller.time.sleep", lambda _seconds: None)

    ctx = StateContext(account_id="ACC_01")
    ctx.current_quest_type = QuestType.NPC

    assert bot._handle_navigate(ctx) == BotState.INTERACT_NPC
    bot._click_npc_interact_bubble.assert_called_once_with("frame")
    bot._press_buff_speed.assert_not_called()


def test_interact_npc_does_not_finish_when_dialog_never_opens(monkeypatch):
    bot = BotController.__new__(BotController)
    bot._find_dialog_buttons = MagicMock(return_value=[])
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._is_npc_interact_bubble_present = MagicMock(return_value=True)
    bot._click_npc_interact_bubble = MagicMock(return_value=True)
    bot._navigate_to_dialogue_npc = MagicMock(return_value=True)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = "frame"
    bot._get_current_location_from_hud = MagicMock(return_value=("Tuong Duong", (180, 180)))
    monkeypatch.setattr("src.bot_controller.time.sleep", lambda _seconds: None)

    ctx = StateContext(account_id="ACC_01")
    ctx.current_quest_type = QuestType.NPC
    ctx.extra.update({
        "quest_target_city": "Tuong Duong",
        "quest_target_npc": "Xa phu",
    })

    assert bot._handle_interact_npc(ctx) == BotState.NAVIGATE
    assert "returning_quest" not in ctx.extra
    assert bot._click_npc_interact_bubble.call_count == 3
    bot._navigate_to_dialogue_npc.assert_called_once_with(
        ctx, "Tuong Duong", "Xa phu"
    )


def test_dialogue_navigation_rejects_intermediate_xa_phu_in_wrong_city(monkeypatch):
    bot = BotController.__new__(BotController)
    bot._checkpoint = MagicMock()
    bot._ensure_npc_dialog_closed = MagicMock()
    bot._find_dialog_buttons = MagicMock(return_value=[])
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._is_npc_interact_bubble_present = MagicMock(return_value=True)
    bot._click_npc_interact_bubble = MagicMock()
    bot._switch_to_city = MagicMock(return_value=True)
    bot._navigate_to_dialogue_npc = MagicMock(return_value=True)
    bot._press_buff_speed = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = "frame"
    bot._get_current_location_from_hud = MagicMock(
        return_value=("Bien Kinh", (158, 185))
    )
    monkeypatch.setattr("src.bot_controller.time.sleep", lambda _seconds: None)

    ctx = StateContext(account_id="ACC_01")
    ctx.current_quest_type = QuestType.NPC
    ctx.extra.update({
        "quest_target_city": "Tuong Duong",
        "quest_target_npc": "Xa phu",
    })

    assert bot._handle_navigate(ctx) == BotState.NAVIGATE
    bot._click_npc_interact_bubble.assert_not_called()
    bot._switch_to_city.assert_called_once_with(
        ctx,
        "Tuong Duong",
        force=True,
        current_location=("Bien Kinh", None),
    )
    bot._navigate_to_dialogue_npc.assert_called_once_with(
        ctx, "Tuong Duong", "Xa phu"
    )


def test_navigate_to_dialogue_npc_selects_area_then_npc_even_from_another_city(monkeypatch):
    """The selected remote NPC starts the game's cross-map auto-path."""
    bot = BotController.__new__(BotController)
    bot._dismiss_unexpected_popups = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {"navigation": {}}
    bot._input = MagicMock()
    bot._adb = MagicMock()
    bot._type_unicode_text = MagicMock()
    bot._click_map_search_result = MagicMock(return_value=True)
    bot._is_map_open = MagicMock(return_value=True)
    bot._ensure_map_open = MagicMock(return_value=True)
    bot._search_and_select_npc = MagicMock(return_value=True)
    bot._close_map = MagicMock()
    bot._press_buff_speed = MagicMock()
    monkeypatch.setattr("src.bot_controller.time.sleep", lambda _seconds: None)

    ctx = StateContext(account_id="ACC_03")
    assert bot._navigate_to_dialogue_npc(ctx, "Duong Chau", "Chu Kim hoan") is True
    bot._type_unicode_text.assert_called_once_with(ctx, "Duong Chau")
    bot._click_map_search_result.assert_called_once_with("Duong Chau")
    bot._search_and_select_npc.assert_called_once_with(
        ctx, "Chu Kim hoan", allow_directional_xa_phu=False
    )


def test_interact_npc_rejects_wrong_city_before_marking_complete(monkeypatch):
    bot = BotController.__new__(BotController)
    bot._get_current_location_from_hud = MagicMock(
        return_value=("Bien Kinh", (158, 185))
    )
    bot._ensure_npc_dialog_closed = MagicMock()
    bot._navigate_to_dialogue_npc = MagicMock(return_value=True)
    monkeypatch.setattr("src.bot_controller.time.sleep", lambda _seconds: None)

    ctx = StateContext(account_id="ACC_01")
    ctx.current_quest_type = QuestType.NPC
    ctx.extra.update({
        "quest_target_city": "Tuong Duong",
        "quest_target_npc": "Xa phu",
    })

    assert bot._handle_interact_npc(ctx) == BotState.NAVIGATE
    assert "returning_quest" not in ctx.extra
    bot._navigate_to_dialogue_npc.assert_called_once_with(
        ctx, "Tuong Duong", "Xa phu"
    )


def test_dialogue_quest_clicks_only_finish_then_returns(monkeypatch):
    bot = BotController.__new__(BotController)
    bot._get_current_location_from_hud = MagicMock(
        return_value=("Tuong Duong", (180, 180))
    )
    bot._find_dialog_buttons = MagicMock(return_value=[(400, 600)])
    bot._find_finish_dialogue_button = MagicMock(return_value=(309, 543))
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = "frame"
    bot._input = MagicMock()
    monkeypatch.setattr("src.bot_controller.time.sleep", lambda _seconds: None)

    ctx = StateContext(account_id="ACC_01")
    ctx.current_quest_type = QuestType.NPC
    ctx.extra.update({
        "quest_target_city": "Tuong Duong",
        "quest_target_npc": "Xa phu",
    })

    assert bot._handle_interact_npc(ctx) == BotState.RETURN_QUEST
    bot._input.tap.assert_called_once_with(309, 543)
    assert ctx.extra["returning_quest"] is True


def test_four_refugee_quest_visits_all_npcs_before_returning(monkeypatch):
    bot = BotController.__new__(BotController)
    bot._get_current_location_from_hud = MagicMock(return_value=("Thành Đô", (180, 180)))
    bot._find_dialog_buttons = MagicMock(return_value=[(400, 600)])
    bot._find_finish_dialogue_button = MagicMock(return_value=(309, 543))
    bot._is_npc_dialog_open = MagicMock(return_value=False)
    bot._navigate_to_dialogue_npc = MagicMock(return_value=True)
    bot._checkpoint = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = "frame"
    bot._input = MagicMock()
    monkeypatch.setattr("src.bot_controller.time.sleep", lambda _seconds: None)

    ctx = StateContext(account_id="ACC_01")
    ctx.current_quest_type = QuestType.NPC
    ctx.extra.update({
        "quest_target_city": "Thành Đô",
        "quest_target_npc": "Nạn dân Giáp",
        "multi_npc_route": list(BotController.FOUR_REFUGEE_NPCS),
        "multi_npc_index": 0,
    })

    for expected_index, expected_npc in enumerate(
        ("Nạn dân Ất", "Nạn dân Bính", "Nạn dân Đinh"), start=1
    ):
        assert bot._handle_interact_npc(ctx) == BotState.NAVIGATE
        assert ctx.extra["multi_npc_index"] == expected_index
        assert ctx.extra["quest_target_npc"] == expected_npc
        assert "returning_quest" not in ctx.extra
        # NAVIGATE consumes this crash-recovery marker before arriving.
        ctx.extra.pop("multi_npc_needs_navigation", None)

    assert bot._handle_interact_npc(ctx) == BotState.RETURN_QUEST
    assert ctx.extra["multi_npc_completed"] is True
    assert ctx.extra["returning_quest"] is True
    assert bot._navigate_to_dialogue_npc.call_count == 3


def test_detects_four_refugee_quest_text():
    text = (
        "Đến 4 cửa thành môn Thành Đô gặp nạn dân Giáp, Ất, Bính, Đinh, "
        "báo họ đến nhận lương thực phân phát."
    )
    assert BotController._is_four_refugee_quest(text) is True


def test_dialogue_quest_uses_safe_visual_button_when_finish_text_is_missed(monkeypatch):
    bot = BotController.__new__(BotController)
    bot._get_current_location_from_hud = MagicMock(
        return_value=("Tuong Duong", (180, 180))
    )
    bot._find_dialog_buttons = MagicMock(return_value=[(400, 600)])
    bot._find_finish_dialogue_button = MagicMock(return_value=None)
    bot._is_npc_dialog_open = MagicMock(return_value=True)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = "frame"
    bot._input = MagicMock()
    monkeypatch.setattr("src.bot_controller.time.sleep", lambda _seconds: None)

    ctx = StateContext(account_id="ACC_01")
    ctx.current_quest_type = QuestType.NPC
    ctx.extra.update({
        "quest_target_city": "Tuong Duong",
        "quest_target_npc": "Xa phu",
    })

    assert bot._handle_interact_npc(ctx) == BotState.INTERACT_NPC
    bot._input.tap.assert_called_once_with(400, 600)
    assert ctx.extra["npc_finish_state_failures"] == 1
    assert "returning_quest" not in ctx.extra

    # Vòng thứ hai không được lặp vô hạn: đóng hộp thoại và quay về Chủ
    # thương hội để xác nhận trạng thái nhiệm vụ thật.
    bot._ensure_npc_dialog_closed = MagicMock(return_value=True)
    assert bot._handle_interact_npc(ctx) == BotState.RETURN_QUEST
    bot._ensure_npc_dialog_closed.assert_called_once()
    assert ctx.extra["returning_quest"] is True


def test_find_commerce_chief_uses_hardcoded_city_coordinates():
    bot = BotController.__new__(BotController)
    bot._input = MagicMock()
    bot._ensure_map_open = MagicMock()
    bot._close_map = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._check_active_commerce_quest = MagicMock(return_value=None)
    bot._is_near_commerce_chief = MagicMock(side_effect=[False, True, True, True])
    bot._is_npc_dialog_open = MagicMock(return_value=True)
    bot._verify_commerce_chief_dialog = MagicMock(return_value=True)
    bot._navigate_to_tuyen_chau_commerce_chief = MagicMock(return_value=True)
    bot._vision = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = "dummy_frame"
    bot._dismiss_unexpected_popups = MagicMock()

    ctx = StateContext(account_id="ACC_01")
    ctx.extra = {}

    from unittest.mock import patch
    with patch("src.bot_controller.time.sleep"):
        next_state = bot._handle_find_commerce_chief(ctx)
    assert next_state == BotState.MEET_NPC
    bot._navigate_to_tuyen_chau_commerce_chief.assert_called_once_with(ctx)
