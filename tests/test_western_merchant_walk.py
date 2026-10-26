from unittest.mock import ANY, MagicMock, call, patch
from src.bot_controller import BotController
from src.state_machine import StateContext
import numpy as np


def test_match_known_city():
    assert BotController._match_known_city("TupenChau") == "Tuyền Châu"
    assert BotController._match_known_city("Thanh Do") == "Thành Đô"
    assert BotController._match_known_city("Bien Kinh") == "Biện Kinh"
    assert BotController._match_known_city("Xa phu") is None


def test_switch_to_city_skips_if_already_there():
    bot = BotController.__new__(BotController)
    bot._get_current_location_from_hud = MagicMock(return_value=("Tuyền Châu", (192, 182)))
    bot._ensure_map_open = MagicMock()
    bot._is_map_open = MagicMock(return_value=True)
    bot._input = MagicMock()

    ctx = StateContext(account_id="ACC_01")
    res = bot._switch_to_city(ctx, "Tuyền Châu")
    assert res is True
    bot._ensure_map_open.assert_not_called()


def test_switch_to_city_skips_with_fuzzy_ocr_typo():
    bot = BotController.__new__(BotController)
    # OCR reads 'TupenChau' instead of 'Tuyền Châu'
    bot._get_current_location_from_hud = MagicMock(return_value=("TupenChau", (192, 182)))
    bot._ensure_map_open = MagicMock()
    bot._input = MagicMock()

    ctx = StateContext(account_id="ACC_01")
    res = bot._switch_to_city(ctx, "Tuyền Châu")
    assert res is True
    bot._ensure_map_open.assert_not_called()


def test_switch_to_city_navigates_when_different():
    bot = BotController.__new__(BotController)
    bot._try_teleport_with_than_hanh_phu = MagicMock(return_value=False)
    has_tapped = False

    def mock_hud(frame=None):
        nonlocal has_tapped
        if bot._input.tap.called:
            return ("Biện Kinh", (179, 184))
        return ("Tuyền Châu", (192, 182))

    bot._get_current_location_from_hud = MagicMock(side_effect=mock_hud)
    bot._dismiss_unexpected_popups = MagicMock(return_value=False)
    bot._ensure_npc_dialog_closed = MagicMock()
    bot._ensure_map_open = MagicMock()
    bot._is_map_open = MagicMock(return_value=False)
    bot._close_map = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._input = MagicMock()
    bot._adb = MagicMock()
    bot._screen = MagicMock()
    from src.text_recognition import TextBox
    bot._text = MagicMock()
    bot._text.read.return_value = [TextBox("Bien Kinh", 0.9, 342, 562), TextBox("le phi 3 luong", 0.9, 500, 500)]
    bot._vision = MagicMock()
    bot._is_npc_dialog_open = MagicMock(return_value=True)
    bot._is_npc_interact_bubble_present = MagicMock(return_value=False)
    bot._check_xa_phu_out_of_money = MagicMock(return_value=False)
    bot._find_dialog_buttons = MagicMock(return_value=[(500, 500)])
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "navigation": {
            "map_tab_find_npc": {"x": 376, "y": 161},
            "map_search_input": {"x": 266, "y": 222},
        },
        "commerce_quest": {
            "xa_phu_cities": {
                "Biện Kinh": {"x": 342, "y": 562},
            }
        }
    }
    bot._input_search_text = MagicMock()
    bot._click_npc_search_result = MagicMock(return_value=True)
    bot._search_tuyen_chau_xa_phu = MagicMock(return_value=True)

    ctx = StateContext(account_id="ACC_01")
    res = bot._switch_to_city(ctx, "Biện Kinh")
    assert res is True
    bot._ensure_map_open.assert_called_once()
    bot._search_tuyen_chau_xa_phu.assert_called_once_with(ctx)
    bot._click_npc_search_result.assert_not_called()


def test_tuyen_chau_minimap_uses_only_north_xa_phu():
    bot = BotController.__new__(BotController)
    bot._search_and_select_npc = MagicMock(return_value=True)
    ctx = StateContext(account_id="ACC_03")

    assert bot._search_tuyen_chau_xa_phu(ctx) is True
    bot._search_and_select_npc.assert_called_once_with(ctx, "Xa phu Bắc")


def test_go_to_city_coordinate_inputs_exact_coords():
    bot = BotController.__new__(BotController)
    bot._dismiss_unexpected_popups = MagicMock()
    bot._switch_to_city = MagicMock(return_value=True)
    bot._ensure_map_open = MagicMock()
    bot._is_map_open = MagicMock(return_value=False)
    bot._close_map = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._input = MagicMock()
    bot._adb = MagicMock()
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "map_coordinate_x": {"x": 704, "y": 778},
            "map_coordinate_y": {"x": 842, "y": 778},
            "map_coordinate_confirm": {"x": 1015, "y": 778},
        }
    }

    ctx = StateContext(account_id="ACC_01")
    res = bot._go_to_city_coordinate(ctx, "Tuyền Châu", 187, 191)
    assert res is True
    bot._switch_to_city.assert_called_once_with(ctx, "Tuyền Châu", force=False)
    # Verify taps X and Y input boxes and confirm
    bot._input.tap.assert_any_call(704, 778)
    bot._input.tap.assert_any_call(842, 778)
    bot._input.tap.assert_any_call(1015, 778)
    assert bot._adb._run.call_count >= 2
    bot._press_buff_speed.assert_called_once()


def test_find_western_merchant_walks_and_finds():
    bot = BotController.__new__(BotController)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "western_merchant_spawns": [
                {"city": "Tuyền Châu", "x": 187, "y": 191}
            ]
        }
    }
    bot._go_to_city_coordinate = MagicMock(return_value=True)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bot._text = MagicMock()
    from src.text_recognition import TextBox
    # Found merchant immediately
    merchant_box = TextBox("Thương nhân Tây Vực", 0.95, 750, 420)
    bot._text.read.return_value = [merchant_box]
    bot._text.best_match.return_value = (0.95, merchant_box, "Thương nhân Tây Vực")
    bot._get_current_location_from_hud = MagicMock(return_value=("Tuyền Châu", (187, 191)))
    bot._input = MagicMock()
    bot._checkpoint = MagicMock()

    ctx = StateContext(account_id="ACC_01")
    res = bot._find_western_merchant(ctx)
    assert res is True
    assert ctx.extra.get("merchant_found_city") == "Tuyền Châu"
    bot._input.tap.assert_called_once_with(750, 455)


def test_find_western_merchant_stops_rotating_when_city_switch_fails():
    bot = BotController.__new__(BotController)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "western_merchant_spawns": [
                {"city": "Tuyền Châu", "x": 187, "y": 191},
                {"city": "Biện Kinh", "x": 180, "y": 185},
                {"city": "Thành Đô", "x": 225, "y": 225},
            ]
        }
    }
    bot._get_current_location_from_hud = MagicMock(
        return_value=("Tuyền Châu", (194, 202))
    )
    bot._go_to_city_coordinate = MagicMock(return_value=False)

    result = bot._find_western_merchant(StateContext(account_id="ACC_03"))

    assert result is False
    # A failed transfer must not produce fake attempts for every later city.
    bot._go_to_city_coordinate.assert_called_once_with(
        ANY, "Tuyền Châu", 187, 191
    )


def test_handle_buy_item_clicks_dong_y_mua_and_confirms():
    from src.state_machine import BotState
    from src.text_recognition import TextBox

    bot = BotController.__new__(BotController)
    bot._app_cfg = MagicMock()
    bot._app_cfg.mode.dry_run = False
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "shop": {
                "dialog_buy_option": {"x": 272, "y": 566},
                "item_slot_linh_duoc": {"x": 397, "y": 190},
                "western_merchant_items": {
                    "Linh Dược Tây Vực": {"x": 397, "y": 190, "row": 1, "col": 2}
                },
            "buy_button": {"x": 538, "y": 485},
            "confirm_button": {"x": 465, "y": 786},
            "close_button": {"x": 702, "y": 128},
            "inventory_close": {"x": 1577, "y": 81},
        }
    }
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox(text="Mua", confidence=0.9, center_x=538, center_y=485),
        TextBox(text="Đồng ý mua", confidence=0.9, center_x=272, center_y=566),
        TextBox(text="Tạp hóa", confidence=0.9, center_x=500, center_y=100),
        TextBox(text="Số lượng mua", confidence=0.9, center_x=500, center_y=600),
        TextBox(text="Đồng ý", confidence=0.9, center_x=465, center_y=786),
    ]
    bot._text.best_match.return_value = None
    bot._vision = MagicMock()
    bot._vision.template_exists.return_value = False
    bot._input = MagicMock()
    bot._adb = MagicMock()
    bot._dismiss_unexpected_popups = MagicMock()

    ctx = StateContext(account_id="ACC_01")
    ctx.extra["requested_shop_item"] = "Linh dược Tây Vực"

    next_state = bot._handle_buy_item(ctx)

    assert next_state == BotState.RETURN_QUEST
    assert ctx.extra.get("returning_quest") is True
    # Verify exact taps:
    # 1. Dialog 'Đồng ý mua' (272, 566)
    # 2. Item slot 2 (397, 190)
    # 3. Buy button (538, 485)
    # 4. Confirm 'Đồng ý' (465, 786)
    # 5. Close X (702, 128)
    bot._input.tap.assert_any_call(272, 566)
    bot._input.tap.assert_any_call(397, 190)
    bot._input.tap.assert_any_call(538, 485)
    bot._input.tap.assert_any_call(465, 786)
    bot._input.tap.assert_any_call(702, 128)


def test_handle_buy_item_uses_fixed_item_position_without_scanning_slots():
    from src.state_machine import BotState
    from src.text_recognition import TextBox

    bot = BotController.__new__(BotController)
    bot._app_cfg = MagicMock()
    bot._app_cfg.mode.dry_run = False
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "shop": {
            "dialog_buy_option": {"x": 272, "y": 566},
            "buy_button": {"x": 538, "y": 485},
            "confirm_button": {"x": 465, "y": 786},
            "close_button": {"x": 702, "y": 128},
                "grid_slots": [
                {"x": 316, "y": 190},  # Slot 1
                {"x": 397, "y": 190},  # Slot 2 (Target)
                {"x": 486, "y": 190},  # Slot 3
                ],
                "western_merchant_items": {
                    "Linh Dược Tây Vực": {"x": 397, "y": 190, "row": 1, "col": 2}
                },
            "tooltip_title_region": {"x1": 700, "y1": 30, "x2": 1100, "y2": 150},
        }
    }
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    bot._text = MagicMock()
    bot._text.read.return_value = [
        TextBox(text="Mua", confidence=0.9, center_x=538, center_y=485),
        TextBox(text="Đồng ý mua", confidence=0.9, center_x=272, center_y=566),
        TextBox(text="Tạp hóa", confidence=0.9, center_x=500, center_y=100),
        TextBox(text="Số lượng mua", confidence=0.9, center_x=500, center_y=600),
        TextBox(text="Đồng ý", confidence=0.9, center_x=465, center_y=786),
    ]
    bot._text.best_match.return_value = None
    bot._vision = MagicMock()
    bot._vision.template_exists.return_value = False
    bot._input = MagicMock()
    bot._adb = MagicMock()
    bot._dismiss_unexpected_popups = MagicMock()

    ctx = StateContext(account_id="ACC_01")
    ctx.extra["requested_shop_item"] = "Linh dược Tây Vực"

    next_state = bot._handle_buy_item(ctx)

    assert next_state == BotState.RETURN_QUEST
    # Click the configured slot directly; never inspect neighboring slots.
    bot._input.tap.assert_any_call(397, 190)
    tapped_coords = [call[0] for call in bot._input.tap.call_args_list]
    assert (316, 190) not in tapped_coords
    assert (486, 190) not in tapped_coords
    # Assert "Mua" was clicked
    bot._input.tap.assert_any_call(538, 485)
    # Assert "Đồng ý" was clicked
    bot._input.tap.assert_any_call(465, 786)


def test_walk_intercity_uses_visible_current_xa_phu_before_destination_map():
    bot = BotController.__new__(BotController)
    bot._get_current_location_from_hud = MagicMock(
        return_value=("Tương Dương", (168, 188))
    )
    bot._screen = MagicMock()
    bot._screen.capture.return_value = object()
    bot._is_standing_next_to_xa_phu = MagicMock(return_value=True)
    bot._switch_to_city = MagicMock(return_value=True)
    bot._trigger_intercity_autopath = MagicMock()
    bot._checkpoint = MagicMock()

    ctx = StateContext(account_id="ACC_03")

    assert bot._walk_intercity_via_world_map(ctx, "Dương Châu") is True
    bot._switch_to_city.assert_called_once_with(
        ctx,
        "Dương Châu",
        force=True,
        current_location=("Tương Dương", (168, 188)),
    )
    bot._trigger_intercity_autopath.assert_not_called()


def test_walk_intercity_when_out_of_money_bypasses_xa_phu():
    bot = BotController.__new__(BotController)
    bot._get_current_location_from_hud = MagicMock(
        side_effect=[
            ("Tương Dương", (168, 188)),
            ("Dương Châu", (200, 200)),
            ("Dương Châu", (200, 200)),
            ("Dương Châu", (200, 200)),
            ("Dương Châu", (200, 200)),
        ]
    )
    bot._screen = MagicMock()
    bot._screen.capture.return_value = object()
    bot._is_standing_next_to_xa_phu = MagicMock(return_value=True)
    bot._switch_to_city = MagicMock()
    bot._trigger_intercity_autopath = MagicMock(return_value=True)
    bot._checkpoint = MagicMock()

    ctx = StateContext(account_id="ACC_03")
    ctx.extra["out_of_money"] = True

    with patch("src.bot_controller.time.time", side_effect=[100.0, 100.0, 100.0, 105.0, 105.0, 105.0, 105.0, 105.0]):
        with patch("src.bot_controller.time.sleep"):
            assert bot._walk_intercity_via_world_map(ctx, "Dương Châu", max_wait=30.0, retry_local_xa_phu=False) is True
    # Must NOT call _switch_to_city, must keep out_of_money flag!
    bot._switch_to_city.assert_not_called()
    assert ctx.extra.get("out_of_money") is True
    bot._trigger_intercity_autopath.assert_called_once_with(ctx, "Dương Châu")


def test_trigger_intercity_autopath_via_world_map():
    bot = BotController.__new__(BotController)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "navigation": {
            "map_tab_find_map": {"x": 172, "y": 175},
            "map_tab_find_npc": {"x": 372, "y": 175},
            "map_search_input": {"x": 266, "y": 222},
            "map_confirm_button": {"x": 1015, "y": 778},
            "map_npc_search_result": {"x": 200, "y": 285},
            "world_map_cities": {
                "Biện Kinh": {"x": 1154, "y": 337},
                "Thành Đô": {"x": 825, "y": 461},
            },
        }
    }
    bot._ensure_map_open = MagicMock()
    bot._is_map_open = MagicMock(return_value=True)
    bot._close_map = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._input = MagicMock()
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bot._text = MagicMock()
    bot._text.read.return_value = []
    bot._adb = MagicMock()
    bot._type_unicode_text = MagicMock()
    bot._click_map_search_result = MagicMock(return_value=True)
    bot._click_npc_search_result = MagicMock(return_value=True)
    bot._get_current_location_from_hud = MagicMock(return_value=("Tuyền Châu", (188, 179)))

    ctx = StateContext(account_id="ACC_01")
    res = bot._trigger_intercity_autopath(ctx, "Biện Kinh")

    assert res is True
    bot._ensure_map_open.assert_called_once()
    # 1. Tìm và chọn đúng map đích.
    bot._input.tap.assert_any_call(172, 175)
    bot._input.tap.assert_any_call(266, 222)
    bot._type_unicode_text.assert_called_once_with(ctx, "Biện Kinh")
    bot._click_map_search_result.assert_called_once_with("Biện Kinh")
    # 2. Không bấm Xác nhận: nút này đóng minimap và làm các click sau rơi lên màn hình game.
    assert call(1015, 778) not in bot._input.tap.call_args_list
    # 3. Khi minimap vẫn mở mới bấm sang tab Tìm NPC.
    bot._input.tap.assert_any_call(372, 175)
    bot._click_npc_search_result.assert_called_once_with("Xa phu")
    bot._close_map.assert_called_once()
    bot._press_buff_speed.assert_called_once()


def test_trigger_intercity_autopath_confirms_selected_map_when_xa_phu_missing():
    """A town such as Nam Thanh Do has no Xa Phu; confirm the selected map instead."""
    bot = BotController.__new__(BotController)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "navigation": {
            "map_tab_find_map": {"x": 172, "y": 175},
            "map_tab_find_npc": {"x": 372, "y": 175},
            "map_search_input": {"x": 266, "y": 222},
            "map_confirm_button": {"x": 1015, "y": 778},
        }
    }
    bot._ensure_map_open = MagicMock()
    # Open after choosing the search result, closed after pressing confirm.
    bot._is_map_open = MagicMock(side_effect=[True, False])
    bot._close_map = MagicMock()
    bot._press_buff_speed = MagicMock()
    bot._input = MagicMock()
    bot._adb = MagicMock()
    bot._type_unicode_text = MagicMock()
    bot._click_map_search_result = MagicMock(return_value=True)
    bot._click_npc_search_result = MagicMock(return_value=False)
    bot._get_current_location_from_hud = MagicMock(
        return_value=("Nam Thanh Do", (195, 181))
    )

    with patch("src.bot_controller.time.sleep"):
        result = bot._trigger_intercity_autopath(
            StateContext(account_id="ACC_03"), "Biện Kinh"
        )

    assert result is True
    bot._input.tap.assert_any_call(1015, 778)
    bot._close_map.assert_not_called()
    bot._press_buff_speed.assert_called_once()


def test_walk_intercity_does_not_wait_when_autopath_failed():
    bot = BotController.__new__(BotController)
    bot._trigger_intercity_autopath = MagicMock(return_value=False)
    bot._get_current_location_from_hud = MagicMock()

    result = bot._walk_intercity_via_world_map(
        StateContext(account_id="ACC_03"), "Biện Kinh", max_wait=240.0
    )

    assert result is False
    # One initial read is allowed to detect whether the character is already
    # standing beside the current city's Xa phu; no wait loop is entered.
    bot._get_current_location_from_hud.assert_called_once()


def test_tooltip_extracts_item_title_ignoring_type_and_noise():
    from src.text_recognition import TextBox, normalize_game_text
    from src.bot_controller import BotController

    bot = BotController.__new__(BotController)
    ignored_tags = (
        "loai", "lual", "loal", "lua", "nang", "gia", "da khoa", "dakhoa", "khoa",
        "ban tiem", "giao dich", "vut bo", "anh huong", "anhhuong", "suc khoe", "suckhoe",
        "choi qua", "180", "phut", "ngay", "thong bao", "he thong", "chu y", "luu y",
        "hanh trang", "sap xep", "thong tin", "tach", "bay ban", "x"
    )

    # Simulated OCR boxes as observed in the real game tooltip:
    # Title: "LongDaiBang"
    # Labels and attributes: "ai:", "500", "Ban tiem, Giao dich, Vut bo"
    boxes = [
        TextBox("ai:", 0.95, 875, 85),
        TextBox("500", 1.00, 991, 106),
        TextBox("LongDaiBang", 0.98, 977, 51),
        TextBox("Ban tiem, Giao dich, Vut bo", 0.91, 966, 130),
    ]

    matched_box = None
    slot_best_score = -1.0
    slot_best_text = ""

    for b in boxes:
        b_norm = normalize_game_text(b.text)
        if any(tag in b_norm for tag in ignored_tags) or b_norm in ignored_tags or b_norm.isdigit():
            continue
        if len(b_norm) < 2:
            continue

        is_match, score = bot._is_matching_shop_item(b.text, "long dai bang")
        if score > slot_best_score:
            slot_best_score = score
            slot_best_text = b.text
        if is_match:
            matched_box = b
            break

    assert matched_box is not None
    assert matched_box.text == "LongDaiBang"
    assert slot_best_score >= 0.85
    # Ensure '500' or 'ai:' is NEVER chosen as matched item
    assert slot_best_text != "500"
    assert slot_best_text != "ai:"


def test_detect_merchant_with_merged_tokens_at_destination():
    from src.text_recognition import TextBox
    bot = BotController.__new__(BotController)
    bot._loader = MagicMock()
    bot._loader.load_ui_coords.return_value = {
        "commerce_quest": {
            "western_merchant_spawns": [
                {"city": "Đại Lý", "x": 178, "y": 187},
            ]
        }
    }
    bot._go_to_city_coordinate = MagicMock(return_value=True)
    bot._screen = MagicMock()
    bot._screen.capture.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bot._text = MagicMock()
    # Simulating OCR output from screenshot: 'ThuongnhanTay Vuc'
    merchant_box = TextBox("ThuongnhanTay Vuc", 0.95, 719, 280)
    bot._text.read.return_value = [merchant_box]
    bot._text.best_match.return_value = None
    bot._get_current_location_from_hud = MagicMock(return_value=("Đại Lý", (178, 188)))
    bot._input = MagicMock()
    bot._checkpoint = MagicMock()
    bot._is_npc_interact_bubble_present = MagicMock(return_value=False)

    ctx = StateContext(account_id="ACC_04")
    with patch("src.bot_controller.time.sleep"):
        res = bot._find_western_merchant(ctx)

    assert res is True
    assert ctx.extra.get("merchant_found_city") == "Đại Lý"
    bot._input.tap.assert_called_with(719, 315)

